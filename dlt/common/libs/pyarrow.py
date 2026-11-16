import base64
import gzip
import uuid
from datetime import datetime, time, timedelta, timezone  # noqa: I251
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Mapping,
    NamedTuple,
    Tuple,
    Optional,
    Union,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    List,
)

from dlt import version
from dlt.common.exceptions import MissingDependencyException, DltException
from dlt.common.schema.typing import C_DLT_ID, C_DLT_LOAD_ID, TColumnSchema, TTableSchemaColumns
from dlt.common import logger
from dlt.common.json import json, custom_encode, map_nested_values_in_place
from dlt.common.destination.capabilities import DestinationCapabilitiesContext
from dlt.common.schema.typing import TColumnType
from dlt.common.schema.utils import (
    get_time_fidelity,
    is_nullable_column,
    dlt_load_id_column,
    time_fidelity_companion_columns,
)
from dlt.common.time import (
    C_UTC_OFFSET_SUFFIX,
    HINT_COMPANION_OF,
    UTC_NAME,
    InvalidLocalTime,
    format_utc_offset,
    get_context_timezone_name,
    get_precision_from_datetime_unit,
    resolve_local_time,
    to_tzinfo,
)
from dlt.common.typing import AnyType, StrStr, TFileOrPath, TDataItems
from dlt.common.normalizers.naming import NamingConvention

try:
    import pyarrow
    import pyarrow.parquet
    import pyarrow.compute
    from pyarrow.parquet import ParquetFile
    from pyarrow import Table
except ModuleNotFoundError:
    raise MissingDependencyException(
        "dlt pyarrow helpers",
        [f"{version.DLT_PKG_NAME}[parquet]"],
        "Install pyarrow to be allowed to load arrow tables, pandas DataFrames and to use parquet"
        " files.",
    )

import ctypes

TAnyArrowItem = Union[pyarrow.Table, pyarrow.RecordBatch]

ARROW_DECIMAL_MAX_PRECISION = 76
ARROW_UUID_EXTENSION_NAME = "arrow.uuid"


class UnsupportedArrowTypeException(DltException):
    """Exception raised when Arrow type conversion failed.

    The setters are used to update the exception with more context
    such as the relevant field and tablea it is caught downstream.
    """

    def __init__(
        self,
        arrow_type: pyarrow.DataType,
        field_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        self.arrow_type = arrow_type
        self._field_name = field_name if field_name else ""
        self._table_name = table_name if table_name else ""

        msg = self.generate_message(self.arrow_type, self._field_name, self._table_name)
        super().__init__(msg)

    @staticmethod
    def generate_message(arrow_type: pyarrow.DataType, field_name: str, table_name: str) -> str:
        msg = f"Arrow type `{arrow_type}`"
        if field_name:
            msg += f" for field `{field_name}`"
        if table_name:
            msg += f" in table `{table_name}`"

        msg += (
            " is unsupported by dlt. See documentation:"
            " https://dlthub.com/docs/dlt-ecosystem/verified-sources/arrow-pandas#supported-arrow-data-types"
        )
        return msg

    def _update_message(self) -> None:
        """Modify the `Exception.args` tuple to update message."""
        msg = self.generate_message(self.arrow_type, self.field_name, self.table_name)
        self.args = (msg,)  # must be a tuple

    @property
    def field_name(self) -> str:
        return self._field_name

    @field_name.setter
    def field_name(self, value: str) -> None:
        self._field_name = value
        self._update_message()

    @property
    def table_name(self) -> str:
        return self._table_name

    @table_name.setter
    def table_name(self, value: str) -> None:
        self._table_name = value
        self._update_message()


class InvalidTimezoneException(DltException):
    """Exception raised when arrow cannot resolve a timezone name."""

    def __init__(self, tz: str, arrow_message: str) -> None:
        self.tz = tz
        super().__init__(
            f"dlt cannot resolve timezone `{tz}`: {arrow_message}. Use a canonical IANA name, for"
            " example `Europe/Berlin`. On Windows install `tzdata`, arrow ships no timezone"
            " database: https://arrow.apache.org/docs/cpp/build_system.html#runtime-dependencies"
        )


class PyToArrowConversionException(DltException):
    """Exception raised when converting data to Arrow based on a TableSchema"""

    def __init__(
        self,
        data_type: Optional[str],
        inferred_arrow_type: Optional[pyarrow.DataType] = None,
        field_name: Optional[str] = None,
        table_name: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        self.data_type = data_type
        self.inferred_arrow_type = inferred_arrow_type
        self._field_name = field_name if field_name else ""
        self._table_name = table_name if table_name else ""
        self._details = details if details else ""

        super().__init__()
        self._update_message()

    @staticmethod
    def generate_message(
        data_type: Optional[str],
        inferred_arrow_type: Optional[pyarrow.DataType],
        field_name: str,
        table_name: str,
        details: str,
    ) -> str:
        msg = "Conversion to arrow failed"
        if field_name:
            msg += f" for field `{field_name}`"
        if table_name:
            msg += f" in table `{table_name}`"

        msg += f" with dlt hint `{data_type=:}` and `{inferred_arrow_type=:}`"
        msg += " " + details
        return msg

    def _update_message(self) -> None:
        """Modify the `Exception.args` tuple to update message."""
        msg = self.generate_message(
            self.data_type,
            self.inferred_arrow_type,
            self.field_name,
            self.table_name,
            self._details,
        )
        self.args = (msg,)  # must be a tuple

    @property
    def field_name(self) -> str:
        return self._field_name

    @field_name.setter
    def field_name(self, value: str) -> None:
        self._field_name = value
        self._update_message()

    @property
    def table_name(self) -> str:
        return self._table_name

    @table_name.setter
    def table_name(self, value: str) -> None:
        self._table_name = value
        self._update_message()

    @property
    def details(self) -> str:
        return self._details

    @details.setter
    def details(self, value: str) -> None:
        self._details = value
        self._update_message()


class ArrowSchemaNormalizationResult(NamedTuple):
    """Named result of should_normalize_arrow_schema

    Fields:
    - should_normalize: whether any normalization is required
    - rename_mapping: mapping from original field names to normalized names
    - rev_mapping: reverse mapping from normalized names to original
    - nullable_updates: fields that require nullable flag updates (by normalized name)
    - columns: potentially filtered/adjusted TTableSchemaColumns
    """

    should_normalize: bool
    rename_mapping: StrStr
    rev_mapping: Dict[str, str]
    nullable_updates: Dict[str, bool]
    columns: TTableSchemaColumns


def get_column_timezone(
    column: TColumnType, tz: str, caps: DestinationCapabilitiesContext = None
) -> Optional[str]:
    """Timezone a timestamp column is stored in, `None` when the column is stored naive."""
    store_aware = column.get("timezone", True)
    if caps is not None:
        # a destination without tz support holds the context wall clock, like a `timezone=False`
        # column; one without naive support holds the instant, like a `timezone=True` column
        if not caps.supports_tz_aware_datetime:
            store_aware = False
        elif not caps.supports_naive_datetime:
            store_aware = True
    return tz if store_aware else None


def get_py_arrow_datatype(
    column: TColumnType,
    caps: DestinationCapabilitiesContext,
    tz: str,
) -> Any:
    column_type = column["data_type"]
    if column_type == "text":
        return pyarrow.string()
    elif column_type == "double":
        return pyarrow.float64()
    elif column_type == "bool":
        return pyarrow.bool_()
    elif column_type == "timestamp":
        timezone = get_column_timezone(column, tz, caps)
        precision = column.get("precision")
        if precision is None:
            precision = caps.timestamp_precision
        return get_py_arrow_timestamp(precision, timezone)
    elif column_type == "bigint":
        return get_pyarrow_int(column.get("precision"))
    elif column_type == "binary":
        return pyarrow.binary(column.get("precision") or -1)
    elif column_type == "json":
        if (nested_type := column.get("x-nested-type")) and caps.supports_nested_types:
            return deserialize_type(nested_type)  # type: ignore[arg-type]
        else:
            return pyarrow.string()
    elif column_type == "decimal":
        precision, scale = column.get("precision"), column.get("scale")
        if (precision is None) and (scale is None):
            precision_tuple = caps.decimal_precision
        elif precision is None:
            precision_tuple = caps.decimal_precision
            logger.warning(
                f"Received decimal column hint `scale={scale}`, but `precision` not set. Will"
                " assume default destination capability `(precision, scale) ="
                f" {caps.decimal_precision}`"
            )
        elif scale is None:
            # setting scale to 0 when unspecified is a common practice across databases
            precision_tuple = (precision, 0)
            logger.warning(
                f"Received decimal column hint `precision={precision}`, but `scale` not set. "
                "Will assume default destination capability `scale=0`"
            )
        else:
            precision_tuple = (precision, scale)

        return get_py_arrow_numeric(precision_tuple)
    elif column_type == "wei":
        return get_py_arrow_numeric(caps.wei_precision)
    elif column_type == "date":
        return pyarrow.date32()
    elif column_type == "time":
        precision = column.get("precision")
        if precision is None:
            precision = caps.timestamp_precision
        return get_py_arrow_time(precision)
    else:
        raise ValueError(column_type)


def get_py_arrow_timestamp(precision: int, tz: str) -> Any:
    tz = tz if tz else None
    if precision == 0:
        return pyarrow.timestamp("s", tz=tz)
    if precision <= 3:
        return pyarrow.timestamp("ms", tz=tz)
    if precision <= 6:
        return pyarrow.timestamp("us", tz=tz)
    return pyarrow.timestamp("ns", tz=tz)


def get_py_arrow_time(precision: int) -> Any:
    if precision == 0:
        return pyarrow.time32("s")
    elif precision <= 3:
        return pyarrow.time32("ms")
    elif precision <= 6:
        return pyarrow.time64("us")
    return pyarrow.time64("ns")


def get_py_arrow_numeric(precision: Tuple[int, int]) -> Any:
    if precision[0] <= 38:
        return pyarrow.decimal128(*precision)
    if precision[0] <= 76:
        return pyarrow.decimal256(*precision)
    # for higher precision use max precision and trim scale to leave the most significant part
    return pyarrow.decimal256(76, max(0, 76 - (precision[0] - precision[1])))


def get_pyarrow_int(precision: Optional[int]) -> Any:
    if precision is None:
        return pyarrow.int64()
    if precision <= 8:
        return pyarrow.int8()
    elif precision <= 16:
        return pyarrow.int16()
    elif precision <= 32:
        return pyarrow.int32()
    return pyarrow.int64()


def get_column_type_from_py_arrow(dtype: pyarrow.DataType) -> TColumnType:
    """Returns (data_type, precision, scale) tuple from pyarrow.DataType"""
    if pyarrow.types.is_string(dtype) or pyarrow.types.is_large_string(dtype):
        return dict(data_type="text")
    elif pyarrow.types.is_floating(dtype):
        return dict(data_type="double")
    elif pyarrow.types.is_boolean(dtype):
        return dict(data_type="bool")
    elif pyarrow.types.is_timestamp(dtype):
        precision = get_precision_from_datetime_unit(dtype.unit)
        timestamp_d: TColumnType = dict(data_type="timestamp", precision=precision)
        if dtype.tz is None:
            timestamp_d["timezone"] = False
        return timestamp_d
    elif pyarrow.types.is_date(dtype):
        return dict(data_type="date")
    elif pyarrow.types.is_time(dtype):
        # Time fields in schema are `DataType` instead of `Time64Type` or `Time32Type`
        precision = get_precision_from_datetime_unit(dtype.unit)
        return dict(data_type="time", precision=precision)
    elif pyarrow.types.is_integer(dtype):
        result: TColumnType = dict(data_type="bigint")
        if dtype.bit_width != 64:  # 64bit is a default bigint
            result["precision"] = dtype.bit_width
        return result
    elif pyarrow.types.is_fixed_size_binary(dtype):
        return dict(data_type="binary", precision=dtype.byte_width)
    elif pyarrow.types.is_binary(dtype) or pyarrow.types.is_large_binary(dtype):
        return dict(data_type="binary")
    elif pyarrow.types.is_decimal(dtype):
        return dict(data_type="decimal", precision=dtype.precision, scale=dtype.scale)
    elif pyarrow.types.is_nested(dtype):
        return get_nested_column_type_from_py_arrow(dtype)
    elif pyarrow.types.is_dictionary(dtype):
        # Dictionary types are essentially categorical encodings. The underlying value_type
        # dictates the "logical" type. We simply delegate to the underlying value_type.
        return get_column_type_from_py_arrow(dtype.value_type)
    elif pyarrow.types.is_null(dtype):
        return {}  # incomplete column, no data_type
    else:
        raise UnsupportedArrowTypeException(arrow_type=dtype)


def py_arrow_to_table_schema_columns(schema: pyarrow.Schema) -> TTableSchemaColumns:
    """Convert a PyArrow schema to a table schema columns dict.

    Args:
        schema (pyarrow.Schema): pyarrow schema

    Returns:
        TTableSchemaColumns: table schema columns
    """
    result: TTableSchemaColumns = {}
    for field in schema:
        try:
            converted_type = get_column_type_from_py_arrow(field.type)
        except UnsupportedArrowTypeException as e:
            # modify attributes inplace to add context instead of re-raising with `raise e`
            e.field_name = field.name
            raise

        result[field.name] = {
            "name": field.name,
            "nullable": field.nullable,
            **converted_type,
        }
    return result


def get_nested_column_type_from_py_arrow(dtype: pyarrow.DataType) -> TColumnType:
    """Creates `json` dlt data type with nested type structure in `x-nested-type` hint.
    Currently the only recognized nested type format is arrow-ipc
    """
    return {"data_type": "json", "x-nested-type": serialize_type(dtype)}  # type: ignore[typeddict-unknown-key]


def serialize_type(dtype: pyarrow.DataType) -> str:
    """Serializes arrow type via arrow ipc as base64 str"""
    schema = pyarrow.schema([pyarrow.field("c", dtype)])
    # mtime=0 keeps gzip deterministic
    compressed = gzip.compress(schema.serialize().to_pybytes(), mtime=0)
    return "arrow-ipc:" + base64.b64encode(compressed).decode("ascii")


def deserialize_type(type_str: str) -> pyarrow.DataType:
    if type_str.startswith("arrow-ipc:"):
        decompressed = gzip.decompress(base64.b64decode(type_str[10:]))
        schema = pyarrow.ipc.read_schema(pyarrow.BufferReader(decompressed))
        return schema.field(0).type
    else:
        raise TypeError("Cannot deserialize pyarrow type, only arrow-ipc is supported")


def remove_null_columns(item: TAnyArrowItem) -> TAnyArrowItem:
    """Remove all columns of datatype pyarrow.null() from the table or record batch.
    Stores removed column names in arrow schema metadata under 'dlt.null_columns' key.
    """
    null_col_names = [field.name for field in item.schema if pyarrow.types.is_null(field.type)]
    if not null_col_names:
        return item
    item = remove_columns(item, null_col_names)
    return add_arrow_metadata(item, {"dlt.null_columns": json.dumps(null_col_names)})


def remove_null_columns_from_schema(schema: pyarrow.Schema) -> Tuple[pyarrow.Schema, bool]:
    """Remove all columns of datatype pyarrow.null() from the schema"""
    fields: List[pyarrow.field] = []
    contains_null: bool = False
    for field in schema:
        if pyarrow.types.is_null(field.type):
            contains_null = True
        else:
            fields.append(field)
    return pyarrow.schema(fields), contains_null


def remove_columns(item: TAnyArrowItem, columns: Sequence[str]) -> TAnyArrowItem:
    """Remove `columns` from Arrow `item`"""
    if not columns:
        return item

    if isinstance(item, pyarrow.Table):
        return item.drop(columns)
    elif isinstance(item, pyarrow.RecordBatch):
        # NOTE: select is available in pyarrow 12 an up
        return item.select([n for n in item.schema.names if n not in columns])  # reverse selection
    else:
        raise ValueError(item)


def append_column(item: TAnyArrowItem, name: str, data: Any) -> TAnyArrowItem:
    """Appends new column to Table or RecordBatch"""
    if isinstance(item, pyarrow.Table):
        return item.append_column(name, data)
    elif isinstance(item, pyarrow.RecordBatch):
        new_field = pyarrow.field(name, data.type)
        return pyarrow.RecordBatch.from_arrays(
            item.columns + [data], schema=item.schema.append(new_field)
        )
    else:
        raise ValueError(item)


def rename_columns(item: TAnyArrowItem, new_column_names: Sequence[str]) -> TAnyArrowItem:
    """Rename arrow columns on Table or RecordBatch, returns same data but with renamed schema"""

    if list(item.schema.names) == list(new_column_names):
        # No need to rename
        return item

    if isinstance(item, pyarrow.Table):
        return item.rename_columns(new_column_names)
    elif isinstance(item, pyarrow.RecordBatch):
        new_fields = [
            field.with_name(new_name) for new_name, field in zip(new_column_names, item.schema)
        ]
        return pyarrow.RecordBatch.from_arrays(item.columns, schema=pyarrow.schema(new_fields))
    else:
        raise TypeError(f"Unsupported data item type: `{type(item)}`")


def fill_empty_source_column_values_with_placeholder(
    table: pyarrow.Table, source_columns: List[str], placeholder: str
) -> pyarrow.Table:
    """
    Replaces empty strings and null values in the specified source columns of an Arrow table with a placeholder string.

    Args:
        table (pa.Table): The input Arrow table.
        source_columns (List[str]): A list of column names to replace empty strings and null values in.
        placeholder (str): The placeholder string to use for replacement.

    Returns:
        pyarrow.Table: The modified Arrow table with empty strings and null values replaced in the specified columns.
    """
    for col_name in source_columns:
        column = table[col_name]
        filled_column = pyarrow.compute.fill_null(column, fill_value=placeholder)
        new_column = pyarrow.compute.replace_substring_regex(
            filled_column, pattern=r"^$", replacement=placeholder
        )
        table = table.set_column(table.column_names.index(col_name), col_name, new_column)
    return table


def should_normalize_arrow_schema(
    schema: pyarrow.Schema,
    columns: TTableSchemaColumns,
    naming: NamingConvention,
    caps: DestinationCapabilitiesContext = None,
    tz: str = None,
    resolve_time_fidelity: bool = True,
) -> ArrowSchemaNormalizationResult:
    """Figure out if any of the normalization steps must be executed. This prevents
    from rewriting arrow tables when no changes are needed. Refer to `normalize_py_arrow_item`
    for a list of normalizations. Note that `column` must be already normalized.

    `resolve_time_fidelity` is `False` while extracting, when a time fidelity column is not yet
    resolved and must be kept exactly as the source wrote it.
    """
    schema, contains_null_cols = remove_null_columns_from_schema(schema)

    rename_mapping = get_normalized_arrow_fields_mapping(schema, naming)
    # no clashes in rename ensured above
    rev_mapping = {v: k for k, v in rename_mapping.items()}
    nullable_mapping = {k: is_nullable_column(v) for k, v in columns.items()}

    # Key is the renamed column name
    nullable_updates: Dict[str, bool] = {}
    column_cast: Dict[str, bool] = {}
    for field in schema:
        norm_name = rename_mapping[field.name]
        # All fields from arrow schema that have nullable set to different value than in columns
        if norm_name in nullable_mapping and field.nullable != nullable_mapping[norm_name]:
            nullable_updates[norm_name] = nullable_mapping[norm_name]
        # Detect arrow columns that require to be normalized
        if norm_name in columns and should_normalize_py_arrow_item_column(
            columns[norm_name], field.type, tz, caps, resolve_time_fidelity
        ):
            column_cast[norm_name] = True

    dlt_load_id_col = naming.normalize_identifier(C_DLT_LOAD_ID)
    dlt_id_col = naming.normalize_identifier(C_DLT_ID)
    dlt_columns = {dlt_load_id_col, dlt_id_col}

    # remove all columns that are dlt columns but are not present in arrow schema. we do not want to add such columns
    # that should happen in the normalizer
    columns = {
        name: column
        for name, column in columns.items()
        if name not in dlt_columns or name in rev_mapping
    }

    # check if nothing to rename
    skip_normalize = (
        (list(rename_mapping.keys()) == list(rename_mapping.values()) == list(columns.keys()))
        and not nullable_updates
        and not contains_null_cols
        and not column_cast
    )
    return ArrowSchemaNormalizationResult(
        not skip_normalize,
        rename_mapping,
        rev_mapping,
        nullable_updates,
        columns,
    )


def normalize_py_arrow_item(
    item: TAnyArrowItem,
    columns: TTableSchemaColumns,
    naming: NamingConvention,
    caps: DestinationCapabilitiesContext,
    tz: str = None,
    resolve_time_fidelity: bool = True,
) -> TAnyArrowItem:
    """Normalize arrow `item` schema according to the `columns`. Note that
    columns must be already normalized.

    0. columns with no data type will be dropped
    1. arrow schema field names will be normalized according to `naming`
    2. arrows columns will be reordered according to `columns`
    3. empty columns will be inserted if they are missing, types will be generated using `caps`
    4. arrow columns with different nullability than corresponding schema columns will be updated
    5. timestamps will be normalized according to timezone flag in dlt column schema
    6. columns of a field in the time fidelity mode will be resolved in their `x-source-timezone`
       and its companion columns filled with the offset and the ambiguity status

    `resolve_time_fidelity` is `False` while extracting: the value is written exactly as the
    source wrote it and only the normalizer resolves it.

    NOTE: nullability is not enforced. it is up to destination to do that.
    """
    tz = tz or get_context_timezone_name()
    item = remove_null_columns(item)
    schema = item.schema
    should_normalize, rename_mapping, rev_mapping, nullable_updates, columns = (
        should_normalize_arrow_schema(schema, columns, naming, caps, tz, resolve_time_fidelity)
    )
    if not should_normalize:
        return item

    # resolutions of the time fidelity fields, by their normalized name. resolved first, so that
    # a companion column is filled whatever its position in `columns`
    resolutions: Dict[str, ArrowLocalTimeResolution] = {}
    if resolve_time_fidelity:
        for column_name, column in columns.items():
            fidelity = get_time_fidelity(column)
            field_name = rev_mapping.get(column_name)
            if fidelity is None or field_name is None:
                continue
            idx = schema.get_field_index(field_name)
            if not pyarrow.types.is_timestamp(schema.field(idx).type):
                continue
            resolutions[column_name] = resolve_arrow_time_fidelity(
                item.column(idx), fidelity[0], fidelity[1]
            )

    new_fields = []
    new_columns = []

    for column_name, column in columns.items():
        # get original field name
        field_name = rev_mapping.pop(column_name, column_name)
        if column.get(HINT_COMPANION_OF) in resolutions:
            # the referent resolved, so the companion holds what it had to resolve
            companion = time_fidelity_companion_values(column, columns, resolutions, naming)
            if companion is not None:
                new_fields.append(
                    pyarrow.field(
                        column_name, pyarrow.string(), nullable=is_nullable_column(column)
                    )
                )
                new_columns.append(companion)
                continue
        if field_name in rename_mapping:
            idx = schema.get_field_index(field_name)
            new_field = schema.field(idx).with_name(column_name)
            if column_name in nullable_updates:
                # Set field nullable to match column
                new_field = new_field.with_nullable(nullable_updates[column_name])

            if (resolution := resolutions.get(column_name)) is not None:
                new_type, new_arrow_column = time_fidelity_timestamp_column(
                    column, new_field.type, resolution, tz, caps
                )
            else:
                # coerce type
                new_type, new_arrow_column = normalize_py_arrow_item_column(
                    column, new_field.type, item.column(idx), tz, caps, resolve_time_fidelity
                )

            # use renamed field
            new_fields.append(new_field.with_type(new_type))
            new_columns.append(new_arrow_column)
        else:
            # column does not exist in pyarrow. create empty field and column
            new_field = pyarrow.field(
                column_name,
                get_py_arrow_datatype(column, caps, tz),
                nullable=is_nullable_column(column),
            )
            new_fields.append(new_field)
            new_columns.append(pyarrow.nulls(item.num_rows, type=new_field.type))

    # add the remaining columns
    for column_name, field_name in rev_mapping.items():
        idx = schema.get_field_index(field_name)
        # use renamed field
        new_fields.append(schema.field(idx).with_name(column_name))
        new_columns.append(item.column(idx))

    # preserve schema metadata (e.g. dlt.null_columns) through normalization rebuild
    return item.__class__.from_arrays(
        new_columns, schema=pyarrow.schema(new_fields, metadata=item.schema.metadata)
    )


def assume_arrow_timezone(
    arrow_column: pyarrow.Array,
    tz: str,
    ambiguous: str = "latest",
    nonexistent: str = "latest",
) -> pyarrow.Array:
    """Labels a naive timestamp column with `tz` without shifting the values.

    `ambiguous` and `nonexistent` are passed to `pyarrow.compute.assume_timezone` and tell how a
    wall clock DST made ambiguous or nonexistent is read. Arrow collapses such a wall clock onto
    the transition edge, so use `resolve_arrow_time_fidelity` when the wall clock must be kept.
    """
    try:
        return pyarrow.compute.assume_timezone(
            arrow_column, tz, ambiguous=ambiguous, nonexistent=nonexistent
        )
    except pyarrow.ArrowInvalid as inv_ex:
        if "timezone database" in str(inv_ex):
            raise InvalidTimezoneException(tz, str(inv_ex)) from inv_ex
        raise


@lru_cache(maxsize=None)
def validate_arrow_timezone(tz: str) -> None:
    """Checks that arrow can resolve `tz`, which `pyarrow.timestamp` never does."""
    assume_arrow_timezone(pyarrow.array([0], type=pyarrow.timestamp("s")), tz)


@lru_cache(maxsize=None)
def _validate_source_timezone(tz: str) -> None:
    """Checks that `tz` is the IANA name the object path accepts as a source timezone.

    Arrow reads a fixed offset such as `+02:00` as a timezone of its own while the object path
    rejects it, so a zone declared with `x-source-timezone` must pass the same check on both
    paths; otherwise the same schema would resolve on arrow data and fail on python objects.
    """
    to_tzinfo(tz)


_TIME_FIDELITY_UNITS: Dict[str, int] = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
"""`int64` ticks per second, per arrow timestamp unit."""


class ArrowLocalTimeResolution(NamedTuple):
    """Result of `resolve_arrow_time_fidelity`."""

    instants: pyarrow.Array
    """Instants in `arrow_column`'s unit, labelled `UTC`."""
    offsets: pyarrow.Array
    """UTC offset in seconds each value was read with, `int64`."""
    statuses: pyarrow.Array
    """`ambiguous`, `nonexistent` or null, per value."""


def _seconds_to_iso(offset_seconds: int) -> str:
    """Formats a UTC offset in seconds the way an ISO 8601 timestamp writes it."""
    return format_utc_offset(timedelta(seconds=offset_seconds))


def offsets_to_iso(offsets: pyarrow.Array) -> pyarrow.Array:
    """Renders an `int64` array of UTC offsets in seconds as ISO 8601 offset strings."""
    if isinstance(offsets, pyarrow.ChunkedArray):
        # `StructArray` below takes plain arrays; offsets never straddle a chunk boundary
        offsets = offsets.combine_chunks()
    distinct = pyarrow.compute.drop_null(pyarrow.compute.unique(offsets)).to_pylist()
    if not distinct:
        return pyarrow.array([None] * len(offsets), type=pyarrow.string())
    # one equality mask per offset, matched by `case_when`, so an arbitrary number of distinct
    # offsets - historical zones change theirs many times - costs a pass each and no Python loop
    conditions = pyarrow.StructArray.from_arrays(
        [
            pyarrow.compute.fill_null(
                pyarrow.compute.equal(offsets, pyarrow.scalar(value, type=pyarrow.int64())), False
            )
            for value in distinct
        ],
        names=[str(i) for i in range(len(distinct))],
    )
    return pyarrow.compute.case_when(
        conditions,
        *[pyarrow.scalar(_seconds_to_iso(value), type=pyarrow.string()) for value in distinct],
        pyarrow.scalar(None, type=pyarrow.string()),
    )


def _local_of_instant(
    instants: pyarrow.Array, unit: str, source_tz: str
) -> pyarrow.Array:
    """Wall clock `instants` (an `int64` array) reads as in `source_tz`, as `int64`."""
    aware = pyarrow.compute.cast(instants, pyarrow.timestamp(unit, UTC_NAME))
    local = pyarrow.compute.local_timestamp(aware.cast(pyarrow.timestamp(unit, source_tz)))
    return pyarrow.compute.cast(local, pyarrow.int64())


def _classify_arrow_wall_clocks(
    walls: pyarrow.Array, unit: str, source_tz: str
) -> Tuple[pyarrow.Array, pyarrow.Array]:
    """Splits naive `walls` (`int64` in `unit`) into ambiguous and nonexistent masks.

    A wall clock is read with the offset in force a day before it and a day after it. For a
    repeated hour those are the readings before and after the clock moved back, for the gap the
    offsets before and after it moved forward and everywhere else the same one offset. An offset
    is a candidate only when it maps the wall clock back to itself, which none does in the gap,
    both do in the repeated hour and exactly one does everywhere else.
    """
    per_second = _TIME_FIDELITY_UNITS[unit]

    def _offset_in_force(day_shift: int) -> pyarrow.Array:
        # the offset in force reads a wall clock into an instant, and instant back into wall clock
        shifted = pyarrow.compute.add(
            walls, pyarrow.scalar(day_shift * 24 * 60 * 60 * per_second, pyarrow.int64())
        )
        return pyarrow.compute.subtract(_local_of_instant(shifted, unit, source_tz), shifted)

    def _is_candidate(offset: pyarrow.Array) -> pyarrow.Array:
        local = _local_of_instant(pyarrow.compute.subtract(walls, offset), unit, source_tz)
        return pyarrow.compute.fill_null(pyarrow.compute.equal(local, walls), False)

    before = _offset_in_force(-1)
    after = _offset_in_force(1)
    before_is_candidate = _is_candidate(before)
    after_is_candidate = _is_candidate(after)
    # a null wall clock is neither ambiguous nor nonexistent; `invert` alone would turn the
    # null-propagated comparison below into true
    present = pyarrow.compute.is_valid(walls)
    ambiguous = pyarrow.compute.and_(
        pyarrow.compute.and_(
            before_is_candidate,
            pyarrow.compute.and_(after_is_candidate, pyarrow.compute.not_equal(before, after)),
        ),
        present,
    )
    nonexistent = pyarrow.compute.and_(
        pyarrow.compute.invert(pyarrow.compute.or_(before_is_candidate, after_is_candidate)),
        present,
    )
    return ambiguous, nonexistent


def _reading(
    walls: pyarrow.Array,
    unit: str,
    source_tz: str,
    day_shift: int,
) -> pyarrow.Array:
    """Offset `walls` are read with when the offset in force `day_shift` days away applies.

    `0` is the offset in force at the wall clock itself, which reads a value that maps to one
    instant correctly. Arrow collapses an ambiguous or nonexistent wall clock onto the transition,
    so those take the reading of the day before or after it instead: `-1` is the offset before the
    nearest transition, which `fold=0` selects, and `1` the offset after it, which `fold=1` does.
    """
    per_second = _TIME_FIDELITY_UNITS[unit]
    shifted = pyarrow.compute.add(
        walls, pyarrow.scalar(day_shift * 24 * 60 * 60 * per_second, pyarrow.int64())
    )
    resolved = pyarrow.compute.assume_timezone(
        pyarrow.compute.cast(shifted, pyarrow.timestamp(unit)),
        source_tz,
        ambiguous="earliest",
        nonexistent="earliest",
    )
    return pyarrow.compute.subtract(shifted, pyarrow.compute.cast(resolved, pyarrow.int64()))


def resolve_arrow_time_fidelity(
    arrow_column: pyarrow.Array,
    source_tz: str,
    policy: str,
    ambiguous: Optional[pyarrow.Array] = None,
    nonexistent: Optional[pyarrow.Array] = None,
) -> ArrowLocalTimeResolution:
    """Resolves a timestamp column in its declared source timezone, per `policy`.

    Returns the canonical instant in the column's own unit, labelled `UTC`, the UTC offset in
    seconds each value was read with and the `ambiguous`/`nonexistent` status of each wall clock.
    The outcome depends only on the values and the declared zone, never on the context timezone
    or on the batch composition, so a replay resolves every value the same way.

    An aware value keeps the offset the source wrote out, exactly as on the object path, and is
    never ambiguous or nonexistent.

    Args:
        arrow_column: Timestamp column, naive or aware.
        source_tz: IANA zone the source writes naive values in.
        policy: One of `TIME_FIDELITY_AMBIGUITY_POLICIES`.
        ambiguous: Mask of the ambiguous wall clocks. Computed when not passed.
        nonexistent: Mask of the nonexistent wall clocks. Computed when not passed.

    Raises:
        InvalidLocalTime: `policy` is `reject` and a wall clock is ambiguous or nonexistent.
        InvalidTimezoneName: `source_tz` is not an IANA name the object path also accepts.
    """
    # both paths read the same declared zone, so they must accept the same names
    _validate_source_timezone(source_tz)
    unit = arrow_column.type.unit
    per_second = _TIME_FIDELITY_UNITS[unit]
    if arrow_column.type.tz is not None:
        # the instant is known: the offset is the wall clock less the instant, which is the offset
        # the source wrote out, and the declared zone only names it
        instants = pyarrow.compute.cast(arrow_column, pyarrow.int64())
        walls = pyarrow.compute.cast(pyarrow.compute.local_timestamp(arrow_column), pyarrow.int64())
        offsets = pyarrow.compute.subtract(walls, instants)
        if pyarrow.compute.any(_late_wall_clocks(instants, unit)).as_py():
            # arrow freezes the offset of a zone past 2038, `zoneinfo` keeps following it
            offsets = _correct_late_aware_values(instants, offsets, unit, arrow_column.type.tz)
        return ArrowLocalTimeResolution(
            pyarrow.compute.cast(instants, pyarrow.timestamp(unit, UTC_NAME)),
            pyarrow.compute.cast(pyarrow.compute.divide(offsets, per_second), pyarrow.int64()),
            pyarrow.array([None] * len(arrow_column), type=pyarrow.string()),
        )

    walls = pyarrow.compute.cast(arrow_column, pyarrow.int64())
    if ambiguous is None or nonexistent is None:
        computed_ambiguous, computed_nonexistent = _classify_arrow_wall_clocks(
            walls, unit, source_tz
        )
        ambiguous = computed_ambiguous if ambiguous is None else ambiguous
        nonexistent = computed_nonexistent if nonexistent is None else nonexistent
    ambiguous = pyarrow.compute.fill_null(ambiguous, False)
    nonexistent = pyarrow.compute.fill_null(nonexistent, False)

    ambiguous_any = pyarrow.compute.any(ambiguous).as_py()
    nonexistent_any = pyarrow.compute.any(nonexistent).as_py()

    # the offset in force at the wall clock reads a value that maps to one instant. a wall clock
    # DST broke is collapsed by arrow onto the transition, so it takes the reading of the day
    # before: `fold=0` for the repeated hour and the offset that shifts the gap forward, the
    # convention of the platforms dlt reads from
    offsets = pyarrow.compute.if_else(
        pyarrow.compute.or_(ambiguous, nonexistent),
        _reading(walls, unit, source_tz, -1),
        _reading(walls, unit, source_tz, 0),
    )
    if policy == "latest" and (ambiguous_any or nonexistent_any):
        offsets = pyarrow.compute.if_else(
            pyarrow.compute.or_(ambiguous, nonexistent),
            _reading(walls, unit, source_tz, 1),
            offsets,
        )
    statuses = pyarrow.compute.if_else(
        ambiguous,
        pyarrow.scalar("ambiguous", type=pyarrow.string()),
        pyarrow.compute.if_else(
            nonexistent, pyarrow.scalar("nonexistent", type=pyarrow.string()),
            pyarrow.scalar(None, type=pyarrow.string()),
        ),
    )
    # arrow reads a timezone past 2038 with the offset it had at the 32-bit epoch boundary, the
    # object path keeps using the rules of the zone. only such wall clocks differ, so they are
    # resolved with the stdlib rather than the arrow rules
    if pyarrow.compute.any(_late_wall_clocks(walls, unit)).as_py():
        offsets, statuses = _correct_late_wall_clocks(
            walls, offsets, statuses, unit, source_tz, policy
        )

    if policy == "reject":
        _raise_invalid_local_time(walls, unit, source_tz, statuses)

    return ArrowLocalTimeResolution(
        pyarrow.compute.cast(
            pyarrow.compute.subtract(walls, offsets), pyarrow.timestamp(unit, UTC_NAME)
        ),
        pyarrow.compute.cast(pyarrow.compute.divide(offsets, per_second), pyarrow.int64()),
        statuses,
    )


# the largest instant arrow reads a timezone rule for: it keeps the offset the zone had at the
# int32 epoch boundary from then on, while `zoneinfo` keeps following the zone
_ARROW_TIMEZONE_LAST_SECOND = 2**31 - 1


def _late_wall_clocks(walls: pyarrow.Array, unit: str) -> pyarrow.Array:
    """Masks the wall clocks arrow cannot read the timezone rules of, past 2038."""
    per_second = _TIME_FIDELITY_UNITS[unit]
    return pyarrow.compute.fill_null(
        pyarrow.compute.greater_equal(
            walls, pyarrow.scalar(_ARROW_TIMEZONE_LAST_SECOND * per_second, pyarrow.int64())
        ),
        False,
    )


def _ticks_to_datetime(ticks: int, unit: str) -> datetime:
    """Converts `ticks` of `unit` since the epoch to a naive datetime, as arrow stores them."""
    per_second = _TIME_FIDELITY_UNITS[unit]
    seconds, remainder = divmod(ticks, per_second)
    return datetime(1970, 1, 1) + timedelta(
        seconds=seconds, microseconds=remainder * 1_000_000 // per_second
    )


def _correct_late_wall_clocks(
    walls: pyarrow.Array,
    offsets: pyarrow.Array,
    statuses: pyarrow.Array,
    unit: str,
    source_tz: str,
    policy: str,
) -> Tuple[pyarrow.Array, pyarrow.Array]:
    """Replaces the offsets and statuses of the wall clocks arrow reads past 2038 with `zoneinfo`.

    Runs the object path `resolve_local_time` on those values, so a wall clock after 2038 resolves
    exactly as it does when it arrives as a python object.
    """
    per_second = _TIME_FIDELITY_UNITS[unit]
    tz = to_tzinfo(source_tz)
    late = _late_wall_clocks(walls, unit)
    corrected_offsets = []
    corrected_statuses = []
    for ticks in pyarrow.compute.filter(walls, late).to_pylist():
        resolution = resolve_local_time(_ticks_to_datetime(ticks, unit), tz, policy)
        corrected_offsets.append(int(resolution.offset.total_seconds()) * per_second)
        corrected_statuses.append(resolution.status)
    return (
        pyarrow.compute.replace_with_mask(
            offsets, late, pyarrow.array(corrected_offsets, type=pyarrow.int64())
        ),
        pyarrow.compute.replace_with_mask(
            statuses, late, pyarrow.array(corrected_statuses, type=pyarrow.string())
        ),
    )


def _correct_late_aware_values(
    instants: pyarrow.Array, offsets: pyarrow.Array, unit: str, column_tz: str
) -> pyarrow.Array:
    """Recomputes, with `zoneinfo`, the offsets arrow derives for an aware value past 2038.

    The instant an aware value holds is exact, arrow only freezes the offset it reads the zone
    with, so replacing the offsets is enough.
    """
    per_second = _TIME_FIDELITY_UNITS[unit]
    tz = to_tzinfo(column_tz)
    late = _late_wall_clocks(instants, unit)
    corrected = []
    for instant in pyarrow.compute.filter(instants, late).to_pylist():
        utc = _ticks_to_datetime(instant, unit).replace(tzinfo=timezone.utc)
        corrected.append(int(utc.astimezone(tz).utcoffset().total_seconds()) * per_second)
    return pyarrow.compute.replace_with_mask(
        offsets, late, pyarrow.array(corrected, type=pyarrow.int64())
    )


def _raise_invalid_local_time(
    walls: pyarrow.Array,
    unit: str,
    source_tz: str,
    statuses: pyarrow.Array,
) -> None:
    """Raises `InvalidLocalTime` for the first wall clock DST made ambiguous or nonexistent."""
    offending = pyarrow.compute.fill_null(
        pyarrow.compute.is_in(
            statuses, value_set=pyarrow.array(["ambiguous", "nonexistent"], type=pyarrow.string())
        ),
        False,
    )
    if not pyarrow.compute.any(offending).as_py():
        return
    index = pyarrow.compute.indices_nonzero(offending)[0].as_py()
    wall = pyarrow.compute.cast(walls[index], pyarrow.timestamp(unit)).as_py()
    raise InvalidLocalTime(wall, source_tz, statuses[index].as_py())


_TIME_FIDELITY_FLOOR_UNITS: Dict[str, str] = {
    "s": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
}
"""`floor_temporal` unit of each arrow timestamp unit."""


def _floor_timestamp_to_precision(instants: pyarrow.Array, precision: int) -> pyarrow.Array:
    """Truncates `instants` to `precision` fractional seconds, like `reduce_datetime_precision`.

    A precision finer than the column's unit cannot truncate anything and leaves the values as is.
    """
    per_second = _TIME_FIDELITY_UNITS[instants.type.unit]
    ticks = per_second // 10**precision if 0 <= precision < 9 else 0
    if ticks <= 0:
        return instants
    if ticks == per_second:
        return pyarrow.compute.floor_temporal(instants, unit="second")
    return pyarrow.compute.floor_temporal(
        instants, unit=_TIME_FIDELITY_FLOOR_UNITS[instants.type.unit], multiple=ticks
    )


def time_fidelity_timestamp_column(
    column: TColumnSchema,
    arrow_type: pyarrow.DataType,
    resolution: ArrowLocalTimeResolution,
    tz: str,
    caps: DestinationCapabilitiesContext = None,
) -> Tuple[pyarrow.DataType, pyarrow.Array]:
    """Value a time fidelity field stores: the canonical instant at the declared precision.

    The instant is truncated to the `precision` hint and then labelled the way the `timezone` hint
    and the destination capabilities ask, so a destination without tz support gets the wall clock
    of the storage timezone, exactly as a value resolved on the object path.
    """
    unit = arrow_type.unit
    values = resolution.instants
    if (precision := column.get("precision")) is not None:
        values = _floor_timestamp_to_precision(values, precision)
    target_tz = get_column_timezone(column, tz, caps)
    if target_tz is None:
        if tz != UTC_NAME:
            values = pyarrow.compute.cast(values, pyarrow.timestamp(unit, tz))
        values = pyarrow.compute.local_timestamp(values)
    else:
        validate_arrow_timezone(target_tz)
        values = pyarrow.compute.cast(values, pyarrow.timestamp(unit, target_tz))
    return pyarrow.timestamp(unit, target_tz), values


def time_fidelity_companion_values(
    column: TColumnSchema,
    columns: TTableSchemaColumns,
    resolutions: Dict[str, ArrowLocalTimeResolution],
    naming: NamingConvention,
) -> Optional[pyarrow.Array]:
    """Values of a companion column of a time fidelity field, `None` when it has none.

    The offset column renders the offset each value was read with, the status column marks the
    wall clocks DST made ambiguous or nonexistent.
    """
    referent_name = column.get(HINT_COMPANION_OF)
    referent = columns.get(referent_name)
    resolution = resolutions.get(referent_name)
    if referent is None or resolution is None:
        return None
    companions = time_fidelity_companion_columns(referent_name, referent, naming)
    for suffix, companion in companions.items():
        if companion["name"] != column.get("name"):
            continue
        if suffix == C_UTC_OFFSET_SUFFIX:
            return offsets_to_iso(resolution.offsets)
        return resolution.statuses
    return None


def should_normalize_py_arrow_item_column(
    column: TColumnSchema,
    arrow_type: pyarrow.DataType,
    tz: str = None,
    caps: DestinationCapabilitiesContext = None,
    resolve_time_fidelity: bool = True,
) -> bool:
    # Only handle timestamp columns
    if not pyarrow.types.is_timestamp(arrow_type):
        return False
    tz = tz or get_context_timezone_name()

    # a time fidelity column always resolves in the normalizer: the instant is read in the
    # declared source zone and the value truncated to the declared precision. while extracting it
    # is written exactly as the source wrote it, so that the normalizer still sees its wall clock
    if get_time_fidelity(column) is not None:
        return bool(resolve_time_fidelity)

    # normalize if tz different
    return arrow_type.tz != get_column_timezone(column, tz, caps)  # type: ignore[no-any-return]


def normalize_py_arrow_item_column(
    column: TColumnSchema,
    arrow_type: pyarrow.Field,
    arrow_column: pyarrow.Array,
    tz: str = None,
    caps: DestinationCapabilitiesContext = None,
    resolve_time_fidelity: bool = True,
) -> Tuple[pyarrow.DataType, pyarrow.Array]:
    """Normalize arrow timestamp column timezone according to dlt schema column convention.

    A column declaring `x-source-timezone`, that is one in the time fidelity mode, additionally
    has its values read in that zone and truncated to the `precision` hint. See
    `resolve_arrow_time_fidelity`.

    Args:
        column: dlt column schema with timezone hint
        arrow_type: actual PyArrow data type
        arrow_column: actual PyArrow column data
        tz: timezone tz-aware columns are labelled with
        caps: destination capabilities, a destination without tz support gets naive columns
        resolve_time_fidelity: resolve a time fidelity column when `True`, keep it as written
            otherwise. Only the normalizer resolves

    Returns:
        Tuple of (modified_type, modified_column) or (arrow_field, arrow_column) if no changes needed
    """
    tz = tz or get_context_timezone_name()
    fidelity = get_time_fidelity(column) if resolve_time_fidelity else None
    if fidelity is not None:
        source_tz, policy = fidelity
        resolution = resolve_arrow_time_fidelity(arrow_column, source_tz, policy)
        return time_fidelity_timestamp_column(column, arrow_type, resolution, tz, caps)

    if not should_normalize_py_arrow_item_column(
        column, arrow_type, tz, caps, resolve_time_fidelity
    ):
        return arrow_type, arrow_column

    unit = arrow_type.unit
    current_tz = arrow_type.tz
    target_tz = get_column_timezone(column, tz, caps)

    if target_tz is not None:
        # `cast` takes any string, so the name must be checked before it reaches a schema
        validate_arrow_timezone(target_tz)
        if current_tz is None:
            # a naive value is read in the target zone, as it is on the object path
            arrow_column = assume_arrow_timezone(arrow_column, target_tz)
        # metadata-only cast, the instant is preserved
        col = pyarrow.compute.cast(arrow_column, pyarrow.timestamp(unit, target_tz))
    else:
        # Need naive
        if current_tz is None:
            col = arrow_column  # already naive
        else:
            # the wall clock is taken in `tz`, as it is on the object path
            if current_tz != tz:
                arrow_column = pyarrow.compute.cast(arrow_column, pyarrow.timestamp(unit, tz))
            col = pyarrow.compute.local_timestamp(arrow_column)

    return pyarrow.timestamp(unit, target_tz), col


def add_dlt_load_id_column(
    item: TAnyArrowItem,
    columns: TTableSchemaColumns,
    caps: DestinationCapabilitiesContext,
    naming: NamingConvention,
    load_id: str,
) -> TAnyArrowItem:
    """
    Adds or replaces the `_dlt_load_id` column.
    """
    dlt_load_id_col_name = naming.normalize_identifier(C_DLT_LOAD_ID)

    idx = item.schema.get_field_index(dlt_load_id_col_name)
    # if the column already exists, get rid of it
    if idx != -1:
        item = remove_columns(item, [dlt_load_id_col_name])

    # get pyarrow.string() type
    pyarrow_string = get_py_arrow_datatype(
        # use already existing column definition or use the default
        # NOTE: the existence of the load id column is ensured by this time
        # since it is added in _compute_tables before files are written
        (
            columns[dlt_load_id_col_name]
            if dlt_load_id_col_name in columns
            else dlt_load_id_column()
        ),
        caps,
        "UTC",  # ts is irrelevant to get pyarrow string, but it's required...
    )

    # Check if destination supports dictionary encoding (default True if not specified)
    use_dictionary = True
    if caps.parquet_format is not None:
        use_dictionary = caps.parquet_format.supports_dictionary_encoding

    # add the column with the new value at previous index or append
    item = add_constant_column(
        item=item,
        name=dlt_load_id_col_name,
        data_type=pyarrow_string,
        value=load_id,
        nullable=(
            columns[dlt_load_id_col_name]["nullable"]
            if dlt_load_id_col_name in columns
            else dlt_load_id_column()["nullable"]
        ),
        index=idx,
        use_dictionary=use_dictionary,
    )

    return item


def get_normalized_arrow_fields_mapping(schema: pyarrow.Schema, naming: NamingConvention) -> StrStr:
    """Normalizes schema field names and returns mapping from original to normalized name. Raises on name collisions"""
    # use normalize_path to be compatible with how regular columns are normalized in dlt.Schema
    norm_f = naming.normalize_path
    name_mapping = {n.name: norm_f(n.name) for n in schema}
    # verify if names uniquely normalize
    normalized_names = set(name_mapping.values())
    if len(name_mapping) != len(normalized_names):
        raise NameNormalizationCollision(
            f"Arrow schema fields normalized from:\n{list(name_mapping.keys())}:\nto:\n"
            f" {list(normalized_names)}"
        )
    return name_mapping


def dlt_column_to_arrow_field(
    column: TColumnSchema,
    caps: DestinationCapabilitiesContext,
    timestamp_timezone: str,
) -> pyarrow.Field:
    """Convert a single dlt column schema to a PyArrow field.

    Args:
        column (TColumnSchema): dlt column schema with at least `name` and `data_type`.
        caps (DestinationCapabilitiesContext): Destination capabilities for type mapping.
        timestamp_timezone (str): Timezone for timestamp columns.

    Returns:
        pyarrow.Field: Corresponding PyArrow field.
    """
    return pyarrow.field(
        column["name"],
        get_py_arrow_datatype(column, caps, timestamp_timezone),
        nullable=column.get("nullable", True),
    )


def columns_to_arrow(
    columns: TTableSchemaColumns,
    caps: DestinationCapabilitiesContext,
    timestamp_timezone: str,
) -> pyarrow.Schema:
    """Convert a table schema columns dict to a pyarrow schema.

    Args:
        columns (TTableSchemaColumns): table schema columns

    Returns:
        pyarrow.Schema: pyarrow schema

    """
    caps = caps or DestinationCapabilitiesContext.generic_capabilities()
    return pyarrow.schema(
        [
            dlt_column_to_arrow_field(column, caps, timestamp_timezone)
            for column in columns.values()
            if column.get("data_type") is not None
        ]
    )


def get_local_dataset_reader(file_paths: Sequence[str]) -> pyarrow.RecordBatchReader:
    """Streams local data files with bounded readahead to limit memory use over throughput."""
    # NOTE: import inline, pyarrow.dataset pulls heavy dependencies
    import pyarrow.dataset

    return (
        pyarrow.dataset.dataset(file_paths)
        .scanner(batch_size=65536, batch_readahead=2, fragment_readahead=1)
        .to_reader()
    )


def get_parquet_metadata(parquet_file: TFileOrPath) -> Tuple[int, pyarrow.Schema]:
    """Gets parquet file metadata (including row count and schema)

    Args:
        parquet_file (str): path to parquet file

    Returns:
        FileMetaData: file metadata
    """
    with pyarrow.parquet.ParquetFile(parquet_file) as reader:
        return reader.metadata.num_rows, reader.schema_arrow


def is_arrow_item(item: Any) -> bool:
    return isinstance(item, (pyarrow.Table, pyarrow.RecordBatch))


def to_arrow_scalar(value: Any, arrow_type: pyarrow.DataType) -> Any:
    """Converts python value to an arrow compute friendly version"""
    return pyarrow.scalar(value, type=arrow_type)


def from_arrow_scalar(arrow_value: pyarrow.Scalar) -> Any:
    """Converts arrow scalar into Python type."""
    return arrow_value.as_py()


TNewColumns = Sequence[Tuple[int, pyarrow.Field, Callable[[pyarrow.Table], Iterable[Any]]]]
"""Sequence of tuples: (field index, field, generating function)"""


def add_constant_column(
    item: TAnyArrowItem,
    name: str,
    data_type: pyarrow.DataType,
    value: Any = None,
    nullable: bool = True,
    index: int = -1,
    use_dictionary: bool = True,
) -> TAnyArrowItem:
    """Add column with a single value to the table.

    Args:
        item: Arrow table or record batch
        name: The new column name
        data_type: The data type of the new column
        nullable: Whether the new column is nullable
        value: The value to fill the new column with
        index: The index at which to insert the new column. Defaults to -1 (append)
        use_dictionary: When True (default), creates a dictionary-encoded column which is
            memory-efficient for repeated values. Set to False for destinations that don't
            support dictionary types (e.g., ADBC drivers for MSSQL).
    Note:
        When use_dictionary=True, the column is created as a DictionaryArray with int8 indices.
        When use_dictionary=False, a regular array filled with the repeated value is created.
    """
    if use_dictionary:
        dictionary = pyarrow.array([value], type=data_type)
        zero_buffer = pyarrow.allocate_buffer(item.num_rows, resizable=False)
        ctypes.memset(zero_buffer.address, 0, item.num_rows)

        indices = pyarrow.Array.from_buffers(
            pyarrow.int8(),
            item.num_rows,
            [None, zero_buffer],  # None validity bitmap means arrow assumes all entries are valid
        )
        column_array = pyarrow.DictionaryArray.from_arrays(indices, dictionary)
    else:
        # Create a regular array filled with the repeated value
        column_array = pyarrow.repeat(pyarrow.scalar(value, type=data_type), item.num_rows)

    field = pyarrow.field(name, column_array.type, nullable=nullable)
    if index == -1:
        return item.append_column(field, column_array)
    return item.add_column(index, field, column_array)


def pq_stream_with_new_columns(
    parquet_file: TFileOrPath, columns: TNewColumns, row_groups_per_read: int = 1
) -> Iterator[pyarrow.Table]:
    """Add column(s) to the table in batches.

    The table is read from parquet `row_groups_per_read` row groups at a time

    Args:
        parquet_file: path or file object to parquet file
        columns: list of columns to add in the form of (insertion index, `pyarrow.Field`, column_value_callback)
            The callback should accept a `pyarrow.Table` and return an array of values for the column.
        row_groups_per_read: number of row groups to read at a time. Defaults to 1.

    Yields:
        `pyarrow.Table` objects with the new columns added.
    """
    with pyarrow.parquet.ParquetFile(parquet_file) as reader:
        n_groups = reader.num_row_groups
        # Iterate through n row groups at a time
        for i in range(0, n_groups, row_groups_per_read):
            tbl: pyarrow.Table = reader.read_row_groups(
                range(i, min(i + row_groups_per_read, n_groups))
            )
            for idx, field, gen_ in columns:
                if idx == -1:
                    tbl = tbl.append_column(field, gen_(tbl))
                else:
                    tbl = tbl.add_column(idx, field, gen_(tbl))
            yield tbl


def cast_arrow_schema_types(
    schema: pyarrow.Schema,
    type_map: Dict[Callable[[pyarrow.DataType], bool], Callable[..., pyarrow.DataType]],
) -> pyarrow.Schema:
    """Returns type-casted Arrow schema.

    Replaces data types for fields matching a type check in `type_map`.
    Type check functions in `type_map` are assumed to be mutually exclusive, i.e.
    a data type does not match more than one type check function.
    """
    for i, e in enumerate(schema.types):
        for type_check, cast_type in type_map.items():
            if type_check(e):
                if callable(cast_type):
                    cast_type = cast_type(e)
                adjusted_field = schema.field(i).with_type(cast_type)
                schema = schema.set(i, adjusted_field)
                break  # if type matches type check, do not do other type checks
    return schema


def concat_batches_and_tables_in_order(
    tables_or_batches: Iterable[Union[pyarrow.Table, pyarrow.RecordBatch]],
    promote_options: str = "none",
) -> pyarrow.Table:
    """Concatenate iterable of tables and batches into a single table, preserving row order.

    Each record batch is converted to a table individually - they are never grouped via
    `Table.from_batches` (which rejects differing nullability/schemas that `concat_tables` can
    otherwise reconcile). All reconciliation happens in `concat_tables` per `promote_options`.

    Args:
        promote_options: PyArrow concat_tables promote_options. "none" (default) requires identical
            schemas and enables zero-copy concat. "default" reconciles nullability and fills missing
            columns. "permissive" additionally promotes across type families (e.g. int64→double).
    """
    tables = []
    for item in tables_or_batches:
        if isinstance(item, pyarrow.Table):
            tables.append(item)
        elif isinstance(item, pyarrow.RecordBatch):
            tables.append(pyarrow.Table.from_batches([item]))
        else:
            raise ValueError(f"Unsupported type: `{type(item)}`")
    # "none" ensures 0 copy concat; "default"/"permissive" reconcile/promote
    return pyarrow.concat_tables(tables, promote_options=promote_options)


# errors raised when arrow schemas cannot be merged/promoted (ArrowTypeError is not an ArrowInvalid)
ARROW_SCHEMA_MERGE_ERRORS = (
    pyarrow.lib.ArrowInvalid,
    pyarrow.lib.ArrowTypeError,
    pyarrow.lib.ArrowNotImplementedError,
)


def reconcile_schema_and_cast(
    target_schema: pyarrow.Schema, table: pyarrow.Table, promote_options: str
) -> Optional[pyarrow.Table]:
    """Computes the unified schema of `target_schema` and `table` under `promote_options`. If a
    unified schema exists, casts `table` to `target_schema` (the first/locked schema) and returns
    it - the cast is unsafe for "permissive" and may lose data. Returns None when no unified schema
    exists, so the caller rotates to a new file. Column sets and order are assumed aligned."""
    safe_cast = promote_options != "permissive"
    try:
        # only validates compatibility (raises e.g. on timestamp vs int64, which a bare cast would
        # silently reinterpret); the unified type itself is not used - we always cast to the target
        pyarrow.unify_schemas(
            [target_schema.remove_metadata(), table.schema.remove_metadata()],
            promote_options=promote_options,
        )
        return table.cast(target_schema, safe=safe_cast)
    except ARROW_SCHEMA_MERGE_ERRORS:
        return None


def concat_and_group_in_compatible_schemas(
    tables_or_batches: Iterable[Union[pyarrow.Table, pyarrow.RecordBatch]],
    promote_options: str,
) -> List[pyarrow.Table]:
    """Groups items into tables. "permissive" locks group schema at
    its first item and unsafe-casts later items into it (accepting data loss) or starts a new table -
    "default" merges runs with safe symmetric concat."""
    tables: List[pyarrow.Table] = []
    current: Optional[pyarrow.Table] = None
    for item in tables_or_batches:
        table = pyarrow.table(item)
        if current is None:
            current = table
            continue
        if promote_options == "permissive":
            fitted = reconcile_schema_and_cast(current.schema, table, promote_options)
            merged = (
                None if fitted is None else concat_batches_and_tables_in_order([current, fitted])
            )
        else:
            try:
                merged = concat_batches_and_tables_in_order([current, table], promote_options)
            except ARROW_SCHEMA_MERGE_ERRORS:
                merged = None
        if merged is None:
            tables.append(current)
            current = table
        else:
            current = merged
    if current is not None:
        tables.append(current)
    return tables


def transpose_rows_to_columns(
    rows: TDataItems, column_names: Iterable[str]
) -> dict[str, Any]:  # dict[str, Sequence[Any]]
    """Transpose rows (data items) into columns. Returns a dictionary of {column_name: column_data}"""
    if not rows:
        return {column_name: () for column_name in column_names}
    # plain zip benchmarked faster than numpy/pandas object-array pivoting
    return dict(zip(column_names, zip(*rows)))


def uuid_to_string(arr: Any) -> Any:  # pyarrow.Array -> pyarrow.Array
    """Convert an array of UUIDs to canonical hyphenated lowercase string array.

    Accepts a `pa.uuid()` extension array or a `fixed_size_binary[16]`. Nulls
    are preserved. Sliced inputs (`storage.offset != 0`) are supported. Falls
    back to pure Python if numpy is missing or the input is sliced.
    """
    pa = pyarrow

    # accept extension arrays (pa.uuid()) and plain fixed_size_binary[16]
    storage = arr.storage if hasattr(arr, "storage") else arr
    n = len(storage)

    try:
        from dlt.common.libs.numpy import numpy as np
    except MissingDependencyException:
        np = None

    # the numpy fast path reads `buffers()[1]` from byte 0 and reuses
    # `buffers()[0]` (bit-packed validity) as-is — both wrong when the input is
    # sliced. iterating `for scalar in storage` honors `storage.offset`, so the
    # pure-python path is correct for sliced inputs as well.
    if np is None or storage.offset != 0:
        # `str(UUID)` is canonical lowercase hyphenated per RFC 4122,
        # byte-identical to the numpy path's output.
        return pa.array(
            [
                None if scalar.as_py() is None else str(uuid.UUID(bytes=scalar.as_py()))
                for scalar in storage
            ],
            type=pa.string(),
        )

    if n == 0:
        return pa.array([], type=pa.string())

    # `bytes.hex()` is lowercase per the Python spec; the slice positions and
    # hyphen placements below match `str(UUID)` exactly so both paths produce
    # identical output.
    raw = storage.buffers()[1].to_pybytes()  # 16*n contiguous bytes
    hexbuf = np.frombuffer(raw.hex().encode("ascii"), dtype=np.uint8).reshape(n, 32)
    out = np.full((n, 36), ord("-"), dtype=np.uint8)
    out[:, 0:8] = hexbuf[:, 0:8]
    out[:, 9:13] = hexbuf[:, 8:12]
    out[:, 14:18] = hexbuf[:, 12:16]
    out[:, 19:23] = hexbuf[:, 16:20]
    out[:, 24:36] = hexbuf[:, 20:32]

    # reuse storage's validity buffer (None if all valid). Output bytes are
    # pure ASCII so the cast to string can never fail, even if null slots in
    # the source data buffer contain garbage.
    fsb36 = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(36), n, [storage.buffers()[0], pa.py_buffer(out.tobytes())]
    )
    return fsb36.cast(pa.string())


def _first_non_none_value(column_data: Any) -> Any:
    for value in column_data:
        if value is not None:
            return value
    return None


def _datetimes_to_arrow(
    column_data: Any,
    column_schema: TColumnSchema,
    arrow_type: Any,
    tz: str,
    caps: DestinationCapabilitiesContext = None,
) -> Any:  # pyarrow.Array
    """Converts datetimes to a timestamp array, reading naive values in `tz`, not in UTC."""
    # pyarrow reads a naive datetime as if it were UTC, so the array is built in the zone the
    # values carry and the timezone hint is applied after, exactly as on the object path
    aware = getattr(_first_non_none_value(column_data), "tzinfo", None) is not None
    source_type = pyarrow.timestamp(arrow_type.unit, UTC_NAME if aware else None)
    return normalize_py_arrow_item_column(
        column_schema, source_type, pyarrow.array(column_data, type=source_type), tz, caps
    )[1]


def convert_array_to_arrow(
    column_data: Any,  # 1-dimensional sequence of column values
    caps: DestinationCapabilitiesContext,
    column_schema: TColumnSchema,
    tz: str,
    safe_arrow_conversion: bool,
    arrow_type: Optional[Any] = None,
) -> Any:  # pyarrow.Array
    """Convert a sequence of column values to a pyarrow array.

    Args:
        column_data: column values
        caps: capabilities of the storage backend
        column_schema: dlt hints about the column (e.g., data type, nullable)
        tz: time zone identifier
        safe_arrow_conversion: if False, truncation and loss of precision is allowed
            ref: https://arrow.apache.org/docs/python/generated/pyarrow.compute.CastOptions.html#pyarrow.compute.CastOptions
        arrow_type: precomputed target arrow type, derived from `column_schema` when None

    Returns:
        an arrow Array
    """
    from dlt.common.libs.pyarrow import pyarrow as pa

    dlt_data_type = column_schema.get("data_type")
    inferred_arrow_type = (
        arrow_type
        if arrow_type is not None
        else (get_py_arrow_datatype(column_schema, caps, tz) if dlt_data_type is not None else None)
    )
    inferred_array = None

    # base case (0): allow pyarrow to infer type, or create array of dlt specified type
    try:
        if (
            tz != UTC_NAME
            and inferred_arrow_type is not None
            and pa.types.is_timestamp(inferred_arrow_type)
        ):
            inferred_array = _datetimes_to_arrow(
                column_data, column_schema, inferred_arrow_type, tz, caps
            )
        else:
            # type=None lets pyarrow infer the type from the data
            inferred_array = pa.array(column_data, type=inferred_arrow_type)
        # pyarrow >=24 infers UUIDs as the `arrow.uuid` extension; coerce to string
        # so destinations see canonical hyphenated text, matching pyarrow <24 behavior
        if inferred_array is not None and _is_arrow_uuid_extension(inferred_array.type):
            inferred_array = uuid_to_string(inferred_array)
    # detailed error handling should happen in fallback cases
    except (pa.ArrowInvalid, pyarrow.ArrowTypeError):
        logger.warning(
            f"Default conversion to `{inferred_arrow_type}` for `data_type={dlt_data_type}` failed."
            " Using fallback strategies."
        )

    def _first_non_none(types_: Sequence[AnyType]) -> bool:
        # the first non-null value guides the fallbacks
        return isinstance(_first_non_none_value(column_data), types_)  # type: ignore[arg-type]

    # case 1 & 2: pyarrow infers the type (e.g., float, string) THEN cast it to the dlt specified type; less constraints than the base case
    # for example, this handles when backends return decimals as floats or strings
    if inferred_array is None and dlt_data_type is not None:
        try:
            raw_array = pa.array(column_data)
            inferred_array = cast_arrow_array_as_column_schema(
                raw_array, caps, column_schema, tz, safe_arrow_conversion
            )
        except PyToArrowConversionException:
            raise
        except Exception as e:
            if (
                (dlt_data_type in ("text", "json"))
                and pa.types.is_string(inferred_arrow_type)
                and _first_non_none((list, dict))
            ):
                # this is handled by fallback case 2 for text/json with nested values
                logger.warning(
                    f"Received `data_type='{dlt_data_type}'`, data requires serialization to"
                    " string, slowing extraction. Cast the JSON field to STRING in your database"
                    " system to improve performance. For example, create and extract data from an"
                    " SQL VIEW that SELECT with CAST."
                )
            else:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details=f"This conversion is currently unsupported by dlt ({str(e)})",
                ) from e

    # case 2: encode Sequence and Mapping types (list, tuples, set, dict, etc.) to JSON strings
    # This logic needs to be before case 3, otherwise pyarrow might infer the deserialized JSON object as a `pyarrow.struct` instead of `pyarrow.string`
    if inferred_array is None and dlt_data_type in (
        "json",
        "text",
    ):
        # depending on the backend, JSON columns are inferred as data_type="text"
        json_serialized_values: list[Union[bytes, None]] = []
        for value in column_data:
            if value is None:
                json_serialized_values.append(None)
                continue
            try:
                json_serialized_values.append(json.dumpb(value))
            except TypeError as e:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details="dlt failed to a JSON-serializable type.",
                ) from e

        inferred_array = pa.array(json_serialized_values).cast(pa.string())

    # case 3: encode Python types unsupported by Arrow. Simple types are converted to strings and complex types to common structures (dict, list)
    # This catches specialized SQL types like `Ranges`
    if inferred_array is None and dlt_data_type is None:
        try:
            inferred_array = pa.array(column_data)
        except (pa.ArrowInvalid, pyarrow.ArrowTypeError) as e:
            # UUID fast path — pyarrow <24 doesn't recognize `uuid.UUID` at
            # inference; build fixed_size_binary[16] from `.bytes` and use the
            # vectorized formatter instead of the slower per-row loop below.
            if _first_non_none((uuid.UUID,)):
                fsb = pa.array(
                    [u.bytes if u is not None else None for u in column_data],
                    type=pa.binary(16),
                )
                inferred_array = uuid_to_string(fsb)
            else:
                logger.warning(
                    f"Type can't be inferred by `pyarrow` {e.args[0]}. Values will be encoded as"
                    " in a loop, slowing extraction."
                )
                encoded_values: list[Union[None, Mapping[Any, Any], Sequence[Any], str]] = []
                for value in column_data:
                    if value is None:
                        encoded_values.append(None)
                        continue
                    try:
                        # the 3 types match those supported by `map_nested_in_place()`
                        if isinstance(value, (tuple, dict, list)):
                            encoded_value = map_nested_values_in_place(custom_encode, value)
                        # convert set to list
                        elif isinstance(value, set):
                            encoded_value = map_nested_values_in_place(custom_encode, list(value))
                        # no nesting
                        else:
                            encoded_value = custom_encode(value)  # type: ignore[assignment]
                        encoded_values.append(encoded_value)
                    except TypeError as e:
                        raise PyToArrowConversionException(
                            data_type=dlt_data_type,
                            inferred_arrow_type=inferred_arrow_type,
                            details="dlt failed to encode values to an Arrow-compatible type.",
                        ) from e

                inferred_array = pa.array(encoded_values)

    return inferred_array


def cast_arrow_array_as_column_schema(
    raw_array: pyarrow.Array,
    caps: DestinationCapabilitiesContext,
    column_schema: TColumnSchema,
    tz: str,
    safe_arrow_conversion: bool,
) -> pyarrow.Array:
    from dlt.common.libs.pyarrow import pyarrow as pa

    dlt_data_type = column_schema.get("data_type")
    inferred_arrow_type = (
        get_py_arrow_datatype(column_schema, caps, tz) if dlt_data_type is not None else None
    )
    inferred_array = None
    try:
        if (
            tz != UTC_NAME
            and inferred_arrow_type is not None
            and pa.types.is_timestamp(inferred_arrow_type)
            and pa.types.is_timestamp(raw_array.type)
        ):
            # casting straight to the target reads a naive value as UTC, so only the unit is cast
            # here and the timezone hint is applied after
            unit_type = pyarrow.timestamp(inferred_arrow_type.unit, raw_array.type.tz)
            _, inferred_array = normalize_py_arrow_item_column(
                column_schema,
                unit_type,
                raw_array.cast(unit_type, safe=safe_arrow_conversion),
                tz,
                caps,
            )
        else:
            inferred_array = raw_array.cast(inferred_arrow_type, safe=safe_arrow_conversion)
    except (pa.ArrowInvalid, pyarrow.ArrowTypeError, pyarrow.ArrowNotImplementedError) as e:
        # TODO add specific error handling as we encounter them
        error_msg = e.args[0]
        if (
            "would cause data loss"
            in error_msg  # specific pyarrow error related to precision loss (i.e., conversion to decimal)
            and dlt_data_type == "decimal"
            and pa.types.is_decimal(inferred_arrow_type)
        ):
            # TODO provide user interface for safe_arrow_conversion=False and include in this error message
            raise PyToArrowConversionException(
                data_type=dlt_data_type,
                inferred_arrow_type=inferred_arrow_type,
                details=(
                    f"Insufficient decimal precision {error_msg}. Consider setting `precision`"
                    " and `scale` hints:"
                    " https://dlthub.com/docs/general-usage/schema/#tables-and-columns"
                ),
            ) from e
        elif dlt_data_type == "timestamp" and "Failed to parse string" in error_msg:
            if "expected a zone offset" in error_msg:
                # expected tz-aware timestamps in column_data
                actual_type = pyarrow.timestamp(inferred_arrow_type.unit, None)

            elif "expected no zone offset" in error_msg:
                # expected naive timestamps in column_data
                actual_type = pyarrow.timestamp(inferred_arrow_type.unit, tz)
            else:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details=f"Timestamp conversion unsupported by dlt ({error_msg})",
                ) from e
            # cast strings to date-times
            raw_array = raw_array.cast(actual_type)
            _, inferred_array = normalize_py_arrow_item_column(
                column_schema, actual_type, raw_array, tz, caps
            )
        elif dlt_data_type == "time" and "function cast_time" in error_msg:
            if "from string to" in error_msg:
                n = len(raw_array)
                is_null = None
                is_null_b = None
                null_count = 0

                data_b = pyarrow.allocate_buffer(n * 8, resizable=False)
                data = memoryview(data_b).cast("q")

                def allocate_lazy_null_mask() -> None:
                    nonlocal is_null, is_null_b
                    if is_null is None:
                        nbytes = (n + 7) // 8
                        is_null_b = pa.allocate_buffer(nbytes, resizable=False)
                        ctypes.memset(is_null_b.address, 0xFF, nbytes)  # start all-valid
                        is_null = memoryview(is_null_b).cast("B")

                for i, scalar in enumerate(raw_array):
                    if not scalar.is_valid:
                        # mark null in bitmap
                        allocate_lazy_null_mask()
                        byte = i >> 3
                        bit = i & 7
                        is_null[byte] &= ~(1 << bit)
                        null_count += 1
                        continue

                    s = scalar.as_py()  # Python str
                    t = time.fromisoformat(s)  # HH:MM | HH:MM:SS | HH:MM:SS.ffffff

                    data[i] = (
                        (t.hour * 3600 + t.minute * 60 + t.second) * 1_000_000
                    ) + t.microsecond

                # Zero-copy build of Arrow int64 with mask; then logical view to time64[us]
                arr_i64 = pyarrow.Array.from_buffers(
                    pa.int64(), n, buffers=[is_null_b, data_b], null_count=null_count
                )  # data zero-copies from NumPy
                inferred_array = arr_i64.view(pa.time64("us"))  # zero-copy reinterpret
                # ts  = pyarrow.compute.strptime(raw_array, format="%H:%M:%S", unit="us")
                # inferred_array = pyarrow.compute.cast(ts, pa.time64("us"))

            elif "from duration" in error_msg:
                # duration and time (if wrapping is not needed) have the same representation
                inferred_array = raw_array.view(pa.time64(raw_array.type.unit))

            else:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details=f"Time conversion unsupported by dlt ({error_msg})",
                ) from e
        # fallback: nested arrow -> json strings when casting to text/json
        elif (
            pa.types.is_nested(raw_array.type)
            and pa.types.is_string(inferred_arrow_type)
            and dlt_data_type in ("text", "json")
        ):
            # log a warning similar to numpy case 2, explaining serialization and destination caps
            logger.warning(
                "Received `data_type='%s'`, Arrow nested values require serialization to STRING."
                " They will be JSON-encoded, which may slow extraction. If your destination"
                " supports nested types, enable them (`supports_nested_types=True`) and provide a"
                " 'json' column with 'x-nested-type' hint to load nested values natively."
                " Otherwise, consider casting to STRING upstream (e.g., in a VIEW).",
                dlt_data_type,
            )
            try:
                # convert nested arrow values to python and json-encode. this mirrors the generic
                # json/text fallback (case 2) used for numpy inputs, but works on arrow arrays.
                py_values = raw_array.to_pylist()
                json_bytes = [None if v is None else json.dumpb(v) for v in py_values]
                inferred_array = pa.array(json_bytes).cast(pa.string())
            except TypeError as enc_err:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details="dlt failed to encode nested values as JSON strings.",
                ) from enc_err
        else:
            raise PyToArrowConversionException(
                data_type=dlt_data_type,
                inferred_arrow_type=inferred_arrow_type,
                details=f"This conversion is currently unsupported by dlt ({error_msg})",
            ) from e
    assert inferred_array
    return inferred_array


def cast_arrow_as_columns_schema(
    item: TAnyArrowItem,
    columns: TTableSchemaColumns,
    caps: DestinationCapabilitiesContext,
    tz: str,
    safe_arrow_conversion: bool = True,
) -> TAnyArrowItem:
    """Cast an Arrow item to match dlt column schema.

    Performs per-column casts using Arrow, with fallbacks equivalent to
    `cast_arrow_array_as_column_schema`. Timestamp timezone normalization is
    applied when required by the dlt schema. The result preserves the input
    item's schema metadata and per-field metadata.

    Args:
        item: Arrow table or record batch to cast.
        columns: dlt table schema columns containing data type hints.
        caps: capabilities of the storage backend.
        tz: time zone identifier used to resolve timestamp hints.
        safe_arrow_conversion: if False, truncation and loss of precision is allowed.

    Returns:
        The same kind of Arrow object (Table or RecordBatch) with columns cast to
        the requested dlt types. Schema and field metadata are preserved.
    """
    from dlt.common.libs.pyarrow import pyarrow as pa

    schema = item.schema
    arrays_out = []
    fields_out = []

    for idx, fld in enumerate(schema):
        name = fld.name
        col_schema = columns[name]
        col_data = item.column(idx)

        # default: keep original
        out_array = col_data
        out_nullable = fld.nullable
        out_type = fld.type

        out_nullable = col_schema.get("nullable", fld.nullable)

        # final cast to the exact target type using fallbacks
        try:
            cast_arr = cast_arrow_array_as_column_schema(
                out_array, caps, col_schema, tz, safe_arrow_conversion
            )
        except PyToArrowConversionException as e:
            e.field_name = name
            raise

        out_array = cast_arr
        out_type = cast_arr.type

        # build new field preserving metadata
        new_field = pa.field(name, out_type, nullable=out_nullable, metadata=fld.metadata)
        fields_out.append(new_field)
        arrays_out.append(out_array)

    # preserve schema-level metadata
    new_schema = pa.schema(fields_out, metadata=schema.metadata)

    return item.__class__.from_arrays(arrays_out, schema=new_schema)


def _struct_timestamp_types(
    rows: TDataItems, arrow_types: Dict[str, Optional[Any]]
) -> Dict[str, Optional[Any]]:
    """Struct field types holding the zone the values carry, which pyarrow reads as UTC otherwise."""
    by_name = isinstance(rows[0], dict)
    struct_types = dict(arrow_types)
    for index, (name, arrow_type) in enumerate(arrow_types.items()):
        if not pyarrow.types.is_timestamp(arrow_type):
            continue
        # a dict row may omit the key and a tuple row may be short, both meaning no value
        values = (
            (row.get(name) for row in rows)
            if by_name
            else (row[index] if index < len(row) else None for row in rows)
        )
        aware = getattr(_first_non_none_value(values), "tzinfo", None) is not None
        struct_types[name] = pyarrow.timestamp(arrow_type.unit, UTC_NAME if aware else None)
    return struct_types


def _row_tuples_to_arrow_struct(
    rows: TDataItems,
    columns: TTableSchemaColumns,
    arrow_types: Dict[str, Optional[Any]],
    tz: str,
    caps: DestinationCapabilitiesContext = None,
) -> Optional[Any]:  # Optional[pyarrow.Table]
    """Converts row tuples to an arrow table in a single pass via a `StructArray`."""
    from dlt.common.libs.pyarrow import pyarrow as pa

    # json values typically need serialization fallbacks which fail the whole struct
    if any(
        arrow_types[name] is None or column.get("data_type") == "json"
        for name, column in columns.items()
    ):
        return None

    # the struct labels timestamps with the target zone, which shifts a naive value. they are
    # built in the zone the values carry and the timezone hint is applied after the flatten
    normalize_timestamps = tz != UTC_NAME and bool(rows)
    try:
        struct_types = (
            _struct_timestamp_types(rows, arrow_types) if normalize_timestamps else arrow_types
        )
        struct_fields = [pa.field(name, struct_types[name]) for name in columns]
        arrays = pa.array(rows, type=pa.struct(struct_fields)).flatten()
    except Exception as e:
        logger.debug(
            "Single-pass arrow conversion not possible, using per-column fallbacks:"
            f" {type(e).__name__}: {e}"
        )
        return None
    if normalize_timestamps:
        arrays = [
            (
                normalize_py_arrow_item_column(columns[field.name], field.type, array, tz, caps)[1]
                if pa.types.is_timestamp(field.type)
                else array
            )
            for field, array in zip(struct_fields, arrays)
        ]
    schema = pa.schema(
        pa.field(name, arrow_types[name], nullable=columns[name].get("nullable", True))
        for name in columns
    )
    return pa.Table.from_arrays(arrays, schema=schema)


def row_tuples_to_arrow(
    rows: TDataItems,
    caps: DestinationCapabilitiesContext,
    columns: TTableSchemaColumns,
    tz: str,
    safe_arrow_conversion: bool = True,
) -> Any:  # pyarrow.Table
    """Converts the rows to an arrow table using the columns schema.
    1. Pivot rows into columns.
    2. Convert columns to pyarrow arrays; coerce type and nullability; exclude types unsupported by arrow
    3. Create table
    4. Remove columns full of null values

    Args:
        rows: data items
        caps: capabilities of the storage backend
        columns: dlt hints about the table columns (e.g., data type, nullabe)
        tz: time zone identifier
        safe_arrow_conversion: if False, truncation and loss of precision is allowed
            ref: https://arrow.apache.org/docs/python/generated/pyarrow.compute.CastOptions.html#pyarrow.compute.CastOptions

    Returns:
        an arrow Table
    """
    from dlt.common.libs.pyarrow import pyarrow as pa

    # pyarrow struct conversion accepts only tuples and dicts, and tuple iteration is
    # ~3x faster than driver row objects (e.g. sqlalchemy Row) in the transpose below
    if rows and not isinstance(rows[0], (tuple, dict)):
        rows = [tuple(row) for row in rows]

    # compute target arrow types once; None when the type must be inferred from data
    arrow_types: Dict[str, Optional[Any]] = {
        name: get_py_arrow_datatype(column, caps, tz) if column.get("data_type") else None
        for name, column in columns.items()
    }

    if (
        arrow_table := _row_tuples_to_arrow_struct(rows, columns, arrow_types, tz, caps)
    ) is not None:
        return arrow_table

    transposed_rows = transpose_rows_to_columns(rows, column_names=columns.keys())

    arrow_arrays = []
    arrow_fields = []
    for column_name, column_data in transposed_rows.items():
        column_schema = columns[column_name]

        try:
            arrow_array = convert_array_to_arrow(
                column_data,
                caps,
                column_schema,
                tz,
                safe_arrow_conversion,
                arrow_types[column_name],
            )
        # TODO if converting to arrow fail, should we raise or skip column?
        except PyToArrowConversionException as e:
            e.field_name = column_name
            raise

        field = pa.field(
            name=column_name, type=arrow_array.type, nullable=column_schema.get("nullable", True)
        )
        arrow_arrays.append(arrow_array)
        arrow_fields.append(field)

    # NOTE careful when casting, modifying types, or enforcing schemas in place. Arrow issues are common
    # This can corrupt the data when writing to Parquet
    # ref: https://github.com/apache/arrow/issues/43146
    # ref: https://github.com/apache/arrow/issues/41667
    arrow_table = pa.Table.from_arrays(arrow_arrays, schema=pa.schema(arrow_fields))
    return arrow_table


class NameNormalizationCollision(ValueError):
    def __init__(self, reason: str) -> None:
        msg = f"Arrow column name collision after input data normalization. {reason}"
        super().__init__(msg)


def add_arrow_metadata(
    item: Union[pyarrow.Table, pyarrow.RecordBatch], metadata: dict[str, Any]
) -> pyarrow.Table:
    # Get current metadata or initialize empty
    schema = item.schema
    current = schema.metadata or {}

    # Convert new metadata to bytes and merge
    update = {k.encode("utf-8"): v.encode("utf-8") for k, v in metadata.items()}
    merged = current.copy()
    merged.update(update)

    # Apply updated schema
    new_schema = schema.with_metadata(merged)

    # Rebuild the object with updated schema
    if isinstance(item, pyarrow.Table):
        return pyarrow.Table.from_arrays(item.columns, schema=new_schema)
    else:  # RecordBatch
        return pyarrow.RecordBatch.from_arrays(item.columns, schema=new_schema)


def set_plus0000_timezone_to_utc(tbl: pyarrow.Table) -> pyarrow.Table:
    """
    Convert any +00:00 timestamp columns to UTC.
    Returns the original table object if nothing needed fixing.
    """
    arrays, fields = [], []
    changed = False

    for col, fld in zip(tbl.columns, tbl.schema):
        if pyarrow.types.is_timestamp(fld.type) and fld.type.tz == "+00:00":
            changed = True
            new_type = pyarrow.timestamp(fld.type.unit, "UTC")
            arrays.append(pyarrow.compute.cast(col, new_type))
            fields.append(pyarrow.field(fld.name, new_type, fld.nullable, fld.metadata))
        else:
            arrays.append(col)
            fields.append(fld)

    if not changed:
        return tbl

    new_schema = pyarrow.schema(fields, metadata=tbl.schema.metadata)
    return pyarrow.Table.from_arrays(arrays, schema=new_schema)


def cast_connectorx_temporal_columns(tbl: pyarrow.Table, tz: Optional[str] = None) -> pyarrow.Table:
    """
    Normalize connectorx temporal columns to microsecond precision.

    connectorx returns timestamps as date64 (milliseconds) on older versions and as
    timestamp[ns] on newer ones (the `arrow_stream` return type), and times as time64[ns].
    dlt and its destinations use microseconds and no database connectorx reads from has
    sub-microsecond resolution, so casting to `us` is lossless. date64 is rescaled to
    timestamp[us]; nanosecond timestamp and time64 columns are truncated to `us` with the
    timezone preserved; other units (ms, s, us) and unrelated columns are left untouched.

    Args:
        tbl: Input Arrow table.
        tz: Optional timezone to annotate date64 columns with (date64 carries no timezone).
            Existing timestamp columns keep their own timezone.

    Returns:
        A new table with the temporal columns cast to microsecond precision, or the original
        table if there was nothing to cast.
    """
    arrays, fields = [], []
    changed = False

    for col, fld in zip(tbl.columns, tbl.schema):
        col_type = fld.type
        if pyarrow.types.is_date64(col_type):
            # date64 is milliseconds since epoch; rescale to us and optionally annotate tz
            new_type = pyarrow.timestamp("us", tz)
        elif pyarrow.types.is_timestamp(col_type) and col_type.unit == "ns":
            # truncate ns timestamps to us, keeping the column's own timezone (other units kept)
            new_type = pyarrow.timestamp("us", col_type.tz)
        elif pyarrow.types.is_time64(col_type) and col_type.unit == "ns":
            new_type = pyarrow.time64("us")
        else:
            arrays.append(col)
            fields.append(fld)
            continue

        changed = True
        # safe=False truncates sub-us precision (lossless here: source DBs carry no nanoseconds)
        arrays.append(pyarrow.compute.cast(col, new_type, safe=False))
        fields.append(pyarrow.field(fld.name, new_type, fld.nullable, fld.metadata))

    if not changed:
        return tbl

    new_schema = pyarrow.schema(fields, metadata=tbl.schema.metadata)
    return pyarrow.Table.from_arrays(arrays, schema=new_schema)


def _is_arrow_uuid_extension(arrow_type: Any) -> bool:
    return (
        isinstance(arrow_type, pyarrow.BaseExtensionType)
        and arrow_type.extension_name == ARROW_UUID_EXTENSION_NAME
    )
