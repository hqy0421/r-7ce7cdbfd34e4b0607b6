from copy import deepcopy
from typing import List

import pytest

from dlt.common.schema import Schema
from dlt.common.schema.typing import TColumnSchema, TTableSchema, TTableSchemaColumns
from dlt.common.exceptions import ValueErrorWithKnownValues
from dlt.common.schema.normalizers import configured_normalizers, import_normalizers
from dlt.common.schema.utils import (
    add_time_fidelity_companion_columns,
    exclude_dlt_entities,
    get_time_fidelity,
    time_fidelity_companion_columns,
    version_table,
    loads_table,
    pipeline_state_table,
    new_column,
)
from dlt.common.time import HINT_COMPANION_OF
from tests.common.utils import load_yml_case

NAME_NORMALIZER_REFS = (
    "tests.common.cases.normalizers.title_case",
    "tests.common.cases.normalizers.sql_upper",
    "tests.common.cases.normalizers.snake_no_x",
)


@pytest.fixture
def ethereum_schema() -> Schema:
    return Schema.from_dict(load_yml_case("schemas/eth/ethereum_schema_v11"))


@pytest.mark.parametrize("name_normalizer_ref", NAME_NORMALIZER_REFS)
def test_exclude_dlt_entities_remove_none(
    ethereum_schema: Schema, name_normalizer_ref: str
) -> None:
    _apply_name_normalizer(ethereum_schema, name_normalizer_ref)

    tables = list(ethereum_schema.tables.values())
    filtered = exclude_dlt_entities(
        tables,
        normalized_dlt_prefix=ethereum_schema._dlt_tables_prefix,
        exclude_dlt_tables=False,
        exclude_dlt_columns=False,
    )

    # verify all tables are still present
    filtered_names = {table["name"] for table in filtered}
    assert ethereum_schema.version_table_name in filtered_names
    assert ethereum_schema.loads_table_name in filtered_names
    blocks_table_name = _normalized_table_name(ethereum_schema, "blocks")
    blocks = _get_table(filtered, blocks_table_name)
    assert blocks_table_name in filtered_names

    # as well as dlt columns
    dlt_load_id_column = _normalized_column_name(ethereum_schema, "_dlt_load_id")
    dlt_id_column = _normalized_column_name(ethereum_schema, "_dlt_id")
    assert dlt_load_id_column in blocks["columns"]
    assert dlt_id_column in blocks["columns"]


@pytest.mark.parametrize("name_normalizer_ref", NAME_NORMALIZER_REFS)
def test_exclude_dlt_entities_remove_tables_only(
    ethereum_schema: Schema, name_normalizer_ref: str
) -> None:
    _apply_name_normalizer(ethereum_schema, name_normalizer_ref)

    tables = list(ethereum_schema.tables.values())
    filtered = exclude_dlt_entities(
        tables,
        normalized_dlt_prefix=ethereum_schema._dlt_tables_prefix,
        exclude_dlt_tables=True,
        exclude_dlt_columns=False,
    )

    filtered_names = {table["name"] for table in filtered}

    # verify tables got removed
    blocks_table_name = _normalized_table_name(ethereum_schema, "blocks")
    assert ethereum_schema.version_table_name not in filtered_names
    assert ethereum_schema.loads_table_name not in filtered_names
    assert blocks_table_name in filtered_names

    # no columns got removed
    dlt_load_id_column = _normalized_column_name(ethereum_schema, "_dlt_load_id")
    dlt_id_column = _normalized_column_name(ethereum_schema, "_dlt_id")
    blocks = _get_table(filtered, blocks_table_name)
    assert dlt_load_id_column in blocks["columns"]
    assert dlt_id_column in blocks["columns"]


@pytest.mark.parametrize("name_normalizer_ref", NAME_NORMALIZER_REFS)
def test_exclude_dlt_entities_remove_columns_only(
    ethereum_schema: Schema, name_normalizer_ref: str
) -> None:
    _apply_name_normalizer(ethereum_schema, name_normalizer_ref)
    blocks_table_name = _normalized_table_name(ethereum_schema, "blocks")
    dlt_load_id_column = _normalized_column_name(ethereum_schema, "_dlt_load_id")
    dlt_id_column = _normalized_column_name(ethereum_schema, "_dlt_id")
    tables = list(ethereum_schema.tables.values())

    filtered = exclude_dlt_entities(
        tables,
        normalized_dlt_prefix=ethereum_schema._dlt_tables_prefix,
        exclude_dlt_tables=False,
        exclude_dlt_columns=True,
    )

    filtered_names = {table["name"] for table in filtered}
    assert ethereum_schema.version_table_name in filtered_names
    assert ethereum_schema.loads_table_name in filtered_names

    blocks = _get_table(filtered, blocks_table_name)
    block_column_names = set(blocks["columns"])
    assert dlt_load_id_column not in block_column_names
    assert dlt_id_column not in block_column_names


@pytest.mark.parametrize("name_normalizer_ref", NAME_NORMALIZER_REFS)
def test_exclude_dlt_entities_remove_tables_and_columns(
    ethereum_schema: Schema, name_normalizer_ref: str
) -> None:
    _apply_name_normalizer(ethereum_schema, name_normalizer_ref)
    blocks_table_name = _normalized_table_name(ethereum_schema, "blocks")
    dlt_load_id_column = _normalized_column_name(ethereum_schema, "_dlt_load_id")
    dlt_id_column = _normalized_column_name(ethereum_schema, "_dlt_id")
    tables = list(ethereum_schema.tables.values())

    filtered = exclude_dlt_entities(
        tables,
        normalized_dlt_prefix=ethereum_schema._dlt_tables_prefix,
        exclude_dlt_tables=True,
        exclude_dlt_columns=True,
    )

    filtered_names = {table["name"] for table in filtered}
    assert ethereum_schema.version_table_name not in filtered_names
    assert ethereum_schema.loads_table_name not in filtered_names
    assert blocks_table_name in filtered_names

    blocks = _get_table(filtered, blocks_table_name)
    block_column_names = set(blocks["columns"])
    assert dlt_load_id_column not in block_column_names
    assert dlt_id_column not in block_column_names


# Helper functions
def _get_table(tables: List[TTableSchema], name: str) -> TTableSchema:
    return next(table for table in tables if table["name"] == name)


def _normalized_table_name(schema: Schema, table_name: str) -> str:
    return schema.naming.normalize_table_identifier(table_name)


def _normalized_column_name(schema: Schema, column_name: str) -> str:
    return schema.naming.normalize_identifier(column_name)


def _apply_name_normalizer(schema: Schema, name_normalizer_ref: str) -> None:
    schema._normalizers_config["names"] = name_normalizer_ref
    schema._normalizers_config["allow_identifier_change_on_table_with_data"] = True
    schema.update_normalizers()


# Tests for internal table column descriptions
def test_version_table_has_column_descriptions() -> None:
    """Test that _dlt_version table has descriptions for all columns"""
    table = version_table()

    # Verify table has description
    assert "description" in table
    assert table["description"] == "Created by DLT. Tracks schema updates"

    # Verify all columns have descriptions
    expected_columns = [
        "version",
        "engine_version",
        "inserted_at",
        "schema_name",
        "version_hash",
        "schema",
    ]
    for col_name in expected_columns:
        assert col_name in table["columns"], f"Column {col_name} missing from version_table"
        col = table["columns"][col_name]
        assert "description" in col, f"Column {col_name} missing description"
        assert isinstance(
            col["description"], str
        ), f"Column {col_name} description should be a string"
        assert len(col["description"]) > 0, f"Column {col_name} description should not be empty"


def test_loads_table_has_column_descriptions() -> None:
    """Test that _dlt_loads table has descriptions for all columns"""
    table = loads_table()

    # Verify table has description
    assert "description" in table
    assert table["description"] == "Created by DLT. Tracks completed loads"

    # Verify all columns have descriptions
    expected_columns = ["load_id", "schema_name", "status", "inserted_at", "schema_version_hash"]
    for col_name in expected_columns:
        assert col_name in table["columns"], f"Column {col_name} missing from loads_table"
        col = table["columns"][col_name]
        assert "description" in col, f"Column {col_name} missing description"
        assert isinstance(
            col["description"], str
        ), f"Column {col_name} description should be a string"
        assert len(col["description"]) > 0, f"Column {col_name} description should not be empty"


def test_pipeline_state_table_has_column_descriptions() -> None:
    """Test that _dlt_pipeline_state table has descriptions for all columns"""
    table = pipeline_state_table(add_dlt_id=True)

    # Verify table has description
    assert "description" in table
    assert table["description"] == "Created by DLT. Tracks pipeline state"

    # Verify all columns have descriptions
    expected_columns = [
        "version",
        "engine_version",
        "pipeline_name",
        "state",
        "created_at",
        "version_hash",
        "_dlt_load_id",
        "_dlt_id",
    ]
    for col_name in expected_columns:
        assert col_name in table["columns"], f"Column {col_name} missing from pipeline_state_table"
        col = table["columns"][col_name]
        assert "description" in col, f"Column {col_name} missing description"
        assert isinstance(
            col["description"], str
        ), f"Column {col_name} description should be a string"
        assert len(col["description"]) > 0, f"Column {col_name} description should not be empty"


def test_pipeline_state_table_without_dlt_id() -> None:
    """Test that _dlt_pipeline_state table without dlt_id has correct columns"""
    table = pipeline_state_table(add_dlt_id=False)

    # Verify _dlt_id is not present when add_dlt_id=False
    assert "_dlt_id" not in table["columns"]

    # Verify _dlt_load_id is still present and has description
    assert "_dlt_load_id" in table["columns"]
    assert "description" in table["columns"]["_dlt_load_id"]


def test_internal_tables_column_descriptions_are_not_empty() -> None:
    """Test that all internal table columns have non-empty descriptions"""
    tables = [
        ("_dlt_version", version_table()),
        ("_dlt_loads", loads_table()),
        ("_dlt_pipeline_state", pipeline_state_table(add_dlt_id=True)),
    ]

    for table_name, table in tables:
        for col_name, col in table["columns"].items():
            assert (
                "description" in col
            ), f"Table {table_name}, column {col_name} missing description"
            assert (
                col["description"] is not None and col["description"].strip() != ""
            ), f"Table {table_name}, column {col_name} has empty description"


def test_get_time_fidelity() -> None:
    assert get_time_fidelity(new_column("ts", "timestamp")) is None
    column = new_column("ts", "timestamp")
    column["x-source-timezone"] = "America/New_York"  # type: ignore[typeddict-unknown-key]
    # the policy defaults to `flag`
    assert get_time_fidelity(column) == ("America/New_York", "flag")
    column["x-local-time-ambiguity"] = "latest"  # type: ignore[typeddict-unknown-key]
    assert get_time_fidelity(column) == ("America/New_York", "latest")
    column["x-local-time-ambiguity"] = "maybe"  # type: ignore[typeddict-unknown-key]
    with pytest.raises(ValueErrorWithKnownValues):
        get_time_fidelity(column)


def test_time_fidelity_companion_columns() -> None:
    _, naming, _ = import_normalizers(configured_normalizers())
    column = new_column("event_ts", "timestamp")
    column["x-source-timezone"] = "America/New_York"  # type: ignore[typeddict-unknown-key]

    columns = time_fidelity_companion_columns("event_ts", column, naming)
    # the offset column always exists, so the shape of the table does not depend on the values
    assert set(columns) == {"utc_offset", "time_status"}
    offset_col = columns["utc_offset"]
    assert offset_col["name"] == "event_ts__utc_offset"
    assert offset_col["data_type"] == "text"
    assert offset_col["nullable"] is True
    assert offset_col[HINT_COMPANION_OF] == "event_ts"
    assert columns["time_status"]["name"] == "event_ts__time_status"

    # a policy that picks one reading does not mark the value, so there is no status column
    column["x-local-time-ambiguity"] = "earliest"  # type: ignore[typeddict-unknown-key]
    assert set(time_fidelity_companion_columns("event_ts", column, naming)) == {"utc_offset"}

    # a column outside of the mode gets none
    assert time_fidelity_companion_columns("ts", new_column("ts", "timestamp"), naming) == {}


def test_add_time_fidelity_companion_columns() -> None:
    _, naming, _ = import_normalizers(configured_normalizers())
    column = new_column("ts", "timestamp")
    column["x-source-timezone"] = "America/New_York"  # type: ignore[typeddict-unknown-key]
    columns = {"ts": column, "id": new_column("id", "bigint")}

    added = add_time_fidelity_companion_columns(columns, naming)
    assert set(added) == {"ts__utc_offset", "ts__time_status"}
    assert list(columns) == ["ts", "id", "ts__utc_offset", "ts__time_status"]

    # a second call adds nothing, so reading a table twice does not grow it
    assert add_time_fidelity_companion_columns(columns, naming) == {}

    # a field that is not a timestamp is not resolved, so it gets no companions
    text_column = new_column("label", "text")
    text_column["x-source-timezone"] = "America/New_York"  # type: ignore[typeddict-unknown-key]
    assert time_fidelity_companion_columns("label", text_column, naming) == {}
    # neither does an incomplete column, whose type is inferred later
    incomplete = {"name": "ts"}  # type: ignore[var-annotated]
    incomplete["x-source-timezone"] = "America/New_York"
    assert time_fidelity_companion_columns("ts", incomplete, naming) == {}
