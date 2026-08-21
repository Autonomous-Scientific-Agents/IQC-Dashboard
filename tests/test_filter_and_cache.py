"""Tests for text-filter SQL building and session-level data caching."""

from pathlib import Path

import pandas as pd
import pytest

from iqc_dashboard.app import (
    ENERGY_UNIT_EV,
    ENERGY_UNIT_KCAL,
    EV_TO_KCAL_MOL,
    DataManager,
    apply_pandas_text_filter,
    build_filter_fingerprint,
    build_selected_descriptor_dataframe,
    build_text_filter_sql,
    convert_descriptor_records_energy_unit,
    descriptor_delta_record_columns,
    fingerprinted_reaction_table,
    is_simple_text_pattern,
    load_session_filtered_data,
)
from iqc_dashboard.descriptor_precompute import build_precomputed_descriptor_dataframe


EXAMPLE_DIR = Path(__file__).parent.parent / "descriptor_kit" / "example"


@pytest.fixture(autouse=True)
def _fresh_duckdb_connection():
    """Drop any mocked connection cached by other test modules."""
    DataManager.get_connection.clear()
    yield


NO_FILTERS = {
    "formula": None,
    "opt_converged": None,
    "smiles_changed": None,
    "number_of_imaginary_max": None,
    "text_filter": None,
}


@pytest.fixture
def special_names_parquet(temp_dir):
    """Parquet file whose names contain ILIKE wildcards and quotes."""
    df = pd.DataFrame(
        {
            "unique_name": [
                "mol_100%_yield",
                "mol_100X_yield",
                "foo_bar",
                "fooXbar",
                "o'brien",
            ],
            "initial_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "opt_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "formula": ["C1", "C2", "C3", "C4", "C5"],
        }
    )
    file_path = Path(temp_dir) / "special_names.parquet"
    df.to_parquet(file_path, index=False)
    return str(file_path)


class TestTextFilterSql:
    def test_is_simple_text_pattern(self):
        assert is_simple_text_pattern("Ni-test")
        assert is_simple_text_pattern("abc*def")
        assert not is_simple_text_pattern("mol_(001|002)")
        assert not is_simple_text_pattern("a.b")
        assert not is_simple_text_pattern("a\\d")

    def test_simple_pattern_uses_parameterized_ilike(self):
        condition, params = build_text_filter_sql("100%_a")
        assert "ILIKE ?" in condition
        assert "ESCAPE" in condition
        # Wildcards must be escaped so they match literally.
        assert params == ["%100\\%\\_a%"] * 3

    def test_regex_pattern_uses_parameterized_regexp(self):
        condition, params = build_text_filter_sql("mol_(001|002)")
        assert "regexp_matches" in condition
        assert "?" in condition
        assert params == ["mol_(001|002)"] * 3

    def test_apply_pandas_text_filter_invalid_regex(self):
        df = pd.DataFrame({"unique_name": ["a"]})
        assert apply_pandas_text_filter(df, "(") is None

    def test_apply_pandas_text_filter_matches(self):
        df = pd.DataFrame({"unique_name": ["mol_001", "mol_002", "other"]})
        filtered = apply_pandas_text_filter(df, "mol_00[12]")
        assert filtered["unique_name"].tolist() == ["mol_001", "mol_002"]


class TestGetFilteredDataTextFilter:
    def _manager(self, temp_dir, parquet_path):
        dm = DataManager(temp_dir)
        dm.parquet_files = [parquet_path]
        return dm

    def test_percent_matches_literally(self, temp_dir, special_names_parquet):
        dm = self._manager(temp_dir, special_names_parquet)
        result = dm.get_filtered_data(text_filter="100%")
        assert result["unique_name"].tolist() == ["mol_100%_yield"]

    def test_underscore_matches_literally(self, temp_dir, special_names_parquet):
        dm = self._manager(temp_dir, special_names_parquet)
        result = dm.get_filtered_data(text_filter="foo_bar")
        assert result["unique_name"].tolist() == ["foo_bar"]

    def test_single_quote_is_safe(self, temp_dir, special_names_parquet):
        dm = self._manager(temp_dir, special_names_parquet)
        result = dm.get_filtered_data(text_filter="o'brien")
        assert result["unique_name"].tolist() == ["o'brien"]

    def test_regex_alternation(self, temp_dir, special_names_parquet):
        dm = self._manager(temp_dir, special_names_parquet)
        result = dm.get_filtered_data(text_filter="^(foo_bar|fooXbar)$")
        assert sorted(result["unique_name"].tolist()) == ["fooXbar", "foo_bar"]

    def test_python_only_regex_falls_back_to_pandas(
        self, temp_dir, special_names_parquet
    ):
        dm = self._manager(temp_dir, special_names_parquet)
        # Lookbehind is unsupported by DuckDB's RE2 engine.
        result = dm.get_filtered_data(text_filter="(?<=foo)Xbar")
        assert result["unique_name"].tolist() == ["fooXbar"]

    def test_invalid_regex_returns_empty(self, temp_dir, special_names_parquet):
        dm = self._manager(temp_dir, special_names_parquet)
        result = dm.get_filtered_data(text_filter="(")
        assert result.empty

    def test_text_filter_combines_with_column_filters(
        self, temp_dir, special_names_parquet
    ):
        dm = self._manager(temp_dir, special_names_parquet)
        result = dm.get_filtered_data(formula="C3", text_filter="foo")
        assert result["unique_name"].tolist() == ["foo_bar"]


class TestSessionFilteredDataCache:
    def test_reuses_cached_frame_for_same_filters(
        self, temp_dir, sample_parquet_file
    ):
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]
        state = {}

        df_first, fingerprint_first = load_session_filtered_data(
            dm, dict(NO_FILTERS), session_state=state
        )
        df_second, fingerprint_second = load_session_filtered_data(
            dm, dict(NO_FILTERS), session_state=state
        )

        assert fingerprint_first == fingerprint_second
        assert df_second is df_first  # no re-query, no copy
        assert len(df_first) == 3

    def test_requeries_when_filters_change(self, temp_dir, sample_parquet_file):
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]
        state = {}

        df_all, fingerprint_all = load_session_filtered_data(
            dm, dict(NO_FILTERS), session_state=state
        )
        converged_filters = dict(NO_FILTERS, opt_converged=True)
        df_converged, fingerprint_converged = load_session_filtered_data(
            dm, converged_filters, session_state=state
        )

        assert fingerprint_all != fingerprint_converged
        assert len(df_all) == 3
        assert len(df_converged) == 2

    def test_fingerprint_tracks_files_and_filters(self):
        base = build_filter_fingerprint("hash-a", dict(NO_FILTERS))
        assert build_filter_fingerprint("hash-a", dict(NO_FILTERS)) == base
        assert build_filter_fingerprint("hash-b", dict(NO_FILTERS)) != base
        assert (
            build_filter_fingerprint("hash-a", dict(NO_FILTERS, formula="H2O"))
            != base
        )


class TestFingerprintedReactionTable:
    def test_error_is_returned_as_payload(self):
        df = pd.DataFrame({"unique_name": ["mol_001"]})
        payload = fingerprinted_reaction_table("test-fingerprint-error", df, ENERGY_UNIT_KCAL)
        assert payload["table"].empty
        assert "G_eV" in payload["error"]

    def test_table_is_returned_for_valid_data(self, sample_parquet_file):
        df = pd.read_parquet(sample_parquet_file)
        # Rename rows into the reaction naming scheme with a CO2 entry.
        df["unique_name"] = [
            "bipy-a_x_reactant_1",
            "bipy-a_x_product_1",
            "co2",
        ]
        payload = fingerprinted_reaction_table("test-fingerprint-ok", df, ENERGY_UNIT_KCAL)
        assert payload["error"] is None
        assert len(payload["table"]) == 1


class TestConvertDescriptorRecordsEnergyUnit:
    def test_kcal_is_passthrough(self):
        records = pd.DataFrame(
            {"deltaG": [-5.0], "deltaG_unit": [ENERGY_UNIT_KCAL]},
            columns=descriptor_delta_record_columns(),
        )
        converted = convert_descriptor_records_energy_unit(records, ENERGY_UNIT_KCAL)
        assert converted is records

    def test_ev_conversion_matches_direct_computation(self):
        def read_example_xyz(name: str) -> str:
            return (EXAMPLE_DIR / name).read_text(encoding="utf-8")

        source_df = pd.DataFrame(
            [
                {
                    "ligand_pair": "bipy-alpha_alkyne-one",
                    "stereo_type": "S",
                    "insertion_type": "Type_I",
                    "reaction_gibbs_kcal": -5.0,
                    "reactant_geometry": read_example_xyz("type_I_reactant.xyz"),
                    "product_geometry": read_example_xyz("type_I_product.xyz"),
                },
            ]
        )
        precomputed_df = build_precomputed_descriptor_dataframe(source_df, workers=1)

        records_kcal = build_selected_descriptor_dataframe(
            precomputed_df, "prod_ni_o1", energy_unit=ENERGY_UNIT_KCAL
        )
        records_ev = build_selected_descriptor_dataframe(
            precomputed_df, "prod_ni_o1", energy_unit=ENERGY_UNIT_EV
        )
        converted = convert_descriptor_records_energy_unit(
            records_kcal, ENERGY_UNIT_EV
        )

        assert len(converted) == len(records_ev) > 0
        assert converted["deltaG"].tolist() == pytest.approx(
            records_ev["deltaG"].tolist()
        )
        assert converted["deltaG"].iloc[0] == pytest.approx(-5.0 / EV_TO_KCAL_MOL)
        assert (converted["deltaG_unit"] == ENERGY_UNIT_EV).all()
        # Descriptor values themselves are unit-independent.
        assert converted["value"].tolist() == pytest.approx(
            records_kcal["value"].tolist()
        )
