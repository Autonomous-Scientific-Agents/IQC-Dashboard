"""Tests for reaction JSON descriptor precomputation."""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from descriptor_kit import (
    DESCRIPTOR_KEYS,
    REGIO_KEYS,
    TDELTA_KEYS,
    compute_descriptors,
    compute_tdelta,
)
from iqc_dashboard.app import ENERGY_UNIT_EV, build_selected_descriptor_dataframe
from iqc_dashboard.descriptor_precompute import (
    PRECOMPUTE_VERSION,
    build_precomputed_descriptor_dataframe,
)


EXAMPLE_DIR = Path(__file__).parent.parent / "descriptor_kit" / "example"


def read_example_xyz(name: str) -> str:
    return (EXAMPLE_DIR / name).read_text(encoding="utf-8")


def build_reaction_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ligand_pair": "bipy-alpha_alkyne-one",
                "stereo_type": "S",
                "insertion_type": "Type_I",
                "reaction_gibbs_kcal": -5.0,
                "reactant_gibbs": -100.0,
                "product_gibbs": -105.0,
                "reactant_geometry": read_example_xyz("type_I_reactant.xyz"),
                "product_geometry": read_example_xyz("type_I_product.xyz"),
                "reactant_configuration": "reactant_conf",
                "product_configuration": "product_conf",
            },
            {
                "ligand_pair": "bipy-alpha_alkyne-one",
                "stereo_type": "S",
                "insertion_type": "Type_II",
                "reaction_gibbs_kcal": -4.0,
                "reactant_gibbs": -101.0,
                "product_gibbs": -105.0,
                "reactant_geometry": read_example_xyz("type_II_reactant.xyz"),
                "product_geometry": read_example_xyz("type_II_product.xyz"),
                "reactant_configuration": "reactant_conf",
                "product_configuration": "product_conf",
            },
        ]
    )


@pytest.fixture(scope="module")
def precomputed_df() -> pd.DataFrame:
    return build_precomputed_descriptor_dataframe(build_reaction_source_df(), workers=1)


def test_precomputed_dataframe_preserves_source_and_dashboard_rows(precomputed_df):
    assert len(precomputed_df) == 4
    assert precomputed_df["reaction_role"].tolist() == [
        "reactant",
        "product",
        "reactant",
        "product",
    ]
    assert precomputed_df["source_json_row"].tolist() == [0, 0, 1, 1]
    assert set(DESCRIPTOR_KEYS).issubset(precomputed_df.columns)
    assert set(TDELTA_KEYS).issubset(precomputed_df.columns)
    assert precomputed_df["descriptor_precomputed"].all()

    source_df = build_reaction_source_df()
    assert precomputed_df.loc[0, "reactant_geometry"] == source_df.loc[
        0,
        "reactant_geometry",
    ]
    assert precomputed_df.loc[1, "product_geometry"] == source_df.loc[
        0,
        "product_geometry",
    ]


def test_precomputed_single_descriptor_matches_descriptor_kit(precomputed_df):
    expected = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
    )

    type_i_rows = precomputed_df[precomputed_df["source_json_row"] == 0]
    assert type_i_rows["prod_ni_o1"].tolist() == pytest.approx(
        [expected["prod_ni_o1"], expected["prod_ni_o1"]]
    )
    assert type_i_rows["reac_sigma_ortho_pyA"].tolist() == pytest.approx(
        [expected["reac_sigma_ortho_pyA"], expected["reac_sigma_ortho_pyA"]]
    )

    with patch(
        "iqc_dashboard.app.compute_selected_single_descriptor_value",
        side_effect=AssertionError("live descriptor calculation should not run"),
    ):
        records = build_selected_descriptor_dataframe(
            precomputed_df,
            "prod_ni_o1",
        )

    assert len(records) == 2
    assert records.loc[0, "value"] == pytest.approx(expected["prod_ni_o1"])
    assert "precomputed descriptor_kit" in records.loc[0, "atom_summary"]

    records_ev = build_selected_descriptor_dataframe(
        precomputed_df,
        "prod_ni_o1",
        energy_unit=ENERGY_UNIT_EV,
    )
    assert records_ev.loc[0, "deltaG"] == pytest.approx(-5.0 / 23.0605)


def test_precomputed_keyword_filters_remain_role_specific(precomputed_df):
    filtered_df = precomputed_df.copy()
    product_mask = filtered_df["reaction_role"] == "product"
    filtered_df.loc[product_mask, "unique_name"] = filtered_df.loc[
        product_mask,
        "unique_name",
    ].str.replace("alpha", "beta", regex=False)

    records = build_selected_descriptor_dataframe(
        filtered_df,
        "prod_ni_o1",
        reactant_keywords=["alpha"],
        product_keywords=["beta"],
    )
    excluded = build_selected_descriptor_dataframe(
        filtered_df,
        "prod_ni_o1",
        reactant_keywords=["beta"],
        product_keywords=["alpha"],
    )

    assert len(records) == 2
    assert excluded.empty


def test_precomputed_tdelta_matches_descriptor_kit_without_recalculation(precomputed_df):
    type_i = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
    )
    type_ii = compute_descriptors(
        read_example_xyz("type_II_reactant.xyz"),
        read_example_xyz("type_II_product.xyz"),
        stereo_type="S",
    )
    expected = compute_tdelta(type_i, type_ii)

    type_i_rows = precomputed_df[precomputed_df["source_json_row"] == 0]
    type_ii_rows = precomputed_df[precomputed_df["source_json_row"] == 1]
    assert type_i_rows["tdelta_ni_Cb"].tolist() == pytest.approx(
        [expected["tdelta_ni_Cb"], expected["tdelta_ni_Cb"]]
    )
    assert type_ii_rows["tdelta_ni_Cb"].isna().all()

    with patch(
        "iqc_dashboard.app.compute_selected_single_descriptor_value",
        side_effect=AssertionError("live descriptor calculation should not run"),
    ):
        records = build_selected_descriptor_dataframe(
            precomputed_df,
            "tdelta_ni_Cb",
        )

    assert len(records) == 1
    assert records.loc[0, "value"] == pytest.approx(expected["tdelta_ni_Cb"])
    assert records.loc[0, "deltaG"] == pytest.approx(-1.0)
    assert "Precomputed regioisomer" in records.loc[0, "atom_summary"]


def _equal_or_both_nan(actual, expected) -> bool:
    if math.isnan(actual) and math.isnan(expected):
        return True
    return actual == pytest.approx(expected)


def _reactant_row(df: pd.DataFrame, source_json_row: int) -> pd.Series:
    rows = df[
        (df["source_json_row"] == source_json_row)
        & (df["reaction_role"] == "reactant")
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_precomputed_dataframe_includes_regio_columns(precomputed_df):
    assert set(REGIO_KEYS).issubset(precomputed_df.columns)


def test_precomputed_version_signals_regio_schema(precomputed_df):
    assert PRECOMPUTE_VERSION >= 2
    assert (precomputed_df["descriptor_precompute_version"] == PRECOMPUTE_VERSION).all()


def test_precomputed_type_i_regio_swaps_and_negates(precomputed_df):
    row = _reactant_row(precomputed_df, 0)  # Type_I reaction
    # The swap must be observable, not vacuously NaN==NaN.
    assert math.isfinite(row["reac_B5_R1"]) and math.isfinite(row["reac_B5_R2"])
    assert row["reac_B5_R1"] != row["reac_B5_R2"]
    assert math.isfinite(row["reac_dvol_alkyne"])
    assert _equal_or_both_nan(row["reac_B5_Ralpha"], row["reac_B5_R2"])
    assert _equal_or_both_nan(row["reac_B5_Rbeta"], row["reac_B5_R1"])
    assert _equal_or_both_nan(row["reac_dvol_alkyne_ab"], -row["reac_dvol_alkyne"])


def test_precomputed_type_ii_regio_passes_through(precomputed_df):
    row = _reactant_row(precomputed_df, 1)  # Type_II reaction
    assert math.isfinite(row["reac_B5_R1"]) and math.isfinite(row["reac_B5_R2"])
    assert row["reac_B5_R1"] != row["reac_B5_R2"]
    assert math.isfinite(row["reac_dvol_alkyne"])
    assert _equal_or_both_nan(row["reac_B5_Ralpha"], row["reac_B5_R1"])
    assert _equal_or_both_nan(row["reac_B5_Rbeta"], row["reac_B5_R2"])
    assert _equal_or_both_nan(row["reac_dvol_alkyne_ab"], row["reac_dvol_alkyne"])


def test_precompute_unknown_insertion_type_yields_nan_regio():
    source = build_reaction_source_df()
    extra = source.iloc[[0]].copy()
    extra["insertion_type"] = "Type_III"
    extra["ligand_pair"] = "other-pair"
    source = pd.concat([source, extra], ignore_index=True)

    out = build_precomputed_descriptor_dataframe(source, workers=1)
    row = _reactant_row(out, 2)  # the unknown-insertion-type reaction
    for key in REGIO_KEYS:
        assert math.isnan(row[key])


def test_regio_columns_round_trip_through_parquet(precomputed_df, tmp_path):
    parquet_path = tmp_path / "regio_descriptors.parquet"
    precomputed_df.to_parquet(parquet_path, index=False)
    loaded_df = pd.read_parquet(parquet_path)

    assert set(REGIO_KEYS).issubset(loaded_df.columns)
    for key in REGIO_KEYS:
        assert str(loaded_df[key].dtype) == "float64"
        loaded = loaded_df[key].to_numpy(float)
        original = precomputed_df[key].to_numpy(float)
        assert ((loaded == original) | (np.isnan(loaded) & np.isnan(original))).all()


def test_precomputed_regio_descriptor_is_selectable_without_recalculation(precomputed_df):
    reactant_row = _reactant_row(precomputed_df, 0)  # Type_I reaction

    with patch(
        "iqc_dashboard.app.compute_selected_single_descriptor_value",
        side_effect=AssertionError("live descriptor calculation should not run"),
    ):
        records = build_selected_descriptor_dataframe(precomputed_df, "reac_B5_Ralpha")

    assert len(records) == 2  # one reactant record per regioisomer reaction
    assert set(records["role"]) == {"reactant"}
    type_i_record = records[records["variant"].astype(str) == "Type_I"]
    assert len(type_i_record) == 1
    assert type_i_record.iloc[0]["value"] == pytest.approx(reactant_row["reac_B5_Ralpha"])
    assert type_i_record.iloc[0]["deltaG"] == pytest.approx(-5.0)


def test_precomputed_dataframe_round_trips_through_parquet(precomputed_df, tmp_path):
    parquet_path = tmp_path / "reaction_descriptors.parquet"
    precomputed_df.to_parquet(parquet_path, index=False)
    loaded_df = pd.read_parquet(parquet_path)

    assert loaded_df.shape == precomputed_df.shape
    assert loaded_df["prod_ni_o1"].tolist() == pytest.approx(
        precomputed_df["prod_ni_o1"].tolist()
    )
    assert loaded_df["reactant_geometry"].tolist() == precomputed_df[
        "reactant_geometry"
    ].tolist()


def test_descriptor_tasks_carry_insertion_type():
    from iqc_dashboard.descriptor_precompute import _descriptor_tasks

    source = build_reaction_source_df()
    tasks = list(_descriptor_tasks(source))
    assert all(len(task) == 4 for task in tasks)
    assert tasks[0][3] == str(source.loc[0, "insertion_type"])
    assert tasks[1][3] == str(source.loc[1, "insertion_type"])
