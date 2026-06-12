"""Tests for reaction JSON descriptor precomputation."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from descriptor_kit import DESCRIPTOR_KEYS, TDELTA_KEYS, compute_descriptors, compute_tdelta
from iqc_dashboard.app import ENERGY_UNIT_EV, build_selected_descriptor_dataframe
from iqc_dashboard.descriptor_precompute import build_precomputed_descriptor_dataframe


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
    )

    type_i_rows = precomputed_df[precomputed_df["source_json_row"] == 0]
    assert type_i_rows["prod_ni_o1"].tolist() == pytest.approx(
        [expected["prod_ni_o1"], expected["prod_ni_o1"]]
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
    )
    type_ii = compute_descriptors(
        read_example_xyz("type_II_reactant.xyz"),
        read_example_xyz("type_II_product.xyz"),
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
