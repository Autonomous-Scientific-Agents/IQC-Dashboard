"""Tests for molecular descriptor calculations."""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from iqc_dashboard.app import (
    build_descriptor_dataframe,
    extract_descriptor_keyword_options,
)


def test_build_descriptor_dataframe_calculates_reactant_n_ni_c_angles():
    """Reactant descriptor returns both N-Ni-C angles."""
    reactant_xyz = """8
reactant
Ni 0.0 0.0 0.0
N -2.0 0.0 0.0
N 2.0 0.0 0.0
C 0.0 2.0 0.0
C -2.0 1.3 0.0
C -3.2 0.0 0.0
C 2.0 1.3 0.0
C 3.2 0.0 0.0
"""
    df = pd.DataFrame(
        {
            "unique_name": ["bipy-test_alkyne-test_reactant_conf1"],
            "initial_smiles": ["reactant-smiles"],
            "opt_smiles": ["reactant-opt-smiles"],
            "opt_xyz": [reactant_xyz],
        }
    )

    descriptors = build_descriptor_dataframe(df)
    angle_records = descriptors[
        descriptors["descriptor_id"] == "reactant_n_ni_c_angles"
    ].sort_values("variant")

    assert angle_records["variant"].tolist() == ["N_A-Ni-C_beta", "N_B-Ni-C_beta"]
    assert angle_records["value"].tolist() == pytest.approx([90.0, 90.0])
    assert angle_records["smiles"].tolist() == [
        "reactant-opt-smiles",
        "reactant-opt-smiles",
    ]


def test_build_descriptor_dataframe_calculates_product_descriptors():
    """Product descriptors identify N donors, O1, C_beta, and the bpy plane."""
    product_xyz = """9
product
Ni 0.0 0.0 0.2
N -2.0 0.0 0.0
N 2.4 0.0 0.0
O 0.0 2.0 0.2
C 0.0 -2.0 0.2
C -2.0 1.3 0.0
C -3.2 0.0 0.0
C 2.4 1.3 0.0
C 3.6 0.0 0.0
"""
    df = pd.DataFrame(
        {
            "unique_name": ["bipy-test_alkyne-test_product_conf1"],
            "initial_smiles": ["product-smiles"],
            "opt_xyz": [product_xyz],
        }
    )

    descriptors = build_descriptor_dataframe(df)
    values = {
        descriptor_id: descriptor_rows["value"].iloc[0]
        for descriptor_id, descriptor_rows in descriptors.groupby("descriptor_id")
    }

    assert values["product_o1_ni_c_beta_angle"] == pytest.approx(180.0)
    assert values["product_mean_n_ni_o1_angle"] == pytest.approx(90.0)
    assert values["product_ni_n_distance_difference"] == pytest.approx(
        ((2.4**2 + 0.2**2) ** 0.5) - ((2.0**2 + 0.2**2) ** 0.5)
    )
    assert values["product_ni_bpy_plane_distance"] == pytest.approx(0.2)


def test_descriptor_keyword_filters_are_role_specific():
    """Descriptor keyword options and filters are separated by reactant/product role."""
    shared_xyz = """5
complex
Ni 0.0 0.0 0.0
N -2.0 0.0 0.0
N 2.0 0.0 0.0
C 0.0 2.0 0.0
O 0.0 -2.0 0.0
"""
    df = pd.DataFrame(
        {
            "unique_name": [
                "bipy-alpha_alkyne-one_reactant_conf1",
                "bipy-beta_alkyne-two_product_conf1",
            ],
            "initial_smiles": ["alpha-reactant", "beta-product"],
            "opt_xyz": [shared_xyz, shared_xyz],
        }
    )

    assert "alpha" in extract_descriptor_keyword_options(df, "reactant")
    assert "beta" not in extract_descriptor_keyword_options(df, "reactant")
    assert "beta" in extract_descriptor_keyword_options(df, "product")

    descriptors = build_descriptor_dataframe(
        df,
        reactant_keywords=["alpha"],
        product_keywords=["beta"],
    )

    assert set(descriptors["role"]) == {"reactant", "product"}
