"""Tests for descriptor_kit-backed molecular descriptor calculations."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from descriptor_kit import DESCRIPTOR_KEYS, REGIO_KEYS, compute_descriptors, compute_tdelta

from iqc_dashboard.app import (
    DESCRIPTOR_DEFINITIONS,
    build_descriptor_delta_hover_html,
    build_descriptor_dataframe,
    build_descriptor_reaction_pairs,
    build_descriptor_value_options,
    build_selected_descriptor_dataframe,
    compact_xyz_for_browser,
    extract_descriptor_keyword_options,
    get_descriptor_definition,
)


EXAMPLE_DIR = Path(__file__).parent.parent / "descriptor_kit" / "example"


def read_example_xyz(name: str) -> str:
    """Read one bundled descriptor_kit example XYZ file."""
    return (EXAMPLE_DIR / name).read_text(encoding="utf-8")


def build_example_reaction_df() -> pd.DataFrame:
    """Build Type_I and Type_II paired reaction rows from descriptor_kit examples."""
    rows = []
    for source_json_row, insertion_type in [(0, "Type_I"), (1, "Type_II")]:
        for reaction_role in ("reactant", "product"):
            xyz_name = f"type_{'I' if insertion_type == 'Type_I' else 'II'}_{reaction_role}.xyz"
            rows.append(
                {
                    "source_json_row": source_json_row,
                    "reaction_role": reaction_role,
                    "unique_name": (
                        f"bipy-alpha_alkyne-one_{reaction_role}_"
                        f"intermediate_{insertion_type}_{source_json_row}"
                    ),
                    "ligand_pair": "bipy-alpha_alkyne-one",
                    "stereo_type": "S",
                    "insertion_type": insertion_type,
                    "reaction_gibbs_kcal": -5.0 + source_json_row,
                    "opt_xyz": read_example_xyz(xyz_name),
                    "initial_smiles": f"{reaction_role}-smiles",
                    "opt_smiles": f"{reaction_role}-opt-smiles",
                }
            )
    return pd.DataFrame(rows)


def test_build_descriptor_dataframe_uses_descriptor_kit_single_reaction():
    """Descriptor dataframe uses descriptor_kit values for paired rows."""
    df = build_example_reaction_df().iloc[:2].copy()

    descriptors = build_descriptor_dataframe(df)
    expected = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
    )

    values = {
        descriptor_id: descriptor_rows["value"].iloc[0]
        for descriptor_id, descriptor_rows in descriptors.groupby("descriptor_id")
    }

    assert values["reac_cc_triple_len"] == pytest.approx(expected["reac_cc_triple_len"])
    assert values["prod_o_ni_bite_Cb"] == pytest.approx(expected["prod_o_ni_bite_Cb"])
    assert set(descriptors["role"]) >= {"reactant", "product"}
    assert "reactant-opt-smiles" in descriptors["smiles"].tolist()
    assert "product-opt-smiles" in descriptors["smiles"].tolist()


def test_build_selected_descriptor_dataframe_uses_product_delta_g_plot_data():
    """Selected product descriptor records plot descriptor value against ΔG."""
    df = build_example_reaction_df().iloc[:2].copy()

    descriptors = build_selected_descriptor_dataframe(df, "prod_ni_o1")
    expected = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
    )

    assert len(descriptors) == 1
    row = descriptors.iloc[0]
    assert row["role"] == "product"
    assert row["value"] == pytest.approx(expected["prod_ni_o1"])
    assert row["deltaG"] == pytest.approx(-5.0)
    assert row["smiles"] == "product-opt-smiles"
    assert "_product_" in row["unique_name"]


def test_build_selected_descriptor_dataframe_uses_reactant_geometry():
    """Reactant descriptors use reactant geometry and the same reaction ΔG."""
    df = build_example_reaction_df().iloc[:2].copy()

    descriptors = build_selected_descriptor_dataframe(df, "reac_cc_triple_len")
    expected = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
    )

    assert len(descriptors) == 1
    row = descriptors.iloc[0]
    assert row["role"] == "reactant"
    assert row["value"] == pytest.approx(expected["reac_cc_triple_len"])
    assert row["deltaG"] == pytest.approx(-5.0)
    assert row["smiles"] == "reactant-opt-smiles"
    assert "_reactant_" in row["unique_name"]


def test_new_alpha_beta_descriptors_are_registered_and_live_calculated():
    """New PyA/PyB descriptors are available through descriptor_kit and the UI path."""
    assert "reac_dB5_bpy" in DESCRIPTOR_KEYS
    assert "reac_sigma_ortho_pyA" in DESCRIPTOR_KEYS
    assert "reac_B5_ortho_pyA" in DESCRIPTOR_KEYS
    assert "reac_abs_dB5_bpy" not in DESCRIPTOR_KEYS

    df = build_example_reaction_df().iloc[:2].copy()
    descriptors = build_selected_descriptor_dataframe(df, "reac_sigma_ortho_pyA")
    expected = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
    )

    assert len(descriptors) == 1
    assert descriptors.iloc[0]["value"] == pytest.approx(expected["reac_sigma_ortho_pyA"])


def test_regio_alpha_beta_descriptors_are_registered_as_reactant_descriptors():
    """The 16 α/β regio descriptors are selectable reactant descriptors."""
    defined_ids = {descriptor["id"] for descriptor in DESCRIPTOR_DEFINITIONS}
    assert set(REGIO_KEYS).issubset(defined_ids)

    for key in REGIO_KEYS:
        definition = get_descriptor_definition(key)
        assert definition is not None
        assert definition["role"] == "reactant"

    expected_units = {
        "reac_dvol_alkyne_ab": "angstrom^3",
        "reac_dvbur_substituent_ab": "percent",
        "reac_dni_c_signed_ab": "angstrom",
        "reac_dbend_alkyne_ab": "deg",
        "reac_dccr_angle_ab": "deg",
        "reac_dnicr_angle_ab": "deg",
        "reac_slippage_ab": "angstrom",
        "reac_dni_firstatom_ab": "angstrom",
        "reac_B5_Ralpha": "angstrom",
        "reac_B5_Rbeta": "angstrom",
        "reac_L_Ralpha": "angstrom",
        "reac_L_Rbeta": "angstrom",
        "reac_bend_Ralpha": "deg",
        "reac_bend_Rbeta": "deg",
        "reac_ni_firstatom_Ralpha": "angstrom",
        "reac_ni_firstatom_Rbeta": "angstrom",
    }
    assert set(expected_units) == set(REGIO_KEYS)
    for key, unit in expected_units.items():
        assert get_descriptor_definition(key)["unit"] == unit, key


def test_regio_descriptors_registered_once_and_always_offered():
    ids = [descriptor["id"] for descriptor in DESCRIPTOR_DEFINITIONS]
    assert set(REGIO_KEYS).issubset(ids)
    assert len(ids) == len(set(ids))  # no double-counting
    for key in REGIO_KEYS:
        assert get_descriptor_definition(key)["role"] == "reactant"


def test_regio_descriptor_labels_render_alpha_beta_tokens():
    """α/β dropdown labels read as Rα/Rβ/α-β, not raw Ralpha/Rbeta/ab tokens."""
    assert get_descriptor_definition("reac_B5_Ralpha")["label"] == "Reactant: B5 Rα"
    assert get_descriptor_definition("reac_B5_Rbeta")["label"] == "Reactant: B5 Rβ"
    assert get_descriptor_definition("reac_dvol_alkyne_ab")["label"].endswith("α/β")

    for key in REGIO_KEYS:
        label = get_descriptor_definition(key)["label"]
        assert "Ralpha" not in label
        assert "Rbeta" not in label
        assert not label.endswith(" ab")


def test_carboxylate_tilt_descriptor_uses_distance_units():
    """The migrated carboxylate tilt is an O2 plane distance, not an angle."""
    descriptor = get_descriptor_definition("prod_carboxylate_tilt")

    assert descriptor is not None
    assert descriptor["unit"] == "angstrom"


def test_build_descriptor_dataframe_includes_tdelta_descriptors():
    """Type_I and Type_II reaction rows produce pair-level tdelta descriptors."""
    df = build_example_reaction_df()

    descriptors = build_descriptor_dataframe(df)
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

    tdelta_rows = descriptors[descriptors["descriptor_id"] == "tdelta_ni_Cb"]

    assert not tdelta_rows.empty
    assert tdelta_rows["role"].iloc[0] == "pair"
    assert tdelta_rows["value"].iloc[0] == pytest.approx(expected["tdelta_ni_Cb"])
    assert "Type_I - Type_II" in tdelta_rows["variant"].iloc[0]


def test_build_selected_descriptor_dataframe_includes_tdelta_delta_g():
    """Selected tdelta descriptor records use Type_I - Type_II ΔG."""
    df = build_example_reaction_df()

    descriptors = build_selected_descriptor_dataframe(df, "tdelta_ni_Cb")
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

    assert len(descriptors) == 1
    row = descriptors.iloc[0]
    assert row["role"] == "pair"
    assert row["value"] == pytest.approx(expected["tdelta_ni_Cb"])
    assert row["deltaG"] == pytest.approx(-1.0)
    assert "Type_I - Type_II" in row["variant"]


def test_descriptor_keyword_filters_are_role_specific():
    """Descriptor keyword options and filters are separated by reactant/product role."""
    df = build_example_reaction_df()
    df.loc[df["reaction_role"] == "product", "unique_name"] = df.loc[
        df["reaction_role"] == "product",
        "unique_name",
    ].str.replace("alpha", "beta", regex=False)

    assert "alpha" in extract_descriptor_keyword_options(df, "reactant")
    assert "beta" not in extract_descriptor_keyword_options(df, "reactant")
    assert "beta" in extract_descriptor_keyword_options(df, "product")

    descriptors = build_descriptor_dataframe(
        df,
        reactant_keywords=["alpha"],
        product_keywords=["beta"],
        max_pairs=1,
    )
    pairs = build_descriptor_reaction_pairs(
        df,
        reactant_keywords=["alpha"],
        product_keywords=["beta"],
        max_pairs=1,
    )

    assert pairs
    assert not descriptors.empty
    assert descriptors["reaction_id"].nunique() == 1


def test_descriptor_value_options_preserve_sorted_species_values():
    """Descriptor value navigation exposes each species value as a selectable row."""
    records = pd.DataFrame(
        [
            {
                "descriptor_id": "prod_ni_o1",
                "role": "product",
                "reaction_id": "10",
                "unique_name": "product_a",
                "value": 1.23456,
            },
            {
                "descriptor_id": "prod_ni_o1",
                "role": "product",
                "reaction_id": "11",
                "unique_name": "product_b",
                "value": 1.5,
            },
        ]
    )

    options = build_descriptor_value_options(records, "Å")

    assert options["_descriptor_value_id"].tolist() == [
        "prod_ni_o1:0",
        "prod_ni_o1:1",
    ]
    assert options["_descriptor_value_position"].tolist() == [0, 1]
    assert "Product | reaction 10 | product_a" in options[
        "_descriptor_value_label"
    ].iloc[0]


def test_descriptor_delta_hover_html_uses_descriptor_x_and_delta_g_y():
    """Hover plot payload contains descriptor values on x and ΔG values on y."""
    records = pd.DataFrame(
        [
            {
                "descriptor_id": "prod_ni_o1",
                "descriptor": "Product: Ni O1",
                "role": "product",
                "variant": "Type_I",
                "value": 1.2345,
                "deltaG": -5.25,
                "unique_name": "product_a",
                "smiles": "C",
                "xyz": read_example_xyz("type_I_product.xyz"),
                "atom_summary": "descriptor_kit",
                "reaction_id": "10",
                "reactant_name": "reactant_a",
                "product_name": "product_a",
            }
        ]
    )

    html = build_descriptor_delta_hover_html(
        records,
        "Product: Ni O1",
        "Å",
        "kcal/mol",
        "ΔG",
    )

    assert '"value": 1.2345' in html
    assert '"deltaG": -5.25' in html
    assert "x: group.map((point) => point.value)" in html
    assert "y: group.map((point) => point.deltaG)" in html


def test_descriptor_delta_hover_html_can_pin_clicked_structure():
    """Clicking a plot point pins its structure until explicitly unpinned."""
    records = pd.DataFrame(
        [
            {
                "descriptor_id": "prod_ni_o1",
                "descriptor": "Product: Ni O1",
                "role": "product",
                "variant": "Type_I",
                "value": 1.2345,
                "deltaG": -5.25,
                "unique_name": "product_a",
                "smiles": "C",
                "xyz": read_example_xyz("type_I_product.xyz"),
                "atom_summary": "descriptor_kit",
                "reaction_id": "10",
                "reactant_name": "reactant_a",
                "product_name": "product_a",
            }
        ]
    )

    html = build_descriptor_delta_hover_html(
        records,
        "Product: Ni O1",
        "Å",
        "kcal/mol",
        "ΔG",
    )

    assert 'plotEl.on("plotly_click"' in html
    assert "pinnedPointIndex !== null" in html
    assert "pinnedPointIndex === pointIndex ? null : pointIndex" in html
    assert 'id="unpin-button"' in html
    assert "Drag the molecule to rotate it." in html


def test_descriptor_delta_hover_html_reindexes_after_skipping_nonfinite_rows():
    """Skipped rows must not leave stale indexes in Plotly hover callbacks."""
    records = pd.DataFrame(
        [
            {
                "descriptor_id": "prod_ni_o1",
                "descriptor": "Product: Ni O1",
                "role": "product",
                "variant": "Type_I",
                "value": float("nan"),
                "deltaG": -5.25,
                "unique_name": "skipped_product",
                "smiles": "C",
                "xyz": read_example_xyz("type_I_product.xyz"),
                "atom_summary": "descriptor_kit",
                "reaction_id": "10",
                "reactant_name": "reactant_a",
                "product_name": "skipped_product",
            },
            {
                "descriptor_id": "prod_ni_o1",
                "descriptor": "Product: Ni O1",
                "role": "product",
                "variant": "Type_I",
                "value": 1.2345,
                "deltaG": -5.25,
                "unique_name": "product_a",
                "smiles": "C",
                "xyz": read_example_xyz("type_I_product.xyz"),
                "atom_summary": "descriptor_kit",
                "reaction_id": "10",
                "reactant_name": "reactant_a",
                "product_name": "product_a",
            },
        ]
    )

    html = build_descriptor_delta_hover_html(
        records,
        "Product: Ni O1",
        "Å",
        "kcal/mol",
        "ΔG",
    )

    assert '"index": 0' in html
    assert "skipped_product" not in html


def test_compact_xyz_for_browser_reduces_coordinate_precision():
    """Browser payloads should not carry unnecessary XYZ decimal precision."""
    xyz = "2\ncomment\nC 1.123456789 -2.987654321 3.111111111\nH 0 0 0\n"

    compact = compact_xyz_for_browser(xyz, digits=3)

    assert compact.splitlines() == [
        "2",
        "comment",
        "C 1.123 -2.988 3.111",
        "H 0.000 0.000 0.000",
    ]


def test_compute_descriptors_includes_regio_keys_with_insertion_type():
    reactant = read_example_xyz("type_I_reactant.xyz")
    product = read_example_xyz("type_I_product.xyz")
    desc = compute_descriptors(
        reactant, product, stereo_type="S", insertion_type="Type_I"
    )
    assert set(REGIO_KEYS).issubset(desc)
    # Type_I negates signed diffs and swaps per-arm values.
    assert desc["reac_dvol_alkyne_ab"] == pytest.approx(-desc["reac_dvol_alkyne"])
    assert desc["reac_B5_Ralpha"] == pytest.approx(desc["reac_B5_R2"])
    assert desc["reac_B5_Rbeta"] == pytest.approx(desc["reac_B5_R1"])


def test_compute_descriptors_regio_is_nan_without_insertion_type():
    reactant = read_example_xyz("type_I_reactant.xyz")
    product = read_example_xyz("type_I_product.xyz")
    desc = compute_descriptors(reactant, product, stereo_type="S")
    for key in REGIO_KEYS:
        assert math.isnan(desc[key]), key


def test_descriptor_keys_include_regio_block():
    assert DESCRIPTOR_KEYS[-len(REGIO_KEYS):] == list(REGIO_KEYS)
    assert len(DESCRIPTOR_KEYS) == len(set(DESCRIPTOR_KEYS))


def test_compute_regio_descriptors_matches_compute_descriptors():
    from descriptor_kit import compute_regio_descriptors

    reactant = read_example_xyz("type_I_reactant.xyz")
    product = read_example_xyz("type_I_product.xyz")
    full = compute_descriptors(
        reactant, product, stereo_type="S", insertion_type="Type_I"
    )
    regio = compute_regio_descriptors(
        reactant, stereo_type="S", insertion_type="Type_I"
    )
    assert set(regio) == set(REGIO_KEYS)
    for key in REGIO_KEYS:
        a, b = full[key], regio[key]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), key


def test_regio_descriptor_is_live_computed_from_geometry():
    df = build_example_reaction_df()  # raw, non-precomputed, has insertion_type
    descriptors = build_selected_descriptor_dataframe(df, "reac_B5_Ralpha")
    expected_type_i = compute_descriptors(
        read_example_xyz("type_I_reactant.xyz"),
        read_example_xyz("type_I_product.xyz"),
        stereo_type="S",
        insertion_type="Type_I",
    )
    assert len(descriptors) == 2  # one reactant record per regioisomer reaction
    assert set(descriptors["role"]) == {"reactant"}
    type_i = descriptors[descriptors["variant"].astype(str) == "Type_I"]
    assert len(type_i) == 1
    assert type_i.iloc[0]["value"] == pytest.approx(expected_type_i["reac_B5_Ralpha"])
