"""Tests for the regiochemistry-resolved (α/β) reactant descriptors.

Mirrors ``carboxylation/random-forest/src/regio.py``: per row, keyed on
``insertion_type``, signed-difference descriptors negate for Type_I while per-arm
descriptors swap R1<->R2 for Type_I; Type_II passes through unchanged.  The
expected values asserted here are exactly what the reference transform produces.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from descriptor_kit import REGIO_KEYS, add_regio_descriptors


SIGNED_DIFF_SOURCES = [
    "reac_dvol_alkyne",
    "reac_dvbur_substituent",
    "reac_dni_c_signed",
    "reac_dbend_alkyne",
    "reac_dccr_angle",
    "reac_dnicr_angle",
    "reac_slippage",
    "reac_dni_firstatom",
]
PER_ARM_SOURCES = [
    ("reac_B5_Ralpha", "reac_B5_Rbeta", "reac_B5_R1", "reac_B5_R2"),
    ("reac_L_Ralpha", "reac_L_Rbeta", "reac_L_R1", "reac_L_R2"),
    ("reac_bend_Ralpha", "reac_bend_Rbeta", "reac_bend_R1", "reac_bend_R2"),
    (
        "reac_ni_firstatom_Ralpha",
        "reac_ni_firstatom_Rbeta",
        "reac_ni_firstatom_R1",
        "reac_ni_firstatom_R2",
    ),
]


def _base_row(insertion_type: str) -> dict:
    """One reaction row with every regio source column set to a distinct value."""
    row = {"insertion_type": insertion_type}
    for offset, source in enumerate(SIGNED_DIFF_SOURCES, start=1):
        row[source] = float(offset)  # 1.0 .. 8.0
    for index, (_, _, r1_col, r2_col) in enumerate(PER_ARM_SOURCES):
        row[r1_col] = 10.0 + index  # 10, 11, 12, 13
        row[r2_col] = 20.0 + index  # 20, 21, 22, 23
    return row


def test_regio_keys_are_the_sixteen_alpha_beta_columns():
    assert list(REGIO_KEYS) == [
        "reac_dvol_alkyne_ab",
        "reac_dvbur_substituent_ab",
        "reac_dni_c_signed_ab",
        "reac_dbend_alkyne_ab",
        "reac_dccr_angle_ab",
        "reac_dnicr_angle_ab",
        "reac_slippage_ab",
        "reac_dni_firstatom_ab",
        "reac_B5_Ralpha",
        "reac_B5_Rbeta",
        "reac_L_Ralpha",
        "reac_L_Rbeta",
        "reac_bend_Ralpha",
        "reac_bend_Rbeta",
        "reac_ni_firstatom_Ralpha",
        "reac_ni_firstatom_Rbeta",
    ]


def test_type_ii_passes_signed_diffs_and_arms_through_unchanged():
    out = add_regio_descriptors(pd.DataFrame([_base_row("Type_II")]))
    for offset, source in enumerate(SIGNED_DIFF_SOURCES, start=1):
        assert out.loc[0, f"{source}_ab"] == pytest.approx(float(offset))
    for index, (alpha, beta, _, _) in enumerate(PER_ARM_SOURCES):
        assert out.loc[0, alpha] == pytest.approx(10.0 + index)  # Rα = R1
        assert out.loc[0, beta] == pytest.approx(20.0 + index)  # Rβ = R2


def test_type_i_negates_signed_diffs_and_swaps_arms():
    out = add_regio_descriptors(pd.DataFrame([_base_row("Type_I")]))
    for offset, source in enumerate(SIGNED_DIFF_SOURCES, start=1):
        assert out.loc[0, f"{source}_ab"] == pytest.approx(-float(offset))
    for index, (alpha, beta, _, _) in enumerate(PER_ARM_SOURCES):
        assert out.loc[0, alpha] == pytest.approx(20.0 + index)  # Rα = R2
        assert out.loc[0, beta] == pytest.approx(10.0 + index)  # Rβ = R1


def test_lowercase_insertion_type_is_normalized_like_titlecase():
    title = add_regio_descriptors(pd.DataFrame([_base_row("Type_I")]))
    lower = add_regio_descriptors(pd.DataFrame([_base_row("type_i")]))
    for key in REGIO_KEYS:
        assert lower.loc[0, key] == pytest.approx(title.loc[0, key])


def test_unknown_insertion_type_yields_nan_regio_values():
    out = add_regio_descriptors(pd.DataFrame([_base_row("")]))
    for key in REGIO_KEYS:
        assert math.isnan(out.loc[0, key])


def test_nan_source_propagates_to_nan_regio_value():
    row = _base_row("Type_II")
    row["reac_slippage"] = float("nan")
    out = add_regio_descriptors(pd.DataFrame([row]))
    assert math.isnan(out.loc[0, "reac_slippage_ab"])


def test_source_columns_are_left_untouched():
    df = pd.DataFrame([_base_row("Type_I")])
    out = add_regio_descriptors(df)
    for source in SIGNED_DIFF_SOURCES:
        assert out.loc[0, source] == pytest.approx(df.loc[0, source])


def test_missing_insertion_type_column_yields_nan_regio_values():
    row = _base_row("Type_I")
    del row["insertion_type"]
    out = add_regio_descriptors(pd.DataFrame([row]))
    for key in REGIO_KEYS:
        assert math.isnan(out.loc[0, key])


def test_mixed_insertion_types_are_resolved_per_row():
    df = pd.DataFrame([_base_row("Type_I"), _base_row("Type_II")])
    out = add_regio_descriptors(df)
    # Type_I row negates, Type_II row keeps.
    assert out.loc[0, "reac_dvol_alkyne_ab"] == pytest.approx(-1.0)
    assert out.loc[1, "reac_dvol_alkyne_ab"] == pytest.approx(1.0)
    # Type_I swaps arms, Type_II keeps.
    assert out.loc[0, "reac_B5_Ralpha"] == pytest.approx(20.0)
    assert out.loc[1, "reac_B5_Ralpha"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "label", ["Type_I", "type_i", "type-i", "TYPE_I", "type_1", "i", "1", "  Type_I  "]
)
def test_type_i_aliases_all_negate(label):
    out = add_regio_descriptors(pd.DataFrame([_base_row(label)]))
    assert out.loc[0, "reac_dvol_alkyne_ab"] == pytest.approx(-1.0)
    assert out.loc[0, "reac_B5_Ralpha"] == pytest.approx(20.0)  # Rα = R2


@pytest.mark.parametrize(
    "label",
    ["Type_II", "type_ii", "type-ii", "TYPE_II", "type_2", "ii", "2", "  Type_II  "],
)
def test_type_ii_aliases_all_pass_through(label):
    out = add_regio_descriptors(pd.DataFrame([_base_row(label)]))
    assert out.loc[0, "reac_dvol_alkyne_ab"] == pytest.approx(1.0)
    assert out.loc[0, "reac_B5_Ralpha"] == pytest.approx(10.0)  # Rα = R1


def test_regio_normalizer_agrees_with_pipeline_on_type_classification():
    """The kit's private normalizer must classify Type_I/II like the pipeline's."""
    from descriptor_kit.regio import _normalize_insertion_type
    from iqc_dashboard.descriptor_precompute import normalize_insertion_type

    for label in [
        "Type_I", "type_i", "Type_II", "type-ii", "i", "ii", "1", "2", "weird", "", None,
    ]:
        kit = _normalize_insertion_type(label)
        pipeline = normalize_insertion_type(label)
        kit_class = kit if kit in {"type_i", "type_ii"} else ""
        pipeline_class = pipeline if pipeline in {"type_i", "type_ii"} else ""
        assert kit_class == pipeline_class


def test_add_regio_descriptors_is_idempotent():
    df = pd.DataFrame([_base_row("Type_I"), _base_row("Type_II"), _base_row("")])
    once = add_regio_descriptors(df)
    twice = add_regio_descriptors(once)
    for key in REGIO_KEYS:
        assert list(twice.columns).count(key) == 1
        first = once[key].to_numpy(float)
        second = twice[key].to_numpy(float)
        assert ((first == second) | (np.isnan(first) & np.isnan(second))).all()


def test_golden_vector_matches_reference_arithmetic():
    """Frozen snapshot of the reference transform (the reference repo is not vendored)."""
    row = {
        "insertion_type": "Type_I",
        "reac_dvol_alkyne": -49.7,
        "reac_dvbur_substituent": 2.5,
        "reac_dni_c_signed": 0.3,
        "reac_dbend_alkyne": 30.9,
        "reac_dccr_angle": -12.0,
        "reac_dnicr_angle": 7.5,
        "reac_slippage": 0.15,
        "reac_dni_firstatom": -0.4,
        "reac_B5_R1": 1.2, "reac_B5_R2": 3.23,
        "reac_L_R1": 2.68, "reac_L_R2": 4.66,
        "reac_bend_R1": 45.6, "reac_bend_R2": 14.7,
        "reac_ni_firstatom_R1": 2.0, "reac_ni_firstatom_R2": 2.1,
    }
    expected = {
        "reac_dvol_alkyne_ab": 49.7,
        "reac_dvbur_substituent_ab": -2.5,
        "reac_dni_c_signed_ab": -0.3,
        "reac_dbend_alkyne_ab": -30.9,
        "reac_dccr_angle_ab": 12.0,
        "reac_dnicr_angle_ab": -7.5,
        "reac_slippage_ab": -0.15,
        "reac_dni_firstatom_ab": 0.4,
        "reac_B5_Ralpha": 3.23, "reac_B5_Rbeta": 1.2,
        "reac_L_Ralpha": 4.66, "reac_L_Rbeta": 2.68,
        "reac_bend_Ralpha": 14.7, "reac_bend_Rbeta": 45.6,
        "reac_ni_firstatom_Ralpha": 2.1, "reac_ni_firstatom_Rbeta": 2.0,
    }
    out = add_regio_descriptors(pd.DataFrame([row]))
    for key in REGIO_KEYS:
        assert out.loc[0, key] == pytest.approx(expected[key]), key


def test_compute_regio_matches_dataframe_transform_row_for_row():
    """The per-row compute_regio equals the vectorized add_regio_descriptors."""
    from descriptor_kit.regio import compute_regio

    df = pd.DataFrame([_base_row("Type_I"), _base_row("Type_II"), _base_row("")])
    out = add_regio_descriptors(df)
    for i in range(len(df)):
        sources = df.iloc[i].to_dict()
        single = compute_regio(sources, sources["insertion_type"])
        for key in REGIO_KEYS:
            a, b = single[key], out.loc[i, key]
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), key


def test_compute_regio_handles_missing_source_keys_as_nan():
    from descriptor_kit.regio import compute_regio

    result = compute_regio({}, "Type_I")
    for key in REGIO_KEYS:
        assert math.isnan(result[key])
