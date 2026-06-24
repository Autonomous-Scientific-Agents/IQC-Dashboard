"""Regiochemistry-resolved (α/β) reactant asymmetry descriptors.

The plain reactant asymmetry descriptors order the two alkyne arms by CIP
priority: ``c1`` = lower-priority carbon (R1), ``c2`` = higher-priority carbon
(R2), and every signed difference is ``f(R1) - f(R2)``.  These α/β columns
instead order the arms by the *regiochemistry of the product* (Cα forms the new
C–C bond to the carboxylate; Cβ binds Ni):

    Type_I  products: Cα = C2   ->  (Rα, Rβ) = (R2, R1)   (arms swapped)
    Type_II products: Cβ = C2   ->  (Rα, Rβ) = (R1, R2)   (arms unchanged)

So, per row, keyed on ``insertion_type``:

  * signed differences ``D = f(R1) - f(R2)``  ->  ``-D`` for Type_I, ``D`` for Type_II;
  * per-arm values swap R1<->R2 for Type_I only.

Because the same reactant pairs with two different products (Type_I and
Type_II), these columns intentionally take *different* values on the two
insertion rows of a (ligand_pair, stereo_type) group — unlike the plain
reactant descriptors, which are constant within the group.

This is the standalone-kit port of the reference
``carboxylation/random-forest/src/regio.py`` transform.  It differs from the
reference in two deliberate, dashboard-friendly ways: it normalizes the
``insertion_type`` spelling itself (so both ``"Type_I"`` and ``"type_i"`` work),
and rows whose insertion type is neither Type_I nor Type_II get ``NaN`` α/β
values instead of raising — matching how the precompute pipeline already skips
non-regioisomer rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Signed-difference α/β columns: new column -> source (R1 - R2) column.
SIGNED_DIFF_MAP = {
    "reac_dvol_alkyne_ab": "reac_dvol_alkyne",
    "reac_dvbur_substituent_ab": "reac_dvbur_substituent",
    "reac_dni_c_signed_ab": "reac_dni_c_signed",
    "reac_dbend_alkyne_ab": "reac_dbend_alkyne",
    "reac_dccr_angle_ab": "reac_dccr_angle",
    "reac_dnicr_angle_ab": "reac_dnicr_angle",
    "reac_slippage_ab": "reac_slippage",
    "reac_dni_firstatom_ab": "reac_dni_firstatom",
}

# Per-arm α/β families: (Rα column, Rβ column, R1 source, R2 source).
PER_ARM_SPECS = [
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

# All 16 α/β column names, in output order (signed diffs first, then per-arm).
REGIO_KEYS = list(SIGNED_DIFF_MAP) + [col for spec in PER_ARM_SPECS for col in spec[:2]]

# Source columns the transform reads (besides ``insertion_type``).
REGIO_SOURCE_KEYS = list(SIGNED_DIFF_MAP.values()) + [
    col for spec in PER_ARM_SPECS for col in spec[2:]
]


def _as_float(value: object) -> float:
    """Coerce to float, mapping missing/invalid to NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compute_regio(source_values, insertion_type: str) -> dict:
    """Regio α/β values for one row from its reac_* sources + insertion_type.

    Row-local equivalent of ``add_regio_descriptors``: signed differences negate
    for Type_I (pass through for Type_II); per-arm values swap R1<->R2 for Type_I.
    Unknown/empty ``insertion_type`` and missing/NaN sources yield NaN.
    """
    norm = _normalize_insertion_type(insertion_type)
    is_t1 = norm == "type_i"
    is_known = norm in {"type_i", "type_ii"}

    out = {}
    for out_col, src in SIGNED_DIFF_MAP.items():
        v = _as_float(source_values.get(src))
        out[out_col] = (-v if is_t1 else v) if is_known else float("nan")
    for alpha, beta, r1, r2 in PER_ARM_SPECS:
        v1 = _as_float(source_values.get(r1))
        v2 = _as_float(source_values.get(r2))
        out[alpha] = (v2 if is_t1 else v1) if is_known else float("nan")
        out[beta] = (v1 if is_t1 else v2) if is_known else float("nan")
    return out


def _normalize_insertion_type(value: object) -> str:
    """Normalize Type I/II labels to ``"type_i"`` / ``"type_ii"`` (else ``""``)."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"type_i", "type_1", "i", "1"}:
        return "type_i"
    if text in {"type_ii", "type_2", "ii", "2"}:
        return "type_ii"
    return ""


def add_regio_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 16 α/β columns to ``df`` (returns a new DataFrame).

    Requires every source column referenced by ``SIGNED_DIFF_MAP`` /
    ``PER_ARM_SPECS`` to be present.  Source columns are left untouched; NaN
    sources propagate to NaN α/β values, and rows whose insertion type is
    neither Type_I nor Type_II — including when the ``insertion_type`` column is
    absent entirely — get NaN α/β values.
    """
    if "insertion_type" in df.columns:
        normalized = df["insertion_type"].map(_normalize_insertion_type)
    else:
        normalized = pd.Series("", index=df.index)
    is_t1 = (normalized == "type_i").to_numpy()
    is_known = normalized.isin(("type_i", "type_ii")).to_numpy()

    new = {}
    for out_col, src in SIGNED_DIFF_MAP.items():
        v = df[src].to_numpy(dtype=float)
        new[out_col] = np.where(is_known, np.where(is_t1, -v, v), np.nan)
    for alpha, beta, r1, r2 in PER_ARM_SPECS:
        v1 = df[r1].to_numpy(dtype=float)
        v2 = df[r2].to_numpy(dtype=float)
        new[alpha] = np.where(is_known, np.where(is_t1, v2, v1), np.nan)
        new[beta] = np.where(is_known, np.where(is_t1, v1, v2), np.nan)

    # Single concat (avoids per-column fragmentation); drop any pre-existing
    # copies first so a re-run / NaN-preinitialised frame stays single-valued.
    df = df.drop(columns=[c for c in new if c in df.columns])
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
