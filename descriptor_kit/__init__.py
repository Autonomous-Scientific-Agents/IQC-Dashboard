"""descriptor_kit — standalone computation of the RF descriptors.

Input: one reaction's geometries (a reactant+product pair, i.e. one
``reactions.parquet`` row, as two xyz blocks).
Output: a flat dict of all single-row descriptors; a separate pair API yields the
regioisomer-Δ (``tdelta_*``) descriptors.

    from descriptor_kit import compute_descriptors, compute_tdelta
    row = compute_descriptors(reactant_xyz, product_xyz, stereo_type="S",
                              insertion_type="Type_I")              # 95 keys
    deltas = compute_tdelta(row_type_I, row_type_II)        # 23 tdelta_* keys

Alkyne C1/C2 labeling is pure CIP (no diaryl golden-rule override).
"""
from .api import (
    compute_descriptors,
    compute_tdelta,
    compute_regio_descriptors,
    add_regio_descriptors,
    DESCRIPTOR_KEYS,
    REACTANT_KEYS,
    PRODUCT_KEYS,
    TDELTA_KEYS,
    REGIO_KEYS,
)

__all__ = [
    "compute_descriptors",
    "compute_tdelta",
    "compute_regio_descriptors",
    "add_regio_descriptors",
    "DESCRIPTOR_KEYS",
    "REACTANT_KEYS",
    "PRODUCT_KEYS",
    "TDELTA_KEYS",
    "REGIO_KEYS",
]
