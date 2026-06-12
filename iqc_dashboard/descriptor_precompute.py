"""Precompute descriptor_kit columns for reaction JSON datasets."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from descriptor_kit import (
    DESCRIPTOR_KEYS,
    TDELTA_KEYS,
    compute_descriptors,
    compute_tdelta,
)


PRECOMPUTE_VERSION = 1
REQUIRED_REACTION_COLUMNS = {
    "ligand_pair",
    "reactant_geometry",
    "product_geometry",
}


def read_reaction_json(json_path: Path) -> pd.DataFrame:
    """Read a supported JSON table and validate its reaction geometry columns."""
    try:
        reaction_df = pd.read_json(json_path)
    except ValueError:
        with json_path.open(encoding="utf-8") as json_file:
            json_data = json.load(json_file)

        if isinstance(json_data, list):
            reaction_df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            for key in ("data", "records", "rows"):
                value = json_data.get(key)
                if isinstance(value, list):
                    reaction_df = pd.DataFrame(value)
                    break
            else:
                reaction_df = pd.DataFrame.from_dict(json_data)
        else:
            raise ValueError("Unsupported JSON table structure.")

    missing_columns = REQUIRED_REACTION_COLUMNS.difference(reaction_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Reaction JSON is missing required columns: {missing_text}")

    return reaction_df.reset_index(drop=True)


def normalize_insertion_type(value: object) -> str:
    """Normalize Type I/II labels for regioisomer grouping."""
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
    return text


def sanitize_name_part(value: object) -> str:
    """Return a compact token suitable for a generated unique name."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    import re

    text = re.sub(r"\s+", "-", str(value).strip())
    return re.sub(r"[^A-Za-z0-9+_.-]+", "-", text).strip("-")


def build_component_unique_name(row: pd.Series, role: str, row_index: int) -> str:
    """Build the same parseable component name used by the dashboard."""
    ligand_pair = sanitize_name_part(row.get("ligand_pair")) or f"reaction-{row_index}"
    config = sanitize_name_part(row.get(f"{role}_configuration"))
    stereo = sanitize_name_part(row.get("stereo_type"))
    insertion = sanitize_name_part(row.get("insertion_type"))
    suffix_parts = [part for part in (config, stereo, insertion, str(row_index)) if part]
    return f"{ligand_pair}_{role}_{'_'.join(suffix_parts)}"


def count_xyz_atoms(xyz_string: object) -> Optional[int]:
    """Return the XYZ atom count when its first line is valid."""
    if xyz_string is None:
        return None
    lines = str(xyz_string).splitlines()
    if not lines:
        return None
    try:
        return int(lines[0].strip())
    except ValueError:
        return None


def role_smiles(row: pd.Series, role: str) -> str:
    """Return the first available role-specific SMILES string."""
    for column in (
        f"{role}_smiles",
        f"{role}_initial_smiles",
        f"{role}_opt_smiles",
        "smiles",
    ):
        value = row.get(column)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return str(value)
    return ""


def _compute_descriptor_row(geometry_pair: tuple[str, str]) -> tuple[dict, int, bool]:
    """Compute all single-reaction descriptors in a worker process."""
    reactant_xyz, product_xyz = geometry_pair
    diagnostics: list[tuple[str, str]] = []
    values = compute_descriptors(
        reactant_xyz,
        product_xyz,
        diagnostics=diagnostics,
    )
    identification_failed = any(key == "_identification" for key, _ in diagnostics)
    return values, len(diagnostics), identification_failed


def _descriptor_tasks(reaction_df: pd.DataFrame) -> Iterable[tuple[str, str]]:
    return (
        (str(reactant_xyz), str(product_xyz))
        for reactant_xyz, product_xyz in reaction_df[
            ["reactant_geometry", "product_geometry"]
        ].itertuples(index=False, name=None)
    )


def compute_single_reaction_descriptors(
    reaction_df: pd.DataFrame,
    workers: int = 1,
    chunksize: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Compute all reac_*/prod_* columns while preserving input row order."""
    total = len(reaction_df)
    descriptor_rows = []
    failure_counts = []
    identification_failures = []

    if workers <= 1:
        results = map(_compute_descriptor_row, _descriptor_tasks(reaction_df))
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(
            _compute_descriptor_row,
            _descriptor_tasks(reaction_df),
            chunksize=max(1, chunksize),
        )

    try:
        for completed, (values, failure_count, identification_failed) in enumerate(
            results,
            start=1,
        ):
            descriptor_rows.append(values)
            failure_counts.append(failure_count)
            identification_failures.append(identification_failed)
            if progress is not None:
                progress(completed, total)
    finally:
        if executor is not None:
            executor.shutdown()

    descriptor_df = pd.DataFrame(descriptor_rows, columns=DESCRIPTOR_KEYS, dtype=float)
    return (
        descriptor_df,
        pd.Series(failure_counts, dtype="int64"),
        pd.Series(identification_failures, dtype="bool"),
    )


def add_tdelta_descriptors(reaction_df: pd.DataFrame) -> pd.DataFrame:
    """Add pair descriptors to the last Type-I row in each dashboard pair group."""
    result = reaction_df.copy()
    for descriptor_key in TDELTA_KEYS:
        result[descriptor_key] = np.nan

    grouped_rows: dict[tuple[str, str], dict[str, int]] = {}
    for row_index, row in result.iterrows():
        insertion_type = normalize_insertion_type(row.get("insertion_type"))
        if insertion_type not in {"type_i", "type_ii"}:
            continue
        group_key = (
            str(row.get("ligand_pair", "")),
            str(row.get("stereo_type", "")),
        )
        grouped_rows.setdefault(group_key, {})[insertion_type] = row_index

    for pair_group in grouped_rows.values():
        type_i_index = pair_group.get("type_i")
        type_ii_index = pair_group.get("type_ii")
        if type_i_index is None or type_ii_index is None:
            continue

        type_i_values = result.loc[type_i_index, DESCRIPTOR_KEYS].to_dict()
        type_ii_values = result.loc[type_ii_index, DESCRIPTOR_KEYS].to_dict()
        tdelta_values = compute_tdelta(type_i_values, type_ii_values)
        for descriptor_key, value in tdelta_values.items():
            result.at[type_i_index, descriptor_key] = value

    return result


def expand_reactions_for_dashboard(reaction_df: pd.DataFrame) -> pd.DataFrame:
    """Expand reaction rows into dashboard-compatible reactant/product rows."""
    component_frames = []
    for role, role_order in (("reactant", 0), ("product", 1)):
        component_df = reaction_df.copy()
        geometry_column = f"{role}_geometry"
        component_df["reaction_role"] = role
        component_df["source_json_row"] = np.arange(len(component_df), dtype=np.int64)
        component_df["unique_name"] = [
            build_component_unique_name(row, role, row_index)
            for row_index, (_, row) in enumerate(component_df.iterrows())
        ]
        component_df["initial_xyz"] = component_df[geometry_column].astype(str)
        component_df["opt_xyz"] = component_df[geometry_column].astype(str)
        component_df["number_of_atoms"] = component_df[geometry_column].apply(
            count_xyz_atoms
        )
        component_df["source_gibbs"] = component_df.get(
            f"{role}_gibbs",
            pd.Series(np.nan, index=component_df.index),
        )
        component_df["formula"] = component_df.get(
            "formula",
            pd.Series(None, index=component_df.index),
        )
        component_df["initial_smiles"] = [
            role_smiles(row, role) for _, row in component_df.iterrows()
        ]
        component_df["opt_smiles"] = component_df["initial_smiles"]
        component_df["task"] = "reaction"
        component_df["calculator"] = "json"
        component_df["opt_converged"] = True
        component_df["smiles_changed"] = False
        component_df["number_of_imaginary"] = 0
        component_df["_component_order"] = (
            component_df["source_json_row"] * 2 + role_order
        )
        component_frames.append(component_df)

    return (
        pd.concat(component_frames, ignore_index=True)
        .sort_values("_component_order", kind="stable")
        .drop(columns="_component_order")
        .reset_index(drop=True)
    )


def build_precomputed_descriptor_dataframe(
    reaction_df: pd.DataFrame,
    workers: int = 1,
    chunksize: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Return dashboard-ready rows containing the source data and all descriptors."""
    reaction_df = reaction_df.reset_index(drop=True).copy()
    missing_columns = REQUIRED_REACTION_COLUMNS.difference(reaction_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Reaction data is missing required columns: {missing_text}")

    descriptor_df, failure_counts, identification_failures = (
        compute_single_reaction_descriptors(
            reaction_df,
            workers=workers,
            chunksize=chunksize,
            progress=progress,
        )
    )
    enriched_df = pd.concat([reaction_df, descriptor_df], axis=1)
    enriched_df["descriptor_precomputed"] = True
    enriched_df["descriptor_precompute_version"] = PRECOMPUTE_VERSION
    enriched_df["descriptor_failure_count"] = failure_counts
    enriched_df["descriptor_identification_failed"] = identification_failures
    enriched_df = add_tdelta_descriptors(enriched_df)
    return expand_reactions_for_dashboard(enriched_df)


def default_worker_count() -> int:
    """Choose a conservative process count for CPU-bound descriptor calculation."""
    return max(1, (os.cpu_count() or 2) - 1)
