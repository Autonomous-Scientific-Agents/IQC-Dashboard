"""Tests for selection state helpers used by Streamlit navigation controls."""

import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from iqc_dashboard.app import (
    build_comparison_match_options,
    build_comparison_match_id,
    move_indexed_selection,
    sync_indexed_selection,
)


def test_move_indexed_selection_updates_value_and_index():
    """Test Next updates both the selectbox value and selected index."""
    state = {
        "selected_molecule_index": 0,
        "selected_molecule_select": "mol_a",
    }
    molecule_names = ["mol_a", "mol_b", "mol_c"]

    result = move_indexed_selection(
        state,
        molecule_names,
        "selected_molecule_select",
        1,
        "selected_molecule_index",
    )

    assert result == 1
    assert state["selected_molecule_index"] == 1
    assert state["selected_molecule_select"] == "mol_b"

def test_move_indexed_selection_clamps_at_bounds():
    """Test navigation does not move outside available options."""
    state = {
        "selected_molecule_index": 2,
        "selected_molecule_select": "mol_c",
    }
    molecule_names = ["mol_a", "mol_b", "mol_c"]

    result = move_indexed_selection(
        state,
        molecule_names,
        "selected_molecule_select",
        1,
        "selected_molecule_index",
    )

    assert result == 2
    assert state["selected_molecule_index"] == 2
    assert state["selected_molecule_select"] == "mol_c"


def test_sync_indexed_selection_repairs_stale_selectbox_value():
    """Test filters or data changes repair stale selected values before rendering."""
    state = {
        "selected_molecule_index": 4,
        "selected_molecule_select": "old_mol",
    }
    molecule_names = ["mol_a", "mol_b"]

    result = sync_indexed_selection(
        state,
        molecule_names,
        "selected_molecule_select",
        "selected_molecule_index",
    )

    assert result == 1
    assert state["selected_molecule_index"] == 1
    assert state["selected_molecule_select"] == "mol_b"


def test_comparison_match_options_follow_source_row_order():
    """Test comparison Next order follows file row order instead of molecule label order."""
    rows = []
    for file_order, file_label in [(0, "method_a.parquet"), (1, "method_b.parquet")]:
        for row_number, molecule_label in [(1, "O"), (2, "O=C=O"), (3, "N")]:
            rows.append(
                {
                    "_comparison_match_id": build_comparison_match_id(
                        molecule_label,
                        1,
                    ),
                    "_comparison_key": molecule_label,
                    "_comparison_occurrence": 1,
                    "_comparison_molecule_label": molecule_label,
                    "_comparison_file_order": file_order,
                    "_comparison_file_label": file_label,
                    "_comparison_row_number": row_number,
                }
            )
    matched_rows = pd.DataFrame(rows).sort_values("_comparison_molecule_label")

    options = build_comparison_match_options(matched_rows)

    assert options["_comparison_molecule_label"].tolist() == ["O", "O=C=O", "N"]


def test_comparison_next_moves_to_next_source_row_match():
    """Test comparison Next advances through the ordered match id list."""
    matched_rows = pd.DataFrame(
        [
            {
                "_comparison_match_id": build_comparison_match_id(label, 1),
                "_comparison_key": label,
                "_comparison_occurrence": 1,
                "_comparison_molecule_label": label,
                "_comparison_file_order": 0,
                "_comparison_row_number": row_number,
            }
            for row_number, label in [(1, "O"), (2, "O=C=O"), (3, "N")]
        ]
    )
    comparison_match_ids = build_comparison_match_options(matched_rows)[
        "_comparison_match_id"
    ].tolist()
    state = {
        "comparison_selected_match_id": comparison_match_ids[0],
        "comparison_selected_match_index": 0,
    }

    result = move_indexed_selection(
        state,
        comparison_match_ids,
        "comparison_selected_match_id",
        1,
        "comparison_selected_match_index",
    )

    assert result == 1
    assert state["comparison_selected_match_index"] == 1
    assert state["comparison_selected_match_id"] == comparison_match_ids[1]
