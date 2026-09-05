"""Regression coverage for alkyne priorities and C1/C2 assignment (issue #7)."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from descriptor_kit.core import cip, geometry


# A outranks B in each pair. The old aromatic duplicate counting and pooled
# sphere traversal ranked all three in the opposite order.
REGRESSION_PAIRS = [
    pytest.param("C#CC", "c1ccccc1", id="propynyl-over-phenyl"),
    pytest.param("C=CC", "C1CCCCC1", id="propenyl-over-cyclohexyl"),
    pytest.param("C=Cc1ccccc1", "C1CCCCC1", id="styryl-over-cyclohexyl"),
]


def _alkyne(arm_a, arm_b, *, reverse_atoms=False):
    """Build explicit-H arms on a marked central alkyne, retaining real rings."""
    mol = Chem.AddHs(Chem.MolFromSmiles(f"[C:1]({arm_a})#[C:2]{arm_b}"))
    if reverse_atoms:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    pair = tuple(
        next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == label)
        for label in (1, 2)
    )
    roots = tuple(
        next(atom.GetIdx() for atom in mol.GetAtomWithIdx(carbon).GetNeighbors()
             if atom.GetIdx() not in pair)
        for carbon in pair
    )
    adj = [{atom.GetIdx() for atom in parent.GetNeighbors()} for parent in mol.GetAtoms()]
    arms = tuple(frozenset(geometry.fragment_bfs(adj, {root}, set(pair))) for root in roots)
    return mol, pair, roots, arms


@pytest.mark.parametrize("arm_a,arm_b", REGRESSION_PAIRS + [
    pytest.param("CC", "C", id="ethyl-over-methyl"),
    pytest.param("F", "C", id="fluorine-over-carbon"),
    pytest.param("C", "[H]", id="carbon-over-hydrogen"),
])
def test_branch_priority_and_reverse_order(arm_a, arm_b):
    mol, _, (ra, rb), (aa, ab) = _alkyne(arm_a, arm_b)
    assert cip._cip_compare_branches(mol, ra, rb, aa, ab) == 1
    assert cip._cip_compare_branches(mol, rb, ra, ab, aa) == -1


@pytest.mark.parametrize("arm_a,arm_b", [
    ("C", "C"),
    ("c1ccccc1", "c1ccccc1"),
    ("[H]", "[H]"),
    ("[C@H](F)Cl", "[C@@H](F)Cl"),
])
def test_constitutionally_equivalent_arms_tie(arm_a, arm_b):
    mol, _, (ra, rb), (aa, ab) = _alkyne(arm_a, arm_b)
    assert cip._cip_compare_branches(mol, ra, rb, aa, ab) == 0
    assert cip._cip_compare_branches(mol, rb, ra, ab, aa) == 0


def _geometry_with_nickel(mol, pair):
    """Exercise the production distance graph and bond-order perception path."""
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    coords = mol.GetConformer().GetPositions()
    nickel = coords[list(pair)].mean(axis=0) + np.array([0.0, 0.0, 3.0])
    return geometry.build_geom(
        [atom.GetSymbol() for atom in mol.GetAtoms()] + ["Ni"],
        np.vstack([coords, nickel]),
    )


@pytest.mark.parametrize("arm_a,arm_b", REGRESSION_PAIRS)
@pytest.mark.parametrize("reverse_atoms", [False, True])
def test_label_alkyne_carbons_uses_correct_priority(arm_a, arm_b, reverse_atoms):
    mol, pair, (ra, rb), (aa, ab) = _alkyne(arm_a, arm_b, reverse_atoms=reverse_atoms)
    geom = _geometry_with_nickel(mol, pair)
    expected = {
        "c1": pair[1], "c2": pair[0],
        "r1_root": rb, "r2_root": ra,
        "r1_atoms": ab, "r2_atoms": aa, "source": "cip",
    }
    assert cip.label_alkyne_carbons(geom, pair, ra, rb, aa, ab) == expected
    assert cip.label_alkyne_carbons(geom, pair[::-1], rb, ra, ab, aa) == expected


@pytest.mark.parametrize("arm", ["C", "c1ccccc1", "[H]"])
def test_symmetric_alkyne_keeps_atom_order_tiebreak(arm):
    mol, pair, (ra, rb), (aa, ab) = _alkyne(arm, arm)
    geom = _geometry_with_nickel(mol, pair)
    result = cip.label_alkyne_carbons(geom, pair, ra, rb, aa, ab)
    assert result["source"] == "symmetric_atom_order"
    assert (result["c1"], result["c2"]) == tuple(sorted(pair))
    assert cip.label_alkyne_carbons(geom, pair[::-1], rb, ra, ab, aa) == result


def test_missing_cip_label_raises_instead_of_falling_back(monkeypatch):
    mol, _, (ra, rb), (aa, ab) = _alkyne("CC", "C")
    monkeypatch.setattr(cip.rdCIPLabeler, "AssignCIPLabels", lambda *args: None)
    with pytest.raises(ValueError, match="could not assign"):
        cip._cip_compare_branches(mol, ra, rb, aa, ab)
