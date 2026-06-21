"""Hammett / Taft sigma subsystem (spec §6).

Faithful copy of ``src/hammett.py`` with package-relative imports.

Perceive a substituent fragment from the metal-free geometry, build a capped
RDKit molecule, canonicalise to SMILES, and look the SMILES up in the curated
literature table in ``sigma_data.py``.

* ``fragment_smiles(geom, root, frag)`` -> canonical SMILES of the fragment
  ``frag`` (which must contain ``root``), capping the broken root->parent bond
  with an explicit H.  Metal-free by construction.
* ``sigma_for_fragment(geom, root, frag, position)``:
    - ``position`` in {4, 6} -> aromatic para / ortho proxy -> ``sigma_p``
    - ``position`` in {3, 5} -> aromatic meta -> ``sigma_m``
    - ``position`` is None   -> alkyne group  -> ``sigma_p`` if tabulated
                                                  (group constant), else Taft
                                                  ``sigma_star``.
  Positions 1/2 -> NaN (skip).  An untabulated fragment returns NaN (a missing
  value never raises); a perception failure inside ``fragment_smiles`` does raise.
"""
from __future__ import annotations
import math

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from . import sigma_data


def fragment_smiles(geom, root_idx, frag_atoms):
    """Canonical SMILES of the capped, metal-free substituent fragment.

    Parameters
    ----------
    geom : Geom
    root_idx : int
        Attachment atom of the fragment (the atom bonded to the parent ring /
        alkyne carbon that is *excluded* from ``frag_atoms``).
    frag_atoms : iterable[int]
        Atom indices of the fragment (must include ``root_idx``; must NOT
        include the parent atom or Ni).

    Returns
    -------
    str
        Canonical SMILES (no explicit Hs in output) of the fragment with the
        broken root->parent bond capped by an explicit H.

    Raises
    ------
    AssertionError / ValueError
        On precondition violation or RDKit perception failure.
    """
    els = geom.elements
    xyz = geom.coords
    adj = geom.adj
    atoms = sorted(frag_atoms)
    assert root_idx in atoms, "root_idx must be in frag_atoms"
    assert geom.ni not in atoms, "fragment must be metal-free (no Ni)"

    remap = {a: i for i, a in enumerate(atoms)}
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms) + 1)
    for i, a in enumerate(atoms):
        assert els[a] != "Ni", "fragment must be metal-free (no Ni)"
        rw.AddAtom(Chem.Atom(els[a]))
        conf.SetAtomPosition(i, [float(c) for c in xyz[a]])
    # Cap the broken root->parent bond with an explicit H placed ~1 A off the
    # root (cap geometry is irrelevant; DetermineBondOrders only needs valid
    # connectivity plus the heavy-atom geometry it already has).
    hidx = rw.AddAtom(Chem.Atom("H"))
    conf.SetAtomPosition(
        hidx,
        [float(xyz[root_idx][0]), float(xyz[root_idx][1]),
         float(xyz[root_idx][2]) + 1.0],
    )
    # Bonds = organic adjacency restricted to the fragment (single placeholders).
    for a in atoms:
        for b in adj[a]:
            if b in remap and b > a:
                rw.AddBond(remap[a], remap[b], Chem.BondType.SINGLE)
    rw.AddBond(remap[root_idx], hidx, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineBondOrders(mol, charge=0)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def sigma_for_fragment(geom, root_idx, frag_atoms, position):
    """Hammett (aromatic) or Taft (aliphatic) sigma of a substituent fragment.

    Parameters
    ----------
    geom : Geom
    root_idx, frag_atoms : as in ``fragment_smiles``.
    position : int | None
        bpy ring position of the attachment ring carbon (1..6) for a bpy
        substituent; ``None`` for an alkyne substituent.

    Returns
    -------
    float
        ``sigma_p`` for positions 4/6, ``sigma_m`` for positions 3/5, the group
        constant (or Taft ``sigma_star`` fallback) for ``position is None``;
        ``float('nan')`` if the fragment / value is untabulated or the position
        is skipped (1/2).
    """
    smiles = fragment_smiles(geom, root_idx, frag_atoms)
    entry = sigma_data.SIGMA_TABLE.get(smiles)
    if entry is None:
        return float("nan")
    # Attachment guard: the capped SMILES alone can alias attachment isomers
    # (e.g. -COOH vs -OCHO both -> formic acid). Require the perceived root
    # element to match the table entry's attachment atom; otherwise it is a
    # different group than tabulated -> NaN.
    attach = entry.get("attach")
    if attach is not None and geom.elements[root_idx] != attach:
        return float("nan")

    if position is None:
        # Alkyne substituent: prefer the aromatic-type group constant (sigma_p);
        # fall back to Taft sigma_star for purely aliphatic groups.
        val = entry.get("sigma_p")
        if val is not None and not _isnan(val):
            return float(val)
        val = entry.get("sigma_star")
        if val is not None and not _isnan(val):
            return float(val)
        return float("nan")

    if position in (4, 6):
        val = entry.get("sigma_p")
    elif position in (3, 5):
        val = entry.get("sigma_m")
    else:
        # positions 1 (N), 2 (bridgehead C), 6 (ortho, skipped per spec)
        return float("nan")
    if val is None or _isnan(val):
        return float("nan")
    return float(val)


def _isnan(x):
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True
