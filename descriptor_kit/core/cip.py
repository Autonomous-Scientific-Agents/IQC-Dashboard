"""Alkyne carbon CIP labeling (C1/C2) — pure CIP, no overrides.

Spec §5.1.  Convention is INVERTED: c1 = carbon bearing the LOWER-priority
substituent, c2 = carbon bearing the HIGHER-priority substituent.

This kit uses PLAIN CIP only.  A symmetric alkyne, whose two substituents are
CIP-indistinguishable, cannot be ordered by CIP; the tie is broken by input atom
number (C1 = the alkyne carbon with the smaller geometry index, source
"symmetric_atom_order").

The alkyne organic fragment ({cA,cB} + both substituents, Ni stripped) is rebuilt
as a metal-free RDKit molecule on *our* distance-based connectivity, bond orders
are perceived with rdDetermineBonds.DetermineBondOrders(charge=0) (neutral,
closed-shell), then the two substituents are ranked by RDKit's true CIP labeler.
"""
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler, rdDetermineBonds


def _build_alkyne_mol(geom, alkyne_pair, r_a_atoms, r_b_atoms):
    """Build a sanitized RDKit Mol of the metal-free alkyne fragment.

    Returns (mol, remap) where remap maps original geom atom index -> mol atom
    index.  Bonds = the organic adjacency restricted to the fragment atoms,
    added as single placeholders; DetermineBondOrders(charge=0) then assigns
    the true orders (recovering the C#C triple bond) on our connectivity.
    """
    els = geom.elements
    xyz = geom.coords
    adj = geom.adj
    cA, cB = alkyne_pair
    frag = sorted({cA, cB} | set(r_a_atoms) | set(r_b_atoms))
    assert cA in frag and cB in frag
    remap = {a: i for i, a in enumerate(frag)}

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(frag))
    for i, a in enumerate(frag):
        rw.AddAtom(Chem.Atom(els[a]))
        conf.SetAtomPosition(i, [float(c) for c in xyz[a]])
    for a in frag:
        for b in adj[a]:
            if b in remap and b > a:
                rw.AddBond(remap[a], remap[b], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineBondOrders(mol, charge=0)
    Chem.SanitizeMol(mol)
    return mol, remap


_PERIODIC = Chem.GetPeriodicTable()


def _copy_branch(rw, mol, atom_indices):
    """Copy one substituent into ``rw`` and return old-to-new atom indices."""
    atom_indices = frozenset(atom_indices)
    remap = {}
    for old_idx in sorted(atom_indices):
        remap[old_idx] = rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old_idx)))
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in remap and end in remap:
            rw.AddBond(remap[begin], remap[end], bond.GetBondType())
    return remap


def _cip_compare_branches(mol, root_a, root_b, atoms_a, atoms_b):
    """Return +1 if substituent A outranks B, -1 if B outranks A, or 0.

    RDKit's true CIP implementation labels stereogenic atoms rather than
    exposing a public ligand-comparison API.  To compare these two ligands, make
    independent copies of them and attach their roots to a tetrahedral carbon
    alongside deuterium and tritium.  For equal root atomic numbers above H, the
    two real ligands are priorities 1/2 and T/D are priorities 3/4.  With the
    fixed neighbour insertion order and clockwise chiral tag below, an ``S``
    label means A outranks B and ``R`` means B outranks A.

    Equivalent ligands leave the probe achiral.  Canonical symmetry ranks
    distinguish that valid tie from an unexpected CIP-labeling failure.
    """
    atoms_a = frozenset(atoms_a)
    atoms_b = frozenset(atoms_b)
    assert root_a in atoms_a, "root_a must be in atoms_a"
    assert root_b in atoms_b, "root_b must be in atoms_b"

    atomic_num_a = mol.GetAtomWithIdx(root_a).GetAtomicNum()
    atomic_num_b = mol.GetAtomWithIdx(root_b).GetAtomicNum()
    if atomic_num_a != atomic_num_b:
        return 1 if atomic_num_a > atomic_num_b else -1
    if atomic_num_a == 1:
        # Geometry input carries ordinary protium only, so two H roots tie.
        return 0

    rw = Chem.RWMol()
    center = rw.AddAtom(Chem.Atom(6))
    remap_a = _copy_branch(rw, mol, atoms_a)
    remap_b = _copy_branch(rw, mol, atoms_b)

    deuterium = Chem.Atom(1)
    deuterium.SetIsotope(2)
    deuterium.SetNoImplicit(True)
    d_idx = rw.AddAtom(deuterium)
    tritium = Chem.Atom(1)
    tritium.SetIsotope(3)
    tritium.SetNoImplicit(True)
    t_idx = rw.AddAtom(tritium)

    # Bond insertion order is part of RDKit's tetrahedral chiral-tag encoding.
    rw.AddBond(center, remap_a[root_a], Chem.BondType.SINGLE)
    rw.AddBond(center, remap_b[root_b], Chem.BondType.SINGLE)
    rw.AddBond(center, d_idx, Chem.BondType.SINGLE)
    rw.AddBond(center, t_idx, Chem.BondType.SINGLE)

    probe = rw.GetMol()
    Chem.SanitizeMol(probe)
    symmetry = Chem.CanonicalRankAtoms(
        probe, breakTies=False, includeChirality=False
    )
    if symmetry[remap_a[root_a]] == symmetry[remap_b[root_b]]:
        return 0

    probe.GetAtomWithIdx(center).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    rdCIPLabeler.AssignCIPLabels(probe, [center])
    center_atom = probe.GetAtomWithIdx(center)
    if not center_atom.HasProp("_CIPCode"):
        raise ValueError("RDKit could not assign the substituent-priority probe")
    code = center_atom.GetProp("_CIPCode").upper()
    if code == "S":
        return 1
    if code == "R":
        return -1
    raise ValueError(f"unexpected tetrahedral CIP code {code!r}")


def _compare_rooted_graphs(mol, parent_a, root_a, parent_b, root_b,
                           max_spheres=None):
    """Compare the rooted bpy environments used to order donor nitrogens.

    This historical sphere traversal remains only for donor-ring ordering.  It
    is deliberately not used for alkyne C1/C2 assignment, which delegates the
    full priority decision to ``rdCIPLabeler`` above.

    A "branch" is the substituent subtree rooted at `root_*` with `parent_*`
    excluded (the atom the substituent is attached to).  Multiple bonds are
    expanded with CIP duplicate (phantom) atoms: a bond of order n between X
    and Y contributes (n-1) duplicate neighbours of Y's atomic number to X and
    vice-versa.  Phantom atoms have atomic number = the duplicated atom's Z but
    no further substituents (children = empty).

    The comparison proceeds breadth-first: at each sphere the multiset of
    atomic numbers of the two frontiers is compared (sorted descending); the
    first difference decides.  Ties expand each frontier node's children in the
    matched order.  It is sufficient to order the constitutionally-distinct bpy
    environments in the curated dataset.
    """
    def children(atom_idx, came_from_idx):
        """Yield (atomic_number, next_atom_idx_or_None) for CIP digraph
        children of `atom_idx`, having arrived from `came_from_idx`.
        Real neighbours (except the came-from atom) plus phantom duplicates
        from bond orders > 1.  Phantoms have next_atom_idx_or_None = None."""
        atom = mol.GetAtomWithIdx(atom_idx)
        out = []
        for bond in atom.GetBonds():
            nb = bond.GetOtherAtom(atom)
            nb_idx = nb.GetIdx()
            order = bond.GetBondTypeAsDouble()
            # Number of phantom duplicates from this bond toward nb.
            n_dup = int(round(order)) - 1 if order > 1 else 0
            if nb_idx == came_from_idx:
                # The bond back to parent: still contributes phantom duplicates
                # (CIP counts the multiple-bond duplicate even toward parent),
                # but not the real traversal back.
                for _ in range(n_dup):
                    out.append((nb.GetAtomicNum(), None))
                continue
            out.append((nb.GetAtomicNum(), nb_idx))
            for _ in range(n_dup):
                out.append((nb.GetAtomicNum(), None))
        return out

    # Each frontier element is a node: (atomic_number, atom_idx_or_None,
    # came_from_idx_or_None).  Phantom/terminal nodes have atom_idx None and
    # produce no children.
    frontier_a = [(mol.GetAtomWithIdx(root_a).GetAtomicNum(), root_a, parent_a)]
    frontier_b = [(mol.GetAtomWithIdx(root_b).GetAtomicNum(), root_b, parent_b)]

    if max_spheres is None:
        max_spheres = 2 * mol.GetNumAtoms() + 4
    for _ in range(max_spheres):
        za = sorted((n[0] for n in frontier_a), reverse=True)
        zb = sorted((n[0] for n in frontier_b), reverse=True)
        if za != zb:
            # Compare lexicographically (pad shorter with 0 = lowest).
            for x, y in zip(za + [0] * (len(zb) - len(za)),
                            zb + [0] * (len(za) - len(zb))):
                if x != y:
                    return 1 if x > y else -1
        # Tie at this sphere: expand, keeping frontier sorted by Z descending so
        # the highest-priority branches are compared first in the next sphere.
        frontier_a.sort(key=lambda n: n[0], reverse=True)
        frontier_b.sort(key=lambda n: n[0], reverse=True)
        next_a, next_b = [], []
        for z, idx, came in frontier_a:
            if idx is None:
                continue
            for cz, cidx in children(idx, came):
                next_a.append((cz, cidx, idx))
        for z, idx, came in frontier_b:
            if idx is None:
                continue
            for cz, cidx in children(idx, came):
                next_b.append((cz, cidx, idx))
        if not next_a and not next_b:
            return 0
        frontier_a, frontier_b = next_a, next_b
    return 0


def label_alkyne_carbons(geom, alkyne_pair, r_a_root, r_b_root,
                         r_a_atoms, r_b_atoms):
    """Assign C1 (lower-priority substituent) and C2 (higher-priority).

    Parameters
    ----------
    geom : Geom
    alkyne_pair : (cA, cB)  the two mutually-bonded alkyne carbons (unordered)
    r_a_root, r_b_root : substituent attachment atoms on cA, cB respectively
    r_a_atoms, r_b_atoms : frozenset substituent fragment atoms (incl. root, H;
                           exclude the alkyne carbons and Ni)

    Returns
    -------
    dict with keys c1,c2,r1_root,r2_root,r1_atoms,r2_atoms,source
        source = "cip" when CIP orders the substituents; "symmetric_atom_order"
        for a symmetric alkyne (CIP-indistinguishable substituents), where the
        tie is broken by input atom number (C1 = smaller-index alkyne carbon).

    Raises on any RDKit perception failure (the caller -> NaN for CIP
    descriptors).
    """
    cA, cB = alkyne_pair
    assert cA != cB, "alkyne carbons must differ"
    assert r_a_root in r_a_atoms, "r_a_root must be in r_a_atoms"
    assert r_b_root in r_b_atoms, "r_b_root must be in r_b_atoms"
    assert cA not in r_a_atoms and cA not in r_b_atoms
    assert cB not in r_a_atoms and cB not in r_b_atoms
    assert geom.ni not in r_a_atoms and geom.ni not in r_b_atoms

    r_a_atoms = frozenset(r_a_atoms)
    r_b_atoms = frozenset(r_b_atoms)

    # --- standard (plain) CIP ranking ---
    # Build the metal-free alkyne molecule with perceived bond orders, then use
    # RDKit's true CIP labeler to compare independent copies of the two arms.
    mol, remap = _build_alkyne_mol(geom, alkyne_pair, r_a_atoms, r_b_atoms)
    cmp = _cip_compare_branches(
        mol,
        root_a=remap[r_a_root],
        root_b=remap[r_b_root],
        atoms_a={remap[idx] for idx in r_a_atoms},
        atoms_b={remap[idx] for idx in r_b_atoms},
    )
    if cmp == 0:
        # Symmetric alkyne: the two substituents are CIP-indistinguishable, so
        # plain CIP cannot order the carbons.  Break the tie by input atom
        # number -> C1 = the alkyne carbon with the SMALLER geometry index.
        if cA < cB:
            c1, c2 = cA, cB
            r1_root, r2_root = r_a_root, r_b_root
            r1_atoms, r2_atoms = r_a_atoms, r_b_atoms
        else:
            c1, c2 = cB, cA
            r1_root, r2_root = r_b_root, r_a_root
            r1_atoms, r2_atoms = r_b_atoms, r_a_atoms
        return {"c1": c1, "c2": c2,
                "r1_root": r1_root, "r2_root": r2_root,
                "r1_atoms": r1_atoms, "r2_atoms": r2_atoms,
                "source": "symmetric_atom_order"}

    if cmp < 0:
        # branch A (on cA) is LOWER priority -> cA = c1
        c1, c2 = cA, cB
        r1_root, r2_root = r_a_root, r_b_root
        r1_atoms, r2_atoms = r_a_atoms, r_b_atoms
    else:
        # branch A is HIGHER priority -> cB bears the lower one -> cB = c1
        c1, c2 = cB, cA
        r1_root, r2_root = r_b_root, r_a_root
        r1_atoms, r2_atoms = r_b_atoms, r_a_atoms

    return {"c1": c1, "c2": c2,
            "r1_root": r1_root, "r2_root": r2_root,
            "r1_atoms": r1_atoms, "r2_atoms": r2_atoms,
            "source": "cip"}


def _build_bpy_mol(geom, bpy_atoms):
    """Build a sanitized metal-free RDKit molecule for the bpy fragment."""
    atoms = sorted(bpy_atoms)
    remap = {a: i for i, a in enumerate(atoms)}

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        assert geom.elements[a] != "Ni", "bpy fragment must be metal-free"
        rw.AddAtom(Chem.Atom(geom.elements[a]))
        conf.SetAtomPosition(i, [float(c) for c in geom.coords[a]])
    for a in atoms:
        for b in geom.adj[a]:
            if b in remap and b > a:
                rw.AddBond(remap[a], remap[b], Chem.BondType.SINGLE)

    mol = rw.GetMol()
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineBondOrders(mol, charge=0)
    Chem.SanitizeMol(mol)
    return mol, remap


def compare_ring_nitrogens(geom, bpy_atoms, n_a, n_b, max_spheres=14):
    """Rank the two bpy donor nitrogens by their ring environments.

    Returns +1 if ``n_a`` outranks ``n_b``, -1 if ``n_b`` outranks ``n_a``, and
    0 when the two rings are constitutionally equivalent.
    """
    mol, remap = _build_bpy_mol(geom, bpy_atoms)
    ia, ib = remap[n_a], remap[n_b]
    sym = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    if sym[ia] == sym[ib]:
        return 0

    cmp = _compare_rooted_graphs(
        mol,
        parent_a=-1,
        root_a=ia,
        parent_b=-1,
        root_b=ib,
        max_spheres=max_spheres,
    )
    if cmp == 0:
        raise ValueError(
            "CIP could not resolve constitutionally-distinct pyridine N's "
            f"within {max_spheres} spheres"
        )
    return cmp
