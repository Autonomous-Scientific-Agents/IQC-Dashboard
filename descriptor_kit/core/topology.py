"""Atom identification for reactant and product (spec §5).

Faithful copy of ``src/topology.py`` with package-relative imports and the
diaryl golden-rule override REMOVED (this kit uses pure CIP).

Reactant: Ni + bpy (2 N donors, two fused pyridine rings) + alkyne (the bonded
C-C pair nearest Ni) with CIP-labeled C1/C2 (via cip.label_alkyne_carbons,
plain CIP).

Product: Ni + bpy + carboxylate metallacycle Ni-O1-C(carb)-Cα-Cβ.

All guardrail preconditions are asserted; violations raise.
"""
from __future__ import annotations

from . import geometry as g
from . import cip
from .contracts import Reactant, Product

# Alkyne C-C bond length window (Angstrom).  C≡C ≈ 1.24-1.31; we allow a margin
# but stay well below a normal C-C single bond (~1.5) so only the triple bond
# qualifies.  Used together with "bonded C-C pair nearest Ni".
_ALKYNE_CC_MIN = 1.0
_ALKYNE_CC_MAX = 1.45


def _connected_components(adj, nodes, exclude):
    """Return list of connected-component sets over `nodes`, never traversing
    into `exclude` (e.g. the Ni index)."""
    nodes = set(nodes) - set(exclude)
    seen = set()
    comps = []
    for start in nodes:
        if start in seen:
            continue
        comp = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in adj[u]:
                if v in nodes and v not in comp:
                    stack.append(v)
        seen |= comp
        comps.append(comp)
    return comps


def _find_6ring(adj, start, elements):
    """Return an ordered list of 6 heavy-atom indices forming a 6-membered
    ring that contains `start`, or None.  Depth-first, heavy atoms only."""
    found = [None]

    def dfs(path):
        if found[0] is not None:
            return
        u = path[-1]
        if len(path) == 6:
            if start in adj[u]:
                found[0] = list(path)
            return
        for v in adj[u]:
            if elements[v] == "H" or v in path:
                continue
            dfs(path + [v])

    dfs([start])
    return found[0]


def _two_donor_ns(geom):
    """The two N atoms nearest Ni (the bpy donors)."""
    ni = geom.ni
    ns = [k for k, e in enumerate(geom.elements) if e == "N"]
    assert len(ns) >= 2, f"need >=2 N atoms, found {len(ns)}"
    ns.sort(key=lambda k: g.dist(geom.coords, ni, k))
    return ns[0], ns[1]


def _bpy_ring_atoms(geom, donors):
    """The 12 bpy ring atoms (10 C + 2 N): the two fused pyridine 6-rings, one
    through each donor N.  Returns (ring_atoms_frozenset, ringA, ringB) where
    ringA/ringB are ordered lists starting at the respective donor N."""
    adj = geom.adj
    els = geom.elements
    nA, nB = donors
    ringA = _find_6ring(adj, nA, els)
    ringB = _find_6ring(adj, nB, els)
    assert ringA is not None, f"no 6-ring through donor N {nA}"
    assert ringB is not None, f"no 6-ring through donor N {nB}"
    ring_atoms = frozenset(ringA) | frozenset(ringB)
    assert len(ring_atoms) == 12, (
        f"bpy ring atoms != 12 (got {len(ring_atoms)}; rings may overlap)")
    return ring_atoms, ringA, ringB


def _bridgehead(geom, ring, donor, other_ring):
    """The bridgehead carbon of `ring`: the ring C adjacent to `donor` that is
    bonded to a carbon in `other_ring` (the inter-ring C-C bond)."""
    adj = geom.adj
    other = set(other_ring)
    cands = [c for c in ring if c != donor and donor in adj[c]
             and geom.elements[c] == "C"]
    for c in cands:
        if any(nb in other for nb in adj[c]):
            return c
    raise ValueError(f"no bridgehead C adjacent to donor {donor}")


def _assign_pyridine_ab(geom, bpy_atoms, donors, ring_a, ring_b, stereo_type):
    """Return the PyA/PyB ring assignment, or ``(None, None)`` if unassigned."""
    n_a, n_b = donors
    try:
        cmp = cip.compare_ring_nitrogens(geom, bpy_atoms, n_a, n_b)
    except Exception:  # noqa: BLE001 - descriptor layer converts unassigned rings to NaN
        return None, None

    if cmp == 0:
        return (ring_a, ring_b) if n_a <= n_b else (ring_b, ring_a)

    higher_ring, lower_ring = (ring_a, ring_b) if cmp > 0 else (ring_b, ring_a)
    stereo = str(stereo_type or "").strip().upper()
    if stereo == "O":
        return higher_ring, lower_ring
    if stereo == "S":
        return lower_ring, higher_ring
    return None, None


def identify_reactant(geom, stereo_type=""):
    """Fill the Reactant dataclass from a reactant Geom (spec §5.1).

    ``stereo_type`` ("S"/"O"/"") is used with donor-N CIP ranking to assign
    PyA/PyB for per-ring descriptor differences.

    Guardrails: exactly 1 Ni; exactly 2 donor N; exactly 2 non-Ni organic
    components; exactly 1 alkyne C-C pair; each alkyne C has exactly one
    substituent root.  Labeling is pure CIP (no golden-rule override).
    """
    els = geom.elements
    adj = geom.adj
    xyz = geom.coords
    ni = geom.ni
    assert els[ni] == "Ni", "geom.ni must point at the Ni atom"

    # --- non-Ni organic components: must be exactly 2 (bpy + alkyne) ---
    all_nodes = [k for k in range(len(els)) if k != ni]
    comps = _connected_components(adj, all_nodes, exclude={ni})
    assert len(comps) == 2, (
        f"reactant: expected exactly 2 non-Ni components, got {len(comps)}")

    # --- donor N + bpy frame ---
    donors = _two_donor_ns(geom)
    bpy_comp = next(c for c in comps if donors[0] in c)
    assert donors[1] in bpy_comp, "both donor N must lie in the same component"
    bpy_atoms = frozenset(bpy_comp)
    bpy_ring_atoms, ringA, ringB = _bpy_ring_atoms(geom, donors)
    assert bpy_ring_atoms <= bpy_atoms, "bpy ring atoms must be in bpy component"

    # --- alkyne: bonded C-C pair nearest Ni ---
    alkyne_comp = next(c for c in comps if c is not bpy_comp)
    cc_pairs = []
    for i in alkyne_comp:
        if els[i] != "C":
            continue
        for j in adj[i]:
            if j > i and j in alkyne_comp and els[j] == "C":
                d = g.dist(xyz, i, j)
                if _ALKYNE_CC_MIN < d < _ALKYNE_CC_MAX:
                    cc_pairs.append((i, j, d))
    assert len(cc_pairs) >= 1, "reactant: no candidate alkyne C-C pair found"
    cc_pairs.sort(key=lambda p: g.dist(xyz, ni, p[0]) + g.dist(xyz, ni, p[1]))
    cA, cB, _ = cc_pairs[0]

    # --- substituent roots: each alkyne C's unique neighbour that is neither
    #     Ni nor the other alkyne carbon (an H counts) ---
    def roots(c):
        return [n for n in adj[c] if n != ni and n != cA and n != cB]
    roots_a, roots_b = roots(cA), roots(cB)
    assert len(roots_a) == 1, (
        f"alkyne C {cA} has {len(roots_a)} substituent roots, expected 1")
    assert len(roots_b) == 1, (
        f"alkyne C {cB} has {len(roots_b)} substituent roots, expected 1")
    r_a_root, r_b_root = roots_a[0], roots_b[0]

    r_a_atoms = frozenset(g.fragment_bfs(adj, {r_a_root}, {cA, cB}))
    r_b_atoms = frozenset(g.fragment_bfs(adj, {r_b_root}, {cA, cB}))
    assert ni not in r_a_atoms and ni not in r_b_atoms

    # --- CIP labeling (c1 = lower-priority substituent carbon, inverted) ---
    lab = cip.label_alkyne_carbons(geom, (cA, cB), r_a_root, r_b_root,
                                   r_a_atoms, r_b_atoms)
    py_a_ring, py_b_ring = _assign_pyridine_ab(
        geom,
        bpy_atoms,
        donors,
        frozenset(ringA),
        frozenset(ringB),
        stereo_type,
    )

    return Reactant(
        geom=geom,
        n_donors=donors,
        bpy_atoms=bpy_atoms,
        bpy_ring_atoms=bpy_ring_atoms,
        c1=lab["c1"],
        c2=lab["c2"],
        r1_root=lab["r1_root"],
        r2_root=lab["r2_root"],
        r1_atoms=lab["r1_atoms"],
        r2_atoms=lab["r2_atoms"],
        cip_source=lab["source"],
        stereo_type=str(stereo_type or "").strip().upper(),
        py_a_ring=py_a_ring,
        py_b_ring=py_b_ring,
    )


def identify_product(geom):
    """Fill the Product dataclass from a product Geom (spec §5.2).

    Guardrails: exactly 1 Ni; >=2 N donors; one C(carb) with 2 O; metallacycle
    Ni-O1-C(carb)-Cα-Cβ closes.
    """
    els = geom.elements
    adj = geom.adj
    xyz = geom.coords
    ni = geom.ni
    assert els[ni] == "Ni", "geom.ni must point at the Ni atom"

    donors = _two_donor_ns(geom)
    bpy_atoms_comp = _connected_components(adj, range(len(els)), exclude={ni})
    bpy_comp = next(c for c in bpy_atoms_comp if donors[0] in c)
    assert donors[1] in bpy_comp, "both donor N must be in the same component"
    bpy_atoms = frozenset(bpy_comp)
    bpy_ring_atoms, _, _ = _bpy_ring_atoms(geom, donors)

    # O1 = nearest O to Ni
    os = [k for k, e in enumerate(els) if e == "O"]
    assert len(os) >= 2, f"product: need >=2 O, found {len(os)}"
    o1 = min(os, key=lambda k: g.dist(xyz, ni, k))

    # C(carb) = C neighbour of O1 with exactly two O neighbours
    ccarb_cands = [c for c in adj[o1] if els[c] == "C"
                   and sum(1 for n in adj[c] if els[n] == "O") == 2]
    assert len(ccarb_cands) == 1, (
        f"product: expected exactly 1 carboxylate C bonded to O1, "
        f"got {len(ccarb_cands)}")
    ccarb = ccarb_cands[0]

    # O2 = C(carb)'s other O
    o2_cands = [o for o in adj[ccarb] if els[o] == "O" and o != o1]
    assert len(o2_cands) == 1, "product: C(carb) must have exactly 2 O"
    o2 = o2_cands[0]

    # Cβ = nearest C to Ni
    cs = [k for k, e in enumerate(els) if e == "C"]
    c_beta = min(cs, key=lambda k: g.dist(xyz, ni, k))

    # Cα = C neighbour of C(carb) that is bonded to Cβ
    ca_cands = [c for c in adj[ccarb] if els[c] == "C" and c_beta in adj[c]]
    assert len(ca_cands) == 1, (
        f"product: expected exactly 1 Cα (C neighbour of C(carb) bonded to Cβ),"
        f" got {len(ca_cands)}")
    c_alpha = ca_cands[0]
    assert c_alpha != c_beta, "Cα and Cβ must differ"

    # Ring closure guardrail
    assert o1 in adj[ccarb], "metallacycle: O1 not bonded to C(carb)"
    assert c_alpha in adj[ccarb], "metallacycle: Cα not bonded to C(carb)"
    assert c_beta in adj[c_alpha], "metallacycle: Cβ not bonded to Cα"

    # Rα / Rβ: BFS off Cα / Cβ excluding C(carb), the other alkene C, and Ni.
    r_alpha = frozenset(g.fragment_bfs(adj, {c_alpha},
                                       {ccarb, c_beta, ni})) - {c_alpha}
    r_beta = frozenset(g.fragment_bfs(adj, {c_beta},
                                      {ccarb, c_alpha, ni})) - {c_beta}

    return Product(
        geom=geom,
        n_donors=donors,
        bpy_atoms=bpy_atoms,
        bpy_ring_atoms=bpy_ring_atoms,
        ccarb=ccarb,
        o1=o1,
        o2=o2,
        c_alpha=c_alpha,
        c_beta=c_beta,
        r_alpha_atoms=r_alpha,
        r_beta_atoms=r_beta,
        metallacycle=(ni, o1, ccarb, c_alpha, c_beta),
    )


def pyridine_rings(obj):
    """Return (ringA, ringB) as two frozensets of 6 ring-atom indices, one
    pyridine ring per bpy donor N. Used to split bpy substituents per ring
    (D3, D14, D15). `obj` is a Reactant or Product (has .geom, .n_donors)."""
    geom = obj.geom
    rings = []
    for d in obj.n_donors:
        r = _find_6ring(geom.adj, d, geom.elements)
        assert r is not None, f"no 6-ring through donor {d}"
        rings.append(frozenset(r))
    assert len(rings[0] & rings[1]) == 0, "pyridine rings overlap"
    return rings[0], rings[1]


def bpy_dihedral_atoms(obj):
    """Return (N_A, bridgeheadA, bridgeheadB, N_B) for the bpy inter-ring
    dihedral N–C2–C2′–N′ (D16/D46). `obj` is a Reactant or Product."""
    geom = obj.geom
    nA, nB = obj.n_donors
    ringA = _find_6ring(geom.adj, nA, geom.elements)
    ringB = _find_6ring(geom.adj, nB, geom.elements)
    assert ringA is not None and ringB is not None, "missing bpy 6-ring"
    bridgeA = _bridgehead(geom, ringA, nA, ringB)
    bridgeB = _bridgehead(geom, ringB, nB, ringA)
    return nA, bridgeA, bridgeB, nB


def bpy_ring_position(obj, ring_atom_idx):
    """Position 1..6 of a bpy ring atom within its pyridine ring.

    Walk from the donor N (=1), bridgehead C (=2), then 3,4,5,6 around the ring.
    `obj` is a Reactant or Product (has .geom, .n_donors).
    """
    geom = obj.geom
    adj = geom.adj
    els = geom.elements
    donors = obj.n_donors

    # Which donor's ring contains ring_atom_idx?
    rings = []
    for d in donors:
        r = _find_6ring(adj, d, els)
        assert r is not None, f"no 6-ring through donor {d}"
        rings.append((d, r))
    for d, r in rings:
        if ring_atom_idx in r:
            ring = r
            donor = d
            other = next(rr for dd, rr in rings if dd != d)
            break
    else:
        raise ValueError(f"atom {ring_atom_idx} is not in any bpy pyridine ring")

    bridge = _bridgehead(geom, ring, donor, other)

    # Order the ring as a cycle starting donor -> bridge, then continue.
    # Build cycle order by walking neighbours within the ring.
    ring_set = set(ring)
    order = [donor, bridge]
    while len(order) < 6:
        u = order[-1]
        nxt = [v for v in adj[u]
               if v in ring_set and v not in order]
        # Avoid jumping straight back to donor before completing the ring.
        if len(order) < 6:
            nxt = [v for v in nxt if not (v == donor)]
        assert nxt, f"ring walk stuck at {u}; ring {ring}"
        order.append(nxt[0])

    pos = order.index(ring_atom_idx) + 1
    return pos
