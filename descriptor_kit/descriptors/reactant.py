"""Reactant descriptors — one public function per output key.

Every public function is named EXACTLY after the descriptor it produces and
returns ``{that_key: float}``.  Shared math (Sterimol of R1/R2, bpy-substituent
enumeration, σ-sums) lives in the private ``_helpers`` below; those are not
descriptors.  Helpers are recomputed per call rather than memoized — it is one
molecule per invocation, so the few extra morfeus calls are negligible and each
descriptor stays fully self-contained.

Each function asserts its preconditions and raises on violation; the orchestrator
(``descriptor_kit.api``) converts a raise into NaN (unless strict=True), so a
finite-but-wrong number is never returned.

Original D-number is noted per function.
"""
from __future__ import annotations
import math

import numpy as np

from ..core import geometry as g
from ..core import topology as topo
from ..core import hammett
from ..core import steric as st


# --------------------------------------------------------------------------- #
# private helpers (NOT descriptors)                                            #
# --------------------------------------------------------------------------- #
_BPY_INCLUDED_POSITIONS = frozenset({3, 4, 5, 6})
_POS_CLASS = {3: "meta", 4: "para", 5: "meta", 6: "ortho"}
_PERPOS_SIGMA_COLUMNS = [
    f"reac_sigma_{cls}_{ring}"
    for ring in ("pyA", "pyB")
    for cls in ("ortho", "meta", "para")
]
_PERPOS_B5_COLUMNS = [
    f"reac_B5_{cls}_{ring}"
    for ring in ("pyA", "pyB")
    for cls in ("ortho", "meta", "para")
]


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _check_alkyne(reactant):
    """Assert the alkyne carbons + substituent roots are well-formed."""
    assert reactant.c1 != reactant.c2, "alkyne carbons c1 and c2 must differ"
    geom = reactant.geom
    assert geom.elements[reactant.c1] == "C", "c1 must be a carbon"
    assert geom.elements[reactant.c2] == "C", "c2 must be a carbon"
    assert reactant.c2 in geom.adj[reactant.c1], "c1 and c2 must be bonded"
    assert reactant.r1_root in geom.adj[reactant.c1], "r1_root must attach to c1"
    assert reactant.r2_root in geom.adj[reactant.c2], "r2_root must attach to c2"


def _check_donors(reactant):
    nA, nB = reactant.n_donors
    geom = reactant.geom
    assert nA != nB, "the two bpy donors must differ"
    assert geom.elements[nA] == "N" and geom.elements[nB] == "N", (
        "both bpy donors must be nitrogen")


def _bpy_substituents(reactant):
    """Yield ``(ring_carbon, position, root, frag_atoms)`` for every bpy
    substituent: a non-ring heavy neighbour of a bpy ring *carbon*, with its
    BFS fragment grown outward (stopping at the ring atoms)."""
    geom = reactant.geom
    els = geom.elements
    adj = geom.adj
    ring = reactant.bpy_ring_atoms
    for ring_atom in sorted(ring):
        if els[ring_atom] != "C":
            continue  # only ring carbons bear substituents (N is the donor)
        position = topo.bpy_ring_position(reactant, ring_atom)
        for nb in sorted(adj[ring_atom]):
            if nb in ring or els[nb] == "H":
                continue  # ring bonds and ring H are not substituents
            frag = frozenset(g.fragment_bfs(adj, {nb}, set(ring)))
            yield ring_atom, position, nb, frag


def _sum_bpy_sigma(reactant, ring_filter=None):
    """Σ σ over bpy substituents at positions 3/4/5/6.  ``ring_filter`` restricts
    the sum to substituents whose ring carbon lies in that ring (D3).  NaN if any
    INCLUDED substituent's sigma is untabulated; positions 1/2 are skipped."""
    geom = reactant.geom
    total = 0.0
    for ring_atom, position, root, frag in _bpy_substituents(reactant):
        if position not in _BPY_INCLUDED_POSITIONS:
            continue
        if ring_filter is not None and ring_atom not in ring_filter:
            continue
        sigma = hammett.sigma_for_fragment(geom, root, frag, position)
        if _isnan(sigma):
            return float("nan")
        total += sigma
    return total


def _perpos_sigma(reactant):
    """Hammett sigma split by PyA/PyB and ortho/meta/para position class."""
    assert reactant.bpy_ring_atoms, "reactant.bpy_ring_atoms is empty"
    ring_a = reactant.py_a_ring
    if ring_a is None or reactant.py_b_ring is None:
        return {col: float("nan") for col in _PERPOS_SIGMA_COLUMNS}

    geom = reactant.geom
    out = {col: 0.0 for col in _PERPOS_SIGMA_COLUMNS}
    for ring_atom, position, root, frag in _bpy_substituents(reactant):
        cls = _POS_CLASS.get(position)
        if cls is None:
            continue
        ring = "pyA" if ring_atom in ring_a else "pyB"
        col = f"reac_sigma_{cls}_{ring}"
        if _isnan(out[col]):
            continue
        sigma = hammett.sigma_for_fragment(geom, root, frag, position)
        out[col] = float("nan") if _isnan(sigma) else out[col] + sigma
    return out


def _perpos_b5(reactant):
    """Sterimol B5 split by PyA/PyB and ortho/meta/para position class."""
    ring_a = reactant.py_a_ring
    if ring_a is None or reactant.py_b_ring is None:
        return {col: float("nan") for col in _PERPOS_B5_COLUMNS}

    geom = reactant.geom
    out = {col: 0.0 for col in _PERPOS_B5_COLUMNS}
    for ring_atom, position, root, frag in _bpy_substituents(reactant):
        cls = _POS_CLASS.get(position)
        if cls is None:
            continue
        ring = "pyA" if ring_atom in ring_a else "pyB"
        out[f"reac_B5_{cls}_{ring}"] += st.sterimol(geom, ring_atom, root, set(frag))["B5"]
    for col, value in out.items():
        assert np.isfinite(value) and value >= 0, f"B5 invalid for {col}: {value}"
    return out


def _sigma_R1(reactant):
    """Alkyne group sigma of R1 (substituent on c1).  position=None -> group
    constant (σ_p-type) or Taft fallback.  NaN if untabulated."""
    assert reactant.r1_root in reactant.r1_atoms, "r1_root must lie in r1_atoms"
    return hammett.sigma_for_fragment(
        reactant.geom, reactant.r1_root, reactant.r1_atoms, None)


def _sigma_R2(reactant):
    """Alkyne group sigma of R2 (substituent on c2)."""
    assert reactant.r2_root in reactant.r2_atoms, "r2_root must lie in r2_atoms"
    return hammett.sigma_for_fragment(
        reactant.geom, reactant.r2_root, reactant.r2_atoms, None)


def _sterimol_R1(reactant):
    """Sterimol {L,B1,B5} of R1 (dummy=c1, attached=r1_root, frag=r1_atoms)."""
    return st.sterimol(reactant.geom, reactant.c1, reactant.r1_root,
                       set(reactant.r1_atoms))


def _sterimol_R2(reactant):
    """Sterimol {L,B1,B5} of R2 (dummy=c2, attached=r2_root, frag=r2_atoms)."""
    return st.sterimol(reactant.geom, reactant.c2, reactant.r2_root,
                       set(reactant.r2_atoms))


def _bends(reactant):
    """(bend1, bend2): deviation from linear at C1 and C2 (degrees)."""
    c = reactant.geom.coords
    bend1 = 180.0 - g.angle(c, reactant.r1_root, reactant.c1, reactant.c2)
    bend2 = 180.0 - g.angle(c, reactant.r2_root, reactant.c2, reactant.c1)
    return bend1, bend2


def _heavy_centroid(geom, atoms):
    heavy = [a for a in atoms if geom.elements[a] != "H"]
    assert heavy, "no heavy atoms to take a centroid over"
    return np.asarray(geom.coords)[heavy].mean(axis=0)


def _vector_angle_deg(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    assert nu > 0 and nv > 0, "zero-length vector in angle computation"
    cosv = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# §8.1 electronic (Hammett / Taft)                                            #
# --------------------------------------------------------------------------- #
def reac_sum_sigma_bpy(reactant):
    """D1: Σ σ of all bpy substituents at positions 3/4/5/6 (both rings)."""
    assert reactant.bpy_ring_atoms, "reactant.bpy_ring_atoms is empty"
    assert len(reactant.bpy_ring_atoms) == 12, (
        f"expected 12 bpy ring atoms, got {len(reactant.bpy_ring_atoms)}")
    return {"reac_sum_sigma_bpy": _sum_bpy_sigma(reactant)}


def reac_sum_sigma_alkyne(reactant):
    """D2: σ(R1) + σ(R2).  NaN if either is untabulated."""
    s1, s2 = _sigma_R1(reactant), _sigma_R2(reactant)
    if _isnan(s1) or _isnan(s2):
        return {"reac_sum_sigma_alkyne": float("nan")}
    return {"reac_sum_sigma_alkyne": s1 + s2}


def reac_dsigma_pyA_pyB(reactant):
    """D3: σ(PyA) − σ(PyB), signed, over positions 3/4/5/6."""
    assert reactant.bpy_ring_atoms, "reactant.bpy_ring_atoms is empty"
    if reactant.py_a_ring is None or reactant.py_b_ring is None:
        return {"reac_dsigma_pyA_pyB": float("nan")}
    sigA = _sum_bpy_sigma(reactant, ring_filter=reactant.py_a_ring)
    sigB = _sum_bpy_sigma(reactant, ring_filter=reactant.py_b_ring)
    if _isnan(sigA) or _isnan(sigB):
        return {"reac_dsigma_pyA_pyB": float("nan")}
    return {"reac_dsigma_pyA_pyB": sigA - sigB}


def reac_dsigma_alkyne(reactant):
    """D4: σ(R1) − σ(R2) (signed).  NaN if either is untabulated."""
    s1, s2 = _sigma_R1(reactant), _sigma_R2(reactant)
    if _isnan(s1) or _isnan(s2):
        return {"reac_dsigma_alkyne": float("nan")}
    return {"reac_dsigma_alkyne": s1 - s2}


# --------------------------------------------------------------------------- #
# §8.2 alkyne bond metric                                                     #
# --------------------------------------------------------------------------- #
def reac_cc_triple_len(reactant):
    """D7: |C1 − C2| (alkyne triple-bond length)."""
    _check_alkyne(reactant)
    return {"reac_cc_triple_len": g.dist(reactant.geom.coords,
                                         reactant.c1, reactant.c2)}


# --------------------------------------------------------------------------- #
# §8.3 steric — reactant                                                      #
# --------------------------------------------------------------------------- #
def reac_bpy_vbur(reactant):
    """D9: fragment-only %V_bur around Ni, bpy fragment."""
    geom = reactant.geom
    pct = st.percent_buried_volume(geom, geom.ni, set(reactant.bpy_atoms))
    assert 0.0 < pct < 100.0, f"D9 %Vbur out of range: {pct}"
    return {"reac_bpy_vbur": pct}


def reac_alkyne_vbur(reactant):
    """D10: %V_bur around Ni of {c1,c2} ∪ R1 ∪ R2."""
    geom = reactant.geom
    include = {reactant.c1, reactant.c2} | set(reactant.r1_atoms) | set(reactant.r2_atoms)
    pct = st.percent_buried_volume(geom, geom.ni, include)
    assert 0.0 < pct < 100.0, f"D10 %Vbur out of range: {pct}"
    return {"reac_alkyne_vbur": pct}


def reac_B5_R1(reactant):
    """D11: Sterimol B5 of R1."""
    return {"reac_B5_R1": _sterimol_R1(reactant)["B5"]}


def reac_B5_R2(reactant):
    """D11: Sterimol B5 of R2."""
    return {"reac_B5_R2": _sterimol_R2(reactant)["B5"]}


def reac_B5_mean(reactant):
    """D11: mean Sterimol B5 of R1, R2."""
    b1 = _sterimol_R1(reactant)["B5"]
    b2 = _sterimol_R2(reactant)["B5"]
    return {"reac_B5_mean": (b1 + b2) / 2.0}


def reac_L_R1(reactant):
    """D12: Sterimol L of R1."""
    return {"reac_L_R1": _sterimol_R1(reactant)["L"]}


def reac_L_R2(reactant):
    """D12: Sterimol L of R2."""
    return {"reac_L_R2": _sterimol_R2(reactant)["L"]}


def reac_L_mean(reactant):
    """D12: mean Sterimol L of R1, R2."""
    l1 = _sterimol_R1(reactant)["L"]
    l2 = _sterimol_R2(reactant)["L"]
    return {"reac_L_mean": (l1 + l2) / 2.0}


def reac_B1_R1(reactant):
    """D13: Sterimol B1 of R1."""
    return {"reac_B1_R1": _sterimol_R1(reactant)["B1"]}


def reac_B1_R2(reactant):
    """D13: Sterimol B1 of R2."""
    return {"reac_B1_R2": _sterimol_R2(reactant)["B1"]}


def reac_B1_mean(reactant):
    """D13: mean Sterimol B1 of R1, R2."""
    b1 = _sterimol_R1(reactant)["B1"]
    b2 = _sterimol_R2(reactant)["B1"]
    return {"reac_B1_mean": (b1 + b2) / 2.0}


def reac_sum_B5_bpy(reactant):
    """D14: Σ Sterimol B5 over all bpy substituents (0 if none)."""
    geom = reactant.geom
    total = 0.0
    for rc, _position, root, frag in _bpy_substituents(reactant):
        total += st.sterimol(geom, rc, root, set(frag))["B5"]
    assert np.isfinite(total) and total >= 0, f"D14 sum B5 invalid: {total}"
    return {"reac_sum_B5_bpy": float(total)}


def reac_dB5_bpy(reactant):
    """D15: ΣB5(PyA) − ΣB5(PyB), signed, over bpy substituents."""
    geom = reactant.geom
    ringA, ringB = reactant.py_a_ring, reactant.py_b_ring
    if ringA is None or ringB is None:
        return {"reac_dB5_bpy": float("nan")}
    sumA = sumB = 0.0
    for rc, _position, root, frag in _bpy_substituents(reactant):
        b5 = st.sterimol(geom, rc, root, set(frag))["B5"]
        if rc in ringA:
            sumA += b5
        elif rc in ringB:
            sumB += b5
        else:
            raise ValueError(
                f"D15: substituent ring carbon {rc} in neither pyridine ring")
    val = sumA - sumB
    assert np.isfinite(val), f"D15 invalid: {val}"
    return {"reac_dB5_bpy": float(val)}


# --------------------------------------------------------------------------- #
# §8.4 bpy geometry                                                           #
# --------------------------------------------------------------------------- #
def reac_bpy_dihedral(reactant):
    """D16: signed dihedral N_A−bridgeA−bridgeB−N_B (bpy inter-ring torsion)."""
    _check_donors(reactant)
    nA, bA, bB, nB = topo.bpy_dihedral_atoms(reactant)
    return {"reac_bpy_dihedral": g.signed_dihedral(reactant.geom.coords,
                                                   nA, bA, bB, nB)}


# --------------------------------------------------------------------------- #
# §8.2 alkyne bends                                                           #
# --------------------------------------------------------------------------- #
def reac_bend_R1(reactant):
    """D17: alkyne bend at C1 = 180 − angle(R1, C1, C2)."""
    _check_alkyne(reactant)
    bend1, _ = _bends(reactant)
    return {"reac_bend_R1": float(bend1)}


def reac_bend_R2(reactant):
    """D17: alkyne bend at C2 = 180 − angle(R2, C2, C1)."""
    _check_alkyne(reactant)
    _, bend2 = _bends(reactant)
    return {"reac_bend_R2": float(bend2)}


def reac_bend_mean(reactant):
    """D17: mean alkyne bend."""
    _check_alkyne(reactant)
    bend1, bend2 = _bends(reactant)
    return {"reac_bend_mean": float((bend1 + bend2) / 2.0)}


# --------------------------------------------------------------------------- #
# §8.5 Ni coordination — reactant                                             #
# --------------------------------------------------------------------------- #
def reac_ni_c_mean(reactant):
    """D18: mean(|Ni−C1|, |Ni−C2|)."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    ni = reactant.geom.ni
    d1 = g.dist(c, ni, reactant.c1)
    d2 = g.dist(c, ni, reactant.c2)
    return {"reac_ni_c_mean": float((d1 + d2) / 2.0)}


def reac_coordplane_rms(reactant):
    """D19: RMS deviation of {Ni, N_A, N_B, C1, C2} from their best-fit plane."""
    _check_alkyne(reactant)
    _check_donors(reactant)
    nA, nB = reactant.n_donors
    ni = reactant.geom.ni
    idxs = [ni, nA, nB, reactant.c1, reactant.c2]
    assert len(set(idxs)) == 5, "the five coordination atoms must be distinct"
    _, _, rms = g.best_fit_plane(reactant.geom.coords, idxs)
    return {"reac_coordplane_rms": float(rms)}


def reac_n_ni_c_sum(reactant):
    """D20: sum of {N_A-Ni-C1, N_A-Ni-C2, N_B-Ni-C1, N_B-Ni-C2}."""
    _check_alkyne(reactant)
    _check_donors(reactant)
    nA, nB = reactant.n_donors
    ni = reactant.geom.ni
    c = reactant.geom.coords
    a = np.asarray([
        g.angle(c, nA, ni, reactant.c1), g.angle(c, nA, ni, reactant.c2),
        g.angle(c, nB, ni, reactant.c1), g.angle(c, nB, ni, reactant.c2),
    ], float)
    return {"reac_n_ni_c_sum": float(a.sum())}


def reac_n_ni_c_std(reactant):
    """D20: population std (ddof=0) of the four N-Ni-C angles."""
    _check_alkyne(reactant)
    _check_donors(reactant)
    nA, nB = reactant.n_donors
    ni = reactant.geom.ni
    c = reactant.geom.coords
    a = np.asarray([
        g.angle(c, nA, ni, reactant.c1), g.angle(c, nA, ni, reactant.c2),
        g.angle(c, nB, ni, reactant.c1), g.angle(c, nB, ni, reactant.c2),
    ], float)
    return {"reac_n_ni_c_std": float(a.std(ddof=0))}


def reac_ni_alkyne_centroid(reactant):
    """D22: dist(Ni, midpoint(C1, C2))."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    ni = reactant.geom.ni
    mid = (c[reactant.c1] + c[reactant.c2]) / 2.0
    return {"reac_ni_alkyne_centroid": float(np.linalg.norm(c[ni] - mid))}


def reac_ni_bpyplane_dist(reactant):
    """D23: |distance of Ni from the bpy heavy-atom plane (10 C + 2 N)|."""
    _check_donors(reactant)
    ring = list(reactant.bpy_ring_atoms)
    assert len(ring) == 12, f"bpy ring must have 12 atoms, got {len(ring)}"
    centroid, normal, _ = g.best_fit_plane(reactant.geom.coords, ring)
    d = g.point_plane_distance(reactant.geom.coords[reactant.geom.ni],
                               centroid, normal)
    return {"reac_ni_bpyplane_dist": float(abs(d))}


# --------------------------------------------------------------------------- #
# §8.9 alkyne steric/geometry asymmetry — reactant                            #
# --------------------------------------------------------------------------- #
def reac_dB5_alkyne(reactant):
    """D48: B5(R1) − B5(R2) (signed)."""
    return {"reac_dB5_alkyne": float(_sterimol_R1(reactant)["B5"]
                                     - _sterimol_R2(reactant)["B5"])}


def reac_dL_alkyne(reactant):
    """D49: L(R1) − L(R2) (signed)."""
    return {"reac_dL_alkyne": float(_sterimol_R1(reactant)["L"]
                                    - _sterimol_R2(reactant)["L"])}


def reac_dvol_alkyne(reactant):
    """D50: vdW_volume(R1) − vdW_volume(R2) (Bondi, signed)."""
    geom = reactant.geom
    v1 = st.vdw_volume(geom, set(reactant.r1_atoms))
    v2 = st.vdw_volume(geom, set(reactant.r2_atoms))
    return {"reac_dvol_alkyne": float(v1 - v2)}


def reac_dvbur_substituent(reactant):
    """D51: %V_bur({c1}∪R1) − %V_bur({c2}∪R2) (signed)."""
    geom = reactant.geom
    pct1 = st.percent_buried_volume(geom, geom.ni,
                                    {reactant.c1} | set(reactant.r1_atoms))
    pct2 = st.percent_buried_volume(geom, geom.ni,
                                    {reactant.c2} | set(reactant.r2_atoms))
    assert 0.0 < pct1 < 100.0 and 0.0 < pct2 < 100.0, (
        f"D51 %Vbur out of range: {pct1}, {pct2}")
    return {"reac_dvbur_substituent": float(pct1 - pct2)}


def reac_dni_c_signed(reactant):
    """D52: |Ni−C1| − |Ni−C2| (signed)."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    ni = reactant.geom.ni
    return {"reac_dni_c_signed": float(g.dist(c, ni, reactant.c1)
                                       - g.dist(c, ni, reactant.c2))}


def reac_dbend_alkyne(reactant):
    """D53: bend1 − bend2 (the two D17 bends)."""
    _check_alkyne(reactant)
    bend1, bend2 = _bends(reactant)
    return {"reac_dbend_alkyne": float(bend1 - bend2)}


def reac_dccr_angle(reactant):
    """D54: angle(C2, C1, R1) − angle(C1, C2, R2)."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    a1 = g.angle(c, reactant.c2, reactant.c1, reactant.r1_root)
    a2 = g.angle(c, reactant.c1, reactant.c2, reactant.r2_root)
    return {"reac_dccr_angle": float(a1 - a2)}


def reac_dnicr_angle(reactant):
    """D55: angle(Ni, C1, R1) − angle(Ni, C2, R2)."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    ni = reactant.geom.ni
    a1 = g.angle(c, ni, reactant.c1, reactant.r1_root)
    a2 = g.angle(c, ni, reactant.c2, reactant.r2_root)
    return {"reac_dnicr_angle": float(a1 - a2)}


def reac_slippage(reactant):
    """D56: η² slippage — signed projection of (foot − midpoint) onto C1→C2."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    ni = reactant.geom.ni
    p1 = c[reactant.c1]
    p2 = c[reactant.c2]
    axis = p2 - p1
    norm = np.linalg.norm(axis)
    assert norm > 1e-9, "C1 and C2 coincide; cannot define an alkyne axis"
    u = axis / norm
    mid = (p1 + p2) / 2.0
    foot = p1 + np.dot(c[ni] - p1, u) * u
    return {"reac_slippage": float(np.dot(foot - mid, u))}


def reac_ni_firstatom_R1(reactant):
    """D57: |Ni − first(R1)| (first = the substituent root on c1)."""
    _check_alkyne(reactant)
    return {"reac_ni_firstatom_R1": float(g.dist(reactant.geom.coords,
                                                 reactant.geom.ni,
                                                 reactant.r1_root))}


def reac_ni_firstatom_R2(reactant):
    """D57: |Ni − first(R2)| (first = the substituent root on c2)."""
    _check_alkyne(reactant)
    return {"reac_ni_firstatom_R2": float(g.dist(reactant.geom.coords,
                                                 reactant.geom.ni,
                                                 reactant.r2_root))}


def reac_dni_firstatom(reactant):
    """D57: |Ni − first(R1)| − |Ni − first(R2)| (signed)."""
    _check_alkyne(reactant)
    c = reactant.geom.coords
    ni = reactant.geom.ni
    d1 = g.dist(c, ni, reactant.r1_root)
    d2 = g.dist(c, ni, reactant.r2_root)
    return {"reac_dni_firstatom": float(d1 - d2)}


def reac_bulky_orientation(reactant):
    """D58: angle between (Ni→bpy centroid) and (C_bulk→R_bulk centroid).

    The bulkier alkyne side is the one with the larger Sterimol B5.
    """
    geom = reactant.geom
    sR1, sR2 = _sterimol_R1(reactant), _sterimol_R2(reactant)
    if sR1["B5"] >= sR2["B5"]:
        c_bulk = reactant.c1
        r_bulk = set(reactant.r1_atoms)
    else:
        c_bulk = reactant.c2
        r_bulk = set(reactant.r2_atoms)
    bpy_centroid = _heavy_centroid(geom, set(reactant.bpy_atoms))
    r_centroid = _heavy_centroid(geom, r_bulk)
    coords = np.asarray(geom.coords)
    v1 = bpy_centroid - coords[geom.ni]
    v2 = r_centroid - coords[c_bulk]
    ang = _vector_angle_deg(v1, v2)
    assert 0.0 <= ang <= 180.0, f"D58 angle out of range: {ang}"
    return {"reac_bulky_orientation": ang}


def reac_alkyne_bpy_dihedral(reactant):
    """D60: angle between the R1−R2 and C1−C2 *lines*, each projected onto the
    bpy plane, folded into [0, 90].
    """
    _check_alkyne(reactant)
    _check_donors(reactant)
    ring = list(reactant.bpy_ring_atoms)
    assert len(ring) == 12, f"bpy ring must have 12 atoms, got {len(ring)}"
    c = reactant.geom.coords
    _, normal, _ = g.best_fit_plane(c, ring)
    n = normal / np.linalg.norm(normal)

    def project(vec):
        return vec - np.dot(vec, n) * n

    rr = project(c[reactant.r2_root] - c[reactant.r1_root])
    cc = project(c[reactant.c2] - c[reactant.c1])
    nrr = np.linalg.norm(rr)
    ncc = np.linalg.norm(cc)
    assert nrr > 1e-9, "R1-R2 projected onto bpy plane is degenerate"
    assert ncc > 1e-9, "C1-C2 projected onto bpy plane is degenerate"
    cosv = float(np.dot(rr, cc) / (nrr * ncc))
    ang = float(np.degrees(np.arccos(np.clip(abs(cosv), 0.0, 1.0))))
    return {"reac_alkyne_bpy_dihedral": ang}


def reac_sigma_ortho_pyA(reactant):
    """D66: Hammett sigma at ortho position 6 of PyA."""
    return {"reac_sigma_ortho_pyA": _perpos_sigma(reactant)["reac_sigma_ortho_pyA"]}


def reac_sigma_meta_pyA(reactant):
    """D66: Hammett sigma sum at meta positions 3/5 of PyA."""
    return {"reac_sigma_meta_pyA": _perpos_sigma(reactant)["reac_sigma_meta_pyA"]}


def reac_sigma_para_pyA(reactant):
    """D66: Hammett sigma at para position 4 of PyA."""
    return {"reac_sigma_para_pyA": _perpos_sigma(reactant)["reac_sigma_para_pyA"]}


def reac_sigma_ortho_pyB(reactant):
    """D66: Hammett sigma at ortho position 6 of PyB."""
    return {"reac_sigma_ortho_pyB": _perpos_sigma(reactant)["reac_sigma_ortho_pyB"]}


def reac_sigma_meta_pyB(reactant):
    """D66: Hammett sigma sum at meta positions 3/5 of PyB."""
    return {"reac_sigma_meta_pyB": _perpos_sigma(reactant)["reac_sigma_meta_pyB"]}


def reac_sigma_para_pyB(reactant):
    """D66: Hammett sigma at para position 4 of PyB."""
    return {"reac_sigma_para_pyB": _perpos_sigma(reactant)["reac_sigma_para_pyB"]}


def reac_B5_ortho_pyA(reactant):
    """D67: Sterimol B5 at ortho position 6 of PyA."""
    return {"reac_B5_ortho_pyA": _perpos_b5(reactant)["reac_B5_ortho_pyA"]}


def reac_B5_meta_pyA(reactant):
    """D67: Sterimol B5 sum at meta positions 3/5 of PyA."""
    return {"reac_B5_meta_pyA": _perpos_b5(reactant)["reac_B5_meta_pyA"]}


def reac_B5_para_pyA(reactant):
    """D67: Sterimol B5 at para position 4 of PyA."""
    return {"reac_B5_para_pyA": _perpos_b5(reactant)["reac_B5_para_pyA"]}


def reac_B5_ortho_pyB(reactant):
    """D67: Sterimol B5 at ortho position 6 of PyB."""
    return {"reac_B5_ortho_pyB": _perpos_b5(reactant)["reac_B5_ortho_pyB"]}


def reac_B5_meta_pyB(reactant):
    """D67: Sterimol B5 sum at meta positions 3/5 of PyB."""
    return {"reac_B5_meta_pyB": _perpos_b5(reactant)["reac_B5_meta_pyB"]}


def reac_B5_para_pyB(reactant):
    """D67: Sterimol B5 at para position 4 of PyB."""
    return {"reac_B5_para_pyB": _perpos_b5(reactant)["reac_B5_para_pyB"]}


# Ordered registry of every reactant descriptor function (one per output key).
ALL = [
    reac_sum_sigma_bpy, reac_sum_sigma_alkyne, reac_dsigma_pyA_pyB,
    reac_dsigma_alkyne, reac_cc_triple_len, reac_bpy_vbur, reac_alkyne_vbur,
    reac_B5_R1, reac_B5_R2, reac_B5_mean,
    reac_L_R1, reac_L_R2, reac_L_mean,
    reac_B1_R1, reac_B1_R2, reac_B1_mean,
    reac_sum_B5_bpy, reac_dB5_bpy, reac_bpy_dihedral,
    reac_bend_R1, reac_bend_R2, reac_bend_mean,
    reac_ni_c_mean, reac_coordplane_rms, reac_n_ni_c_sum, reac_n_ni_c_std,
    reac_ni_alkyne_centroid, reac_ni_bpyplane_dist,
    reac_dB5_alkyne, reac_dL_alkyne, reac_dvol_alkyne, reac_dvbur_substituent,
    reac_dni_c_signed, reac_dbend_alkyne, reac_dccr_angle, reac_dnicr_angle,
    reac_slippage, reac_ni_firstatom_R1, reac_ni_firstatom_R2,
    reac_dni_firstatom, reac_bulky_orientation, reac_alkyne_bpy_dihedral,
    reac_sigma_ortho_pyA, reac_sigma_meta_pyA, reac_sigma_para_pyA,
    reac_sigma_ortho_pyB, reac_sigma_meta_pyB, reac_sigma_para_pyB,
    reac_B5_ortho_pyA, reac_B5_meta_pyA, reac_B5_para_pyA,
    reac_B5_ortho_pyB, reac_B5_meta_pyB, reac_B5_para_pyB,
]
