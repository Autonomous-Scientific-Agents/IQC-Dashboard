"""Product descriptors — one public function per output key.

Every public function is named EXACTLY after the descriptor it produces and
returns ``{that_key: float}``.  The shared precondition guard ``_check`` (not a
descriptor) validates the metallacycle topology; each function calls it.

Each function raises on a precondition violation; the orchestrator converts a
raise into NaN (unless strict=True).  Provenance (original D-number) noted per
function.
"""
from __future__ import annotations
import itertools

import numpy as np

from ..core import geometry as g
from ..core import topology
from ..core import steric as st


def _check(product):
    """Shared precondition guard (NOT a descriptor): the metallacycle atoms
    exist, are distinct, and match the documented topology.  Returns the unpacked
    anchor atoms ``(geom, ni, o1, o2, ccarb, ca, cb, nA, nB)``."""
    geom = product.geom
    ni = geom.ni
    assert geom.elements[ni] == "Ni", "product.geom.ni must point at Ni"
    o1, o2, ccarb = product.o1, product.o2, product.ccarb
    ca, cb = product.c_alpha, product.c_beta
    nA, nB = product.n_donors
    key = {ni, o1, o2, ccarb, ca, cb, nA, nB}
    assert len(key) == 8, f"product anchor atoms not distinct: {key}"
    assert geom.elements[o1] == "O" and geom.elements[o2] == "O", "o1/o2 must be O"
    assert geom.elements[ccarb] == "C", "ccarb must be C"
    assert geom.elements[ca] == "C" and geom.elements[cb] == "C", "Cα/Cβ must be C"
    assert geom.elements[nA] == "N" and geom.elements[nB] == "N", "donors must be N"
    assert product.metallacycle == (ni, o1, ccarb, ca, cb), (
        "metallacycle order must be (ni, o1, ccarb, c_alpha, c_beta)")
    return geom, ni, o1, o2, ccarb, ca, cb, nA, nB


# --- §8.6 Ni coordination (product) -----------------------------------------
def prod_ni_o1(product):
    """D24: |Ni−O1|."""
    geom, ni, o1, *_ = _check(product)
    return {"prod_ni_o1": g.dist(geom.coords, ni, o1)}


def prod_ni_Cb(product):
    """D25: |Ni−Cβ|."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_ni_Cb": g.dist(geom.coords, ni, cb)}


def prod_o_ni_bite_Cb(product):
    """D26: ∠(O1, Ni, Cβ) (bite angle)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_o_ni_bite_Cb": g.angle(geom.coords, o1, ni, cb)}


def prod_n_ni_o_mean(product):
    """D27: mean(∠(N_A, Ni, O1), ∠(N_B, Ni, O1))."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_n_ni_o_mean": 0.5 * (g.angle(c, nA, ni, o1) + g.angle(c, nB, ni, o1))}


def prod_n_ni_c_mean(product):
    """D28: mean(∠(N_A, Ni, Cβ), ∠(N_B, Ni, Cβ))."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_n_ni_c_mean": 0.5 * (g.angle(c, nA, ni, cb) + g.angle(c, nB, ni, cb))}


def prod_ni_n_mean(product):
    """D29: mean(|Ni−N_A|, |Ni−N_B|)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_ni_n_mean": 0.5 * (g.dist(c, ni, nA) + g.dist(c, ni, nB))}


def prod_abs_dni_n(product):
    """D30: |‖Ni−N_A‖ − ‖Ni−N_B‖|."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_abs_dni_n": abs(g.dist(c, ni, nA) - g.dist(c, ni, nB))}


def prod_ni_bpyplane_dist(product):
    """D31: |distance of Ni from the bpy heavy-atom plane| (product)."""
    geom, ni, *_ = _check(product)
    ring = list(product.bpy_ring_atoms)
    assert len(ring) == 12, f"bpy ring atoms != 12 (got {len(ring)})"
    centroid, normal, _rms = g.best_fit_plane(geom.coords, ring)
    d = g.point_plane_distance(geom.coords[ni], centroid, normal)
    return {"prod_ni_bpyplane_dist": abs(d)}


def prod_coordplane_rms(product):
    """D32: RMS deviation of {Ni, N_A, N_B, O1, Cβ} from best-fit plane."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    _c, _n, rms = g.best_fit_plane(geom.coords, [ni, nA, nB, o1, cb])
    return {"prod_coordplane_rms": rms}


def prod_tau4(product):
    """D33: τ₄ = (360 − (α+β))/141 (α,β = two largest L-Ni-L angles among
    donors {N_A, N_B, O1, Cβ})."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    donors = [nA, nB, o1, cb]
    angles = [g.angle(c, a, ni, b) for a, b in itertools.combinations(donors, 2)]
    assert len(angles) == 6, f"expected 6 L-Ni-L angles, got {len(angles)}"
    angles.sort(reverse=True)
    alpha, beta = angles[0], angles[1]
    return {"prod_tau4": (360.0 - (alpha + beta)) / 141.0}


def prod_metallacycle_perimeter(product):
    """D34: |Ni−O1| + |O1−Ccarb| + |Ccarb−Cα| + |Cα−Cβ| + |Cβ−Ni|."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    perim = (g.dist(c, ni, o1) + g.dist(c, o1, ccarb) + g.dist(c, ccarb, ca)
             + g.dist(c, ca, cb) + g.dist(c, cb, ni))
    return {"prod_metallacycle_perimeter": perim}


def prod_cremer_pople_Q(product):
    """D35: Cremer-Pople amplitude Q of [Ni, O1, Ccarb, Cα, Cβ]."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_cremer_pople_Q": g.cremer_pople_Q(geom.coords,
                                                    list(product.metallacycle))}


def prod_dih_ni_o_c_Ca(product):
    """D36: signed dihedral Ni−O1−Ccarb−Cα."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_dih_ni_o_c_Ca": g.signed_dihedral(geom.coords, ni, o1, ccarb, ca)}


def prod_dih_o_c_ca_cb(product):
    """D36: signed dihedral O1−Ccarb−Cα−Cβ."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_dih_o_c_ca_cb": g.signed_dihedral(geom.coords, o1, ccarb, ca, cb)}


# --- §8.7 Carboxylate / CO2 (product) ----------------------------------------
def prod_newcc_Ca(product):
    """D37: |Ccarb−Cα| (the newly formed C-C bond)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_newcc_Ca": g.dist(geom.coords, ccarb, ca)}


def prod_cc_metallacycle_len(product):
    """D38: |Cα−Cβ| (former alkyne, now alkene, in the metallacycle)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_cc_metallacycle_len": g.dist(geom.coords, ca, cb)}


def prod_cc_alternation(product):
    """D39: |Ccarb−Cα| − |Cα−Cβ| (signed)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_cc_alternation": g.dist(c, ccarb, ca) - g.dist(c, ca, cb)}


def prod_o_c_o_angle(product):
    """D40: ∠(O1, Ccarb, O2)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_o_c_o_angle": g.angle(geom.coords, o1, ccarb, o2)}


def prod_co_asym(product):
    """D41: |Ccarb−O1| − |Ccarb−O2| (signed)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_co_asym": g.dist(c, ccarb, o1) - g.dist(c, ccarb, o2)}


def prod_co_len_mean(product):
    """D42: mean(|Ccarb−O1|, |Ccarb−O2|)."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    return {"prod_co_len_mean": 0.5 * (g.dist(c, ccarb, o1) + g.dist(c, ccarb, o2))}


def prod_carboxylate_tilt(product):
    """D43: out-of-plane distance of dangling O2 from the metallacycle plane."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    centroid, normal, _rms = g.best_fit_plane(c, [ni, o1, ccarb, ca, cb])
    return {"prod_carboxylate_tilt": abs(g.point_plane_distance(c[o2], centroid, normal))}


def prod_ni_ccarb(product):
    """D44: |Ni−Ccarb|."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    return {"prod_ni_ccarb": g.dist(geom.coords, ni, ccarb)}


def prod_ni_alkene_centroid(product):
    """D45: dist(Ni, midpoint(Cα, Cβ))."""
    geom, ni, o1, o2, ccarb, ca, cb, nA, nB = _check(product)
    c = geom.coords
    mid = 0.5 * (c[ca] + c[cb])
    return {"prod_ni_alkene_centroid": float(np.linalg.norm(c[ni] - mid))}


# --- §8.8 bpy (product) ------------------------------------------------------
def prod_bpy_dihedral(product):
    """D46: bpy inter-ring dihedral N_A−bridgeA−bridgeB−N_B (product geometry)."""
    _check(product)
    nA, bridgeA, bridgeB, nB = topology.bpy_dihedral_atoms(product)
    return {"prod_bpy_dihedral": g.signed_dihedral(product.geom.coords,
                                                   nA, bridgeA, bridgeB, nB)}


# --- §8.6 steric (product) ---------------------------------------------------
def prod_metallacycle_vbur(product):
    """D47: %V_bur around Ni of {o1,ccarb,o2,c_alpha,c_beta} ∪ Rα ∪ Rβ."""
    geom = product.geom
    include = ({product.o1, product.ccarb, product.o2,
                product.c_alpha, product.c_beta}
               | set(product.r_alpha_atoms) | set(product.r_beta_atoms))
    pct = st.percent_buried_volume(geom, geom.ni, include)
    assert 0.0 < pct < 100.0, f"D47 %Vbur out of range: {pct}"
    return {"prod_metallacycle_vbur": pct}


# Ordered registry of every product descriptor function (one per output key).
ALL = [
    prod_ni_o1, prod_ni_Cb, prod_o_ni_bite_Cb, prod_n_ni_o_mean,
    prod_n_ni_c_mean, prod_ni_n_mean, prod_abs_dni_n, prod_ni_bpyplane_dist,
    prod_coordplane_rms, prod_tau4, prod_metallacycle_perimeter,
    prod_cremer_pople_Q, prod_dih_ni_o_c_Ca, prod_dih_o_c_ca_cb,
    prod_newcc_Ca, prod_cc_metallacycle_len, prod_cc_alternation,
    prod_o_c_o_angle, prod_co_asym, prod_co_len_mean, prod_carboxylate_tilt,
    prod_ni_ccarb, prod_ni_alkene_centroid, prod_bpy_dihedral,
    prod_metallacycle_vbur,
]
