"""Geometry primitives: xyz parsing, covalent-graph construction, and the
distance/angle/dihedral/plane/ring helpers every descriptor is built on.

Faithful copy of ``src/geometry.py`` with package-relative imports.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from .constants import COVALENT_RADII, HEAVY_BOND_SCALE, MAX_DEGREE
from .contracts import Geom


def parse_xyz(block):
    L = block.splitlines()
    n = int(L[0].split()[0])
    els = []
    xs = []
    for line in L[2 : 2 + n]:
        p = line.split()
        els.append(p[0])
        xs.append([float(p[1]), float(p[2]), float(p[3])])
    return els, np.asarray(xs, float)


def dist(coords, i, j):
    return float(np.linalg.norm(coords[i] - coords[j]))


def build_geom(elements, coords):
    n = len(elements)
    D = cdist(coords, coords)
    adj = [set() for _ in range(n)]
    heavy = [k for k in range(n) if elements[k] not in ("H", "Ni")]
    for a in range(len(heavy)):
        for b in range(a + 1, len(heavy)):
            i, j = heavy[a], heavy[b]
            cutoff = HEAVY_BOND_SCALE * (
                COVALENT_RADII[elements[i]] + COVALENT_RADII[elements[j]]
            )
            if D[i, j] < cutoff:
                adj[i].add(j)
                adj[j].add(i)
    for h in range(n):                       # H monovalent: bond to nearest non-Ni heavy
        if elements[h] != "H":
            continue
        k = min(heavy, key=lambda k: D[h, k])
        adj[h].add(k)
        adj[k].add(h)
    ni = [k for k in range(n) if elements[k] == "Ni"]
    assert len(ni) == 1, f"expected exactly one Ni, got {len(ni)}"
    # overbonded sanity (heavy degree only; H always degree 1)
    for k in range(n):
        if elements[k] in ("Ni", "H"):
            continue
        if len(adj[k]) > MAX_DEGREE[elements[k]]:
            raise ValueError(f"overbonded {elements[k]}{k}: degree {len(adj[k])}")
    return Geom(elements=elements, coords=coords, adj=adj, ni=ni[0])


_GEOM_EPS = 1e-9


def angle(coords, a, b, c):
    u = coords[a] - coords[b]
    v = coords[c] - coords[b]
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < _GEOM_EPS or nv < _GEOM_EPS:
        # Coincident atoms — angle is undefined; surface as NaN rather than
        # silently returning 0/90/180 from a divide-by-zero.
        return float("nan")
    cosv = np.dot(u, v) / (nu * nv)
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))


def signed_dihedral(coords, a, b, c, d):
    p0, p1, p2, p3 = coords[a], coords[b], coords[c], coords[d]
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1 = np.linalg.norm(b1)
    if n1 < _GEOM_EPS:
        # Middle bond has zero length — dihedral is undefined.
        return float("nan")
    b1u = b1 / n1
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def best_fit_plane(coords, idxs):
    P = coords[list(idxs)]
    c = P.mean(0)
    _U, _S, Vt = np.linalg.svd(P - c)
    normal = Vt[-1]
    d = (P - c) @ normal
    return c, normal / np.linalg.norm(normal), float(np.sqrt(np.mean(d**2)))


def point_plane_distance(point, centroid, normal):
    return float(np.dot(point - centroid, normal / np.linalg.norm(normal)))


def cremer_pople_Q(coords, ring):
    P = coords[list(ring)]
    N = len(ring)
    geo = P.mean(0)
    R = P - geo
    j = np.arange(N)
    cosw = np.cos(2 * np.pi * j / N)
    sinw = np.sin(2 * np.pi * j / N)
    R1 = (R * sinw[:, None]).sum(0)
    R2 = (R * cosw[:, None]).sum(0)
    n = np.cross(R1, R2)
    n /= np.linalg.norm(n)
    z = R @ n
    return float(np.sqrt(np.sum(z**2)))


def fragment_bfs(adj, roots, stop):
    seen = set(roots)
    frontier = list(roots)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v in seen or v in stop:
                    continue
                seen.add(v)
                nxt.append(v)
        frontier = nxt
    return seen
