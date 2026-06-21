"""Frozen data contracts shared across the kit.

Verbatim copy of ``src/contracts.py``.  ``Geom`` is the parsed molecule (elements
+ coordinates + covalent graph + Ni index); ``Reactant`` / ``Product`` are the
identified, atom-mapped views the descriptor functions consume.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Geom:
    elements: list[str]          # length N, e.g. "C","H","Ni"
    coords: np.ndarray           # (N,3) float
    adj: list[set[int]]          # organic covalent graph; Ni present as node but with NO edges
    ni: int                      # index of the (unique) Ni atom
    # dist(i,j) helper lives in geometry.py, not here

@dataclass
class Reactant:
    geom: Geom
    n_donors: tuple[int, int]    # the two bpy N indices bonded to Ni (unordered)
    bpy_atoms: frozenset[int]    # full bpy fragment incl substituents + H
    bpy_ring_atoms: frozenset[int]   # the 12 ring atoms (10 C + 2 N)
    c1: int                      # alkyne carbon with LOWER-priority substituent (CIP, inverted)
    c2: int                      # alkyne carbon with HIGHER-priority substituent
    r1_root: int                 # substituent attachment atom on c1
    r2_root: int                 # substituent attachment atom on c2
    r1_atoms: frozenset[int]     # R1 fragment (excludes c1, c2, Ni); incl H
    r2_atoms: frozenset[int]     # R2 fragment
    cip_source: str              # "cip" or "symmetric_atom_order" (pure CIP; no override)
    stereo_type: str = ""        # "S" / "O" / "" from the reaction row
    py_a_ring: frozenset[int] | None = None
    py_b_ring: frozenset[int] | None = None

@dataclass
class Product:
    geom: Geom
    n_donors: tuple[int, int]
    bpy_atoms: frozenset[int]
    bpy_ring_atoms: frozenset[int]
    ccarb: int                   # carboxylate carbon (2 O neighbours)
    o1: int                      # Ni-bound O (nearest O to Ni)
    o2: int                      # dangling O
    c_alpha: int                 # alkene C bonded to ccarb
    c_beta: int                  # alkene C bonded to Ni (nearest C to Ni)
    r_alpha_atoms: frozenset[int]
    r_beta_atoms: frozenset[int]
    metallacycle: tuple[int,int,int,int,int]  # (ni, o1, ccarb, c_alpha, c_beta)
