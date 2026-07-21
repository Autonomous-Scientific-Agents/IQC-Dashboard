#!/usr/bin/env python3

"""check_dist_mat.py

Read .xyz files from a directory, compute pairwise distances, run checks,
and copy passing/failed files into numbered destinations when names collide.

Ni-neighbour expectation is chosen from filename:
- 'reactant' -> expect 2 N, 2 C
- 'product'  -> expect 2 N, 1 C, 1 O
"""

from pathlib import Path
import argparse
import sys
import shutil
from typing import List, Tuple

import numpy as np


COVALENT_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 1.0,
    'F': 0.57,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Br': 1.20,
    'I': 1.39,
    'Ni': 1.24,
}

FALLBACK_RADIUS = 0.75

# Optional per-element maximum allowed bond distances (Angstrom).
# Keys are lowercase element symbols sorted in a tuple string form: ('el1','el2').
# This prevents counting chemically-unreasonable long distances as bonds
# for specific element pairs (example: Ni-H should be fairly short to be
# considered a bonded neighbour).
PER_ELEMENT_MAX_BOND = {
    ('h', 'ni'): 3.0,  # Ni-H should be quite close to be considered bonded
}

def max_allowed_bond(el1: str, el2: str, default_max: float) -> float:
    """Return the maximum allowed bond distance for the element pair.

    Looks up a per-pair override in PER_ELEMENT_MAX_BOND and falls back to
    the provided default_max (or very large if default_max is None).
    """
    if default_max is None:
        default_max = float('inf')
    try:
        k = tuple(sorted([str(el1).lower(), str(el2).lower()]))
        return PER_ELEMENT_MAX_BOND.get(k, default_max)
    except Exception:
        return default_max


def parse_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], np.empty((0, 3))
    start = 0
    try:
        int(lines[0].split()[0])
        start = 2
    except Exception:
        start = 0
    elems = []
    coords = []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        elems.append(parts[0])
        coords.append(tuple(map(float, parts[1:4])))
    return elems, np.asarray(coords, dtype=float)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def covalent_radius(el: str) -> float:
    return COVALENT_RADII.get(el, FALLBACK_RADIUS)


def find_too_close(elems: List[str], coords: np.ndarray, factor: float, hard_min: float):
    d = distance_matrix(coords)
    n = len(elems)
    bad = []
    for i in range(n):
        for j in range(i + 1, n):
            thresh = max(hard_min, factor * (covalent_radius(elems[i]) + covalent_radius(elems[j])))
            if d[i, j] < thresh:
                bad.append((i, j, elems[i], elems[j], float(d[i, j]), float(thresh)))
    return bad


def check_ni_neighbors(elems: List[str], coords: np.ndarray, neighbours: int, factor: float = 1.0, hard_min: float = 0.3, max_bond_distance: float = None):
    """Find up to `neighbours` atoms closest to each Ni that are within a
    reasonable bonding distance.

    The distance threshold for considering a neighbor is:
        max(hard_min, factor * (r_cov(Ni) + r_cov(other)))

    Only atoms with d <= threshold are counted/returned. This prevents very
    distant atoms from being treated as nearest neighbours just because the
    system is sparse.
    """
    d = distance_matrix(coords)
    results = []
    for i, el in enumerate(elems):
        if el.lower() == 'ni':
            # sort other atoms by distance to this Ni
            idxs = np.argsort(d[i])
            idxs = [int(k) for k in idxs if k != i]

            neigh = []
            for k in idxs:
                # stop once we've collected the requested number of close neighbours
                if len(neigh) >= neighbours:
                    break
                other_el = elems[k]
                tmp_factor = factor
                if 'h' in other_el.lower():
                    tmp_factor = 0.1
                thresh = max(hard_min, tmp_factor * 2 * (covalent_radius(el) + covalent_radius(other_el)))
                # apply per-pair maximum allowed bond distance (if provided)
                pair_max = max_allowed_bond(el, other_el, max_bond_distance if max_bond_distance is not None else thresh)
                dist = float(d[i, k])
                if dist <= thresh and dist <= pair_max:
                    neigh.append((k, other_el, dist))
            cnt_ni = sum(1 for _, e, _ in neigh if e.lower() == 'ni')
            cnt_n = sum(1 for _, e, _ in neigh if e.lower() == 'n')
            cnt_c = sum(1 for _, e, _ in neigh if e.lower() == 'c')
            cnt_o = sum(1 for _, e, _ in neigh if e.lower() == 'o')
            results.append((i, neigh, cnt_ni, cnt_n, cnt_c, cnt_o))
    return results


def copy_with_suffix(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    base = src.stem
    suffix = src.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{base}_{i}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1


def extract_label(name: str) -> str:
    """Return a normalized label for a filename.

    The label is the stem with a single trailing numeric suffix removed
    (e.g. 'A_1' -> 'A', 'bipy-..._2' -> 'bipy-...'). If no trailing
    numeric suffix is present the full stem is returned.
    """
    stem = Path(name).stem
    parts = stem.split('_')
    if len(parts) == 1:
        return stem
    # drop a trailing numeric suffix if present (e.g. A_B_1 -> A_B)
    if parts[-1].isdigit():
        parts = parts[:-1]
        if not parts:
            return stem
    # if we have at least two components return the first two (A_B_... -> A_B)
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    # fallback to the single remaining part
    return parts[0]


def expected_counts_from_name(name: str):
    name = name.lower()
    if 'product' in name:
        return {'Ni': 2, 'N': 2, 'C': 1, 'O': 1}
    # default / reactant
    if 'reactant' in name:
        return {'Ni': 2, 'N': 2, 'C': 2, 'O': 0}
    return {'Ni': 2, 'N': 2, 'C': 2, 'O': 0}


def main():
    p = argparse.ArgumentParser(description='Check .xyz distance matrices')
    p.add_argument('--dir', '-d', default='geometries', help='input directory with .xyz files')
    p.add_argument('--outdir', '-o', default='passed_geos', help='directory to copy passing files')
    p.add_argument('--faildir', '-f', default='failed_geos', help='directory to copy failing files')
    p.add_argument('--factor', type=float, default=0.6, help='scaling factor for covalent-sum threshold')
    p.add_argument('--min', type=float, default=0.6, help='hard minimum threshold (Angstrom)')
    p.add_argument('--neigh', type=int, default=4, help='number of closest neighbours to inspect for Ni')
    args = p.parse_args()

    base = Path(args.dir)
    outdir = Path(args.outdir)
    faildir = Path(args.faildir)
    outdir.mkdir(parents=True, exist_ok=True)
    faildir.mkdir(parents=True, exist_ok=True)

    if not base.is_dir():
        print(f"Directory {base} not found")
        sys.exit(1)
        sys.exit(1)

    files = sorted(base.glob('*.xyz'))
    if not files:
        print(f'No .xyz files in {base}')
        return
    print(f'Found {len(files)} .xyz files in {base}')

    total = 0
    passed = 0
    failed_any = 0
    fail_parse = 0
    fail_too_close = 0
    fail_ni_pattern = 0

    for f in files:
        total += 1
        elems, coords = parse_xyz(f)
        # print(f"\nFile: {f.name}  (atoms: {len(elems)})")
        reasons = set()

        if len(elems) == 0 or coords.size == 0:
            print('  parse/unreadable')
            reasons.add('parse')

        if len(elems) > 1:
            too_close = find_too_close(elems, coords, args.factor*.7, args.min)
            if too_close:
                # print('  Too-close atom pairs found')
                reasons.add('too_close')
        else:
            print('  Insufficient atoms to check distances')

        ni_checks = check_ni_neighbors(elems, coords, args.neigh, args.factor, args.min) if len(elems) else []
        if not ni_checks:
            print('  No Ni atoms')
        else:
            expected = expected_counts_from_name(f.name)
            for i, neigh, _, cnt_n, cnt_c, cnt_o in ni_checks:
                exp_N = expected.get('N', 2)
                exp_C = expected.get('C', 2)
                exp_O = expected.get('O', 0)
                ok = (cnt_n == exp_N and cnt_c == exp_C and cnt_o == exp_O)
                # print(f'  Ni idx {i}: Ni_neigh={cnt_ni}, N_neigh={cnt_n}, C_neigh={cnt_c}, O_neigh={cnt_o} (expected Ni={exp_Ni}, N={exp_N}, C={exp_C}, O={exp_O})')
                if not ok:
                    # print(f.name)
                    # print(f'  Ni idx {i}: N_neigh={cnt_n}, C_neigh={cnt_c}, O_neigh={cnt_o} (expected N={exp_N}, C={exp_C}, O={exp_O})')
                    reasons.add('ni_pattern')

        if reasons:
            failed_any += 1
            if 'parse' in reasons:
                fail_parse += 1
            if 'too_close' in reasons:
                fail_too_close += 1
            if 'ni_pattern' in reasons:
                fail_ni_pattern += 1
            # If a file with the same label already exists in tested_geos,
            # don't copy this failing file into the faildir.
            tested_dir = Path('temp_pass')
            try:
                skip_copy = False
                if tested_dir.is_dir():
                    my_label = extract_label(f.name)
                    for t in tested_dir.glob('*.xyz'):
                        if extract_label(t.name) == my_label:
                            skip_copy = True
                            break
                if skip_copy:
                    print(f'  -> skipping copy to {faildir}: matching label in {tested_dir}')
                    pass
                else:
                    dest = copy_with_suffix(f, faildir)
            except Exception as e:
                print('  -> copy failed:', e)
        else:
            passed += 1
            try:
                dest = copy_with_suffix(f, outdir)
            except Exception as e:
                print('  -> copy failed:', e)

    print('\nSummary:')
    print('  total:', total)
    print('  passed:', passed)
    print('  failed_any:', failed_any)
    print('  fail_parse:', fail_parse)
    print('  fail_too_close:', fail_too_close)
    print('  fail_ni_pattern:', fail_ni_pattern)


if __name__ == '__main__':
    main()


