# descriptor_kit

Self-contained computation of the random-forest descriptors. **Input:** one
reaction's geometries — a reactant + product pair (i.e. one `reactions.parquet`
row), as two standard xyz blocks. **Output:** a flat `dict` of that pair's
descriptors.

Every output key is produced by **exactly one function**, named identically to
the key it returns (`reac_B5_R1`, `prod_tau4`, `tdelta_ni_Cb`, …). 102 functions
total: 54 `reac_*` + 25 `prod_*` + 23 `tdelta_*`.

## Usage

```python
from descriptor_kit import compute_descriptors, compute_tdelta

# Single reaction (reactant xyz + product xyz) -> 79 reac_*/prod_* descriptors.
row = compute_descriptors(reactant_xyz, product_xyz, stereo_type="S")
#   {"reac_sum_sigma_bpy": -1.66, ..., "prod_metallacycle_vbur": 34.63}

# Regioisomer Δ: pass the two compute_descriptors results for the Type_I and
# Type_II rows of one (ligand_pair, stereo_type) -> 23 tdelta_* descriptors.
deltas = compute_tdelta(row_type_I, row_type_II)
```

The directory that *contains* the `descriptor_kit/` folder must be importable —
run from there, or add it to `sys.path`. The bundled demo does this for you and
runs fully offline from committed example geometries (no data files needed):

```bash
pip install -r descriptor_kit/requirements.txt        # one-time
python descriptor_kit/example/run_example.py          # or: python -m descriptor_kit.example.run_example
```

### Failure policy
`compute_descriptors(..., strict=False)` (default) mirrors the production
pipeline: a descriptor whose preconditions fail becomes `NaN` and the rest are
unaffected. Pass `strict=True` to let the underlying exception propagate, or
`diagnostics=[]` to collect `(key, reason)` tuples for every failure.

## Layout

```
descriptor_kit/
├── requirements.txt       # pip install -r requirements.txt
├── api.py                 # compute_descriptors, compute_tdelta + orchestration
├── core/                  # frozen primitives (copied from src/, lightly adapted)
│   ├── constants.py       #   radii / tolerances
│   ├── contracts.py       #   Geom, Reactant, Product dataclasses
│   ├── geometry.py        #   parse_xyz, build_geom, dist/angle/dihedral/plane/CP/bfs
│   ├── topology.py        #   identify_reactant / identify_product, ring helpers
│   ├── cip.py             #   pure-CIP alkyne C1/C2 labeling
│   ├── hammett.py         #   fragment_smiles, sigma_for_fragment
│   ├── sigma_data.py      #   curated Hammett/Taft sigma table
│   └── steric.py          #   Sterimol / %V_bur / Bondi vdW volume (morfeus)
├── descriptors/
│   ├── reactant.py        # 54 reac_* functions (take a Reactant)
│   ├── product.py         # 25 prod_* functions (take a Product)
│   └── pair.py            # 23 tdelta_* functions (take two compute_descriptors results)
└── example/
    ├── run_example.py     # offline demo
    └── *.xyz              # bundled Type_I / Type_II reactant+product geometries
```

The pipeline is `parse_xyz → build_geom (covalent graph) → identify_reactant /
identify_product → descriptor functions`. Within a descriptor module, shared math
(Sterimol of R1/R2, bpy-substituent enumeration, σ-sums, the `_check`
precondition guard) lives in private `_helpers` — those are not descriptors and
are recomputed per call, keeping each descriptor function self-contained.

## Dependencies

Python ≥ 3.10, plus `numpy`, `scipy`, `rdkit` (CIP perception / SMILES), and
`morfeus-ml` (Sterimol, BuriedVolume — note the import name is `morfeus`). See
`requirements.txt`; tested with numpy 2.4.4 / scipy 1.17.1 / rdkit 2026.03.1 /
morfeus-ml 0.8.0 on Python 3.13. 

Verified standalone: copied to an empty directory with only these packages
installed (no repo present), `import descriptor_kit` and both entry points run
and produce the full descriptor set.
