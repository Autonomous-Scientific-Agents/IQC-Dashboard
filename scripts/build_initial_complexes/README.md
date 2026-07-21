# Building scripts for initial complex geometries

These scripts are a pipeline for generating, validating, and creating reactant and product stereoisomers of nickel-bipyridine-alkyne complex geometries for CO2 insertion studies. This workflow uses machine learning interactive potentials (MACE-MP) with Architector for initial structure generation, followed by geometric validation and stereoisomer generation.

## Overview

This directory contains three main scripts that work sequentially:

1. **`mace_geoms.py`** - Generate initial 3D geometries from ligand SMILES and coordination information using Architector + MACE-MP
2. **`check_dist_mat.py`** - Validate geometries using distance matrices and coordination pattern checks
3. **`get_opposite_S_O_geo.py`** - Analyze geometry types and create mirrored stereoisomers

## Prerequisites

### Required Python Packages
- `architector` - Complex geometry builder
- `mace-torch` - MACE machine learning potential
- `ase` - Atomic simulation environment
- `numpy`
- `pathlib`
- `json`

### Required Input Files

**`ligand_sets.json`** - Dictionary defining ligand combinations for each complex
- Format: `{complex_id: {"reactant": [...], "productA": [...], "productB": [...]}}`
- Each ligand entry contains:
  - `smiles`: SMILES string for the ligand
  - `coordList`: List of atom indices that coordinate to metal

Example structure:
```json
{
  "bipy-aaaaaaaa_a-C2H2-a": {
    "reactant": [
      {"smiles": "[CH]#[CH]", "coordList": [0,1]},
      {"smiles": "C1=CC=NC(=C1)C=2C=CC=CN=2", "coordList": [3,11]}
    ],
    "productA": [...]
  }
}
```

---

## Script 1: mace_geoms.py

Generates initial 3D geometries for Ni complexes using Architector with MACE-MP as the energy calculator.

### Usage

```bash
python mace_geoms.py \
    --start-entry 0 \
    --num-entries 100 \
    --core-type tetrahedral \
    --mol-type reactant \
    --nprocs 120 \
    --ligands-file ./ligand_sets.json
```

### Arguments

- `--start-entry`: Starting entry index in ligand_sets.json (0-based) [required]
- `--num-entries`: Number of entries to process from start [required]
- `--core-type`: Initial metal coordination geometry, one of:
  - `tetrahedral` (default)
  - `square_planar`
  - `seesaw`
- `--mol-type`: Molecule type to generate, one of:
  - `reactant` (default) - Ni(0) with alkyne
  - `productA` or `productB` - CO2 inserted between Ni(II) and an alkenyl carbon
- `--nprocs`: Number of parallel workers (default: 120)
- `--ligands-file`: Path to ligand definitions JSON (default: `./ligand_sets.json`)
- `--output-dir`: Directory to save generated geometries (default: `./geometries`)

### How It Works

1. **Preloads MACE-MP model** in the parent process for efficient sharing across workers
2. **Distributes ligand entries** across multiple processes using a fixed random seed for reproducibility
3. For each complex:
   - Sets up Architector input dictionaries, including:
     - 'core', with Ni center
     - 'parameters', with MACE_MP calculator and oxidation state: 0 for reactants, +2 for products
     - 'ligands' with bipyridine and alkyne/carboxylated alkene ligand SMILES, lists specifying coordinating atom indices, and setting ligand type to bidentate_cis
   - Uses Architector to generate a structure
4. **Writes XYZ files** to the specified output directory (default: `geometries/`)
   - Format: `{complex_id}_{mol_type}.xyz`
   - Comment line includes structure key and energy

### Output

- XYZ files written to the directory specified by `--output-dir` (default: `geometries/`)
- Each file contains: atom count, energy comment, and Cartesian coordinates

### Performance

- Uses multiprocessing for parallelization (default: 120 workers)
- Automatically skips already-generated structures (resumable)
- MACE-MP model is preloaded once and shared via fork for efficiency

### Notes

- The script suppresses most output from Architector for cleaner logs
- Failed structures are silently skipped (reported only at process completion)
- Triple bonds in alkynes are converted to double bonds as a workaround for Architector
  being unable to coordinate to a bond and requiring coordination to both alkynyl carbons

---

## Script 2: check_dist_mat.py

Validates generated geometries by checking interatomic distances, covalent bonding patterns, and Ni coordination environments.

### Usage

```bash
python check_dist_mat.py \
    --dir geometries \
    --outdir passed_geos \
    --faildir failed_geos \
    --factor 0.6 \
    --min 0.6 \
    --neigh 4
```

### Arguments

- `--dir`, `-d`: Input directory with .xyz files (default: `geometries`)
- `--outdir`, `-o`: Output directory for passing geometries (default: `passed_geos`)
- `--faildir`, `-f`: Output directory for failing geometries (default: `failed_geos`)
- `--factor`: Scaling factor for covalent radii sum threshold (default: 0.6)
- `--min`: Hard minimum distance threshold in Angstroms (default: 0.6)
- `--neigh`: Number of nearest neighbors to check for Ni (default: 4)

### Validation Checks

1. **Atom-Pair Distance Check**
   - Flags atoms that are too close together
   - Threshold: `max(hard_min, factor × (r_cov(A) + r_cov(B)))`
   - Special per-element maximum bond distances (e.g., Ni-H < 3.0 Å)

2. **Ni Coordination Pattern Check**
   - Identifies N nearest neighbors to each Ni atom
   - For **reactants** expects: 2 N, 2 C, 0 O
   - For **products** expects: 2 N, 1 C, 1 O
   - Only counts atoms within reasonable bonding distance

3. **Filename-Based Expectations**
   - Automatically determines expected coordination from filename
   - Files with "reactant" → Ni(N)2(C)2 pattern
   - Files with "product" → Ni(N)2(C)(O) pattern

### How It Works

1. Reads all .xyz files from specified directory (e.g., `geometries/`)
2. For each structure:
   - Computes pairwise distance matrix
   - Checks for unreasonably close atoms
   - Validates Ni coordination environment
3. Copies geometries to appropriate output directory:
   - **Passed** → `passed_geos/` (default)
   - **Failed** → `failed_geos/` with collision handling

### Output

- **Passed geometries**: Copied to `--outdir` (default: `passed_geos/`)
- **Failed geometries**: Copied to `--faildir` (default: `failed_geos/`)
  - Files with same label in `temp_pass/` are skipped (avoids duplicates)
  - Numeric suffixes added automatically for name collisions

### Output Statistics

The script prints summary statistics:
```
Found N .xyz files in geometries
Total: X
Passed: Y
Failed: Z
  - Parse errors: A
  - Too-close atoms: B
  - Ni pattern mismatch: C
```

### Covalent Radii Reference

Built-in radii (Angstroms):
- H: 0.31, C: 0.76, N: 0.71, O: 1.00, Ni: 1.24
- F: 0.57, Cl: 1.02, Br: 1.20, I: 1.39
- P: 1.07, S: 1.05

---

## Script 3: get_opposite_S_O_geo.py

Analyzes coordination geometry (square planar vs. tetrahedral) and creates mirrored stereoisomers for asymmetric ligand combinations.

### Usage

```bash
python get_opposite_S_O_geo.py
```

**Note:** This script operates on .xyz files in the current working directory. Run it inside the directory containing validated geometries (e.g., `cd passed_geos && python ../get_opposite_S_O_geo.py`).

### What It Does

1. **Geometry Classification**
   - Identifies Ni atom and 2 closest N atoms (bipyridine) and 2 closest C atoms (alkyne)
   - Calculates angle between coordination planes:
     - Plane 1: Ni + 2 N atoms
     - Plane 2: Ni + 2 C atoms
   - Classifies as:
     - **Square planar**: angle < 30°
     - **Tetrahedral**: angle > 60°
     - **Intermediate**: 30° ≤ angle ≤ 60°
    (note the cutoffs were tightened to 25 and 75 for postprocessed geometries in corresponding paper)
2. **Bipyridine Ligand Identification**
   - Uses distance-based connectivity to identify all atoms in bipyridine
   - Automatically adjusts bond distance thresholds if validation fails
   - Handles various substituents (Cl, Br, F, I, OCH₃, NO₂, etc.)
   - Fixes hypervalent atoms by keeping only shortest bonds

3. **Ligand Asymmetry Detection**
   - Parses filename to determine if bipyridine or alkyne ligands are symmetric
   - Bipyridine: Checks if substituent pattern is palindromic (e.g., `bipy-maaaaaaa` is asymmetric)
   - Alkyne: Checks if left/right substituents match (e.g., `o-C2H2-f` is asymmetric)

4. **Stereoisomer Generation**
   - Creates mirrored geometry **only if**:
     - BOTH bipyridine AND alkyne are asymmetric
     - Geometry is NOT perpendicular (angle ≠ 90° ± 10°)
   - Mirror operation:
     - Defines mirror plane perpendicular to N₁-N₂ line, passing through Ni
     - Reflects only bipyridine atoms through this plane
     - Swaps N₁ ↔ N₂ positions, creating opposite stereochemistry (S ↔ O)

### Output Files

1. **Individual mirrored files**: `{basename}_mirrored.xyz`
2. **Concatenated files** (for easy viewing in Molden):
   - `all_originals.xyz` - All original geometries
   - `all_mirrored.xyz` - All mirrored geometries

3. **Flagged subdirectories** (for geometry mismatches):
   - `flagged_tetrahedral/` - Expected square planar but found tetrahedral
   - `flagged_intermediate/` - Expected planar/tetrahedral but found intermediate
   - `validation_warnings/` - Potential ligand assignment issues

### Output Summary

- Total files analyzed
- Geometry mismatches found
- Validation warnings (after auto-correction)
- Auto-corrected files
- Mirrored structures created

### Alkyne Functional Group Codes

The script recognizes these substituent codes in filenames:
- `a`: H
- `b`: NO₂
- `c`: Cl
- `d`: OCH₃
- `e`: N(CH₃)₂
- `f`: CH₃
- `g`: CH₂CH₃
- `h`: C(O)OH
- `i`: OCOCH₃
- `j`: F
- `k`: CF₃
- `l`: C(CH₃)₃
- `m`: OH
- `n`: NH₂
- `o`: phenyl
- `p`: -CH₂-phenyl-OCH₃
- `q`: -CH₂-phenyl-OCF₃

---

## Complete Workflow Example

```bash
# Step 1: Generate initial geometries (reactants, tetrahedral core)
python mace_geoms.py \
    --start-entry 0 \
    --num-entries 500 \
    --core-type tetrahedral \
    --mol-type reactant \
    --nprocs 32 \
    --output-dir geos_tetrahedral_reactant

# Step 2: Validate generated geometries
python check_dist_mat.py \
    --dir geos_tetrahedral_reactant \
    --outdir passed_geos_tetrahedral_reactant \
    --faildir failed_geos_tetrahedral_reactant \
    --factor 0.6 \
    --min 0.6

# Step 3: Generate stereoisomers
cd passed_geos_tetrahedral_reactant
python ../get_opposite_S_O_geo.py

# Step 4: Repeat for square planar core
cd ..
python mace_geoms.py \
    --start-entry 0 \
    --num-entries 500 \
    --core-type square_planar \
    --mol-type reactant \
    --nprocs 32 \
    --output-dir geos_square_planar_reactant

python check_dist_mat.py \
    --dir geos_square_planar_reactant \
    --outdir passed_geos_square_planar_reactant \
    --faildir failed_geos_square_planar_reactant

cd passed_geos_square_planar_reactant
python ../get_opposite_S_O_geo.py

# Step 5: Process products (Type I and Type II)
cd ..
python mace_geoms.py \
    --start-entry 0 \
    --num-entries 500 \
    --mol-type productA \
    --core-type tetrahedral \
    --nprocs 32 \
    --output-dir geos_tetrahedral_productA

python check_dist_mat.py \
    --dir geos_tetrahedral_productA \
    --outdir passed_geos_tetrahedral_productA \
    --faildir failed_geos_tetrahedral_productA
    
cd passed_geos_tetrahedral_productA
python ../get_opposite_S_O_geo.py
```

## Directory Structure After Workflow

```
build_initial_complexes/
├── ligand_sets.json
├── mace_geoms.py
├── check_dist_mat.py
├── get_opposite_S_O_geo.py
├── geos_tetrahedral_reactant/     (raw MACE output)
├── geos_square_planar_reactant/   (raw MACE output)
├── geos_tetrahedral_productA/     (raw MACE output)
├── passed_geos_tetrahedral_reactant/
│   ├── *.xyz                      (validated structures)
│   ├── *_mirrored.xyz             (stereoisomers)
│   ├── all_originals.xyz
│   ├── all_mirrored.xyz
│   ├── flagged_tetrahedral/
│   ├── flagged_intermediate/
│   └── validation_warnings/
├── failed_geos_tetrahedral_reactant/
├── passed_geos_square_planar_reactant/
│   └── ...
├── passed_geos_tetrahedral_productA/
│   └── ...
└── README.md
```

---

## Troubleshooting

### mace_geoms.py Issues

1. **MACE model fails to load**
   - Ensure `mace-torch` is installed: `pip install mace-torch`
   - Model will auto-download on first run (~1 GB)

2. **Architector errors**
   - Some ligand combinations may fail due to steric clashes
   - Failed structures are skipped automatically

3. **Memory issues with high `--nprocs`**
   - Reduce number of workers

### check_dist_mat.py Issues

1. **Too many failures**
   - Try adjusting `--factor` (default 0.6, try 0.5-0.7)
   - Check `failed_geos/` files manually to diagnose issues

2. **Wrong Ni coordination pattern**
   - Verify filename contains "reactant" or "product"
   - Check that MACE optimization converged properly

### get_opposite_S_O_geo.py Issues

1. **Validation warnings about alkyne atoms**
   - Script will attempt auto-correction with different thresholds
   - Files with persistent warnings moved to `validation_warnings/`
   - May indicate unusual bonding pattern or failed MACE optimization

2. **No mirrored geometries created**
   - Check that ligands are actually asymmetric (parse filename)
   (if script is adapted for new ligand sets with different naming conventions
   user may need to update logic for asymmetry check)
   - If angle ≈ 90°, mirror will not be substantially different (e.g., energetically)

3. **Geometry type mismatch**
   - Expected type is inferred from directory name
   - Files flagged and moved to `flagged_*` subdirectories
   - May indicate need for different `--core-type` in Step 1

---

## Authors & Citation

This pipeline is part of the supporting information for:

Elliott, S. N.; Reji, R.; Chen, Q.; Zheng, X.; Riaz, S.; Yan, X.; Pham, T. D.; Huerta, E. A.; Glusac, K. D.; Keçeli, M. Data-Driven Discovery of Structure–Reactivity Relationships in Oxidative Addition at Zerovalent Nickel Centers. *Manuscript in preparation*, 2026.

Please cite this work when using this code.
