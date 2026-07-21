# process_opt_geometries.py

Pipeline for processing, validating, and classifying optimized geometries of nickel complexes (each
with a pair of a bipyridinyl and alkynyl ligand), including: 
    (1) S or O stereochemistry (which assigns how the two ligands are oriented with respect to one another)
    (2) Type-I or Type-II reaction type (which describes at which side of the alkyne CO2 inserties)
    (3) Overall configuration (e.g., square planar, tetrahedral, or intermediate),
and calculating reaction energies for CO2 insertion reactions to product an output reaction energy parquet.

## Overview

This script performs three main tasks:

1. **Filter optimized geometries** - Validates that computed geometries are optimized, through convergence and frequencies, and that they match expected SMILES structures for reactants and products
2. **Classify geometries** - Determines molecular configuration (square planar, tetrahedral, or intermediate), stereochemistry (S or O), and insertion type (Type I or Type II)
3. **Calculate reaction energies** - Pairs corresponding reactant and product geometries and computes Gibbs free energies of reaction

## Prerequisites

### Required Python Packages
- `pandas`
- `numpy`
- `scipy`
- `rdkit`
- `automol`

### Required Input Files

The script expects the following files in the working directory:

1. **`iqc_thermo_results_20260404_095146_reduced.parquet`** - IQC (Inverse Quantum Chemistry) output containing optimized geometries and thermochemistry
   - Must contain columns: `unique_name`, `opt_xyz`, `number_of_imaginary`, `G_eV`

2. **`full_large_smiles_dct.json`** - Dictionary of expected SMILES structures
   - Format: `{pair_id: {"reactant_smiles": str, "product1_smiles": str, "product2_smiles": str}}`
   - Used to validate that optimized geometries match intended molecular structures

## How to Run

### Basic Usage

```bash
python process_opt_geometries.py
```

The script runs with default parameters:
- Input file: `iqc_thermo_results_20260404_095146_reduced.parquet`
- Parallel workers: 10

### Command-Line Options

You can customize the input file and number of parallel workers:

```bash
python process_opt_geometries.py --input-file <path_to_parquet> --max-workers <N>
```

**Arguments:**
- `--input-file`: Path to input IQC parquet file (default: `iqc_thermo_results_20260404_095146_reduced.parquet`)
- `--max-workers`: Number of parallel workers for geometry classification (default: 10)
  - Recommended: Set according to available CPU cores (e.g., use 4-8 workers on a typical laptop, or 16-32 on a server)
  - Higher values speed up processing but increase memory usage

**Examples:**

```bash
# Use custom input file with default workers
python process_opt_geometries.py --input-file my_results.parquet

# Use 20 parallel workers with default input file
python process_opt_geometries.py --max-workers 20

# Specify both options
python process_opt_geometries.py --input-file my_results.parquet --max-workers 16

# View all available options
python process_opt_geometries.py --help
```

### Resumable Processing

The script is designed to be resumable. It saves progress periodically (every 500 structures) and skips already-processed geometries on subsequent runs. This is useful for large datasets or if processing is interrupted.

To resume from a previous run, simply re-run the script in the same directory with the output parquet files present.

## Output Files

The script generates the following output files:

### 1. `matched_geometries.parquet`
- Contains all geometries that successfully match expected SMILES structures
- Includes original data plus a `match_type` column:
  - `0`: Matches reactant SMILES
  - `1`: Matches product1 SMILES
  - `2`: Matches product2 SMILES

### 2. `labeled_geometries.parquet`
- Contains geometries that passed validation with classification labels
- Additional columns:
  - **`geometry_type`**: `square_planar`, `tetrahedral`, or `intermediate`
    - Classification based on dihedral angle between coordination planes:
      - Square planar: dihedral < 25°
      - Tetrahedral: dihedral > 75°
      - Intermediate: 25° ≤ dihedral ≤ 75°
  - **`stereo_label`**: `S`, `O`, or empty string
    - S (same): High-priority groups on ligands have same clockwise orientation
    - O (opposite): High-priority groups on ligands have opposite clockwise orientation
    - Empty for symmetric ligands with no stereochemistry
  - **`insertion_type`**: `Type_I`, `Type_II`, or empty string
    - Type I: CO₂ inserted at higher-priority carbon (CIP rules)
    - Type II: CO₂ inserted at lower-priority carbon
    - Empty for reactants
  - **`ligand_pair`**: Ligand identifier extracted from `unique_name`
  - **`mol_category`**: `reactant` or `product`
  - **`max_angle`**: Largest bond angle around Ni (indicates trans geometry if ~180°)
  - **`dihedral_angle`**: Dihedral between coordination planes (used for geometry classification)

### 3. `bad_geometries.parquet`
- Contains geometries that failed validation checks
- Additional columns:
  - **`ni_coordination_flag`**: Boolean indicating improper Ni coordination
  - **`alkyne_valence_flag`**: Boolean indicating improper alkyne/product valence
  - **`bad_flag`**: Description of validation failure

### 4. `reaction_data.json`
- Reaction-level data pairing reactants with products
- Contains:
  - `ligand_pair`: Identifier for the ligand system
  - `stereo_type`: `S`, `O`, or empty
  - `insertion_type`: `Type_I` or `Type_II`
  - `reaction_gibbs_kcal`: ΔG°ᵣₓₙ in kcal/mol (includes CO₂ reference energy and PBE correction)
  - `reactant_gibbs`: Reactant G in eV
  - `product_gibbs`: Product G in eV
  - `reactant_geometry`: Reactant XYZ structure
  - `product_geometry`: Product XYZ structure
  - `reactant_configuration`: Reactant geometry type
  - `product_configuration`: Product geometry type

## Methodology Details

### Geometry Classification Algorithm

1. **Nickel Coordination Analysis**
   - Identifies Ni atom and its 4 nearest non-hydrogen neighbors
   - Validates proper coordination sphere (no extra close atoms)
   - Identifies 2 nitrogen atoms (from bipyridine ligand) and 2 alkyne/product atoms

2. **Stereochemistry Determination** (using CIP-like priority rules)
   - Compares priority of substituents on bipyridine nitrogens
   - Compares priority of alkyne carbons (or O vs C for products)
   - Determines spatial relationship between high-priority groups:
     - If high-priority N and high-priority C/O are closer: **O** (opposite)
     - If high-priority N and high-priority C/O are farther: **S** (same)
   - For Type II insertions, labels are reversed

3. **Insertion Type Classification** (products only)
   - Traces connectivity from inserted CO₂ oxygen to find insertion carbon
   - Compares priority of insertion carbon vs. the other alkyne carbon
   - **Type I**: Insertion at higher-priority carbon
   - **Type II**: Insertion at lower-priority carbon

4. **Geometry Type Classification**
   - Calculates dihedral angle between two coordination planes:
     - Plane 1: Ni + 2 nitrogen atoms
     - Plane 2: Ni + 2 alkyne/product atoms
   - Classifies as square planar, tetrahedral, or intermediate based on angle

### Reaction Energy Calculation

The script calculates ΔG_rxn using:

```
ΔG_rxn_ = (G_product - G_reactant - G_CO2) × 23.061 kcal/eV + correction
```

Where:
- **G_CO2 = -23.199012350 eV** (MACE-MP-0a-large with dispersion)
- **correction = -20.35 kcal/mol** (PBE-determined benchmark correction)

For each ligand pair, the script finds:
- Lowest-energy O-stereoisomer reactant and products (Type I and Type II)
- Lowest-energy S-stereoisomer reactant and products (Type I and Type II)
- If symmetric in the bipyridine (i.e. no stereoisomerism) lowest energy reactant and product (Type I and Type II)
- If symmetric in the alkyne (i.e. no stereoisomerism or difference
in Type I and Type II) overall lowest energy reactant and product  
- Calculates reaction energies for all valid reactant-product pairs

## Progress Monitoring

The script prints progress updates including:
- Number of already-labeled and already-flagged structures
- Number of pending structures to classify
- Progress every 500 structures
- Summary of matched geometries by match type
- Final counts of labeled and bad geometries

Example output:
```
Already labeled: 2500
Already flagged bad: 150
Pending to classify: 1000
Processed 500/1000 pending structures...
Total labeled rows saved: 3000
Total bad rows saved: 200
```

## Validation Checks

The script performs the following validation checks:

1. **SMILES Structure Matching**: Optimized geometry must match expected connectivity
2. **Imaginary Frequencies**: Structures with imaginary frequencies are excluded
3. **Ni Coordination**: Validates no extra atoms are coordinating with Ni
4. **Alkyne Valence**: Validates proper bonding patterns for alkyne/product atoms

Structures failing validation are saved to `bad_geometries.parquet` with diagnostic information.

## Parallelization

The script uses `ThreadPoolExecutor` for parallel classification of geometries. The number of workers can be set using the `--max-workers` command-line argument (default: 10).

**Recommendations:**
- Laptop/Desktop (4-8 cores): Use 4-8 workers
- Workstation (16+ cores): Use 12-20 workers
- HPC node (32+ cores): Use 20-32 workers

Higher worker counts increase speed but also memory usage. If you encounter memory issues, reduce the number of workers.

## Troubleshooting

### Common Issues

1. **Missing input files**: Ensure your input parquet file (or the default `iqc_thermo_results_20260404_095146_reduced.parquet`) and `full_large_smiles_dct.json` are in the working directory

2. **RDKit connectivity errors**: Some geometries may fail RDKit's automatic bond determination. These are caught and flagged in `bad_geometries.parquet`

3. **Memory issues**: For very large datasets, consider reducing `max_workers` or processing in batches



## Authors & Citation

This script is part of the supporting information for:

Elliott, S. N.; Reji, R.; Chen, Q.; Zheng, X.; Riaz, S.; Yan, X.; Pham, T. D.; Huerta, E. A.; Glusac, K. D.; Keçeli, M. Data-Driven Discovery of Structure–Reactivity Relationships in Oxidative Addition at Zerovalent Nickel Centers. *Manuscript in preparation*, 2026.

Please cite this work when using this code.
