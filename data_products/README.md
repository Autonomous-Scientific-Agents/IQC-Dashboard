# IQC-Dashboard Data Products

This directory contains curated data products from computational chemistry studies using the Interactive Quantum Chemistry (IQC) Dashboard workflow. Each subdirectory represents a distinct chemical system or study.

## Available Data Products

### [Ni_CO2](./Ni_CO2/)

Data for Ni(0)–bipyridine complexes with coordinated alkynes, Ni(II)–bipyridine complexes formed by CO2  insertion into the Ni–alkenyl bond, and the reaction energies for each corresponding pair.

**Contents:**

1. **`iqc_thermo_results_20260404_095146_reduced.parquet`** (291 MB, 58,157 entries, 34 columns)
   - Raw IQC output containing optimized geometries and thermochemical properties determined with MACE-MP-0a-Large
   - Includes both reactant (bipyridine-Ni(0)-alkyne) and product (bipyridine-Ni(II)-carboxylated alkene) geometries
   - **Data Quality:** 99.8% of geometries have 0 imaginary frequencies
   - Key columns:
     - `unique_name`: Identifier for each geometry (format: `{bipyridine-ligand-identifier}_{alkyne-identifier}_{mol_type}_{config}`)
     - `opt_xyz`: Optimized geometry in XYZ format
     - `G_eV`: Gibbs free energy in eV
     - `total_energy_eV`: Total electronic energy in eV
     - `number_of_imaginary`: Number of imaginary frequencies (used for validation)
     - `formula`: Molecular formula
   - Additional columns:
     - **Provenance**: `date` (calculation timestamp), `hostname` (compute node), `mpi_rank`, `initial_xyz`, `initial_smiles`, `initial_energy_eV`, `initial_symmetry`
     - **Optimization details**: `opt_converged` (optimization success flag), `warnings` (calculation warnings), `smiles_changed` (structure validation flag), `opt_steps` (number of steps), `opt_time` (wall time), `opt_forces` (final forces), `opt_sym` (point group symmetry)
     - **Thermochemistry properties**: `H_eV` (enthalpy), `S_eV/K` (entropy), `E_ZPE_eV` (zero-point energy)
     - **Vibrational data**: `frequencies_cm^-1`, `vib_energies`, `vib_time` (calculation time)
   - The input geometries for these were generated using the pipeline in ../scripts/build_initial_complexes/ 
   - This file serves as input for the geometry classification and reaction pairing pipeline in ../scripts/process_initial_complexes/

2. **`reaction_data_06_12_raw.parquet`** (28 MB, 15,003 entries, 10 columns)
   - Reaction-level data pairing reactants with products
   - One row per unique reaction pathway (ligand pair + stereochemistry + insertion type)
   - Columns:
     - `ligand_pair`: Identifier for bipyridine-alkyne combination
     - `stereo_type`: Stereochemistry label, labelling the orientation of the bipyridine and alkyne ligands with respect to one another (`S`, `O`, or empty for symmetric ligands)
       - **S (same)**: High-priority groups on ligands have same clockwise orientation
       - **O (opposite)**: High-priority groups on ligands have opposite clockwise orientation
     - `insertion_type`: CO2 insertion site (`Type_I` or `Type_II`)
       - **Type I**: CO2 inserted at higher-priority carbon (CIP rules)
       - **Type II**: CO2 inserted at lower-priority carbon
     - `reaction_gibbs_kcal`: ΔG in kcal/mol
     - `reactant_gibbs`: Absolute Gibbs energy of the reactant in eV 
     - `product_gibbs`:  Absolute Gibbs energy of the product in eV 
     - `reactant_geometry`: XYZ coordinates of lowest-energy reactant conformer
     - `product_geometry`: XYZ coordinates of lowest-energy product conformer
     - `reactant_configuration`: Ni coordination geometry (`square_planar`, `tetrahedral`, or `intermediate`)
     - `product_configuration`: Ni coordination geometry for product
   - Generated from `iqc_thermo_results_20260404_095146_reduced.parquet` using the geometry classification pipeline in in ../scripts/process_initial_complexes (without the PBE benchmark correction factor)

3. **`reaction_data_06_12_corrected.parquet`** (28 MB, 15,003 entries, 10 columns)
   - Same structure as `reaction_data_06_12_raw.parquet` but Includes PBE benchmark correction (-20.35 kcal/mol)
   - Recommended dataset for analysis and machine learning model training

**Benchmark Data:**

4. **`PBE_reaction_energy_benchmark/mace_mp.parquet`** (107 KB, 51 entries, 10 columns)
   - Subset of 50 reactions computed with MACE-MP-0a-Large for benchmarking
   - Same columns as `reaction_data_06_12_raw.parquet`
   - Used with pbe.parquet to derive the -20.35 kcal/mol PBE correction factor
    - This systematic offset is applied as the correction in `reaction_data_06_12_corrected.parquet`

5. **`PBE_reaction_energy_benchmark/pbe.parquet`** (93 KB, 51 entries, 8 columns)
   - Same 50 reactions as mace_mp.parquet recomputed with PBE/def2-TZVP 


**Study Overview:**

This dataset explores structure-reactivity relationships for CO2 insertion at zerovalent nickel centers coordinated by bipyridine and alkyne ligands. The study systematically varies:
- Bipyridine substituents (8 positions, various electron-donating/withdrawing groups)
- Alkyne substituents (left and right positions, various functional groups)
- Metal coordination geometry (square planar vs. tetrahedral)
- Stereochemistry (S vs. O orientations for asymmetric ligands)
- Insertion regiochemistry (Type I vs. Type II)

**Processing Pipeline:**

```
MACE-MP geometry generation
         ↓
   IQC optimization
         ↓
iqc_thermo_results_*.parquet 
         ↓
Geometry classification
(process_opt_geometries.py)
         ↓
reaction_data_*.parquet 
```

**Usage Example:**

```python
import pandas as pd

# Load reaction data
df = pd.read_parquet('Ni_CO2/reaction_data_06_12_corrected.parquet')

# Filter for Type I insertions with S stereochemistry
type_I_S = df[(df['insertion_type'] == 'Type_I') & (df['stereo_type'] == 'S')]

# Analyze reaction energies
print(f"Mean ΔG: {type_I_S['reaction_gibbs_kcal'].mean():.2f} kcal/mol")
print(f"Range: {type_I_S['reaction_gibbs_kcal'].min():.2f} to {type_I_S['reaction_gibbs_kcal'].max():.2f} kcal/mol")
```

---

## Citation

Data products in this directory are part of the supporting information for:

Elliott, S. N.; Reji, R.; Chen, Q.; Zheng, X.; Riaz, S.; Yan, X.; Pham, T. D.; Huerta, E. A.; Glusac, K. D.; Keçeli, M. Data-Driven Discovery of Structure–Reactivity Relationships in Oxidative Addition at Zerovalent Nickel Centers. *Manuscript in preparation*, 2026.

Please cite this work when using these data products.

## Contributing New Data Products

