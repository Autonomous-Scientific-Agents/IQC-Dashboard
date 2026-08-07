
import os
import json
import argparse
import io
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr

from architector import build_complex
from mace.calculators import mace_mp


_MACE_CALC = None
def get_mace_calculator():
    global _MACE_CALC
    if _MACE_CALC is None:
        _MACE_CALC = mace_mp(model="large")
    return _MACE_CALC


class SuppressOutput:
    def __enter__(self):
        self._stdout = os.dup(1)
        self._stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)

    def __exit__(self, *args):
        os.dup2(self._stdout, 1)
        os.dup2(self._stderr, 2)
        os.close(self._stdout)
        os.close(self._stderr)

def write_structure_xyz(outdict: dict, filename: str = "structure.xyz"):
    """
    Write all structures from the output dictionary to XYZ format.
    Each structure gets its own file with a counter appended.
    Comment line includes the energy.
    
    Args:
        outdict: Output dictionary from build_complex
        filename: Base filename for XYZ output (counter will be appended)
    """
    if not outdict:
        raise ValueError("Output dictionary is empty")
    
    # Split filename into base and extension
    base, ext = os.path.splitext(filename)
    num_structures = len(outdict)
    
    written_files = []
    for idx, (key, struct_data) in enumerate(outdict.items(), 1):
        try:
            # Get atoms object and energy
            atoms = struct_data['ase_atoms']
            energy = struct_data.get('energy', 0.0)  # Default to 0.0 if energy not present
            
            # Create filename with counter if multiple structures
            if num_structures > 1:
                current_filename = f"{base}_{idx}{ext}"
            else:
                current_filename = filename
            
            # Write XYZ file
            with open(current_filename, 'w') as f:
                # Number of atoms
                f.write(f"{len(atoms)}\n")
                f.write(f"Structure {key}, Energy = {energy:.6f}\n")
                for symbol, pos in zip(atoms.get_chemical_symbols(), 
                                     atoms.get_positions()):
                    f.write(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

            written_files.append(current_filename)
            print(f"Wrote structure {idx}/{num_structures} to {current_filename}")
            
        except (KeyError, AttributeError):
            print(f"Error in structure data format for key {key}:")
            print(f"Structure data: {struct_data}")
            raise
    
    return written_files


def chunk_dict(data, start_idx, num_entries):
    """
    Distribute dictionary entries into non-overlapping, evenly sized, randomized chunks.
    The randomization uses a fixed seed for reproducibility.
    start_idx: the starting index for this chunk (0-based)
    num_entries: number of entries to include in this chunk
    """
    import random
    keys = list(data.keys())
    random.Random(42).shuffle(keys)
    total = len(keys)
    end_idx = min(start_idx + num_entries, total)
    chunk_keys = keys[start_idx:end_idx]
    return {k: data[k] for k in chunk_keys}

def process_single_ligand(key, out_dir, ligand_sets, process_num, selected_mol_type='reactant', core_type='tetrahedral'):
    """Process a single ligand entry."""
    # Create base input dictionary
    base_dict = {
        'core': {
            'metal': 'Ni',
            'coreCN': 4,
            'coreType': core_type,
            'smiles': '[Ni]',
        },
        'parameters': {
            'calculator': None,
            'metal_ox': 0 if selected_mol_type == 'reactant' else 2,
            'is_actinide': False,
            'original_metal': 'Ni',
        }
    }
    
    print(f"\nProcessor {process_num} processing key: {key}")
    base_dict['parameters']['calculator'] = get_mace_calculator()
    
    # Check if ligand_sets is a dict or list
    with SuppressOutput():
        if isinstance(ligand_sets, dict):
            items_to_process = ligand_sets.items()
        elif isinstance(ligand_sets, list):
            items_to_process = enumerate(ligand_sets)
        else:
            return

        os.makedirs(out_dir, exist_ok=True)
        for mol_typ, ligand_set in items_to_process:
            if selected_mol_type != mol_typ: 
               continue

            # Check if files already exist
            base_file = os.path.join(out_dir, f"{key}_{mol_typ}.xyz")
            if os.path.exists(base_file):
                print(f"  Skipping {key}_{mol_typ}: File already exists")
                continue
            
            # Create a new input dictionary for this ligand set
            input_dict = base_dict.copy()
            input_dict['ligands'] = ligand_set
            if mol_typ == 'reactant':
                fixed_ligand_set = []
                for ligand in ligand_set:
                   lig_dct = {
                       'smiles': ligand['smiles'].replace(']#[',']=['),
                       'coordList': ligand['coordList'],
                       'ligType': 'bi_cis',
                   }
                   fixed_ligand_set.append(lig_dct.copy())
                input_dict['ligands'] = fixed_ligand_set
            if not input_dict['ligands']:
                continue
            
            try:
                # Build the complex while suppressing any prints from build_complex
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                try:
                    with redirect_stdout(buf_out), redirect_stderr(buf_err):
                        result = build_complex(input_dict)
                except Exception:
                    # Report the exception (outside the redirected context) and continue
                    result = build_complex(input_dict)
                    if not result:
                        continue

                try:
                    # Save result to individual XYZ file in geometries directory
                    base_filename = os.path.join(out_dir, f"{key}_{mol_typ}.xyz")
                    write_structure_xyz(result, base_filename)
                except Exception:
                    continue

            except Exception:
                continue

    print(f"\nCompleted processor {process_num} on key: {key}.")

def read_ligands_file(filename='ligand_sets.json'):
    """Read and parse the JSON file containing ligand information."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filename}")
        print(f"Details: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading {filename}: {e}")
        return None

def main():
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process ligand entries across processors')
    parser.add_argument('--start-entry', type=int, required=True,
                      help='Starting entry number (0-based)')
    parser.add_argument('--num-entries', type=int, required=True,
                      help='Number of entries to process on this node')
    parser.add_argument('--core-type', type=str, default='tetrahedral',
                      help='Type of core to use for ligands')
    parser.add_argument('--mol-type', type=str, default='reactant',
                      help='Type of molecule to process (e.g., reactant, product)')
    parser.add_argument('--nprocs', type=int, default=120,
                      help='Number of processors to use')
    parser.add_argument('--ligands-file', type=str, default='./ligand_sets.json',
                      help='Path to the JSON file containing ligand information')
    parser.add_argument('--output-dir', type=str, default='geometries',
                      help='Directory to save generated geometries')
    args = parser.parse_args()
    
    # Read all ligands
    ligands_data = read_ligands_file(filename=args.ligands_file)
    nprocs = args.nprocs
    try:
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            pass
        print('Preloading MACE model in parent process...')
        get_mace_calculator()
        print('Preload complete.')
    except Exception as e:
        print(f'Warning: Preloading MACE failed: {e}')

    if ligands_data is None:
        print("Error reading ligands file")
        return
    
    total_entries = len(ligands_data)
    print(f"Total ligand entries in JSON: {total_entries}")
    if args.start_entry < 0 or args.start_entry >= total_entries:
        print(f"Error: Starting entry {args.start_entry} is out of range. Total entries: {total_entries}")
        return
    if args.start_entry + args.num_entries > total_entries:
        print("Warning: Requested more entries than available. Processing until end.")
        args.num_entries = total_entries - args.start_entry
    
    # Get this node's entries
    node_data = chunk_dict(ligands_data, args.start_entry, args.num_entries)
    print(f"Processing entries {args.start_entry} to {args.start_entry + args.num_entries - 1}")
    print(f"Total entries in this node: {len(node_data)}")

    # Set up multiprocessing (120 processors)
    pool = mp.Pool(processes=nprocs)
    
    try:
        # Create process pool and map individual entries to processes
        process_args = [(key, args.output_dir, ligand_sets, i, args.mol_type, args.core_type) for i, (key, ligand_sets) in enumerate(node_data.items())]
        pool.starmap(process_single_ligand, process_args)
    finally:
        pool.close()
        pool.join()
        
    print(f"Finished processing entries {args.start_entry} to {args.start_entry + args.num_entries - 1}")

if __name__ == '__main__':
    main()
