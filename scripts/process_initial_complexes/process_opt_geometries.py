""" process parquet of initial complex geometries
(1) check if the optimized geometry matches the expected SMILES structure for reactants and products
(2) classify reactant and product type and configuration
(3) pair corresponding reactant and product geometries for each reaction, and determine reaction energy
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem


def filter_opt_geometries(df, smiles_dct):
    """Filters the input dataframe to include only rows where the 
    optimized geometry matches the expected SMILES structure for reactants and products.

    :param df: Input dataframe containing optimized geometries and related information.
    :param smiles_dct: Dictionary containing expected SMILES structures for reactants and products.
    :return: Filtered dataframe with matched geometries and their match types.
    """
    # List to collect matched rows with their match type
    matched_rows = []
    no_entries = []
    no_opt_xyz = []
    no_match = []
    count = 0

    for row in df.itertuples(index=False):
        count += 1
        # if count%1000 == 0:
        #    print(f"Processed {count} rows...")

        unique_name = getattr(row, 'unique_name', None)
        pair = '_'.join(unique_name.split('_')[:2])
        mol_type = unique_name.split('_')[2].replace('A','').replace('B','') 
        opt_xyz = getattr(row, 'opt_xyz', None)

        if opt_xyz is None or row.number_of_imaginary > 0:
            no_opt_xyz.append(unique_name)
            continue
        
        m1 = Chem.MolFromXYZBlock(opt_xyz)
        rdDetermineBonds.DetermineConnectivity(m1)
        AllChem.Compute2DCoords(m1)
        for b in m1.GetBonds():
            b.SetBondType(Chem.BondType.UNSPECIFIED)

        if pair not in smiles_dct:
            no_entries.append(pair)
            continue

        if 'product2_smiles' not in smiles_dct[pair] or 'product1_smiles' not in smiles_dct[pair] or 'reactant_smiles' not in smiles_dct[pair]:
            print(f"Missing SMILES entries for pair {pair} in smiles_dct.")
            continue
        
        entry = smiles_dct[pair]
        compare_mols = [None, None, None]

        if mol_type == 'reactant' and entry['reactant_smiles'] is not None:
            compare_mols[0] = Chem.MolFromSmiles(entry['reactant_smiles'], sanitize=False) 
        if mol_type == 'product' and entry['product1_smiles'] is not None:
            compare_mols[1] = Chem.MolFromSmiles(entry['product1_smiles'], sanitize=False) 
        if mol_type == 'product' and entry['product2_smiles'] is not None:
            compare_mols[2] = Chem.MolFromSmiles(entry['product2_smiles'], sanitize=False) 

        match_found = False
        for idx, cmp_mol in enumerate(compare_mols):
            if cmp_mol is None:
                continue
            Chem.SanitizeMol(cmp_mol, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            cmp_mol = Chem.AddHs(cmp_mol)
            for b in cmp_mol.GetBonds():
                b.SetBondType(Chem.BondType.UNSPECIFIED)
            same = m1.HasSubstructMatch(cmp_mol) and cmp_mol.HasSubstructMatch(m1)
            if same:
                row_dict = row._asdict()
                match_type = idx
                row_dict['match_type'] = match_type
                matched_rows.append(row_dict)
                match_found = True
                break  # Only take the first match
        if not match_found:
            no_match.append(unique_name)

    # Create the new dataframe with matched rows
    df_matched = pd.DataFrame(matched_rows)
    print(f"\nTotal matched rows: {len(df_matched)}")
    if len(df_matched) > 0:
        print(f"Match type distribution:\n{df_matched['match_type'].value_counts()}")

    print(f"Total rows with no opt_xyz: {len(no_opt_xyz)}")
    print(f"Total rows with no entries: {len(no_entries)}, {len(set(no_entries))}")
    print(f"Total rows with no match: {len(no_match)}")
    return df_matched


def classify_ni_geometry(xyz_str, determine_stereo=True):
    """
    Parse XYZ string, find Ni atom, identify 4 bound atoms, and classify geometry.
    1. For products, insertion type I or II (based on CIP rules)
    2. S or O stereochemistry (based on CIP rules)
    3. Geometry classification based on dihedral angle between two planes
        - Square planar: dihedral < 25
        - Tetrahedral: dihedral > 75
        - Intermediate: 25 <= dihedral <= 75
    
    :param xyz_str: XYZ string of the molecule
    :param determine_stereo: Whether to determine stereochemistry (default: True)
    :return: Tuple of (geometry classification, stereochemistry label)
    """
   
    ######### 
    # Functions 
    #########
    def get_shared_ring_with_ni(xyz_str, n_idx1, n_idx2, ni_idx):
        """
        Find the ring containing both nitrogen atoms and Ni (the chelate ring).
        Returns list of atom indices in the ring, or empty list if not found.
        """
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        
        # Build molecule from XYZ
        mol = Chem.MolFromXYZBlock(xyz_str)
        if mol is None:
            return []
        
        rdDetermineBonds.DetermineConnectivity(mol)
        
        # Initialize ring information
        Chem.GetSSSR(mol)
        
        # Get all rings in the molecule
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        # Find ring containing both N atoms and Ni
        for ring in atom_rings:
            if n_idx1 in ring and n_idx2 in ring and ni_idx in ring:
                return list(ring)
        
        return []
    
    def get_insertion_type(xyz_str, o_idx, alk_idx, excluded_atoms):
        """
        Determine if co2 inserted at the higher priority carbon
        """
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        
        # Build molecule from XYZ
        mol = Chem.MolFromXYZBlock(xyz_str)
        if mol is None:
            return 'unknown'
        
        rdDetermineBonds.DetermineConnectivity(mol)
        
        # Get neighbors of o_idx and alk_idx, excluding excluded_atoms
        insertion_carbon_idx = None
        excluded_atoms = set(excluded_atoms) | {o_idx}

        neighbors_o = set(
            idx.GetIdx() for idx in mol.GetAtomWithIdx(o_idx).GetNeighbors() 
            if idx.GetIdx() not in excluded_atoms)
        for alpha in neighbors_o.copy():
            excluded_atoms.add(alpha)
            if mol.GetAtomWithIdx(alpha).GetSymbol() == 'Ni':
                continue
            beta_neighbors = set(
                idx.GetIdx() for idx in mol.GetAtomWithIdx(alpha).GetNeighbors() 
                if idx.GetIdx() not in excluded_atoms and idx.GetIdx() != o_idx
            )
            for beta in beta_neighbors:
                if mol.GetAtomWithIdx(beta).GetSymbol() == 'O':
                    excluded_atoms.add(beta)
                else:
                    insertion_carbon_idx = beta
                    break
        insertion_is_type_one = compare_atom_priority_excluding_chelate(
            xyz_str, insertion_carbon_idx, alk_idx, excluded_atoms
        )
        return insertion_is_type_one
    
    def compare_atom_priority_excluding_chelate(xyz_str, atom_idx1, atom_idx2, excluded_atoms):
        """
        Compare two atoms using CIP-like rules, excluding atoms in excluded_atoms set.
        Returns: 1 if atom_idx1 has higher priority, -1 if atom_idx2, 0 if tied
        """
        import automol as am
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        # Build molecule and graph
        mol = Chem.MolFromXYZBlock(xyz_str)
        if mol is None:
            return 0
        rdDetermineBonds.DetermineConnectivity(mol)
        excluded_atoms = set(excluded_atoms) | {atom_idx1, atom_idx2}
        # Build automol graph from RDKit connectivity (to avoid VanderWaalsRadii error for Ni)
        # Extract atom symbols and coordinates
        conf = mol.GetConformer()
        symbols = tuple(atom.GetSymbol() for atom in mol.GetAtoms())
        coords = tuple(
            tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
        )
        
        # Extract bonds from RDKit
        bond_keys = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_keys.append(frozenset({i, j}))
        
        # Create automol graph from symbols and bonds
        graph = am.graph.base.from_data(
            {i: s for i, s in enumerate(symbols)},
            bond_keys
        )
        
        # Pre-compute bond orders once (expensive operation)
        bond_orders = am.graph.kekule_bond_orders(graph)
        
        # Pre-compute atomic priorities
        priority_dct = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
            'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'Ni': 28
        }
        atom_priorities = [priority_dct.get(sym, 0) for sym in symbols]
        
        # Memoization cache for comparisons
        comparison_cache = {}
        
        def _get_neighbor_atoms(idx, graph, visited, exclude_set):
            """Get neighbors, excluding visited and excluded atoms."""
            expanded_idxs = []
            ngbs = am.graph.atom_neighbor_atom_keys(graph, idx)
            
            for ngb in ngbs:
                if ngb in visited or ngb in exclude_set:
                    continue
                bond_key = frozenset({idx, ngb})
                order = bond_orders.get(bond_key, 1)
                for _ in range(int(order)):
                    expanded_idxs.append(ngb)
            
            return expanded_idxs
        
        def _get_neighbor_priorities(idx, graph, visited, exclude_set):
            """Get sorted neighbor priorities, excluding atoms in exclude_set."""
            ngbs = _get_neighbor_atoms(idx, graph, visited, exclude_set)
            if not ngbs:
                return [], []
            
            prios = [atom_priorities[ngb] for ngb in ngbs]
            paired = sorted(zip(prios, ngbs), key=lambda x: x[0], reverse=True)
            prios, ngbs = zip(*paired)
            return list(ngbs), list(prios)
        
        def _determine_winner(idx_i, idx_j, visited_i, visited_j, depth=0):
            """Recursively determine winner by CIP rules."""
            # Check cache first
            cache_key = (idx_i, idx_j, frozenset(visited_i), frozenset(visited_j))
            if cache_key in comparison_cache:
                return comparison_cache[cache_key]
            
            # Depth limit
            if depth > 20:
                return 0
            
            # Early termination: compare atomic numbers directly first
            if depth == 0:
                prio_i = atom_priorities[idx_i]
                prio_j = atom_priorities[idx_j]
                if prio_i != prio_j:
                    result = 1 if prio_i > prio_j else -1
                    comparison_cache[cache_key] = result
                    return result
            
            new_visited_i = visited_i | {idx_i}
            new_visited_j = visited_j | {idx_j}
            
            ngbs_i, prios_i = _get_neighbor_priorities(idx_i, graph, visited_i, excluded_atoms)
            ngbs_j, prios_j = _get_neighbor_priorities(idx_j, graph, visited_j, excluded_atoms)
            # Pad the shorter list with None (lower priority than any atom)
            max_len = max(len(prios_i), len(prios_j))
            prios_i = list(prios_i) + [0] * (max_len - len(prios_i))
            prios_j = list(prios_j) + [0] * (max_len - len(prios_j))
            ngbs_i = list(ngbs_i) + [None] * (max_len - len(ngbs_i))
            ngbs_j = list(ngbs_j) + [None] * (max_len - len(ngbs_j))
            
            # Compare immediate neighbor priorities
            for i in range(max_len):
                if prios_i[i] > prios_j[i]:
                    comparison_cache[cache_key] = 1
                    return 1
                elif prios_j[i] > prios_i[i]:
                    comparison_cache[cache_key] = -1
                    return -1
            
            # Keep going until a difference is found
            for i in range(max_len):
                if ngbs_i[i] is not None and ngbs_j[i] is not None:
                    result = _determine_winner(ngbs_i[i], ngbs_j[i], new_visited_i, new_visited_j, depth + 1)
                    if result != 0:
                        comparison_cache[cache_key] = result
                        return result
            
            comparison_cache[cache_key] = 0
            return 0
        
        return _determine_winner(atom_idx1, atom_idx2, set(), set())

    ######### 
    # Read in XYZ and find Ni and 4 nearest neighbors
    #########
    if not xyz_str or pd.isna(xyz_str):
        return 'unknown'
    
    lines = str(xyz_str).strip().split('\n')
    if len(lines) < 3:
        return 'unknown'
    
    # Parse atoms
    atoms = []
    ni_idx = None
    for i, line in enumerate(lines[2:]):  # Skip first 2 lines (count + comment)
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((elem, x, y, z))
        if elem == 'Ni' and ni_idx is None:
            ni_idx = len(atoms) - 1
    if ni_idx is None or len(atoms) < 5:
        return 'unknown'
    
    # Get Ni position
    ni_pos = np.array(atoms[ni_idx][1:])
    
    # Find 4 nearest NON-HYDROGEN neighbors to Ni
    distances = []
    for i, (elem, x, y, z) in enumerate(atoms):
        if i == ni_idx:
            continue
        pos = np.array([x, y, z])
        dist = np.linalg.norm(pos - ni_pos)
        distances.append((i, dist, elem))
    distances.sort(key=lambda x: x[1])
    neighbor_indices = [distances[i][0] for i in range(len(distances)) if distances[i][2] != 'H']
    if len(neighbor_indices) < 4:
        return 'unknown'
    else:
        neighbor_indices = neighbor_indices[:4]
    neighbor_positions = np.array([atoms[i][1:] for i in neighbor_indices])

    # Get the two N index and positions
    n_indices = [i for i, (elem, _, _, _) in enumerate(atoms) if elem.upper() == 'N' and i in neighbor_indices]
    if len(n_indices) < 2:
        return 'unknown'
    alk_idxs_list = set(neighbor_indices) - set(n_indices)
    alk_idxs_list = list(alk_idxs_list)

    # Create coords array for easy position lookups
    coords = np.array([atom[1:] for atom in atoms])

    # Do some additional checks on the geometry
    ni_coordination_flag = False
    alkyne_valence_flag = False

    # Check that no extra hydrogens are coordinating with the Ni
    ni_h_bond_threshold = 1.5  # Angstrom
    ni_heavy_bond_threshold = 2.0  # Angstrom
    hydrogen_distances = [dist for _, dist, elem in distances if elem.upper() == 'H']
    closest_hydrogen = min(hydrogen_distances) if hydrogen_distances else None
    if closest_hydrogen is not None and closest_hydrogen < ni_h_bond_threshold:
        ni_coordination_flag = True
    
    # Check that no extra heavy atoms are coordinating with the Ni
    heavy_distances = [dist for _, dist, elem in distances if elem.upper() != 'H']
    if len(heavy_distances) > 4:
        if heavy_distances[4] < ni_heavy_bond_threshold:
            ni_coordination_flag = True

    # Check that no atoms are over-coordinating with the alkyne-side atoms
    alkyne_metal_close_threshold = 2.5  # Angstrom
    alkyne_heavy_close_threshold = 2.2  # Angstrom
    alkyne_hyd_close_threshold = 1.4  # Angstrom
    alkyne_heavy_close_threshold_with_o = 2.2  # Angstrom (looser for product-like C/O coordination)
    alkyne_hyd_close_threshold_with_o = 1.6  # Angstrom (looser for product-like C/O coordination)
    if len(alk_idxs_list) != 2:
        alkyne_valence_flag = True
    else:
        alk_a, alk_b = alk_idxs_list
        has_o_in_alkyne = any(atoms[idx][0].upper() == 'O' for idx in alk_idxs_list)
        required_map = {
            alk_a: {ni_idx, alk_b},
            alk_b: {ni_idx, alk_a},
        }
        for center_idx in (alk_a, alk_b):
            center_pos = np.array(atoms[center_idx][1:])
            close_atoms = []
            heavy_cutoff = alkyne_heavy_close_threshold_with_o if has_o_in_alkyne else alkyne_heavy_close_threshold
            hyd_cutoff = alkyne_hyd_close_threshold_with_o if has_o_in_alkyne else alkyne_hyd_close_threshold
            for idx, (_, x, y, z) in enumerate(atoms):
                if idx == center_idx:
                    continue
                dist = np.linalg.norm(np.array([x, y, z]) - center_pos)
                if dist <= alkyne_metal_close_threshold and atoms[idx][0].upper() == 'NI':
                    close_atoms.append(idx)
                elif dist <= heavy_cutoff and atoms[idx][0].upper() != 'H':
                    close_atoms.append(idx)
                elif dist <= hyd_cutoff and atoms[idx][0].upper() == 'H':
                    close_atoms.append(idx)
            if has_o_in_alkyne:
                # Product-like mode: only enforce Ni is present and total close count is Ni + 2 others.
                if ni_idx not in close_atoms or (len(close_atoms) != 3 and atoms[center_idx][0].upper() == 'C'):
                    print(f"Alkyne valence issue at atom {center_idx} ({atoms[center_idx][0]}): close atoms {close_atoms}, has O in alkyne: {has_o_in_alkyne}")
                    alkyne_valence_flag = True
                    break
                if ni_idx not in close_atoms or (len(close_atoms) not in [2, 3] and atoms[center_idx][0].upper() == 'O'):
                    print(f"Alkyne valence issue at atom {center_idx} ({atoms[center_idx][0]}): close atoms {close_atoms}, has O in alkyne: {has_o_in_alkyne}")
                    alkyne_valence_flag = True
                    break
            else:
                required = required_map[center_idx]
                if not required.issubset(set(close_atoms)):
                    print(f"Alkyne valence issue at atom {center_idx} ({atoms[center_idx][0]}): close atoms {close_atoms}, required: {required}")
                    alkyne_valence_flag = True
                    break
                additional_close = [idx for idx in close_atoms if idx not in required]
                if len(additional_close) != 1:
                    alkyne_valence_flag = True
                    break
    
    #####################
    # TYPE I or TYPE II insertion
    #####################
    insertion_label = ''
    has_o = [atoms[idx][0].upper() == 'O' for idx in alk_idxs_list]
    if any(has_o):
        insertion_type = get_insertion_type(
            xyz_str,
            alk_idxs_list[has_o.index(True)], 
            alk_idxs_list[1 - has_o.index(True)], {})
        if insertion_type == 1:
            insertion_label = 'Type_I'
        elif insertion_type == -1:
            insertion_label = 'Type_II'
        else:
            insertion_label = ''
    
    #####################
    # S or O stereochemistry 
    #####################
    # determine if highest priority groups on each ligand are 
    # close to each other (opposite clockwiseness) or far from each other (same clockwiseness)
    if determine_stereo:
        # Compare the two nitrogens, excluding chelate ring atoms AND Ni
        chelate_ring = get_shared_ring_with_ni(xyz_str, n_indices[0], n_indices[1], ni_idx)
        chelate_set = set(chelate_ring) if chelate_ring else set()
        excluded_for_comparison = chelate_set | {ni_idx}
        n_priority_result = compare_atom_priority_excluding_chelate(xyz_str, n_indices[0], n_indices[1], excluded_for_comparison)
    
        # Assign high and low priority nitrogen indices
        if n_priority_result > 0:
            high_pri_n_idx = n_indices[0]
            low_pri_n_idx = n_indices[1]
        elif n_priority_result < 0:
            high_pri_n_idx = n_indices[1]
            low_pri_n_idx = n_indices[0]
        else:
            # Tied - default to first one
            high_pri_n_idx = None
            low_pri_n_idx = None

        # get the higher priority C index (or O if a product)
        alk_idxs = set(neighbor_indices) - set(n_indices)
        alk_idxs_list = list(alk_idxs)
    
        if len(alk_idxs_list) == 2:
            # Check if there's an oxygen
            if any(has_o):
                # If there's an O, it's higher priority
                high_pri_alk_idx = alk_idxs_list[has_o.index(True)]
                low_pri_alk_idx = alk_idxs_list[1 - has_o.index(True)]

            else:
                # Both are carbons - compare using CIP-like rules excluding chelate and Ni
                alk_priority_result = compare_atom_priority_excluding_chelate(
                    xyz_str, alk_idxs_list[0], alk_idxs_list[1], excluded_for_comparison
                )
                if alk_priority_result > 0:
                    high_pri_alk_idx = alk_idxs_list[0]
                    low_pri_alk_idx = alk_idxs_list[1]
                elif alk_priority_result < 0:
                    high_pri_alk_idx = alk_idxs_list[1]
                    low_pri_alk_idx = alk_idxs_list[0]
                else:
                    # Tied - default to first
                    high_pri_alk_idx = alk_idxs_list[0]
                    low_pri_alk_idx = alk_idxs_list[1]
        elif len(alk_idxs_list) == 1:
            high_pri_alk_idx = alk_idxs_list[0]
            low_pri_alk_idx = None
        else:
            high_pri_alk_idx = None
            low_pri_alk_idx = None

        # Determine stereochemistry label (S or O) based on spatial arrangement
        # Only add if there's a priority difference for both alk and nitrogen atoms
        n_has_priority = (high_pri_n_idx is not None and low_pri_n_idx is not None 
                          and high_pri_n_idx != low_pri_n_idx)
        alk_has_priority = (high_pri_alk_idx is not None and low_pri_alk_idx is not None 
                            and high_pri_alk_idx != low_pri_alk_idx)
        if n_has_priority and alk_has_priority:
            # Calculate distances from high_pri_alk to both nitrogens
            high_alk_pos = coords[high_pri_alk_idx]
            high_n_pos = coords[high_pri_n_idx]
            low_n_pos = coords[low_pri_n_idx]
            
            dist_to_high_n = np.linalg.norm(high_alk_pos - high_n_pos)
            dist_to_low_n = np.linalg.norm(high_alk_pos - low_n_pos)
            
            if dist_to_high_n < dist_to_low_n and insertion_label in ['Type_I', '']:
                stereo_label = 'O'  # opposite clockwiseness
            elif dist_to_high_n < dist_to_low_n and insertion_label in ['Type_II']:
                stereo_label = 'S'  # Same clockwiseness
            elif dist_to_high_n > dist_to_low_n and insertion_label in ['Type_I', '']:
                stereo_label = 'S'  # same clockwiseness
            elif dist_to_high_n > dist_to_low_n and insertion_label in ['Type_II']:
                stereo_label = 'O'  # opposite clockwiseness
    else: 
        stereo_label = ''


    ##########
    # Geometry configuration (tetrahedral, square planar, intermediate) 
    ##########
    vectors = neighbor_positions - ni_pos
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Find the pair of atoms with the largest angle (likely trans if exists)
    max_angle = 0
    best_pair = (0, 1)
    for i in range(4):
        for j in range(i+1, 4):
            cos_angle = np.dot(vectors_norm[i], vectors_norm[j])
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            if angle > max_angle:
                max_angle = angle
                best_pair = (i, j)
    
    # Split atoms into two pairs
    pair1 = tuple(n_indices)  
    pair2 = tuple(alk_idxs_list)
    # Plane 1: Ni + atoms from pair1
    p1_a = atoms[pair1[0]][1:]
    p1_b = atoms[pair1[1]][1:]
    v1_a = p1_a - ni_pos
    v1_b = p1_b - ni_pos
    normal1 = np.cross(v1_a, v1_b)
    norm1 = np.linalg.norm(normal1)
    
    # Plane 2: Ni + atoms from pair2
    p2_a = atoms[pair2[0]][1:]
    p2_b = atoms[pair2[1]][1:]
    v2_a = p2_a - ni_pos
    v2_b = p2_b - ni_pos
    normal2 = np.cross(v2_a, v2_b)
    norm2 = np.linalg.norm(normal2)
    # Check for degenerate cases
    if norm1 < 1e-6 or norm2 < 1e-6:
        return ('unknown', stereo_label)
    
    # Normalize the normals
    normal1 = normal1 / norm1
    normal2 = normal2 / norm2
    
    # Dihedral angle between the two planes
    cos_dihedral = np.dot(normal1, normal2)
    cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
    dihedral_angle = np.degrees(np.arccos(abs(cos_dihedral)))  # Use abs to get 0-90° range
    
    # Classification based on dihedral angle
    # Square planar
    if dihedral_angle <= 25:
        return ('square_planar', stereo_label, insertion_label, ni_coordination_flag, alkyne_valence_flag, max_angle, dihedral_angle)
    
    # Tetrahedral
    if dihedral_angle >= 75:
        return ('tetrahedral', '', insertion_label, ni_coordination_flag, alkyne_valence_flag, max_angle, dihedral_angle)
    
    # Intermediate: 
    else:
        return ('intermediate', stereo_label, insertion_label, ni_coordination_flag, alkyne_valence_flag, max_angle, dihedral_angle)


def classify_pending_row(row_dict):
    """Classify one pending row and return either a good-row or bad-row payload."""
    xyz_str = row_dict.get('opt_xyz', None)
    name = row_dict.get('unique_name', 'N/A')

    determine_stereo = True
    try:
        bipy_subs = name.split('_')[0].split('-')[1]
        if bipy_subs == bipy_subs[::-1]:  # Symmetric substitution - no stereochemistry
            determine_stereo = False
        alk_subs = name.split('_')[1].split('-C2H2-')
        if alk_subs == alk_subs[::-1]:  # Symmetric substitution - no stereochemistry
            determine_stereo = False
    except Exception:
        determine_stereo = True

    result = classify_ni_geometry(xyz_str, determine_stereo=determine_stereo)
    if len(result) == 7:
        geometry_type, stereo_label, insertion_type, ni_coord_flag, alkyne_val_flag, max_angle, dihedral_angle = result
    else:
        geometry_type = 'unknown'
        stereo_label = ''
        insertion_type = ''
        ni_coord_flag = True
        alkyne_val_flag = True
        max_angle = None
        dihedral_angle = None

    out_row = row_dict.copy()
    out_row['geometry_type'] = geometry_type
    out_row['stereo_label'] = stereo_label
    out_row['insertion_type'] = insertion_type
    out_row['max_angle'] = max_angle
    out_row['dihedral_angle'] = dihedral_angle

    if ni_coord_flag or alkyne_val_flag:
        print_msg = 'bad Ni coordination' if ni_coord_flag else ''
        print_msg += ' and ' if ni_coord_flag and alkyne_val_flag else ''
        print_msg += 'bad alkyne valence' if alkyne_val_flag else ''
        out_row['ni_coordination_flag'] = bool(ni_coord_flag)
        out_row['alkyne_valence_flag'] = bool(alkyne_val_flag)
        out_row['bad_flag'] = print_msg
        return 'bad', out_row, name

    return 'good', out_row, name


def get_reaction_parquet(df_processed):
    ligand_pairs = set(df_processed['ligand_pair'].to_list())
    reaction_dct_lst = []
    # MACE_MP-0a-large, with dispersion, determined CO2 energy in eV 
    co2_energy = -23.199012350 
    ev_to_kcal = 23.061
    # pbe determined benchmark correction to reaction energy (kcal/mol)
    correction = -20.35 
    for pair in ligand_pairs:
        pair_data = df_processed[df_processed['ligand_pair'] == pair]
        reactant_rows = pair_data[pair_data['mol_category']=='reactant']
        prod1_rows = pair_data[pair_data['insertion_type']=='Type_I']
        prod_rows = pair_data[(pair_data['insertion_type']=='') & (pair_data['mol_category'].isin(['product']))]
        prod1_rows = pd.concat([prod1_rows, prod_rows])
        prod2_rows = pair_data[pair_data['insertion_type']=='Type_II']
        o_reactant_rows = reactant_rows[reactant_rows['stereo_label'] == 'O']
        s_reactant_rows = reactant_rows[reactant_rows['stereo_label'] == 'S']
        a_reactant_rows = reactant_rows[reactant_rows['stereo_label'] == '']
        o_reactant_rows = pd.concat([o_reactant_rows, a_reactant_rows])
        s_reactant_rows = pd.concat([s_reactant_rows, a_reactant_rows])
        o_reactant_rows = o_reactant_rows.sort_values(by='G_eV')
        s_reactant_rows = s_reactant_rows.sort_values(by='G_eV')
        min_o_reactant = o_reactant_rows.iloc[0] if len(o_reactant_rows) > 0 else None
        min_s_reactant = s_reactant_rows.iloc[0] if len(s_reactant_rows) > 0 else None

        o_p1_rows = prod1_rows[prod1_rows['stereo_label'] == 'O']
        s_p1_rows = prod1_rows[prod1_rows['stereo_label'] == 'S']
        a_p1_rows = prod1_rows[prod1_rows['stereo_label'] == '']
        o_p1_rows = pd.concat([o_p1_rows, a_p1_rows])
        s_p1_rows = pd.concat([s_p1_rows, a_p1_rows])
        o_p1_rows = o_p1_rows.sort_values(by='G_eV')
        s_p1_rows = s_p1_rows.sort_values(by='G_eV')
        min_o_p1 = o_p1_rows.iloc[0] if len(o_p1_rows) > 0 else None
        min_s_p1 = s_p1_rows.iloc[0] if len(s_p1_rows) > 0 else None

        o_p2_rows = prod2_rows[prod2_rows['stereo_label'] == 'O']
        s_p2_rows = prod2_rows[prod2_rows['stereo_label'] == 'S']
        a_p2_rows = prod2_rows[prod2_rows['stereo_label'] == '']
        o_p2_rows = pd.concat([o_p2_rows, a_p2_rows])
        s_p2_rows = pd.concat([s_p2_rows, a_p2_rows])
        o_p2_rows = o_p2_rows.sort_values(by='G_eV')
        s_p2_rows = s_p2_rows.sort_values(by='G_eV')
        min_o_p2 = o_p2_rows.iloc[0] if len(o_p2_rows) > 0 else None
        min_s_p2 = s_p2_rows.iloc[0] if len(s_p2_rows) > 0 else None

        has_stereo = any(row.stereo_label in ['O', 'S'] for row in pair_data.itertuples(index=False))

        if min_o_reactant is not None and min_o_p1 is not None:
            reaction_energy = (min_o_p1['G_eV'] - min_o_reactant['G_eV'] - co2_energy) * ev_to_kcal + correction

            reaction_dct_lst.append({
                'ligand_pair': pair,
                'stereo_type': 'O' if has_stereo else '',
                'insertion_type': 'Type_I',
                'reaction_gibbs_kcal': reaction_energy,
                'reactant_gibbs': min_o_reactant['G_eV'],
                'product_gibbs': min_o_p1['G_eV'],
                'reactant_geometry': min_o_reactant['opt_xyz'],
                'product_geometry': min_o_p1['opt_xyz'],
                'reactant_configuration': min_o_reactant['geometry_type'],
                'product_configuration': min_o_p1['geometry_type'],
            })
        if has_stereo and min_s_reactant is not None and min_s_p1 is not None:
            unique_s = True
            if min_o_reactant is not None and min_o_p1 is not None:
                if (min_s_reactant.unique_name == min_o_reactant.unique_name) and (min_s_p1.unique_name == min_o_p1.unique_name):
                    unique_s = False
            if unique_s:
                reaction_energy = (min_s_p1['G_eV'] - min_s_reactant['G_eV'] - co2_energy) * ev_to_kcal + correction
                reaction_dct_lst.append({
                    'ligand_pair': pair,
                    'stereo_type': 'S',
                    'insertion_type': 'Type_I',
                    'reaction_gibbs_kcal': reaction_energy,
                    'reactant_gibbs': min_s_reactant['G_eV'],
                    'product_gibbs': min_s_p1['G_eV'],
                    'reactant_geometry': min_s_reactant['opt_xyz'],
                    'product_geometry': min_s_p1['opt_xyz'],
                    'reactant_configuration': min_s_reactant['geometry_type'],
                    'product_configuration': min_s_p1['geometry_type'],
                })
        if min_o_reactant is not None and min_o_p2 is not None:
            reaction_energy = (min_o_p2['G_eV'] - min_o_reactant['G_eV'] - co2_energy) * ev_to_kcal + correction
            reaction_dct_lst.append({
                'ligand_pair': pair,
                'stereo_type': 'O' if has_stereo else '',
                'insertion_type': 'Type_II',
                'reaction_gibbs_kcal': reaction_energy,
                'reactant_gibbs': min_o_reactant['G_eV'],
                'product_gibbs': min_o_p2['G_eV'],
                'reactant_geometry': min_o_reactant['opt_xyz'],
                'product_geometry': min_o_p2['opt_xyz'],
                'reactant_configuration': min_o_reactant['geometry_type'],
                'product_configuration': min_o_p2['geometry_type'],
            })

        if has_stereo and min_s_reactant is not None and min_s_p2 is not None:
            unique_s = True
            if min_o_reactant is not None and min_o_p2 is not None:
                if (min_s_reactant.unique_name == min_o_reactant.unique_name) and (min_s_p2.unique_name == min_o_p2.unique_name):
                    unique_s = False
            if unique_s:
                reaction_energy = (min_s_p2['G_eV'] - min_s_reactant['G_eV'] - co2_energy) * ev_to_kcal + correction
                reaction_dct_lst.append({
                    'ligand_pair': pair,
                    'stereo_type': 'S',
                    'insertion_type': 'Type_II',
                    'reaction_gibbs_kcal': reaction_energy,
                    'reactant_gibbs': min_s_reactant['G_eV'],
                    'product_gibbs': min_s_p2['G_eV'],
                    'reactant_geometry': min_s_reactant['opt_xyz'],
                    'product_geometry': min_s_p2['opt_xyz'],
                    'reactant_configuration': min_s_reactant['geometry_type'],
                    'product_configuration': min_s_p2['geometry_type'],
                })
    return pd.DataFrame(reaction_dct_lst)

if __name__ == "__main__":
    # IQC output parquet file
    path = Path("./iqc_thermo_results_20260404_095146_reduced.parquet")
    df = pd.read_parquet(path)

    # Dictionary of expected SMILES
    with open('full_large_smiles_dct.json', 'r') as f:
        smiles_dct = json.load(f)

    # Updated parquet with matched geometries and their match types 
    match_file = 'matched_geometries.parquet'
    if not Path(match_file).exists():
        matched_df = filter_opt_geometries(df, smiles_dct)
        matched_df.to_parquet(match_file, index=False)
    else:
        matched_df = pd.read_parquet(match_file)

    label_file = 'labeled_geometries.parquet'
    label_cols = ['geometry_type', 'stereo_label', 'insertion_type']
    bad_file = 'bad_geometries.parquet'
    bad_cols = label_cols + ['ni_coordination_flag', 'alkyne_valence_flag', 'bad_flag']

    if Path(label_file).exists():
        df_labeled = pd.read_parquet(label_file)
    else:
        df_labeled = matched_df.head(0).copy()
        for col in label_cols:
            df_labeled[col] = pd.Series(dtype='object')
        df_labeled.to_parquet(label_file, index=False)

    for col in label_cols:
        if col not in df_labeled.columns:
            df_labeled[col] = pd.Series(dtype='object')

    if Path(bad_file).exists():
        df_bad = pd.read_parquet(bad_file)
    else:
        df_bad = matched_df.head(0).copy()
        for col in bad_cols:
            df_bad[col] = pd.Series(dtype='object')
        df_bad.to_parquet(bad_file, index=False)

    for col in bad_cols:
        if col not in df_bad.columns:
            df_bad[col] = pd.Series(dtype='object')

    labeled_names = set(df_labeled['unique_name']) if 'unique_name' in df_labeled.columns else set()
    bad_names = set(df_bad['unique_name']) if 'unique_name' in df_bad.columns else set()
    processed_names = labeled_names | bad_names
    pending_df = matched_df[~matched_df['unique_name'].isin(processed_names)]
    total_pending = len(pending_df)
    print(f"Already labeled: {len(labeled_names)}")
    print(f"Already flagged bad: {len(bad_names)}")
    print(f"Pending to classify: {total_pending}")

    rows_to_append = []
    bad_rows_to_append = []
    pending_records = pending_df.to_dict(orient='records')
    max_workers = 10
    flush_every = 500

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(classify_pending_row, row_dict): row_dict
            for row_dict in pending_records
        }

        for idx, future in enumerate(as_completed(future_to_row), start=1):
            row_dict = future_to_row[future]
            name = row_dict.get('unique_name', 'N/A')
            try:
                row_type, out_row, name = future.result()
            except Exception as exc:
                print(f"Name: {name} - classification failed with error: {exc}")
                out_row = row_dict.copy()
                out_row['geometry_type'] = 'unknown'
                out_row['stereo_label'] = ''
                out_row['insertion_type'] = ''
                out_row['ni_coordination_flag'] = True
                out_row['alkyne_valence_flag'] = True
                out_row['bad_flag'] = f"classification_error: {exc}"
                row_type = 'bad'

            if row_type == 'bad':
                print(f"Name: {name} - flagged as {out_row['bad_flag']}.")
                bad_rows_to_append.append(out_row)
            else:
                rows_to_append.append(out_row)

            # Periodically flush to parquet so reruns can resume from partial progress.
            if idx % flush_every == 0:
                print(f"Processed {idx}/{total_pending} pending structures...")
                if rows_to_append:
                    batch_df = pd.DataFrame(rows_to_append)
                    df_labeled = pd.concat([df_labeled, batch_df], ignore_index=True)
                    df_labeled.to_parquet(label_file, index=False)
                    rows_to_append = []
                if bad_rows_to_append:
                    bad_batch_df = pd.DataFrame(bad_rows_to_append)
                    df_bad = pd.concat([df_bad, bad_batch_df], ignore_index=True)
                    df_bad.to_parquet(bad_file, index=False)
                    bad_rows_to_append = []

    if rows_to_append:
        batch_df = pd.DataFrame(rows_to_append)
        df_labeled = pd.concat([df_labeled, batch_df], ignore_index=True)
        df_labeled.to_parquet(label_file, index=False)

    if bad_rows_to_append:
        bad_batch_df = pd.DataFrame(bad_rows_to_append)
        df_bad = pd.concat([df_bad, bad_batch_df], ignore_index=True)
        df_bad.to_parquet(bad_file, index=False)

    print(f"Total labeled rows saved: {len(df_labeled)}")
    print(f"Total bad rows saved: {len(df_bad)}")


    # Add ligand_pair column to labeled dataframe for easier reaction-level grouping
    if 'ligand_pair' not in df_labeled.columns:
        df_labeled['ligand_pair'] = df_labeled['unique_name'].apply(lambda x: '_'.join(x.split('_')[:2]) if isinstance(x, str) else 'N/A')
        df_labeled.to_parquet(label_file, index=False)
    if 'mol_category' not in df_labeled.columns:
        df_labeled['mol_category'] = df_labeled['unique_name'].apply(lambda x: 'reactant' if 'reactant' in x else ('product' if 'product' in x else 'N/A') if isinstance(x, str) else 'N/A')
        df_labeled.to_parquet(label_file, index=False)

    reaction_json_file = 'reaction_data_06_12.json'
    if not Path(reaction_json_file).exists():
        df_labeled = pd.read_parquet(label_file)
        reaction_df = get_reaction_parquet(df_labeled)
        reaction_df.to_json(reaction_json_file)
        print(f"Reaction-level data saved to {reaction_json_file}")

