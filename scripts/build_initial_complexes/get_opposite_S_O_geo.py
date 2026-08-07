import numpy as np
import os
import glob

def read_xyz(filename):
    """Read xyz file and return atom types and coordinates."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    atoms = []
    coords = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return atoms, np.array(coords), comment

def write_xyz(filename, atoms, coords, comment):
    """Write xyz file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}\n")

def find_closest_atoms(ni_coord, atom_type, atoms, coords, n=2):
    """Find the n closest atoms of a specific type to Ni."""
    indices = [i for i, atom in enumerate(atoms) if atom == atom_type]
    
    if len(indices) < n:
        raise ValueError(f"Not enough {atom_type} atoms found")
    
    distances = []
    for idx in indices:
        dist = np.linalg.norm(coords[idx] - ni_coord)
        distances.append((dist, idx))
    
    distances.sort()
    return [idx for dist, idx in distances[:n]]

def plane_normal(p1, p2, p3):
    """Calculate normal vector of a plane defined by three points."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    # Normalize
    normal = normal / np.linalg.norm(normal)
    return normal

def angle_between_planes(normal1, normal2):
    """Calculate angle between two planes given their normal vectors."""
    cos_angle = np.abs(np.dot(normal1, normal2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def identify_bipyridine_atoms(atoms, coords, n_indices, threshold_multiplier=1.0):
    """Identify all atoms that are part of the bipyridine ligand.
    Uses a distance-based approach to find connected atoms to the N atoms.
    
    Args:
        atoms: List of atom types
        coords: Array of coordinates
        n_indices: Indices of the N atoms
        threshold_multiplier: Multiplier for bond distance thresholds (default 1.0)
    """
    
    bipy_atoms = set(n_indices)
    
    # Build connectivity based on distance thresholds
    base_thresholds = {
        ('N', 'C'): 1.5,
        ('C', 'C'): 1.7,
        ('C', 'H'): 1.3,
        ('N', 'H'): 1.3,
        ('C', 'Cl'): 1.85,  # C-Cl bond
        ('C', 'Br'): 2.0,   # C-Br bond
        ('C', 'F'): 1.5,    # C-F bond
        ('C', 'I'): 2.2,    # C-I bond
        ('N', 'Cl'): 1.85,  # N-Cl bond (if any N-substituted)
        ('N', 'F'): 1.5,    # N-F bond (if any N-substituted)
        ('C', 'O'): 1.5,    # C-O bond (if any O-substituted)
        ('N', 'O'): 1.5,    # N-O bond (e.g., nitro groups)
    }
    
    # Apply threshold multiplier
    bond_thresholds = {k: v * threshold_multiplier for k, v in base_thresholds.items()}
    
    max_bond_distance = 1.7 * threshold_multiplier  # Maximum distance to consider atoms connected
    
    # Iteratively add connected atoms starting from N atoms
    to_check = list(n_indices)
    checked = set()
    
    while to_check:
        current = to_check.pop(0)
        if current in checked:
            continue
        checked.add(current)
        
        current_atom = atoms[current]
        current_coord = coords[current]
        
        for i, (atom, coord) in enumerate(zip(atoms, coords)):
            if i == current or i in bipy_atoms:
                continue
            
            # Skip Ni
            if atom == 'Ni':
                continue
                
            dist = np.linalg.norm(coord - current_coord)
            
            # Check if likely bonded
            pair = tuple(sorted([current_atom, atom]))
            threshold = bond_thresholds.get(pair, max_bond_distance)
            
            if dist < threshold:
                bipy_atoms.add(i)
                if i not in checked:
                    to_check.append(i)
    
    # Check for hypervalent atoms and fix connectivity
    bipy_atoms = fix_hypervalent_atoms(atoms, coords, list(bipy_atoms), bond_thresholds)
    
    return list(bipy_atoms)

def fix_hypervalent_atoms(atoms, coords, bipy_indices, bond_thresholds):
    """Fix hypervalent atoms by keeping only the shortest bonds.
    
    For example, if a hydrogen appears to have 2 bonds (likely one real bond
    and one hydrogen bond), keep only the shortest one.
    """
    # Expected max valency for each element
    max_valency = {
        'H': 1,
        'C': 4,
        'N': 4,  # Can be 3 or 4
        'O': 2,
        'F': 1,
        'Cl': 1,
        'Br': 1,
        'I': 1,
    }
    
    # Build connectivity map with distances
    connectivity = {idx: [] for idx in bipy_indices}
    
    for i in bipy_indices:
        for j in bipy_indices:
            if i >= j:
                continue
            
            dist = np.linalg.norm(coords[i] - coords[j])
            pair = tuple(sorted([atoms[i], atoms[j]]))
            threshold = bond_thresholds.get(pair, 1.7)
            
            if dist < threshold:
                connectivity[i].append((j, dist))
                connectivity[j].append((i, dist))
    
    for idx in bipy_indices:
        atom_type = atoms[idx]
        max_bonds = max_valency.get(atom_type, 6)  # Default to 6 for unknown types
        
        connections = connectivity[idx]
        
        if len(connections) > max_bonds:
            # Atom is hypervalent - keep only the shortest bonds
            connections_sorted = sorted(connections, key=lambda x: x[1])
            
            # Mark the longer "bonds" for removal from the other atoms' connectivity
            for other_idx, dist in connections_sorted[max_bonds:]:
                # Remove this connection from the other atom's list
                connectivity[other_idx] = [(idx2, d) for idx2, d in connectivity[other_idx] if idx2 != idx]
    
    # Rebuild connectivity and check if any atoms became isolated
    # An atom is isolated if it has no connections to other bipyridine atoms
    isolated_atoms = set()
    
    for idx in bipy_indices:
        if len(connectivity[idx]) == 0 and atoms[idx] != 'N':  # N atoms are starting points, keep them
            isolated_atoms.add(idx)
    
    # Remove isolated atoms (likely from broken hydrogen bonds)
    valid_bipy_atoms = set(bipy_indices) - isolated_atoms
    
    return list(valid_bipy_atoms)

def create_mirror_plane(ni_coord, n_coords):
    """Create a mirror plane that swaps N1↔N2.
    The plane is perpendicular to the N1-N2 line and passes through Ni."""
    
    # Vector from N1 to N2
    n1_to_n2 = n_coords[1] - n_coords[0]
    
    # The mirror plane normal is perpendicular to the N1-N2 line
    # So the normal is parallel to the N1-N2 vector
    mirror_normal = n1_to_n2 / np.linalg.norm(n1_to_n2)
    
    # The plane passes through Ni
    mirror_point = ni_coord
    
    return mirror_normal, mirror_point

def reflect_point_through_plane(point, plane_normal, plane_point):
    """Reflect a point through a plane defined by a normal and a point."""
    # Distance from point to plane
    d = np.dot(plane_normal, point - plane_point)
    # Reflected point
    reflected = point - 2 * d * plane_normal
    return reflected

def create_mirrored_geometry(atoms, coords, ni_idx, n_indices, bipy_indices):
    """Create a new geometry with the bipyridine mirrored."""
    
    ni_coord = coords[ni_idx]
    n_coords = [coords[idx] for idx in n_indices]
    
    # Define mirror plane
    mirror_normal, mirror_point = create_mirror_plane(ni_coord, n_coords)
    
    # Create new coordinates
    new_coords = coords.copy()
    
    # Mirror only bipyridine atoms
    for idx in bipy_indices:
        new_coords[idx] = reflect_point_through_plane(coords[idx], mirror_normal, mirror_point)
    
    return new_coords

def classify_geometry(angle):
    """Classify geometry based on plane angle.
    Square planar: ~0°
    Tetrahedral: ~90°
    """
    if angle < 30:
        return "square_planar"
    elif angle > 60:
        return "tetrahedral"
    else:
        return "intermediate"

def parse_filename_for_asymmetry(filename):
    """Parse filename to determine if bipyridine or alkyne ligands are asymmetric.
    
    Filename format: bipy-{substituents}_{alkyne}_{type}.xyz
    Example: bipy-maaaaaaa_o-C2H2-f_reactant.xyz
    
    Returns: (bipy_asymmetric, alkyne_asymmetric)
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    # Extract bipyridine substituents (after 'bipy-')
    bipy_part = parts[0]
    if 'bipy-' in bipy_part:
        bipy_substituents = bipy_part.split('bipy-')[1]
        # Check if palindrome (symmetric)
        bipy_asymmetric = bipy_substituents != bipy_substituents[::-1]
    else:
        bipy_asymmetric = False
    
    # Extract alkyne substituents
    alkyne_asymmetric = False
    if len(parts) > 1:
        alkyne_part = parts[1]
        # Format: left-C2H2-right
        if '-C2H2-' in alkyne_part:
            alkyne_parts = alkyne_part.split('-C2H2-')
            if len(alkyne_parts) == 2:
                left, right = alkyne_parts
                alkyne_asymmetric = left != right
    
    return bipy_asymmetric, alkyne_asymmetric

def get_alkyne_functional_group_info(group_letter):
    """Get information about alkyne functional groups from letter code.
    
    Returns: (name, expected_non_H_atom_count)
    """
    groups = {
        'a': ('H', 0),                    # Just H
        'b': ('NO2', 2),                  # N + 2 O
        'c': ('Cl', 1),                   # 1 Cl
        'd': ('OCH3', 2),                 # O + 1 C
        'e': ('N(CH3)2', 3),              # N + 2 C
        'f': ('CH3', 1),                  # 1 C
        'g': ('CH2CH3', 2),               # 2 C
        'h': ('C(O)OH', 3),               # 2 C + 2 O (one is carboxylic)
        'i': ('OCOCH3', 4),               # 2 O + 2 C
        'j': ('F', 1),                    # 1 F
        'k': ('CF3', 4),                  # 1 C + 3 F
        'l': ('C(CH3)3', 4),              # 4 C
        'm': ('OH', 1),                   # 1 O
        'n': ('NH2', 1),                  # 1 N
        'o': ('phenyl', 6),               # 6 C
        'p': ('-CH2-phenyl-OCH3', 9),    # 8 C + 1 O
        'q': ('-CH2-phenyl-OCF3', 11),   # 8 C + 1 O + 3 F (though OCF3 structure unclear)
    }
    return groups.get(group_letter, ('unknown', 0))

def validate_alkyne_assignment(filename, atoms, bipy_indices):
    """Validate that bipyridine assignment doesn't incorrectly include alkyne atoms.
    
    Uses filename to determine expected alkyne functional groups.
    Returns: (is_valid, warning_message)
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('_')
    
    if len(parts) < 2:
        return True, None
    
    alkyne_part = parts[1]
    if '-C2H2-' not in alkyne_part:
        return True, None
    
    alkyne_parts = alkyne_part.split('-C2H2-')
    if len(alkyne_parts) != 2:
        return True, None
    
    left_letter, right_letter = alkyne_parts
    left_name, left_atoms = get_alkyne_functional_group_info(left_letter)
    right_name, right_atoms = get_alkyne_functional_group_info(right_letter)
    
    # Expected alkyne ligand atoms: 2 C (from C2H2) + functional groups
    # Plus 2 H on the alkyne carbons (unless substituted)
    expected_alkyne_non_H = 2 + left_atoms + right_atoms
    
    # Count atoms not in bipyridine and not Ni
    non_bipy_atoms = [i for i in range(len(atoms)) if i not in bipy_indices and atoms[i] != 'Ni']
    
    # Count non-H atoms in alkyne region
    alkyne_non_h = [i for i in non_bipy_atoms if atoms[i] != 'H']
    
    # Check if we have approximately the right number (allow some tolerance for H counting)
    if len(alkyne_non_h) < expected_alkyne_non_H - 2:
        warning = f"Warning: Expected ~{expected_alkyne_non_H} non-H atoms in alkyne ({left_name}-C2H2-{right_name}), but found {len(alkyne_non_h)}. May be including alkyne atoms in bipyridine."
        return False, warning
    
    return True, None

def try_alternative_thresholds(filename, atoms, coords, n_indices):
    """Try different connectivity thresholds to find best bipyridine assignment.
    
    Returns: (best_bipy_indices, threshold_used, validation_passed)
    """
    # Try different threshold multipliers
    multipliers_to_try = [1.0, 0.95, 0.90, 0.85, 1.05, 1.10]
    
    best_result = None
    best_score = float('inf')  # Lower is better (difference from expected)
    
    for multiplier in multipliers_to_try:
        bipy_indices = identify_bipyridine_atoms(atoms, coords, n_indices, threshold_multiplier=multiplier)
        is_valid, warning = validate_alkyne_assignment(filename, atoms, bipy_indices)
        
        if is_valid:
            # Found a valid assignment!
            return bipy_indices, multiplier, True
        
        # Calculate how far off we are (rough score)
        non_bipy = [i for i in range(len(atoms)) if i not in bipy_indices and atoms[i] != 'Ni']
        alkyne_non_h = [i for i in non_bipy if atoms[i] != 'H']
        
        # Extract expected count from warning message if possible
        base_name = os.path.splitext(os.path.basename(filename))[0]
        parts = base_name.split('_')
        if len(parts) >= 2:
            alkyne_part = parts[1]
            if '-C2H2-' in alkyne_part:
                alkyne_parts = alkyne_part.split('-C2H2-')
                if len(alkyne_parts) == 2:
                    left_letter, right_letter = alkyne_parts
                    left_name, left_atoms = get_alkyne_functional_group_info(left_letter)
                    right_name, right_atoms = get_alkyne_functional_group_info(right_letter)
                    expected_alkyne_non_H = 2 + left_atoms + right_atoms
                    
                    score = abs(len(alkyne_non_h) - expected_alkyne_non_H)
                    
                    if score < best_score:
                        best_score = score
                        best_result = (bipy_indices, multiplier, False)
    
    # Return best attempt even if none passed validation
    if best_result:
        return best_result
    else:
        # Fallback to default
        return identify_bipyridine_atoms(atoms, coords, n_indices, 1.0), 1.0, False

def should_create_mirror(filename, angle, tolerance=10.0):
    """Determine if a mirrored geometry should be created.
    
    Only create mirror if:
    1. BOTH bipyridine AND alkyne ligands are asymmetric
    2. The angle is not ~90° (within tolerance)
    """
    # Check for asymmetry
    bipy_asym, alkyne_asym = parse_filename_for_asymmetry(filename)
    has_asymmetry = bipy_asym and alkyne_asym  # Both must be asymmetric
    
    # Check if angle is close to 90°
    is_perpendicular = abs(angle - 90.0) < tolerance
    
    return has_asymmetry and not is_perpendicular

def analyze_structure(filename):
    """Analyze a single xyz file."""
    atoms, coords, comment = read_xyz(filename)
    
    # Find Ni atom
    ni_indices = [i for i, atom in enumerate(atoms) if atom == 'Ni']
    if len(ni_indices) == 0:
        print(f"No Ni atom found in {filename}")
        return None
    
    ni_idx = ni_indices[0]
    ni_coord = coords[ni_idx]
    
    # Find closest 2 nitrogens
    n_indices = find_closest_atoms(ni_coord, 'N', atoms, coords, n=2)
    
    # Find closest 2 carbons
    c_indices = find_closest_atoms(ni_coord, 'C', atoms, coords, n=2)
    
    # Get coordinates
    n_coords = [coords[idx] for idx in n_indices]
    c_coords = [coords[idx] for idx in c_indices]
    
    # Calculate plane normals
    # Plane 1: Ni + 2 closest N atoms
    normal1 = plane_normal(ni_coord, n_coords[0], n_coords[1])
    
    # Plane 2: Ni + 2 closest C atoms
    normal2 = plane_normal(ni_coord, c_coords[0], c_coords[1])
    
    # Calculate angle between planes
    angle = angle_between_planes(normal1, normal2)
    
    # Classify geometry
    geometry_type = classify_geometry(angle)
    
    # Print distances for verification
    n_dists = [np.linalg.norm(coords[idx] - ni_coord) for idx in n_indices]
    c_dists = [np.linalg.norm(coords[idx] - ni_coord) for idx in c_indices]
    
    # Identify bipyridine atoms
    bipy_indices = identify_bipyridine_atoms(atoms, coords, n_indices)
    
    # Validate alkyne assignment
    is_valid, warning = validate_alkyne_assignment(filename, atoms, bipy_indices)
    
    # If validation fails, try alternative thresholds
    threshold_used = 1.0
    auto_corrected = False
    if not is_valid and warning:
        bipy_indices_alt, threshold_alt, validation_passed = try_alternative_thresholds(filename, atoms, coords, n_indices)
        
        if validation_passed:
            # Successfully found a better threshold!
            bipy_indices = bipy_indices_alt
            threshold_used = threshold_alt
            auto_corrected = True
            warning = None  # Clear the warning since we fixed it
        elif threshold_alt != 1.0:
            # Didn't pass validation but found a better assignment
            bipy_indices = bipy_indices_alt
            threshold_used = threshold_alt
            auto_corrected = True
            # Re-validate to get updated warning message
            is_valid, warning = validate_alkyne_assignment(filename, atoms, bipy_indices)
    
    return {
        'filename': filename,
        'ni_index': ni_idx,
        'n_indices': n_indices,
        'c_indices': c_indices,
        'n_distances': n_dists,
        'c_distances': c_dists,
        'angle': angle,
        'geometry_type': geometry_type,
        'normal1': normal1,
        'normal2': normal2,
        'bipy_indices': bipy_indices,
        'atoms': atoms,
        'coords': coords,
        'comment': comment,
        'validation_warning': warning,
        'threshold_used': threshold_used,
        'auto_corrected': auto_corrected
    }

if __name__ == '__main__':
    # Get all xyz files in current directory (exclude previously generated files)
    all_files = glob.glob('*.xyz')
    xyz_files = [f for f in all_files if not f.endswith('_mirrored.xyz') and 
                 f not in ['all_originals.xyz', 'all_mirrored.xyz']]
    
    # Determine expected geometry from directory path
    cwd = os.getcwd()
    if 'square_planar' in cwd:
        expected_geometry = 'square_planar'
    elif 'tetrahedral' in cwd:
        expected_geometry = 'tetrahedral'
    elif 'seesaw' in cwd:
        expected_geometry = 'seesaw'
    else:
        expected_geometry = 'unknown'
    
    print(f"Found {len(xyz_files)} xyz file(s)")
    print(f"Expected geometry from directory: {expected_geometry}\n")
    
    flagged_files = []
    validation_warnings = []  # Store files with validation warnings
    auto_corrected_files = []  # Store files that were auto-corrected
    mirrored_structures = []  # Store (original_atoms, original_coords, mirrored_coords, comment) for mirrored structures
    
    for xyz_file in sorted(xyz_files):
        print(f"Analyzing: {xyz_file}")
        print("=" * 60)
        
        result = analyze_structure(xyz_file)
        
        if result:
            print(f"Ni atom index: {result['ni_index']}")
            print("\nClosest 2 N atoms:")
            for i, (idx, dist) in enumerate(zip(result['n_indices'], result['n_distances'])):
                print(f"  N {i+1}: atom index {idx}, distance = {dist:.4f} Å")
            
            print("\nClosest 2 C atoms:")
            for i, (idx, dist) in enumerate(zip(result['c_indices'], result['c_distances'])):
                print(f"  C {i+1}: atom index {idx}, distance = {dist:.4f} Å")
            
            print(f"\nPlane 1 normal (Ni + 2 N): {result['normal1']}")
            print(f"Plane 2 normal (Ni + 2 C): {result['normal2']}")
            print(f"\nAngle between planes: {result['angle']:.2f}°")
            print(f"Classified geometry: {result['geometry_type']}")
            
            # Check for inconsistency
            if expected_geometry != 'unknown' and result['geometry_type'] != expected_geometry:
                print("\n⚠️  WARNING: Geometry mismatch!")
                print(f"   Expected: {expected_geometry}, Found: {result['geometry_type']}")
                flagged_files.append((xyz_file, expected_geometry, result['geometry_type']))
            
            # Identify and print bipyridine atoms
            print(f"\nBipyridine ligand contains {len(result['bipy_indices'])} atoms")
            print(f"Indices: {sorted(result['bipy_indices'])}")
            
            # Print threshold info if auto-corrected
            if result.get('auto_corrected'):
                print(f"  ✓ Auto-corrected using threshold multiplier: {result['threshold_used']:.2f}")
                auto_corrected_files.append((xyz_file, result['threshold_used']))
            
            # Print validation warning if present
            if result.get('validation_warning'):
                print(f"\n⚠️  {result['validation_warning']}")
                validation_warnings.append((xyz_file, result['validation_warning']))
            
            # Check if we should create a mirrored geometry
            bipy_asym, alkyne_asym = parse_filename_for_asymmetry(xyz_file)
            should_mirror = should_create_mirror(xyz_file, result['angle'])
            
            print("\nAsymmetry analysis:")
            print(f"  Bipyridine asymmetric: {bipy_asym}")
            print(f"  Alkyne asymmetric: {alkyne_asym}")
            print(f"  Angle close to 90°: {abs(result['angle'] - 90.0) < 10.0}")
            print(f"  Should create mirror: {should_mirror}")
            
            # Create mirrored geometry only if needed
            if should_mirror:
                mirrored_coords = create_mirrored_geometry(
                    result['atoms'], 
                    result['coords'], 
                    result['ni_index'],
                    result['n_indices'],
                    result['bipy_indices']
                )
                
                # Save mirrored geometry
                base_name = os.path.splitext(xyz_file)[0]
                mirrored_filename = f"{base_name}_mirrored.xyz"
                write_xyz(
                    mirrored_filename, 
                    result['atoms'], 
                    mirrored_coords, 
                    result['comment'] + " (mirrored bipyridine)"
                )
                print(f"\n✓ Created mirrored geometry: {mirrored_filename}")
                
                # Store for concatenated files
                mirrored_structures.append({
                    'filename': xyz_file,
                    'atoms': result['atoms'],
                    'original_coords': result['coords'],
                    'mirrored_coords': mirrored_coords,
                    'comment': result['comment']
                })
            else:
                reason = []
                if not (bipy_asym and alkyne_asym):
                    if not bipy_asym and not alkyne_asym:
                        reason.append("both ligands symmetric")
                    elif not bipy_asym:
                        reason.append("bipyridine symmetric")
                    elif not alkyne_asym:
                        reason.append("alkyne symmetric")
                if abs(result['angle'] - 90.0) < 10.0:
                    reason.append("perpendicular geometry (~90°)")
                print(f"\n⊘ Skipped mirror creation: {', '.join(reason)}")
        
        print("\n")
    
    # Summary of flagged files
    if flagged_files:
        print("=" * 60)
        print("SUMMARY: Flagged Files with Geometry Mismatches")
        print("=" * 60)
        
        # Create subdirectories for mismatched geometries
        tetrahedral_dir = "flagged_tetrahedral"
        intermediate_dir = "flagged_intermediate"
        
        os.makedirs(tetrahedral_dir, exist_ok=True)
        os.makedirs(intermediate_dir, exist_ok=True)
        
        moved_count = {'tetrahedral': 0, 'intermediate': 0}
        
        for filename, expected, found in flagged_files:
            print(f"  {filename}")
            print(f"    Expected: {expected}, Found: {found}")
            
            # Move file to appropriate subdirectory
            if found == 'tetrahedral':
                dest_dir = tetrahedral_dir
                moved_count['tetrahedral'] += 1
            elif found == 'intermediate':
                dest_dir = intermediate_dir
                moved_count['intermediate'] += 1
            else:
                continue
            
            # Move the file
            src = filename
            dst = os.path.join(dest_dir, filename)
            try:
                os.rename(src, dst)
            except Exception as e:
                print(f"    Error moving file: {e}")
        
        print()
        print(f"Moved {moved_count['tetrahedral']} files to {tetrahedral_dir}/")
        print(f"Moved {moved_count['intermediate']} files to {intermediate_dir}/")
        print()
    
    # Summary of validation warnings
    if validation_warnings:
        print("=" * 60)
        print("SUMMARY: Files with Validation Warnings")
        print("=" * 60)
        print(f"Found {len(validation_warnings)} file(s) with potential ligand assignment issues:\n")
        
        # Create subdirectory for validation warnings
        validation_dir = "validation_warnings"
        os.makedirs(validation_dir, exist_ok=True)
        
        moved_count = 0
        
        for filename, warning in validation_warnings:
            print(f"  {filename}")
            print(f"    {warning}")
            
            # Move the file
            src = filename
            dst = os.path.join(validation_dir, filename)
            try:
                os.rename(src, dst)
                moved_count += 1
            except FileNotFoundError:
                # File might have already been moved to flagged directory
                print("    Note: File not found (may have been moved to flagged directory)")
            except Exception as e:
                print(f"    Error moving file: {e}")
            print()
        
        print(f"Moved {moved_count} files to {validation_dir}/")
        print()
    
    # Summary of auto-corrected files
    if auto_corrected_files:
        print("=" * 60)
        print("SUMMARY: Auto-Corrected Files")
        print("=" * 60)
        print(f"Successfully auto-corrected {len(auto_corrected_files)} file(s) by adjusting connectivity thresholds:\n")
        
        for filename, threshold in auto_corrected_files:
            print(f"  {filename}")
            print(f"    Threshold multiplier: {threshold:.2f}")
        print()
    
    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total files analyzed: {len(xyz_files)}")
    print(f"Geometry mismatches: {len(flagged_files)}")
    print(f"Validation warnings (after auto-correction): {len(validation_warnings)}")
    print(f"Auto-corrected files: {len(auto_corrected_files)}")
    print(f"Mirrored structures created: {len(mirrored_structures)}")
    print()
    
    # Create concatenated XYZ files for molden viewing
    if mirrored_structures:
        print("=" * 60)
        print(f"Creating concatenated XYZ files for {len(mirrored_structures)} mirrored structures")
        print("=" * 60)
        
        # Write all originals to one file
        with open('all_originals.xyz', 'w') as f:
            for struct in mirrored_structures:
                f.write(f"{len(struct['atoms'])}\n")
                f.write(f"{struct['comment']} - {struct['filename']}\n")
                for atom, coord in zip(struct['atoms'], struct['original_coords']):
                    f.write(f"{atom:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}\n")
        
        # Write all mirrored to another file
        with open('all_mirrored.xyz', 'w') as f:
            for struct in mirrored_structures:
                f.write(f"{len(struct['atoms'])}\n")
                f.write(f"{struct['comment']} (mirrored) - {struct['filename']}\n")
                for atom, coord in zip(struct['atoms'], struct['mirrored_coords']):
                    f.write(f"{atom:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}\n")
        
        print(f"✓ Created all_originals.xyz with {len(mirrored_structures)} structures")
        print(f"✓ Created all_mirrored.xyz with {len(mirrored_structures)} structures")
        print("\nYou can now open both files in molden and scroll through them to compare.")
        print()
