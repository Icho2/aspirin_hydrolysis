 #!/Users/mauricioagurcia/my_venv/bin python3

import os
from rdkit import Chem
from rdkit.Chem import AllChem

m = Chem.MolFromMolFile('data/aspirin.mol')

mol = Chem.AddHs(m)

# Set up ETKDGv3 parameters for conformer generation
params = AllChem.ETKDGv3()
params.randomSeed = 42  # Ensures reproducibility
params.pruneRmsThresh = 0.5  # Drops conformers closer than 0.5 Å RMSD
params.numThreads = 0  # Uses all available CPU cores

# Generate 50 initial conformers
num_confs = 50
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

# Optimize conformers using the MMFF94 force field
# Returns a list of tuples: (not_converged_flag, energy_in_kcal_mol)
results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant="MMFF94s")

# Pair conformer IDs with their energies and sort from lowest to highest
conformer_energies = []
for conf_id, (not_converged, energy) in zip(conf_ids, results):
    conformer_energies.append((conf_id, energy))

conformer_energies.sort(key=lambda x: x[1])

# Print the lowest-energy conformer ID and its energy
lowest_id, lowest_energy = conformer_energies[0]
print(f"Lowest energy conformer ID: {lowest_id} ({lowest_energy:.2f} kcal/mol)")

for conf_id, energy in conformer_energies:
    # Build the file path (e.g., "data/conformer_0.sdf")
    file_path = os.path.join("data", f"conformer_{conf_id}.sdf")
    
    # Initialize a writer for this specific file
    writer = Chem.SDWriter(file_path)
    
    # Optional: Attach meta-data properties directly to the file
    mol.SetProp("_Name", f"Conformer_{conf_id}")
    mol.SetDoubleProp("Energy_kcal_mol", energy)
    
    # Write and close
    writer.write(mol, confId=conf_id)
    writer.close()

print(f"Successfully saved {len(conformer_energies)} conformers to the 'data/' directory.")
