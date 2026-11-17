from ase.io import Trajectory

# 1. Open the original trajectory file in read mode
input_traj = Trajectory("NEB_aspirin_hydrolysis.traj", "r")

# 2. Open a new trajectory file in write mode
output_traj = Trajectory("NEB_aspirin_hydrolysis_cut.traj", "w")

# Define how often to save a frame (e.g., keep every 10th frame)
stride = 5

# 3. Iterate through the frames using Python slicing or a step loop
for atoms in input_traj[::stride]:
    output_traj.write(atoms)

# Optional: Close files explicitly (good practice)
input_traj.close()
output_traj.close()

print(f"Successfully reduced trajectory from {len(input_traj)} to {len(input_traj[::stride])} frames.")
