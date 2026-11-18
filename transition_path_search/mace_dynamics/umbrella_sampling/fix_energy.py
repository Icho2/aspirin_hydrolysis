from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

# Load all frames from your file
frames = read('traj_window_00_cv_-4.07.traj', index=':')

for atoms in frames:
  if atoms.calc is not None and 'energy' in atoms.calc.results:
    # Extract scalar float from the numpy array
    energy_val = float(atoms.get_potential_energy().item())
    forces_val = (
        atoms.get_forces() if 'forces' in atoms.calc.results else None
    )

    # Re-attach a clean SinglePointCalculator with scalar energy
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy_val, forces=forces_val
    )

  # Also check atoms.info in case energy is stored there (common in extxyz)
  if 'energy' in atoms.info:
    atoms.info['energy'] = float(atoms.info['energy'])

write('traj_window_00_cv_-4.07.traj', frames)
