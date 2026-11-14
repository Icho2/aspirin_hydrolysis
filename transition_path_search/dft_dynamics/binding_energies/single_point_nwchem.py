from ase import Atoms
from ase.calculators.nwchem import NWChem
from ase.optimize import BFGS
from ase.io import read, write

# 1. Define the geometry of the molecule
atoms = read("cluster_0_15waters_opted.xyz")

# 2. Configure the NWChem calculator
# NWChem-specific blocks (like dft, scf, mp2) are passed via dictionaries
calc = NWChem(
    label='calc/CO_init_image',
    memory='1024 mb',
    theory='dft',
    basis='6-31+G*',
    dft=dict(
        xc='B3LYP',
        maxiter=2000,
        convergence=dict(energy='1e-5', density='1e-4')
    )
)

# 3. Attach the calculator to the Atoms object
atoms.calc = calc

# 4. Run the single-point energy calculation
energy = atoms.get_potential_energy()

# 5. Print the result (ASE automatically converts Hartrees to eV)
print(f'Single-Point Energy: {energy:.6f} eV')
