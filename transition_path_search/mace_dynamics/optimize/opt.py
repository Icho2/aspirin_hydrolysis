from ase.constraints import FixAtoms, Hookean
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_off

# 1. Read structure
atoms = read("cluster_1_15waters.pdb")

# 2. Identify aspirin and water oxygen indices (0-based)
aspirin_indices = list(range(21))
water_o_indices = [
    atom.index for atom in atoms if atom.symbol == "O" and atom.index >= 21
]

# 3. Get aspirin's center of mass coordinates [x, y, z]
aspirin_com = atoms[aspirin_indices].get_center_of_mass()

# 4. Anchor the spherical wall to aspirin's COM coordinate
# (Keep one central aspirin carbon fixed so aspirin stays centered in the sphere)
constraints = [FixAtoms(indices=[0])]
constraints += [
    Hookean(a1=o_idx, a2=aspirin_com, rt=6.5, k=2.0)
    for o_idx in water_o_indices
]
atoms.set_constraint(constraints)

# 5. Run optimization
atoms.calc = mace_off(model="medium", device="cuda")
opt = BFGS(atoms, trajectory="optimization.traj")
opt.run(fmax=0.01)

write("cluster_0_15waters_opted.xyz", atoms)

