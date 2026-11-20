from ase import io
from ase.mep import NEB
from ase.optimize import MDMin
from mace.calculators import mace_off

# Read initial and final states:
traj_file = io.read('NEB_new_guess.traj', index=":")
# Make a band consisting of 5 images:
images = [traj_file[0]]
images += [traj_file[0].copy(), traj_file[0].copy(), traj_file[0].copy()]
images += [traj_file[1]]
neb = NEB(images)
# Interpolate linearly the potisions of the three middle images:
neb.interpolate(apply_constraint=True)
# Set calculators:
for image in images[0:5]:
    image.calc = mace_off(model="medium", device="cpu")
# Optimize:
optimizer = MDMin(neb, trajectory='A2B.traj')
optimizer.run(fmax=0.04)
