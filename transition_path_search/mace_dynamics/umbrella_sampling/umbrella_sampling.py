from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.calculators.plumed import Plumed
import numpy as np
import time
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
from ase.md import MDLogger
from mace.calculators import mace_off
from ase.constraints import FixAtoms, Hookean


calculator = mace_off(model='medium', device='cuda')
init_conf = read('cluster_0_15waters_opted.xyz')


aspirin_indices = list(range(21))
water_o_indices = [
    atom.index for atom in init_conf if atom.symbol == "O" and atom.index >= 21
]

# 3. Get aspirin's center of mass coordinates [x, y, z]
aspirin_com = init_conf[aspirin_indices].get_center_of_mass()

# 4. Anchor the spherical wall to aspirin's COM coordinate
# (Keep one central aspirin carbon fixed so aspirin stays centered in the sphere)
constraints = [FixAtoms(indices=[0])]
constraints += [
    Hookean(a1=o_idx, a2=aspirin_com, rt=6.5, k=2.0)
    for o_idx in water_o_indices
]

init_conf.set_constraint(constraints)

MaxwellBoltzmannDistribution(init_conf, temperature_K=300)

timestep = 0.5

target_cv = -0.5
kappa = 200

plumed_setup = [
    "UNITS LENGTH=A TIME=ps ENERGY=kj/mol",
    f"d_break: DISTANCE ATOMS=5,8",
    f"d_form: DISTANCE ATOMS=5,52",
    "cv: COMBINE ARG=d_break,d_form COEFFICIENTS=1.0,-1.0 PERIODIC=NO",
    # Harmonic umbrella restraint
    f"restraint: RESTRAINT ARG=cv AT={target_cv} KAPPA={kappa}",
    # Optional: Prevent the 15 water molecules (e.g., atoms 22 to 66) from evaporating
    "com_asp: COM ATOMS=1-21",
    "d_wat: DISTANCE ATOMS=com_asp,22",
    "uwall: UPPER_WALLS ARG=d_wat AT=6.0 KAPPA=100.0",
    # Output CV values every 10 steps for WHAM/MBAR analysis
    f"PRINT STRIDE=10 ARG=cv,d_break,d_form,restraint.bias FILE=COLVAR_{target_cv:.2f}.dat"
]

init_conf.calc = Plumed(
    calc=calculator,
    input=plumed_setup,
    timestep=0.5 * units.fs,
    kT=300 * units.kB,        # Thermal energy in eV units (crucial for Well-Tempered MetaD)
    atoms=init_conf,
    log="plumed.log"
)

dyn = Langevin(
    init_conf,
    timestep=0.5 * units.fs,
    temperature_K=300,  # Target temperature in Kelvin
    friction=0.01 / units.fs,  # Friction coefficient (0.002 to 0.02 / fs is typical)
    trajectory="md_bias_nvt.traj",  # Output trajectory file
    loginterval=1,
)

dyn.attach(
    MDLogger(dyn, init_conf, "-", header=True, stress=False, peratom=False),
    interval=1000,
)

dyn.run(1000000)
