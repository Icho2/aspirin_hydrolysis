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

plumed_setup = [
    "RESTART",
    # Align PLUMED units with ASE standard units (Angstrom, fs, kJ/mol)
    f"UNITS LENGTH=A TIME={1/(1000 * units.fs)} ENERGY={units.mol/units.kJ}",

    # Define a Collective Variable (CV) — for example, a distance between atoms 1 and 2
    "d: DISTANCE ATOMS=5,11 NOPBC",

    "c_form: COORDINATION GROUPA=5 GROUPB=22-66:3 R_0=2.8 NN=4 MM=8 NOPBC",

    # Configure Well-Tempered Metadynamics
    "METAD "
    "ARG=d "                  # Collective variable to bias
    "SIGMA=0.2 "              # Gaussian width (in units of the CV)
    "HEIGHT=1.2 "             # Initial Gaussian height (kJ/mol)
    "PACE=100 "               # Frequency (in steps) of depositing Gaussians
    "BIASFACTOR=15 "          # Well-tempered bias factor (DeltaT / T + 1)
    "TEMP=300 "               # Simulation temperature (K)
    "FILE=HILLS"              # Output file for deposited hills
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
    interval=10,
)

dyn.run(100000)
