from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.calculators.plumed import Plumed
import numpy as np
import time
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
from ase.md import MDLogger

calculator = MACECalculator(model_path='/home/ichop/mace/mace/calculators/foundations_models/2023-12-03-mace-mp.model', device='cuda')
init_conf = read('cluster_0_1waters.xyz', '0')
MaxwellBoltzmannDistribution(init_conf, temperature_K=300)

timestep = 0.5

plumed_setup = [
    # Align PLUMED units with ASE standard units (Angstrom, fs, kJ/mol)
    f"UNITS LENGTH=A TIME={1/(1000 * units.fs)} ENERGY={units.mol/units.kJ}",

    # Define a Collective Variable (CV) — for example, a distance between atoms 1 and 2
    "d: DISTANCE ATOMS=5,7 NOPBC",

    "c_form: COORDINATION GROUPA=5 GROUPB=22 R_0=1.9 NN=6 MM=12 NOPBC",

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
    atoms=init_conf
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

dyn.run(5000)
