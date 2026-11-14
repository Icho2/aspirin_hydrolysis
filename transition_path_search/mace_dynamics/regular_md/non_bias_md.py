from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.calculators.plumed import Plumed
import numpy as np
import time
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
from ase.calculators.mixing import LinearCombinationCalculator
from tblite.ase import TBLite
from mace.calculators import mace_off
from ase.md import MDLogger


#calculator = MACECalculator(model_path='/home/ichop/mace/mace/calculators/foundations_models/2023-12-03-mace-mp.model', device='cuda')
calculator = mace_off(model='medium', device='cuda')
init_conf = read('cluster_0_1waters.xyz', '0')
MaxwellBoltzmannDistribution(init_conf, temperature_K=300)

timestep = 0.5

xtb_solv = TBLite(method="GFN2-xTB", solvation=("alpb", "water"))
xtb_gas = TBLite(method="GFN2-xTB")

#solvated_calc = LinearCombinationCalculator(
#    [calculator, xtb_solv, xtb_gas], [1.0, 1.0, -1.0]
#)

#init_conf.calc = solvated_calc
init_conf.calc = calculator

dyn = Langevin(
    init_conf,
    timestep=0.5 * units.fs,
    temperature_K=300,  # Target temperature in Kelvin
    friction=0.01 / units.fs,  # Friction coefficient (0.002 to 0.02 / fs is typical)
    trajectory="md_nvt.traj",  # Output trajectory file
    loginterval=1,
)

dyn.attach(
    MDLogger(dyn, init_conf, "-", header=True, stress=False, peratom=False),
    interval=10,
)

dyn.run(5000)
