from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms, Hookean
from mace.calculators import mace_off
from ase.md import MDLogger
import numpy as np
import os

# 1. Initialize MACE calculator (shared across all windows)
calculator = mace_off(model='medium', device='cuda')

# 2. Load NEB images/path (replace 'neb_path.traj' with your actual NEB file, 
#    or pass a list of image filenames like ['image_00.xyz', 'image_01.xyz', ...])
#neb_images = read('NEB_aspirin_hydrolysis.traj', index=':')  # Reads all images along the path
neb_images = read('traj_window_12_cv_-0.73.traj', index=':')  # Reads all images along the path

print(f"Loaded {len(neb_images)} NEB windows to run.")

# Common setup constants
kappa = 1000.0
total_steps = 10000  # Adjust steps per window as needed (e.g., 10k steps = 5 ps at 0.5 fs)
timestep = 0.5        # fs

os.makedirs("optimized_geometry_around_ts", exist_ok=True)
os.makedirs("colvar_around_ts", exist_ok=True)

CVs = np.round(np.arange(-0.75, 0.75, 0.05), 2)

#for i, img in enumerate(neb_images):
for i in CVs:
    print(f"\n--- Starting Umbrella Window {i+1} of {len(neb_images)} ---")
    
    # Calculate the CV value directly from this specific NEB image geometry
    # d_break: distance between atoms 5 and 8
    pos = neb_images[0].get_positions()
    #d_break_val = np.linalg.norm(pos[4] - pos[6])
    
    # d_form: distance between atoms 5 and 52
    #d_form_val = np.linalg.norm(pos[4] - pos[21])
    
    # Target CV defined as d_break - d_form (matching your PLUMED setup)
    #target_cv = d_break_val - d_form_val
    target_cv = i
    #print(f"Image {i}: d_break={d_break_val:.3f}, d_form={d_form_val:.3f} -> Target CV={target_cv:.3f}")

    # Set up constraints (same as your original script)
    aspirin_indices = list(range(21))
    water_o_indices = [
        atom.index for atom in neb_images[0] if atom.symbol == "O" and atom.index >= 21
    ]
    aspirin_com = neb_images[0][aspirin_indices].get_center_of_mass()

    constraints = [FixAtoms(indices=[0])]
    constraints += [
        Hookean(a1=o_idx, a2=aspirin_com, rt=6.5, k=2.0)
        for o_idx in water_o_indices
    ]
    neb_images[0].set_constraint(constraints)

    # Initialize velocities
    MaxwellBoltzmannDistribution(neb_images[0], temperature_K=300)

    # Unique filenames for this window based on its target CV
    colvar_filename = f"colvar_around_ts/COLVAR_window_{i:.2f}_cv_{target_cv:.2f}.dat"
    traj_filename = f"optimized_geometry_around_ts/geom_traj_window_{i:.2f}_cv_{target_cv:.2f}.traj"
    #colvar_filename = f"COLVAR_window_36_cv_{target_cv:.2f}.dat"
    #traj_filename = f"traj_window_36_cv_{target_cv:.2f}.traj"

    # PLUMED configuration for this specific window
    plumed_setup = [
        "UNITS LENGTH=A TIME=ps ENERGY=kj/mol",
        f"d_break: DISTANCE ATOMS=5,7",
        f"d_form: DISTANCE ATOMS=5,22",
        "cv: COMBINE ARG=d_break,d_form COEFFICIENTS=1.0,-1.0 PERIODIC=NO",
        f"restraint: RESTRAINT ARG=cv AT={target_cv:.4f} KAPPA={kappa}",
        "com_asp: COM ATOMS=1-21",
        "d_wat: DISTANCE ATOMS=com_asp,22",
        "uwall: UPPER_WALLS ARG=d_wat AT=6.0 KAPPA=100.0",
        f"PRINT STRIDE=10 ARG=cv,d_break,d_form,restraint.bias FILE={colvar_filename}"
    ]

    # Attach PLUMED calculator
    neb_images[0].calc = Plumed(
        calc=calculator,
        input=plumed_setup,
        timestep=timestep * units.fs,
        kT=300 * units.kB,
        atoms=neb_images[0],
        log=f"logss/plumed_window_{i:.2f}.log"
    )

    # Setup Geometry Optimization
    from ase.optimize import BFGS

    opt = BFGS(neb_images[0], trajectory=traj_filename, logfile="opt.log")


    # Run the umbrella sampling simulation for this window
    try:
        opt.run(fmax=0.01)
        write(f"optimized_geometry_around_ts/window_{i:.2f}_cv_{target_cv:.2f}.xyz", neb_images[0])
        print(f"Successfully finished window {i+1}")
    except Exception as e:
        print(f"Error encountered in window {i+1}: {e}")
        continue
