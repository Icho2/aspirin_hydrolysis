import pypackmol

# Define your Packmol input instructions as a string
packmol_input = """
tolerance 2.0
filetype pdb
output cluster_6waters.pdb

structure data/conformer_0.sdf
  number 1
  center
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure water.pdb
  number 6
  inside sphere 0. 0. 0. 6.0
end structure
"""

# Run Packmol directly from Python
pypackmol.packmol(packmol_input)
print("Solvation complete: cluster_6waters.pdb")

