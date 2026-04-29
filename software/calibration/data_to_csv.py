from pathlib import Path

import kineticstoolkit as ktk
from nextwheel import read_dat
import pandas as pd
import numpy as np

# Find matrix A and offsets
ROOT = Path.cwd()
directory = ROOT / "matrix_A_offset"
z = np.load(ROOT/"matrix_A_offset.npz")
A = z["A"]
off = np.load(directory/"offset.npz")
print(off)
offset = off["offset"]
print(offset)

encoder_variation = 0.087890625  # degree by click
wheel_diameter = 0.6096  # m




