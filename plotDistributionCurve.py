import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os



F = 0.005  # sensor area (m²)

# Average diameter Di (mm)
Di = np.array([
    0.359, 0.455, 0.551, 0.656, 0.771,
    0.917, 1.131, 1.331, 1.506, 1.665,
    1.912, 2.259, 2.589, 2.869, 3.205,
    3.544, 3.916, 4.350, 4.859, 5.373
])

# Fall velocity v(Di) (m/s)
vDi = np.array([
    1.435, 1.862, 2.267, 2.692, 3.154,
    3.717, 4.382, 4.986, 5.423, 5.907,
    6.315, 7.009, 7.546, 7.903, 8.258,
    8.556, 6.784, 8.965, 9.076, 9.137
])

# Diameter interval Delta_Di (mm)
delta_Di = np.array([
    0.092, 0.100, 0.091, 0.119, 0.112,
    0.120, 0.233, 0.197, 0.153, 0.133,
    0.197, 0.364, 0.223, 0.284, 0.374,
    0.319, 0.429, 0.446, 0.266, 0.455
])

# === Load all CSV files ===
path = "data/"
all_files = glob.glob(os.path.join(path, "RD-*.csv"))

if not all_files:
    raise FileNotFoundError("⚠️ No CSV files found in the 'data/' folder!")

total_counts = np.zeros(20)
total_time = 0

for file in all_files:
    df = pd.read_csv(file)
    drop_cols = [f"n{i}" for i in range(1, 21)]

    
    df_valid = df[df[drop_cols].sum(axis=1) > 0]

    if not df_valid.empty:
        total_counts += df_valid[drop_cols].sum().values
        total_time += df_valid["Interval [s]"].sum()

# === Compute Overall N(D) ===
# N(Di) = ni / (F * total_time * v(Di) * ΔDi)
NDi = total_counts / (F * total_time * vDi * delta_Di)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(Di, NDi, marker='o', linewidth=2, label='Overall N(D)')

plt.yscale('log')
plt.xlabel("Drop Diameter D (mm)", fontsize=12)
plt.ylabel("Number Density N(D) [1/(m³·mm)]", fontsize=12)
plt.title("Drop Size Distribution N(D)", fontsize=14)
plt.grid(True, which='both', linestyle='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
