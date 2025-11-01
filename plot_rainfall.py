import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === Constants ===
F = 0.005  # sensor area (m²)
vDi = np.array([0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0,
                13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0])  # fall velocities (example)
delta_Di = np.array([0.1] * 20)  # bin width (example)
Di = np.linspace(0.25, 5.5, 20)  # drop diameters (mm)

# === Load all CSV files ===
path = "data/"
all_files = glob.glob(os.path.join(path, "RD-*.csv"))

if not all_files:
    raise FileNotFoundError("No CSV files found in the data folder!")

# Initialize total counts
total_counts = np.zeros(20)
total_time = 0

for file in all_files:
    df = pd.read_csv(file)
    drop_cols = [f"n{i}" for i in range(1, 21)]

    # Sum all drop counts
    total_counts += df[drop_cols].sum().values
    total_time += df["Interval [s]"].sum()

# === Compute overall N(D) ===
NDi = total_counts / (F * total_time * vDi * delta_Di)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(Di, NDi, marker='o', color='b', linewidth=2)
plt.yscale('log')
plt.xlabel("Drop Diameter D (mm)")
plt.ylabel("Number Density N(D) [1/(m³·mm)]")
plt.title("Overall Drop Size Distribution N(D)")
plt.grid(True, which='both', linestyle='--', lw=0.5)
plt.show()
