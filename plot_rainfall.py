import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === Constants ===
F = 0.005  # sensor area (m²)
vDi = np.array([0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0,
                13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0])  # fall velocity (example)
delta_Di = np.array([0.1] * 20)  # bin width (example)
Di = np.linspace(0.25, 5.5, 20)  # drop diameters (mm)

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

    # Add only nonzero intervals
    df_valid = df[df[drop_cols].sum(axis=1) > 0]
    if not df_valid.empty:
        total_counts += df_valid[drop_cols].sum().values
        total_time += df_valid["Interval [s]"].sum()

# === Compute Overall N(D) ===
NDi = total_counts / (F * total_time * vDi * delta_Di)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(Di, NDi, marker='o', color='blue', linewidth=2, label='Overall N(D)')
plt.yscale('log')  
plt.xlabel("Drop Diameter D (mm)", fontsize=12)
plt.ylabel("Number Density N(D) [1/(m³·mm)]", fontsize=12)
plt.title("Drop Size Distribution N(D)", fontsize=14)
plt.grid(True, which='both', linestyle='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
