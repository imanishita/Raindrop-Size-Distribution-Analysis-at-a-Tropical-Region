import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Sensor area (m²)
F = 0.005  

# Define 20-class mean diameters Di (mm)
Di = np.array([
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5,
    4.0, 4.5, 5.0, 5.5, 6.0
])

# === Load all files ===
path = "data/"
files = sorted(glob.glob(os.path.join(path, "RD-*.csv")))

all_times = []
all_RI = []

for file in files:
    df = pd.read_csv(file)

    # Identify drop count columns (n1–n20)
    drop_cols = [f"n{i}" for i in range(1, 21)]

    # Extract counts
    N = df[drop_cols].values  # shape: (rows, 20)

    # Extract time interval
    t = df["Interval [s]"].values  # (rows,)

    # Compute RI for each row
    # RI formula: (π/6)*3.6e3*(1/(F*t))*Σ(n_i * D_i^3)
    RI = (np.pi / 6) * 3.6e3 * (1 / (F * t)) * np.sum(N * (Di ** 3), axis=1)

    # Store results
    all_RI.extend(RI)
    all_times.extend(pd.to_datetime(df["YYYY-MM-DD"] + " " + df["hh:mm:ss"]))

# ========= Plotting Rain Intensity ==========
plt.figure(figsize=(12, 5))
plt.plot(all_times, all_RI)
plt.xlabel("Time")
plt.ylabel("Rain Intensity (mm/h)")
plt.title("Rain Intensity Curve (RI vs Time)")
plt.grid(True)
plt.tight_layout()
plt.show()
