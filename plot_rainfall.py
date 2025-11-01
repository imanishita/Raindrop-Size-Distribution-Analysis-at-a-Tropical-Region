import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Constants
F = 0.005  # sensor area (m²)
path = "data/"  # folder containing your converted CSVs

# Typical diameter bins (mm) for 20-channel disdrometer
Di = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5,
               1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

# Bin width (ΔD) for each class (mm)
delta_Di = np.gradient(Di)

# Fall velocity (m/s) approximation for each diameter (Gunn–Kinzer relation)
vDi = 3.78 * (1 - np.exp(-0.25 * Di**1.3))

# Loop through all CSV files
for file in glob.glob(os.path.join(path, "RD-*.csv")):
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"⚠️ Skipped {file} (empty file)")
            continue

        # Pick one time interval (you can loop for more later)
        row = df.iloc[0]

        # Extract n1–n20
        ni = row[["n1","n2","n3","n4","n5","n6","n7","n8","n9","n10",
                  "n11","n12","n13","n14","n15","n16","n17","n18","n19","n20"]].values

        t = row["Interval [s]"]

        # Compute N(Di)
        NDi = ni / (F * t * vDi * delta_Di)

        # Plot N(D)
        plt.figure(figsize=(8, 6))
        plt.plot(Di, NDi, marker='o')
        plt.yscale('log')
        plt.xlabel("Drop Diameter D (mm)")
        plt.ylabel("Number Density N(D) [1/(m³·mm)]")
        plt.title(f"Drop Size Distribution N(D) — {os.path.basename(file)}")
        plt.grid(True, which='both', linestyle='--', lw=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
