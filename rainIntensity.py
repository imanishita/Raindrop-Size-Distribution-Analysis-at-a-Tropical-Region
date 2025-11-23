import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- constants ---
F = 0.005  # sensor area [m²]


Di = np.array([
    0.359, 0.455, 0.551, 0.656, 0.771,
    0.917, 1.131, 1.331, 1.506, 1.665,
    1.912, 2.259, 2.589, 2.869, 3.205,
    3.544, 3.916, 4.350, 4.859, 5.373
])

path = "data/"

all_times = []
all_RI = []

for file in sorted(glob.glob(os.path.join(path, "RD-*.csv"))):
    df = pd.read_csv(file)

    # --- get RI for each interval ---

    if "RI [mm/h]" in df.columns:
        # 1) easiest: use RI column already provided by the instrument
        RI = df["RI [mm/h]"].values
    else:
        # 2) or compute RI from the formula:
        # RI = (π/6) * 3.6·10³ * (1/(F·t)) * Σ( n_i · D_i³ )
        drop_cols = [f"n{i}" for i in range(1, 21)]
        N = df[drop_cols].values                         # shape (rows, 20)
        t = df["Interval [s]"].values                    # seconds
        factor = (np.pi / 6.0) * 3.6e3 / (F * t)         # shape (rows,)
        RI = factor * np.sum(N * (Di**3), axis=1)        # shape (rows,)

    # time axis
    times = pd.to_datetime(df["YYYY-MM-DD"] + " " + df["hh:mm:ss"])

    all_times.append(times)
    all_RI.append(RI)

# put everything into one long series
all_times = np.concatenate(all_times)
all_RI = np.concatenate(all_RI)

# sort by time just in case
order = np.argsort(all_times)
all_times = all_times[order]
all_RI = all_RI[order]

# --- plot: TOP CURVE (Rain intensity vs time) ---
plt.figure(figsize=(10, 5))
plt.step(all_times, all_RI, where="post")
plt.xlabel("Time")
plt.ylabel("Rain intensity RI [mm/h]")
plt.title("Rain Intensity Curve (RI vs Time)")
plt.grid(True)
plt.tight_layout()
plt.show()
