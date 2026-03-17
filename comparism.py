import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ------------------------------------------------------------
# CONSTANTS (RD-80 Manual)
# ------------------------------------------------------------

F = 0.005  # sensor area (m²)

Di = np.array([
    0.359, 0.455, 0.551, 0.656, 0.771,
    0.917, 1.131, 1.331, 1.506, 1.665,
    1.912, 2.259, 2.589, 2.869, 3.205,
    3.544, 3.916, 4.350, 4.859, 5.373
])  # in mm

# Convert diameters mm → meters
Di_m = Di / 1000.0


# ------------------------------------------------------------
# LOAD ALL CSV FILES
# ------------------------------------------------------------

path = "data/"
files = sorted(glob.glob(os.path.join(path, "RD-*.csv")))

all_times = []
ri_formula = []
ri_inst = []

for file in files:
    df = pd.read_csv(file)

    # Drop count columns n1–n20
    drop_cols = [f"n{i}" for i in range(1, 21)]
    df["drop_sum"] = df[drop_cols].sum(axis=1)

    # Filter rows with real drops
    df_valid = df[df["drop_sum"] > 0]

    if df_valid.empty:
        continue

    # Time
    times = pd.to_datetime(df_valid["YYYY-MM-DD"] + " " + df_valid["hh:mm:ss"])
    all_times.append(times.values)

    # Drop counts (matrix), time interval (sec)
    N = df_valid[drop_cols].values
    t = df_valid["Interval [s]"].values

    # --------------------------------------------------------
    # CORRECTED RD-80 MANUAL FORMULA (Di converted to meters)
    # --------------------------------------------------------

    factor = (np.pi / 6) * 3.6e3 / (F * t)  # 3600 / (F * t)
    RI_f = factor * np.sum(N * (Di_m ** 3), axis=1)

    ri_formula.append(RI_f)
    ri_inst.append(df_valid["RI [mm/h]"].values)

# Combine all results
all_times = np.concatenate(all_times)
ri_formula = np.concatenate(ri_formula)
ri_inst = np.concatenate(ri_inst)

# Sort by time
order = np.argsort(all_times)
all_times = all_times[order]
ri_formula = ri_formula[order]
ri_inst = ri_inst[order]


# ------------------------------------------------------------
# RI VALUES SUMMARY
# ------------------------------------------------------------

print("\n------------------ RI VALUES SUMMARY ------------------")
print(f"Instrument RI: min={ri_inst.min():.3e}, median={np.median(ri_inst):.3e}, max={ri_inst.max():.3e}")
print(f"Formula RI:    min={ri_formula.min():.3e}, median={np.median(ri_formula):.3e}, max={ri_formula.max():.3e}")

# Ratio check
median_inst = np.median(ri_inst)
median_formula = np.median(ri_formula)

ratio = (median_formula / median_inst) if median_inst > 0 else float('inf')
print(f"\nRatio (Median Formula RI / Instrument RI) = {ratio:.3e}")
print("--------------------------------------------------------\n")


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

plt.figure(figsize=(12,6))
plt.plot(all_times, ri_inst, label="Instrument RI", lw=1)
plt.plot(all_times, ri_formula, label="Formula RI", lw=1)

plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Rain Intensity (mm/h)")
plt.title("Comparison: RI (Formula) vs RI (Instrument)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
