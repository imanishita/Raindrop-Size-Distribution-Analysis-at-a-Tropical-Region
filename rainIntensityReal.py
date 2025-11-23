import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

path = "data/"

all_files = glob.glob(os.path.join(path, "RD-*.csv"))

full_data = []

for file in all_files:
    df = pd.read_csv(file)
    
    # Combine date and time into one timestamp
    df["timestamp"] = df["YYYY-MM-DD"] + " " + df["hh:mm:ss"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Keep only timestamp + RI
    df_small = df[["timestamp", "RI [mm/h]"]]
    
    full_data.append(df_small)

# Combine all 12 files into one big dataframe
full_df = pd.concat(full_data).sort_values("timestamp")

# Plot RI vs time
plt.figure(figsize=(10, 5))
plt.plot(full_df["timestamp"], full_df["RI [mm/h]"])
plt.xlabel("Time")
plt.ylabel("Rainfall Intensity RI (mm/h)")
plt.title("Rainfall Intensity Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
