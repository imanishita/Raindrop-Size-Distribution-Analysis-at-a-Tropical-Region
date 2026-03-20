import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
F = 0.005  # sensor area [m²]

Di = np.array([
    0.359, 0.455, 0.551, 0.656, 0.771,
    0.917, 1.131, 1.331, 1.506, 1.665,
    1.912, 2.259, 2.589, 2.869, 3.205,
    3.544, 3.916, 4.350, 4.859, 5.373
])

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("Loading processed data...")
df = pd.read_csv("processed_data.csv", low_memory=False)
print(f"  Total rows loaded : {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# FIX TIMESTAMP
# ─────────────────────────────────────────────────────────────────
print("Fixing timestamp...")
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# Filter to valid year range
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]

# Sort and reset index — MUST reset so array positions stay aligned
df = df.sort_values(by='timestamp').reset_index(drop=True)

print(f"  Rows after filter : {len(df):,}")
print(f"  Time range        : {df['timestamp'].min()}  to  {df['timestamp'].max()}")
print(f"  Years present     : {sorted(df['timestamp'].dt.year.unique().tolist())}")

# ─────────────────────────────────────────────────────────────────
# COMPUTE RAINFALL INTENSITY
# ─────────────────────────────────────────────────────────────────
if 'RI' in df.columns:
    print("Using instrument RI column...")
    df['RI_plot'] = pd.to_numeric(df['RI'], errors='coerce').fillna(0)
else:
    print("Computing RI from DSD drop counts...")
    drop_cols = [f'n{i}' for i in range(1, 21)]
    for c in drop_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    if 'interval' in df.columns:
        t = pd.to_numeric(df['interval'], errors='coerce').fillna(30).values
    else:
        t = np.full(len(df), 30)

    N             = df[drop_cols].values
    factor        = (np.pi / 6.0) * 3.6e3 / (F * t)
    df['RI_plot'] = factor * np.sum(N * (Di ** 3), axis=1)

# ─────────────────────────────────────────────────────────────────
# RESAMPLE TO HOURLY — covers ALL years, keeps plot clean
# This replaces the N_POINTS=5000 slice which was cutting off data
# ─────────────────────────────────────────────────────────────────
print("Resampling to hourly for full 2010-2015 view...")

df_hourly = (df.set_index('timestamp')['RI_plot']
               .resample('1h').mean()
               .fillna(0)
               .reset_index())
df_hourly.columns = ['timestamp', 'RI_hourly']

# Light smoothing on hourly data
df_hourly['RI_smooth'] = df_hourly['RI_hourly'].rolling(window=6, min_periods=1).mean()

print(f"  Hourly points     : {len(df_hourly):,}")
print(f"  Hourly time range : {df_hourly['timestamp'].min()}  to  {df_hourly['timestamp'].max()}")

# ─────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────
print("Plotting full Rain Intensity Curve (2010-2015)...")

fig, ax = plt.subplots(figsize=(16, 6))

# Alternating year background bands
year_colors = ['#f0f4ff', '#ffffff']
for i, year in enumerate(range(2010, 2016)):
    ax.axvspan(pd.Timestamp(f'{year}-01-01'),
               pd.Timestamp(f'{year+1}-01-01'),
               alpha=0.4, color=year_colors[i % 2], zorder=0)

# Main RI curve
ax.plot(df_hourly['timestamp'], df_hourly['RI_smooth'],
        color='#4361ee', linewidth=0.9, label='Smoothed RI (hourly avg)', zorder=2)

ax.fill_between(df_hourly['timestamp'], df_hourly['RI_smooth'],
                alpha=0.2, color='#4361ee', zorder=2)

# Force x-axis to show exactly 2010-2015
ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2015-12-31'))
ax.set_ylim(bottom=0)

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Rain Intensity RI [mm/h]', fontsize=11)
ax.set_title('Rain Intensity Curve — Full Dataset (2010–2015)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.35, linewidth=0.5)
ax.legend(fontsize=9)

# Year labels at top of plot
y_top = ax.get_ylim()[1]
for year in range(2010, 2016):
    ax.text(pd.Timestamp(f'{year}-07-01'), y_top * 0.96,
            str(year), ha='center', va='top',
            fontsize=11, color='#555555', fontweight='bold', zorder=3)

plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()