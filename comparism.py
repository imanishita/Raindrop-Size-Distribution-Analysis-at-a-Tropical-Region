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
# LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
print("Loading processed data...")
df = pd.read_csv("processed_data.csv", low_memory=False)
print(f"  Raw rows : {len(df):,}")

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values(by='timestamp').reset_index(drop=True)
print(f"  Filtered rows : {len(df):,}")
print(f"  Years present : {sorted(df['timestamp'].dt.year.unique().tolist())}")

# ─────────────────────────────────────────────────────────────────
# 1. INSTRUMENT RI
#    The RI column is CUMULATIVE (RAT). Diff it to get per-interval
#    rainfall in mm per 30s, then multiply by 120 to get mm/h.
# ─────────────────────────────────────────────────────────────────
print("Deriving instrument RI from cumulative column...")
df['RI_instrument'] = df['RI'].diff().clip(lower=0) * 120  # mm/30s → mm/h
df['RI_instrument'] = df['RI_instrument'].fillna(0)

print(f"  Instrument RI — max: {df['RI_instrument'].max():.2f}  "
      f"mean (rain only): {df[df['RI_instrument']>0]['RI_instrument'].mean():.2f} mm/h")

# ─────────────────────────────────────────────────────────────────
# 2. COMPUTED RI FROM DSD
# ─────────────────────────────────────────────────────────────────
print("Computing RI from DSD drop counts...")
drop_cols = [f'n{i}' for i in range(1, 21)]
for c in drop_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# Use 30s interval if column missing or has bad values
if 'Interval [s]' in df.columns:
    t = pd.to_numeric(df['Interval [s]'], errors='coerce').fillna(30).values
else:
    t = np.full(len(df), 30)
t = np.where(t <= 0, 30, t)   # guard against zero/negative intervals

N                = df[drop_cols].values
factor           = (np.pi / 6.0) * 3.6e3 / (F * t)
df['RI_computed'] = factor * np.sum(N * (Di ** 3), axis=1)

print(f"  Computed RI   — max: {df['RI_computed'].max():.2f}  "
      f"mean (rain only): {df[df['RI_computed']>0]['RI_computed'].mean():.2f} mm/h")

# ─────────────────────────────────────────────────────────────────
# RESAMPLE TO HOURLY for clean full-range plot
# ─────────────────────────────────────────────────────────────────
print("Resampling to hourly for full 2010-2015 view...")
df_h = (df.set_index('timestamp')[['RI_instrument','RI_computed']]
          .resample('1h').mean()
          .fillna(0)
          .reset_index())

# Light smoothing
df_h['inst_smooth'] = df_h['RI_instrument'].rolling(6, min_periods=1).mean()
df_h['comp_smooth'] = df_h['RI_computed'].rolling(6, min_periods=1).mean()

print(f"  Hourly points : {len(df_h):,}")

# ─────────────────────────────────────────────────────────────────
# PLOT — TWO PANELS stacked
# ─────────────────────────────────────────────────────────────────
print("Plotting comparison...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.suptitle('Instrument vs Computed Rainfall Intensity (2010–2015)',
             fontsize=14, fontweight='bold')

year_colors = ['#f0f4ff', '#ffffff']

for ax in axes:
    for i, year in enumerate(range(2010, 2016)):
        ax.axvspan(pd.Timestamp(f'{year}-01-01'),
                   pd.Timestamp(f'{year+1}-01-01'),
                   alpha=0.35, color=year_colors[i % 2], zorder=0)
    ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2015-12-31'))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35, linewidth=0.5)

# Panel 1 — Instrument RI
axes[0].plot(df_h['timestamp'], df_h['inst_smooth'],
             color='#4361ee', linewidth=0.9, label='Instrument RI (from cumulative diff)')
axes[0].fill_between(df_h['timestamp'], df_h['inst_smooth'],
                     alpha=0.2, color='#4361ee')
axes[0].set_ylabel('RI [mm/h]', fontsize=10)
axes[0].set_title('Instrument RI', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)

# Year labels panel 1
y_top = axes[0].get_ylim()[1]
for year in range(2010, 2016):
    axes[0].text(pd.Timestamp(f'{year}-07-01'), y_top * 0.96,
                 str(year), ha='center', va='top',
                 fontsize=10, color='#555555', fontweight='bold')

# Panel 2 — Computed RI
axes[1].plot(df_h['timestamp'], df_h['comp_smooth'],
             color='#f72585', linewidth=0.9, label='Computed RI (from DSD formula)')
axes[1].fill_between(df_h['timestamp'], df_h['comp_smooth'],
                     alpha=0.2, color='#f72585')
axes[1].set_ylabel('RI [mm/h]', fontsize=10)
axes[1].set_xlabel('Time', fontsize=10)
axes[1].set_title('Computed RI (DSD formula)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)

# Year labels panel 2
y_top2 = axes[1].get_ylim()[1]
for year in range(2010, 2016):
    axes[1].text(pd.Timestamp(f'{year}-07-01'), y_top2 * 0.96,
                 str(year), ha='center', va='top',
                 fontsize=10, color='#555555', fontweight='bold')

plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()