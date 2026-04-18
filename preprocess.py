import pandas as pd
import numpy as np

INPUT_FILE = "merged_data.csv"
OUTPUT_FILE = "processed_data.csv"

# ─────────────────────────────────────────────────────────────────
# RD-80 PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────
Di = np.array([
    0.359, 0.455, 0.551, 0.656, 0.771,
    0.917, 1.131, 1.331, 1.506, 1.665,
    1.912, 2.259, 2.589, 2.869, 3.205,
    3.544, 3.916, 4.350, 4.859, 5.373
])  # drop diameters [mm]

F     = 0.005    # sensor area [m²]
F_cm2 = F * 1e4  # sensor area [cm²] = 50 cm²
t_s   = 30       # sampling interval [seconds]

# RI cap for Kolkata region
RI_MAX = 100.0   # mm/h — anything above this is sensor noise

DROP_COLS = [f'n{i}' for i in range(1, 21)]

print("=" * 62)
print("  PREPROCESSING PIPELINE — WITH PHYSICS-BASED CLEANING")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading raw data …")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Original shape    : {df.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — RENAME COLUMNS
# ─────────────────────────────────────────────────────────────────
df.rename(columns={
    'RI [mm/h]': 'RI',
    'RA [mm]': 'RA',
    'RAT [mm]': 'RAT'
}, inplace=True)

# ─────────────────────────────────────────────────────────────────
# STEP 3 — COERCE NUMERIC TYPES
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Coercing numeric types …")

df['RI'] = pd.to_numeric(df['RI'], errors='coerce')
for c in DROP_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# ─────────────────────────────────────────────────────────────────
# STEP 4 — CREATE TIMESTAMP
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Creating timestamp …")

if 'YYYY-MM-DD' in df.columns and 'hh:mm:ss' in df.columns:
    df['timestamp'] = pd.to_datetime(
        df['YYYY-MM-DD'].astype(str).str.strip() + ' ' +
        df['hh:mm:ss'].astype(str).str.strip(),
        errors='coerce'
    )
else:
    raise ValueError("Date/Time columns not found!")

df = df.dropna(subset=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"  Valid timestamps  : {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — REMOVE NO-RAIN ROWS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Removing no-rain rows (total drops = 0) …")

df['total_drops'] = df[DROP_COLS].sum(axis=1)
before = len(df)
df = df[df['total_drops'] > 0]
print(f"  Removed           : {before - len(df):,} zero-rain rows")
print(f"  Remaining         : {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — PHYSICS-BASED CLEANING (NEW)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Physics-based cleaning …")
before = len(df)

# Cap extreme rainfall intensity
# Values > 100 mm/h are sensor noise / corruption.
ri_flagged = (df['RI'] > RI_MAX).sum()
df = df[df['RI'] <= RI_MAX]
print(f"  RI > {RI_MAX} mm/h removed       : {ri_flagged:,} rows")

total_removed = before - len(df)
print(f"\n  Total rows cleaned : {total_removed:,}  ({total_removed/before*100:.2f}%)")
print(f"  Rows remaining     : {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 7 — TIME FEATURES
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 7] Creating time features …")

df['hour']  = df['timestamp'].dt.hour
df['day']   = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# ─────────────────────────────────────────────────────────────────
# STEP 8 — SEASON LABELS
# ─────────────────────────────────────────────────────────────────
def get_season(month):
    if month in [6, 7, 8, 9]:
        return "monsoon"
    elif month in [10, 11]:
        return "post-monsoon"
    elif month in [3, 4, 5]:
        return "pre-monsoon"
    else:
        return "winter"

df['season'] = df['month'].apply(get_season)

# ─────────────────────────────────────────────────────────────────
# STEP 9 — FINAL VALIDATION
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 9] Final validation …")

df = df.dropna(subset=['RI'])
# Recalculate total_drops after all filtering
df['total_drops'] = df[DROP_COLS].sum(axis=1)

print(f"  Final shape       : {df.shape}")
print(f"  RI range          : {df['RI'].min():.3f} – {df['RI'].max():.3f} mm/h")
print(f"  RI mean           : {df['RI'].mean():.3f} mm/h")
print(f"  RI 99th pctile    : {df['RI'].quantile(0.99):.3f} mm/h")
print(f"  RI 95th pctile    : {df['RI'].quantile(0.95):.3f} mm/h")
print(f"  Year range        : {df['timestamp'].dt.year.min()} – {df['timestamp'].dt.year.max()}")
print(f"  Seasons           : {df['season'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────
# STEP 10 — SAVE
# ─────────────────────────────────────────────────────────────────
print(f"\n[STEP 10] Saving to {OUTPUT_FILE} …")
df.to_csv(OUTPUT_FILE, index=False)

print(f"  Saved successfully!")
print("\n" + "=" * 62)
print("  PREPROCESSING COMPLETE")
print("=" * 62)