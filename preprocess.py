import pandas as pd
import numpy as np

INPUT_FILE  = "merged_data.csv"
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

RI_MAX    = 100.0   # mm/h — Kolkata realistic ceiling
DROP_COLS = [f'n{i}' for i in range(1, 21)]

print("=" * 62)
print("  PREPROCESSING PIPELINE")
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
    'RA [mm]'  : 'RA',
    'RAT [mm]' : 'RAT'
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
# STEP 5 — REMOVE TRULY EMPTY ROWS
# ─────────────────────────────────────────────────────────────────
# FIX: The RD-80 can report a non-zero RI (from its internal RAT
# accumulator / smoothing) even when all 20 drop-count bins are 0.
# These rows carry real rainfall signal and must NOT be dropped.
#
# Old logic (WRONG):  drop if total_drops == 0
# New logic (CORRECT): drop only if total_drops == 0 AND RI == 0
#                      → keep rows where RI > 0 regardless of drop counts
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Removing truly empty rows …")
df['total_drops'] = df[DROP_COLS].sum(axis=1)

before = len(df)
has_drops = df['total_drops'] > 0
has_ri    = df['RI'].fillna(0) > 0
df = df[has_drops | has_ri].reset_index(drop=True)

removed = before - len(df)
print(f"  Removed (both drop=0 AND RI=0) : {removed:,} rows")
print(f"  Kept rows with RI>0 but drops=0: {(~has_drops & has_ri).sum():,} rows (instrument interpolation)")
print(f"  Remaining                       : {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — PHYSICS-BASED CLEANING
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Physics-based cleaning …")
before = len(df)

# 6A: Cap extreme RI
ri_flagged = (df['RI'] > RI_MAX).sum()
df = df[df['RI'] <= RI_MAX]
print(f"  6A — RI > {RI_MAX} mm/h removed : {ri_flagged:,} rows")

# 6B: Large-drop spike on n20 (5.373mm drops — extremely rare naturally)
n20_flagged = (df['n20'] >= 50).sum()
df = df[df['n20'] < 50]
print(f"  6B — n20 >= 50 removed         : {n20_flagged:,} rows")

# 6C: Single-bin dominance (sensor malfunction indicator)
# Real rain has a bell-curve DSD; one bin holding >=70% is a glitch.
# Only apply this check to rows that actually have drops to check.
has_drops_mask = df['total_drops'] > 0
df.loc[has_drops_mask, 'max_bin']   = df.loc[has_drops_mask, DROP_COLS].max(axis=1)
df.loc[has_drops_mask, 'dominance'] = (
    df.loc[has_drops_mask, 'max_bin'] /
    (df.loc[has_drops_mask, 'total_drops'] + 1e-6)
)
df['dominance'] = df['dominance'].fillna(0)  # rows with no drops get 0 dominance

dom_flagged = (df['dominance'] >= 0.70).sum()
df = df[df['dominance'] < 0.70]
print(f"  6C — Single-bin dom >= 70%     : {dom_flagged:,} rows")

df.drop(columns=['max_bin', 'dominance'], errors='ignore', inplace=True)

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
    if month in [6, 7, 8, 9]:   return "monsoon"
    if month in [10, 11]:        return "post-monsoon"
    if month in [3, 4, 5]:       return "pre-monsoon"
    return "winter"

df['season'] = df['month'].apply(get_season)

# ─────────────────────────────────────────────────────────────────
# STEP 9 — FINAL VALIDATION
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 9] Final validation …")
df = df.dropna(subset=['RI'])
df['total_drops'] = df[DROP_COLS].sum(axis=1)   # recalculate after filtering

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