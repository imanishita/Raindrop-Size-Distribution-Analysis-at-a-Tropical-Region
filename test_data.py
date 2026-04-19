import pandas as pd
import numpy as np

OUTPUT_FILE = "processed_data.csv"
INPUT_FILE = "merged_data.csv"

# Instead of the whole script, just read the relevant part of merged_data
df = pd.read_csv(INPUT_FILE, low_memory=False)
df.rename(columns={
    'RI [mm/h]': 'RI',
    'RA [mm]'  : 'RA',
    'RAT [mm]' : 'RAT'
}, inplace=True)

df['RI'] = pd.to_numeric(df['RI'], errors='coerce')

df['timestamp'] = pd.to_datetime(
    df['YYYY-MM-DD'].astype(str).str.strip() + ' ' +
    df['hh:mm:ss'].astype(str).str.strip(),
    errors='coerce'
)

# find the max for 2015-07-24
mask_day = df['timestamp'].dt.date == pd.to_datetime('2015-07-24').date()
print("Initial max for 24-Jul:", df.loc[mask_day, 'RI'].max())

DROP_COLS = [f'n{i}' for i in range(1, 21)]
for c in DROP_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['total_drops'] = df[DROP_COLS].sum(axis=1)

has_drops = df['total_drops'] > 0
has_ri    = df['RI'].fillna(0) > 0
df = df[has_drops | has_ri].reset_index(drop=True)

mask_day = df['timestamp'].dt.date == pd.to_datetime('2015-07-24').date()
print("After drops/RI > 0:", df.loc[mask_day, 'RI'].max())

df = df[df['RI'] <= 100.0]
mask_day = df['timestamp'].dt.date == pd.to_datetime('2015-07-24').date()
print("After RI <= 100:", df.loc[mask_day, 'RI'].max())

df = df[df['n20'] < 50]
mask_day = df['timestamp'].dt.date == pd.to_datetime('2015-07-24').date()
print("After n20 < 50:", df.loc[mask_day, 'RI'].max())

has_drops_mask = df['total_drops'] > 0
df.loc[has_drops_mask, 'max_bin']   = df.loc[has_drops_mask, DROP_COLS].max(axis=1)
df.loc[has_drops_mask, 'dominance'] = (
    df.loc[has_drops_mask, 'max_bin'] /
    (df.loc[has_drops_mask, 'total_drops'] + 1e-6)
)
df['dominance'] = df['dominance'].fillna(0)
df = df[df['dominance'] < 0.70]

mask_day = df['timestamp'].dt.date == pd.to_datetime('2015-07-24').date()
print("After dominance < 0.7:", df.loc[mask_day, 'RI'].max())
