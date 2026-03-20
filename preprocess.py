import pandas as pd

INPUT_FILE = "merged_data.csv"
OUTPUT_FILE = "processed_data.csv"

print("Loading data...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Original shape:", df.shape)

# -----------------------------
# RENAME COLUMNS
# -----------------------------
df.rename(columns={
    'RI [mm/h]': 'RI',
    'RA [mm]': 'RA',
    'RAT [mm]': 'RAT'
}, inplace=True)

# -----------------------------
# REMOVE NO-RAIN ROWS
# -----------------------------
drop_cols = [f'n{i}' for i in range(1, 21)]
df['total_drops'] = df[drop_cols].sum(axis=1)
df = df[df['total_drops'] > 0]

print("After removing no-rain rows:", df.shape)

# -----------------------------
# CREATE TIMESTAMP
# -----------------------------
print("Creating timestamp...")

if 'YYYY-MM-DD' in df.columns and 'hh:mm:ss' in df.columns:
    
    df['timestamp'] = pd.to_datetime(
        df['YYYY-MM-DD'].astype(str).str.strip() + ' ' +
        df['hh:mm:ss'].astype(str).str.strip(),
        errors='coerce'
    )

else:
    raise ValueError("Date/Time columns not found!")

# Remove invalid timestamps only
df = df.dropna(subset=['timestamp'])

print("Timestamp created!")

# -----------------------------
# TIME FEATURES
# -----------------------------
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# -----------------------------
# SEASON
# -----------------------------
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

# -----------------------------
# FINAL CLEAN
# -----------------------------
df = df.dropna(subset=['RI'])

print("Final shape:", df.shape)

# -----------------------------
# SAVE
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("Saved as processed_data.csv")