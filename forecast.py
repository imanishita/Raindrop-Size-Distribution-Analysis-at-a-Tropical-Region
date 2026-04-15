import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "processed_data.csv"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

N_COLS = [f'n{i}' for i in range(1, 21)]

print("=" * 60)
print("FULL DAY FORECAST — 22 JULY 2015")
print("=" * 60)

# -----------------------------
# STEP 1 — LOAD & CLEAN
# -----------------------------
df = pd.read_csv(DATA_FILE, low_memory=False)

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['RI'] = pd.to_numeric(df['RI'], errors='coerce')
df = df.dropna(subset=['RI'])

# Remove unrealistic spikes
df = df[df['RI'] < 200]

# Keep rain only
df = df[(df[N_COLS].sum(axis=1) > 0) & (df['RI'] > 0)].reset_index(drop=True)

# -----------------------------
# STEP 2 — FEATURES
# -----------------------------
df['total_drops'] = df[N_COLS].sum(axis=1)
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month

# Lag features
for lag in [1, 2, 3]:
    df[f'RI_lag{lag}'] = df['RI'].shift(lag)

df = df.dropna(subset=['RI_lag1','RI_lag2','RI_lag3']).reset_index(drop=True)

FEATURES = N_COLS + [
    'total_drops',
    'hour',
    'month',
    'RI_lag1',
    'RI_lag2',
    'RI_lag3'
]

TARGET = 'RI'

# -----------------------------
# STEP 3 — TRAIN / TEST SPLIT
# -----------------------------
train_df = df[df['timestamp'].dt.year <= 2014]

test_df = df[
    (df['timestamp'].dt.year == 2015) &
    (df['timestamp'].dt.month == 7) &
    (df['timestamp'].dt.day == 22)
]

if len(test_df) == 0:
    raise ValueError("No data for 22 July 2015!")

print(f"Train samples: {len(train_df):,}")
print(f"Test samples : {len(test_df):,} (FULL DAY)")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# -----------------------------
# STEP 4 — TRAIN MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 5 — PREDICT (FULL DAY)
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 6 — METRICS
# -----------------------------
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== FULL DAY RESULTS ===")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f} mm/h")

# -----------------------------
# STEP 7 — DAILY SUMMARY
# -----------------------------
print("\n=== DAILY SUMMARY (22 JULY 2015) ===")

print(f"Actual Mean RI    : {y_test.mean():.2f} mm/h")
print(f"Predicted Mean RI : {y_pred.mean():.2f} mm/h")

print(f"Actual Max RI     : {y_test.max():.2f} mm/h")
print(f"Predicted Max RI  : {y_pred.max():.2f} mm/h")

# Total rainfall (mm) = sum(RI * time interval hours)
interval_hr = 30 / 3600 

actual_total = np.sum(y_test * interval_hr)
pred_total   = np.sum(y_pred * interval_hr)
"""
print(f"Actual Total Rain : {actual_total:.2f} mm/h")
print(f"Predicted Total   : {pred_total:.2f} mm/h")
"""
# -----------------------------
# STEP 8 — FULL DAY PLOT
# -----------------------------
plt.figure(figsize=(14,6))

plt.plot(test_df['timestamp'], y_test,
         label="Actual RI", color='green', linewidth=2)

plt.plot(test_df['timestamp'], y_pred,
         label="Predicted RI", color='red', linestyle='--', linewidth=2)

plt.xlabel("Time (22 July 2015)")
plt.ylabel("Rainfall Intensity (mm/h)")
plt.title("Full Day Rainfall Prediction — 22 July 2015")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "full_day_22_july_2015.png")
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"\nPlot saved at: {plot_path}")