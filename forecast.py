import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "processed_data.csv"

N_COLS = [f'n{i}' for i in range(1, 21)]

# -----------------------------
# STEP 1 — LOAD
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE, low_memory=False)

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# STEP 2 — USE ACTUAL RI (FIX)
# -----------------------------
print("Using instrument RI...")

# Ensure numeric
df['RI'] = pd.to_numeric(df['RI'], errors='coerce')
df = df.dropna(subset=['RI'])

# Remove extreme outliers (optional but recommended)
df = df[df['RI'] < 200]

# -----------------------------
# STEP 3 — FEATURES
# -----------------------------
df['total_drops'] = df[N_COLS].sum(axis=1)
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month

# Lag features
for lag in [1, 2, 3]:
    df[f'RI_lag{lag}'] = df['RI'].shift(lag)

df = df.dropna(subset=['RI_lag1','RI_lag2','RI_lag3'])

features = N_COLS + ['total_drops', 'hour', 'month',
                    'RI_lag1', 'RI_lag2', 'RI_lag3']

# -----------------------------
# STEP 4 — TRAIN / TEST SPLIT
# -----------------------------
print("Splitting data...")

train_df = df[df['timestamp'].dt.year <= 2014]

test_df = df[
    (df['timestamp'].dt.year == 2015) &
    (df['timestamp'].dt.month == 7)
]

if len(test_df) == 0:
    raise ValueError("No July 2015 data!")

print(f"Train samples: {len(train_df):,}")
print(f"Test samples : {len(test_df):,}")

X_train = train_df[features]
y_train = train_df['RI']

X_test = test_df[features]
y_test = test_df['RI']

# -----------------------------
# STEP 5 — TRAIN MODEL
# -----------------------------
print("Training model...")

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 6 — PREDICT
# -----------------------------
print("Predicting...")

y_pred = model.predict(X_test)

# -----------------------------
# STEP 7 — EVALUATION
# -----------------------------
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== JULY FORECAST RESULT =====")
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f} mm/h")

# -----------------------------
# STEP 8 — PRINT VALUES (IMPORTANT)
# -----------------------------
print("\nSample Actual vs Predicted:")
for i in range(10):
    print(f"Actual: {y_test.values[i]:.2f}  |  Predicted: {y_pred[i]:.2f}")

# -----------------------------
# STEP 9 — PLOT (TIME SERIES)
# -----------------------------
plt.figure(figsize=(12, 5))

plt.plot(test_df['timestamp'], y_test,
         label="Actual RI", color='green')

plt.plot(test_df['timestamp'], y_pred,
         label="Predicted RI", color='red', linestyle='--')

plt.xlabel("Time")
plt.ylabel("Rainfall Intensity (mm/h)")
plt.title("Rainfall Prediction — July 2015")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 10 — SCATTER
# -----------------------------
plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred, alpha=0.4)

max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'k--')

plt.xlabel("Actual RI")
plt.ylabel("Predicted RI")
plt.title("Actual vs Predicted (July 2015)")

plt.grid(True)
plt.show()