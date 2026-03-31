import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_FILE = "processed_data.csv"   
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# RD-80 physical constants (identical to mlModel.py & comparism.py)
F      = 0.005            # sensor area [m²]
F_cm2  = F * 1e4          # → 50 cm²
t_s    = 30               # sampling interval [s]

Di = np.array([0.359,0.455,0.551,0.656,0.771,0.917,1.131,1.331,1.506,
               1.665,1.912,2.259,2.589,2.869,3.205,3.544,3.916,4.350,
               4.859,5.373])   # drop diameters [mm]

Vi = np.array([0.87,1.34,1.79,2.17,2.55,2.90,3.20,3.46,3.72,
               4.01,4.60,5.39,6.34,7.06,7.58,8.01,8.35,8.61,
               8.81,8.94])     # fall velocities [m/s]

dDi = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
                0.250,0.500,0.750,0.500,0.500,0.500,0.500,0.500,0.500,
                0.500,0.500])  # bin widths [mm]

N_COLS = [f'n{i}' for i in range(1, 21)]

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  JULY 2015 FORECAST  |  Train: 2010–2014")
print("=" * 62)
print("\n[1] Loading data …")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"    Raw rows     : {len(df):,}")

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

print(f"    Clean rows   : {len(df):,}")
print(f"    Years present: {sorted(df['timestamp'].dt.year.unique().tolist())}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE RI TARGET  (DSD physics formula)
# ─────────────────────────────────────────────────────────────────
print("\n[2] Computing RI from DSD physics formula …")

N_mat  = df[N_COLS].values.astype(float)
factor = (np.pi / 6.0) * (3.6e3 / (F_cm2 * t_s))
df['RI_computed'] = factor * np.sum(N_mat * (Di ** 3), axis=1)

# Cap extreme values (same threshold as repo)
df.loc[df['RI_computed'] > 300, 'RI_computed'] = np.nan
df['RI_computed'] = df['RI_computed'].fillna(0)

# Keep only rain intervals
df = df[(df[N_COLS].sum(axis=1) > 0) & (df['RI_computed'] > 0)].reset_index(drop=True)

print(f"    Rain intervals: {len(df):,}")
print(f"    RI mean (all) : {df['RI_computed'].mean():.3f} mm/h")
print(f"    RI max  (all) : {df['RI_computed'].max():.3f} mm/h")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n[3] Engineering features …")

N_mat = df[N_COLS].values.astype(float)
ND    = N_mat / (F * t_s * Vi * dDi)   # drop size distribution [m⁻³ mm⁻¹]

df['total_drops'] = N_mat.sum(axis=1)
df['log_drops']   = np.log1p(df['total_drops'])
df['LWC']         = (np.pi/6) * 1e-3 * (ND * Di**3 * dDi).sum(axis=1)
df['Dm']          = ( (ND * Di**4 * dDi).sum(axis=1) /
                      ((ND * Di**3 * dDi).sum(axis=1) + 1e-12) )
df['D_mean']      = (N_mat * Di).sum(axis=1) / (df['total_drops'] + 1e-12)
df['Z']           = (ND * Di**6 * dDi).sum(axis=1)
df['log_Z']       = np.log1p(df['Z'])
df['hour']        = df['timestamp'].dt.hour
df['month']       = df['timestamp'].dt.month

# Lag features for forecasting (temporal context)
for lag in [1, 2, 3]:
    df[f'RI_lag{lag}'] = df['RI_computed'].shift(lag)

df = df.dropna(subset=['RI_lag1', 'RI_lag2', 'RI_lag3']).reset_index(drop=True)

FEATURES = (N_COLS +
            ['total_drops', 'log_drops', 'LWC', 'Dm', 'D_mean',
             'Z', 'log_Z', 'hour', 'month',
             'RI_lag1', 'RI_lag2', 'RI_lag3'])
TARGET = 'RI_computed'

print(f"    Features used : {len(FEATURES)}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — TEMPORAL TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────
print("\n[4] Temporal split: Train 2010–2014 | Test July 2015 …")

train_df = df[df['timestamp'].dt.year <= 2014]
test_df  = df[(df['timestamp'].dt.year == 2015) &
              (df['timestamp'].dt.month == 7)]

if len(test_df) == 0:
    raise ValueError("❌  No July 2015 data found in processed_data.csv!")

print(f"    Train samples : {len(train_df):,}")
print(f"    Test samples  : {len(test_df):,}  (July 2015)")

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].values
y_test  = test_df[TARGET].values

# ─────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────────
print("\n[5] Training Random Forest (150 trees, max_depth=12) …")

model = RandomForestRegressor(
    n_estimators = 150,
    max_depth    = 12,
    max_features = 'sqrt',
    min_samples_split = 5,
    min_samples_leaf  = 2,
    n_jobs       = -1,
    random_state = 42
)
model.fit(X_train, y_train)
print("    Training complete.")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — PREDICT
# ─────────────────────────────────────────────────────────────────
print("\n[6] Predicting July 2015 …")
y_pred = model.predict(X_test)

# ─────────────────────────────────────────────────────────────────
# STEP 7 — EVALUATION + MEAN RI COMPARISON
# ─────────────────────────────────────────────────────────────────
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

mean_actual    = y_test.mean()
mean_predicted = y_pred.mean()
mean_error_pct = abs(mean_predicted - mean_actual) / mean_actual * 100

print("\n" + "=" * 62)
print("  JULY 2015 FORECAST RESULTS")
print("=" * 62)
print(f"\n  Model Performance:")
print(f"    R²   : {r2:.4f}")
print(f"    RMSE : {rmse:.4f} mm/h")
print(f"    MAE  : {mae:.4f} mm/h")

print(f"\n  ── Mean RI Comparison (July 2015) ──────────────────")
print(f"    Actual    mean RI : {mean_actual:.4f} mm/h")
print(f"    Predicted mean RI : {mean_predicted:.4f} mm/h")
print(f"    Difference        : {abs(mean_predicted - mean_actual):.4f} mm/h")
print(f"    Error %           : {mean_error_pct:.2f}%")

print(f"\n  Distribution (July 2015 test set):")
print(f"    Actual    — min: {y_test.min():.2f}  max: {y_test.max():.2f}  std: {y_test.std():.2f}")
print(f"    Predicted — min: {y_pred.min():.2f}  max: {y_pred.max():.2f}  std: {y_pred.std():.2f}")

print(f"\n  Sample Actual vs Predicted (first 10 rows):")
print(f"  {'#':>3}  {'Actual':>10}  {'Predicted':>10}  {'Error':>10}")
print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}")
for i in range(min(10, len(y_test))):
    err = y_pred[i] - y_test[i]
    print(f"  {i+1:>3}  {y_test[i]:>10.2f}  {y_pred[i]:>10.2f}  {err:>+10.2f}")

# ─────────────────────────────────────────────────────────────────
# STEP 8 — FEATURE IMPORTANCE (TOP 10)
# ─────────────────────────────────────────────────────────────────
importances = model.feature_importances_
top10_idx   = np.argsort(importances)[::-1][:10]
print(f"\n  Top 10 Features by Importance:")
for rank, idx in enumerate(top10_idx, 1):
    print(f"    {rank:>2}. {FEATURES[idx]:<18} {importances[idx]:.4f}")

# ─────────────────────────────────────────────────────────────────
# STEP 9 — PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[7] Saving plots …")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Random Forest — July 2015 Rainfall Intensity Forecast\n"
             f"Train: 2010–2014  |  Test: July 2015  |  R²={r2:.4f}  RMSE={rmse:.2f} mm/h",
             fontsize=13, fontweight='bold', y=0.98)

# Use GridSpec for tighter control over spacing
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.32,
              top=0.90, bottom=0.08, left=0.07, right=0.97)

# ── Plot 1: Time Series ──
ax = fig.add_subplot(gs[0, 0])
ax.plot(test_df['timestamp'].values, y_test,
        color='#2dc653', linewidth=0.9, label='Actual RI', alpha=0.9)
ax.plot(test_df['timestamp'].values, y_pred,
        color='#f72585', linewidth=0.9, linestyle='--',
        label='Predicted RI', alpha=0.85)
ax.axhline(mean_actual,    color='#2dc653', linewidth=1.5,
           linestyle=':', label=f'Actual mean = {mean_actual:.2f} mm/h')
ax.axhline(mean_predicted, color='#f72585', linewidth=1.5,
           linestyle=':', label=f'Predicted mean = {mean_predicted:.2f} mm/h')
ax.set_title('Time Series — July 2015', fontweight='bold', fontsize=10, pad=8)
ax.set_xlabel('Timestamp', fontsize=8)
ax.set_ylabel('Rainfall Intensity (mm/h)', fontsize=8)
ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
ax.grid(True, alpha=0.3)
# Use only ~8 evenly spaced date ticks to avoid crowding
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=7)

# ── Plot 2: Scatter ──
ax = fig.add_subplot(gs[0, 1])
ax.scatter(y_test, y_pred, alpha=0.3, s=12, color='#4361ee', edgecolors='none')
lim = [0, max(y_test.max(), y_pred.max()) * 1.05]
ax.plot(lim, lim, 'k--', linewidth=1.3, label='Perfect fit')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title('Actual vs Predicted (Scatter)', fontweight='bold', fontsize=10, pad=8)
ax.set_xlabel('Actual RI (mm/h)', fontsize=8)
ax.set_ylabel('Predicted RI (mm/h)', fontsize=8)
ax.text(0.05, 0.93,
        f'R²   = {r2:.4f}\nRMSE = {rmse:.2f} mm/h\nMAE  = {mae:.2f} mm/h\nn = {len(y_test):,}',
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# ── Plot 3: RI Distribution Overlay + Mean lines ──
ax = fig.add_subplot(gs[1, 0])
ax.hist(y_test, bins=50, color='#2dc653', alpha=0.55, density=True,
        label='Actual RI', edgecolor='white', linewidth=0.3)
ax.hist(y_pred, bins=50, color='#4361ee', alpha=0.55, density=True,
        label='Predicted RI', edgecolor='white', linewidth=0.3)
ax.axvline(mean_actual,    color='#1a9e45', linewidth=2,
           linestyle='-',  label=f'Actual mean = {mean_actual:.2f}')
ax.axvline(mean_predicted, color='#c0006a', linewidth=2,
           linestyle='--', label=f'Predicted mean = {mean_predicted:.2f}')
ax.set_title('RI Distribution — July 2015', fontweight='bold', fontsize=10, pad=8)
ax.set_xlabel('Rainfall Intensity (mm/h)', fontsize=8)
ax.set_ylabel('Density', fontsize=8)
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
ax.grid(True, alpha=0.3)

# ── Plot 4: Mean RI Bar Chart ──
ax = fig.add_subplot(gs[1, 1])
categories = ['Actual Mean RI', 'Predicted Mean RI']
values     = [mean_actual, mean_predicted]
colors     = ['#2dc653', '#4361ee']
bars = ax.bar(categories, values, color=colors, width=0.45,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(values) * 0.02,   # dynamic offset
            f'{val:.2f} mm/h', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title(f'Mean RI Comparison — July 2015\nError = {mean_error_pct:.2f}%',
             fontweight='bold', fontsize=10, pad=8)
ax.set_ylabel('Mean Rainfall Intensity (mm/h)', fontsize=8)
ax.set_ylim(0, max(values) * 1.30)   # extra headroom for labels
ax.tick_params(axis='x', labelsize=9)
ax.grid(True, alpha=0.3, axis='y')
plot_path = os.path.join(PLOT_DIR, "july2015_forecast.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Plot saved: {plot_path}")

print("\n" + "=" * 62)
print("  DONE")
print("=" * 62)
