"""
╔══════════════════════════════════════════════════════════════════╗
║   Random Forest — 1-Hour Rainfall Intensity Forecasting          ║
║   Input  : DSD features at time t  (n1–n20 + derived + lags)    ║
║   Output : Computed RI at time t + 1 hour                        ║
║   Data   : RD-80 Disdrometer | Tropical Region | 2010–2015       ║
╚══════════════════════════════════════════════════════════════════╝

TARGET: Computed RI (same DSD physics formula as comparism.py)
        RI = (π/6) × (3.6×10³ / F_cm2 × t) × Σ(nᵢ × Dᵢ³)
        Shifted 120 steps forward → 120 × 30s = 1 hour ahead

        Expected output stats:
          max  ~300 mm/h
          mean ~41  mm/h  (consistent with comparism.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.ensemble  import RandomForestRegressor
from sklearn.metrics   import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FILE      = "processed_data.csv"
PLOT_DIR       = "plots"
FORECAST_STEPS = 120          # 120 × 30s = 1 hour
os.makedirs(PLOT_DIR, exist_ok=True)

# RD-80 physical constants — identical to comparism.py & rf_current_ri.py
F       = 0.005
F_cm2   = F * 1e4             # 0.005 m² → 50 cm²
t_s     = 30

Di  = np.array([0.359,0.455,0.551,0.656,0.771,0.917,1.131,1.331,1.506,
                1.665,1.912,2.259,2.589,2.869,3.205,3.544,3.916,4.350,
                4.859,5.373])
Vi  = np.array([0.87,1.34,1.79,2.17,2.55,2.90,3.20,3.46,3.72,
                4.01,4.60,5.39,6.34,7.06,7.58,8.01,8.35,8.61,8.81,8.94])
dDi = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
                0.250,0.500,0.750,0.500,0.500,0.500,0.500,0.500,0.500,
                0.500,0.500])
N_COLS = [f'n{i}' for i in range(1, 21)]

P = {'bg':'#f8f9fa','panel':'#ffffff','border':'#dee2e6','text':'#1a1a2e',
     'sub':'#6c757d','blue':'#4361ee','red':'#f72585','green':'#2dc653',
     'orange':'#ff6b35','purple':'#7209b7'}

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(P['panel'])
    ax.tick_params(colors=P['sub'], labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor(P['border'])
    if title:  ax.set_title(title,   color=P['text'], fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=P['sub'],  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=P['sub'],  fontsize=9)
    ax.grid(True, color=P['border'], linewidth=0.5, alpha=0.8)

def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
    plt.close()
    print(f"    Saved: {path}")

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  RANDOM FOREST — 1-HOUR RI FORECASTING")
print("=" * 62)
print("\n[STEP 1] Loading & cleaning …")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Raw rows      : {len(df):,}")

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

print(f"  Clean rows    : {len(df):,}")
print(f"  Years         : {sorted(df['timestamp'].dt.year.unique().tolist())}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE RI (same formula as comparism.py)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Computing RI (DSD physics formula) …")

N_mat  = df[N_COLS].values.astype(float)
factor = (np.pi / 6.0) * (3.6e3 / (F_cm2 * t_s))
df['RI_computed'] = factor * np.sum(N_mat * (Di ** 3), axis=1)

df.loc[df['RI_computed'] > 300, 'RI_computed'] = np.nan
df['RI_computed'] = df['RI_computed'].fillna(0)

print(f"  RI max  : {df['RI_computed'].max():.3f} mm/h")
print(f"  RI mean : {df[df['RI_computed']>0]['RI_computed'].mean():.3f} mm/h")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING + LAG FEATURES
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Engineering features + lag windows …")

N_mat = df[N_COLS].values.astype(float)
ND    = N_mat / (F * t_s * Vi * dDi)

df['total_drops'] = N_mat.sum(axis=1)
df['log_drops']   = np.log1p(df['total_drops'])
df['LWC']         = (np.pi/6)*1e-3*(ND*Di**3*dDi).sum(axis=1)
df['Dm']          = ((ND*Di**4*dDi).sum(axis=1) /
                     ((ND*Di**3*dDi).sum(axis=1) + 1e-12))
df['D_mean']      = (N_mat*Di).sum(axis=1) / (df['total_drops'] + 1e-12)
df['Z']           = (ND*Di**6*dDi).sum(axis=1)
df['log_Z']       = np.log1p(df['Z'])
df['hour']        = df['timestamp'].dt.hour
df['month']       = df['timestamp'].dt.month

BASE_FEATS = (N_COLS +
              ['total_drops','log_drops','LWC','Dm','D_mean','Z','log_Z',
               'hour','month'])

# Lag features: t-1, t-2, t-3 (gives model short-term memory)
for lag in [1, 2, 3]:
    for col in ['RI_computed','total_drops','LWC','Dm','log_Z']:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Rolling mean/std over last 3 intervals
for col in ['RI_computed','LWC','log_Z']:
    df[f'{col}_rmean3'] = df[col].shift(1).rolling(3).mean()
    df[f'{col}_rstd3']  = df[col].shift(1).rolling(3).std().fillna(0)

LAG_FEATS = [c for c in df.columns if '_lag' in c or '_rmean' in c or '_rstd' in c]
FEATURES  = BASE_FEATS + LAG_FEATS

# ─────────────────────────────────────────────────────────────────
# STEP 4 — CREATE 1-HOUR AHEAD TARGET
# ─────────────────────────────────────────────────────────────────
print(f"\n[STEP 4] Creating target: RI at t + 1 hour …")
print(f"  shift(-{FORECAST_STEPS}) × 30s = {FORECAST_STEPS*30//3600} hour ahead")

df['RI_1hr'] = df['RI_computed'].shift(-FORECAST_STEPS)

# Drop NaN rows from lags and target
df = df.dropna(subset=FEATURES + ['RI_1hr']).reset_index(drop=True)

# Keep rain intervals only
df = df[(df[N_COLS].sum(axis=1) > 0) & (df['RI_computed'] > 0)].reset_index(drop=True)

X = df[FEATURES].values
y = df['RI_1hr'].values

print(f"  Samples       : {len(X):,}")
print(f"  Features      : {len(FEATURES)}")
print(f"  Target max    : {y.max():.3f} mm/h")
print(f"  Target mean   : {y.mean():.3f} mm/h")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — TIME-AWARE SPLIT (no shuffle)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Time-aware split (first 80% train / last 20% test) …")

split           = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"  Train : {len(X_train):,}   Test : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Training Random Forest (200 trees) …")

rf = RandomForestRegressor(
    n_estimators      = 200,
    max_depth         = 15,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features      = 'sqrt',
    n_jobs            = -1,
    random_state      = 42
)
rf.fit(X_train, y_train)
print("  Training complete.")

# ─────────────────────────────────────────────────────────────────
# STEP 7 — EVALUATE
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 7] Evaluating …")

y_pred = rf.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f} mm/h")
print(f"  MAE  = {mae:.4f} mm/h")
print(f"\n  Predicted RI stats:")
print(f"  Pred max   : {y_pred.max():.3f} mm/h")
print(f"  Pred mean  : {y_pred.mean():.3f} mm/h")
print(f"  Actual max : {y_test.max():.3f} mm/h")
print(f"  Actual mean: {y_test.mean():.3f} mm/h")

# ─────────────────────────────────────────────────────────────────
# STEP 8 — PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 8] Generating plots …")

# Plot 1 — Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(8, 7), facecolor=P['bg'])
ax.scatter(y_test, y_pred, alpha=0.3, s=15,
           color=P['red'], edgecolors='none')
lim = [0, max(y_test.max(), y_pred.max()) * 1.05]
ax.plot(lim, lim, '--', color=P['green'], linewidth=1.5, label='Perfect fit')
ax.set_xlim(lim); ax.set_ylim(lim)
style(ax, 'Actual vs Predicted RI — 1 Hour Ahead',
      'Actual RI t+1hr (mm/h)', 'Predicted RI t+1hr (mm/h)')
ax.text(0.05, 0.93,
        f'R²   = {r2:.4f}\n'
        f'RMSE = {rmse:.3f} mm/h\n'
        f'MAE  = {mae:.3f} mm/h\n'
        f'Horizon = 1 hr (120 × 30s)',
        transform=ax.transAxes, color=P['text'], fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=P['bg'], edgecolor=P['border']))
ax.legend(fontsize=9)
plt.tight_layout(); save('forecast_1hr_scatter.png')

# Plot 2 — Time series + residuals
n = min(300, len(y_test))
fig, axes2 = plt.subplots(2, 1, figsize=(15, 9), facecolor=P['bg'])
fig.suptitle('1-Hour RI Forecasting — Time Series',
             color=P['text'], fontsize=13, fontweight='bold')

axes2[0].plot(range(n), y_test[:n],  color=P['green'], linewidth=1.5,
              label='Actual RI (t+1hr)',    alpha=0.9)
axes2[0].plot(range(n), y_pred[:n],  color=P['red'],   linewidth=1.0,
              label='Predicted RI (t+1hr)', alpha=0.85, linestyle='--')
style(axes2[0], 'Predicted vs Actual (first 300 test samples)',
      'Sample Index', 'RI (mm/h)')
axes2[0].legend(fontsize=9)

res = y_test[:n] - y_pred[:n]
axes2[1].fill_between(range(n), res, 0, where=res >= 0,
                      color=P['green'], alpha=0.5, label='Under-predicted')
axes2[1].fill_between(range(n), res, 0, where=res <  0,
                      color=P['red'],   alpha=0.5, label='Over-predicted')
axes2[1].axhline(0, color=P['text'], linewidth=0.8, linestyle='--')
style(axes2[1], 'Residuals', 'Sample Index', 'Residual (mm/h)')
axes2[1].legend(fontsize=9)
plt.tight_layout(); save('forecast_1hr_timeseries.png')

# Plot 3 — Feature importance
importances = rf.feature_importances_
idx         = np.argsort(importances)[::-1][:25]
top_names   = [FEATURES[i] for i in idx]
top_imp     = importances[idx]

fig, ax = plt.subplots(figsize=(12, 8), facecolor=P['bg'])
bars = ax.barh(range(len(top_names)), top_imp[::-1],
               color=P['red'], edgecolor=P['border'], alpha=0.85)
for b in list(bars)[::-1][:3]: b.set_color(P['orange'])
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names[::-1], color=P['text'], fontsize=9)
for b, v in zip(bars[::-1], top_imp[::-1]):
    ax.text(v + max(top_imp)*0.005, b.get_y()+b.get_height()/2,
            f'{v:.4f}', va='center', color=P['sub'], fontsize=8)
style(ax, '1-Hour Forecasting — Feature Importance (Top 25)\n'
         'Lag features show temporal contribution', 'Score', 'Feature')
plt.tight_layout(); save('forecast_1hr_feature_importance.png')

# Plot 4 — RI distribution: actual vs predicted
fig, ax = plt.subplots(figsize=(9, 5), facecolor=P['bg'])
ax.hist(y_test, bins=60, color=P['green'], edgecolor=P['border'],
        alpha=0.6, density=True, label='Actual RI t+1hr')
ax.hist(y_pred, bins=60, color=P['red'],   edgecolor=P['border'],
        alpha=0.6, density=True, label='Predicted RI t+1hr')
style(ax, 'RI Distribution — Actual vs Predicted (1hr ahead)',
      'RI (mm/h)', 'Density')
ax.legend(fontsize=9)
ax.text(0.97, 0.96,
        f'Actual  mean = {y_test.mean():.2f} mm/h\n'
        f'Predict mean = {y_pred.mean():.2f} mm/h',
        transform=ax.transAxes, ha='right', va='top',
        color=P['text'], fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=P['bg'], edgecolor=P['border']))
plt.tight_layout(); save('forecast_1hr_distribution.png')

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  1-HOUR FORECASTING MODEL — FINAL REPORT")
print("=" * 62)
print(f"  Target            : Computed RI (DSD physics formula)")
print(f"  Forecast horizon  : 1 hour (120 × 30s steps)")
print(f"  Training samples  : {len(X_train):,}")
print(f"  Test samples      : {len(X_test):,}")
print(f"  Features used     : {len(FEATURES)}")
print(f"")
print(f"  RI Statistics:")
print(f"  Actual max        : {y_test.max():.3f} mm/h")
print(f"  Actual mean       : {y_test.mean():.3f} mm/h")
print(f"  Predicted mean    : {y_pred.mean():.3f} mm/h")
print(f"")
print(f"  Model Performance:")
print(f"  R² Score          : {r2:.4f}")
print(f"  RMSE              : {rmse:.4f} mm/h")
print(f"  MAE               : {mae:.4f} mm/h")
print(f"")
print(f"  Top 5 features:")
for rank, i in enumerate(idx[:5], 1):
    print(f"    {rank}. {FEATURES[i]:<22} → {importances[i]:.4f}")
print(f"\n  Plots saved to: {PLOT_DIR}/")
print("=" * 62)