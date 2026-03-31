

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FILE = "processed_data.csv"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# RD-80 physical constants — same as comparism.py
F       = 0.005           # sensor area [m²]
F_cm2   = F * 1e4         # convert to cm² → 50 cm²
t_s     = 30              # sampling interval [seconds]

Di  = np.array([0.359,0.455,0.551,0.656,0.771,0.917,1.131,1.331,1.506,
                1.665,1.912,2.259,2.589,2.869,3.205,3.544,3.916,4.350,
                4.859,5.373])        # drop diameters [mm]
Vi  = np.array([0.87,1.34,1.79,2.17,2.55,2.90,3.20,3.46,3.72,
                4.01,4.60,5.39,6.34,7.06,7.58,8.01,8.35,8.61,
                8.81,8.94])          # fall velocities [m/s]
dDi = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
                0.250,0.500,0.750,0.500,0.500,0.500,0.500,0.500,0.500,
                0.500,0.500])        # bin widths [mm]

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
print("  RANDOM FOREST — CURRENT RI PREDICTION")
print("=" * 62)
print("\n[STEP 1] Loading & cleaning data …")

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
# STEP 2 — COMPUTE RI TARGET (same formula as comparism.py)
# ─────────────────────────────────────────────────────────────────
# This is the SAME formula used in comparism.py that gives:
#   max  ~300 mm/h
#   mean ~41  mm/h
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Computing RI target (DSD physics formula) …")

N_mat  = df[N_COLS].values.astype(float)
factor = (np.pi / 6.0) * (3.6e3 / (F_cm2 * t_s))
df['RI_computed'] = factor * np.sum(N_mat * (Di ** 3), axis=1)

# Cap at 300 mm/h — same as comparism.py
df.loc[df['RI_computed'] > 300, 'RI_computed'] = np.nan
df['RI_computed'] = df['RI_computed'].fillna(0)

# Keep only rain intervals (drop counts > 0 AND RI > 0)
df = df[(df[N_COLS].sum(axis=1) > 0) & (df['RI_computed'] > 0)].reset_index(drop=True)

print(f"  Rain intervals: {len(df):,}")
print(f"  RI max        : {df['RI_computed'].max():.3f} mm/h")
print(f"  RI mean       : {df['RI_computed'].mean():.3f} mm/h")
print(f"  RI std        : {df['RI_computed'].std():.3f} mm/h")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Engineering features …")

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

FEATURES = (N_COLS +
            ['total_drops','log_drops','LWC','Dm','D_mean','Z','log_Z',
             'hour','month'])
TARGET   = 'RI_computed'

X = df[FEATURES].values
y = df[TARGET].values

print(f"  Features      : {len(FEATURES)}")
print(f"  Samples       : {len(X):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Train / Test split (80% / 20%) …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"  Train samples : {len(X_train):,}")
print(f"  Test samples  : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Training Random Forest (200 trees) …")

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
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Evaluating model …")

y_pred = rf.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"  R²   = {r2:.4f}   (1.0 = perfect)")
print(f"  RMSE = {rmse:.4f} mm/h")
print(f"  MAE  = {mae:.4f} mm/h")

print("  Running 5-fold cross validation …")
cv = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f"  CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

# Predicted vs actual RI statistics
print(f"\n  Predicted RI stats (test set):")
print(f"  Pred max  : {y_pred.max():.3f} mm/h")
print(f"  Pred mean : {y_pred.mean():.3f} mm/h")
print(f"  Actual max: {y_test.max():.3f} mm/h")
print(f"  Actual mean:{y_test.mean():.3f} mm/h")

# ─────────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 7] Generating plots …")

# Plot 1 — Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(8, 7), facecolor=P['bg'])
ax.scatter(y_test, y_pred, alpha=0.3, s=15,
           color=P['blue'], edgecolors='none')
lim = [0, max(y_test.max(), y_pred.max()) * 1.05]
ax.plot(lim, lim, '--', color=P['green'], linewidth=1.5, label='Perfect fit')
ax.set_xlim(lim); ax.set_ylim(lim)
style(ax, 'Actual vs Predicted RI (Computed, Current Interval)',
      'Actual RI (mm/h)', 'Predicted RI (mm/h)')
ax.text(0.05, 0.93,
        f'R²   = {r2:.4f}\n'
        f'RMSE = {rmse:.3f} mm/h\n'
        f'MAE  = {mae:.3f} mm/h\n'
        f'n    = {len(y_test):,}',
        transform=ax.transAxes, color=P['text'], fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=P['bg'], edgecolor=P['border']))
ax.legend(fontsize=9)
plt.tight_layout(); save('rf_actual_vs_predicted.png')

# Plot 2 — Residuals + time series sample
residuals = y_test - y_pred
fig, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor=P['bg'])

axes2[0].hist(residuals, bins=60, color=P['blue'],
              edgecolor=P['border'], alpha=0.85, density=True)
axes2[0].axvline(0, color=P['orange'], linewidth=1.5, linestyle='--')
style(axes2[0], 'Residual Distribution', 'Residual (mm/h)', 'Density')
axes2[0].text(0.97, 0.96,
              f'μ = {residuals.mean():.3f}\nσ = {residuals.std():.3f}',
              transform=axes2[0].transAxes, ha='right', va='top',
              color=P['text'], fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', facecolor=P['bg'],
                        edgecolor=P['border']))

n = min(300, len(y_test))
axes2[1].plot(range(n), y_test[:n],  color=P['green'], linewidth=1.2,
              label='Actual RI',     alpha=0.9)
axes2[1].plot(range(n), y_pred[:n],  color=P['blue'],  linewidth=1.0,
              label='Predicted RI',  alpha=0.85, linestyle='--')
style(axes2[1], 'Sample Predictions (first 300 test points)',
      'Sample', 'RI (mm/h)')
axes2[1].legend(fontsize=9)

plt.tight_layout(); save('rf_residuals.png')

# Plot 3 — Feature importance
importances = rf.feature_importances_
idx         = np.argsort(importances)[::-1][:20]
top_names   = [FEATURES[i] for i in idx]
top_imp     = importances[idx]

fig, ax = plt.subplots(figsize=(11, 7), facecolor=P['bg'])
bars = ax.barh(range(len(top_names)), top_imp[::-1],
               color=P['blue'], edgecolor=P['border'], alpha=0.85)
for b in list(bars)[::-1][:3]: b.set_color(P['orange'])
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names[::-1], color=P['text'], fontsize=9)
for b, v in zip(bars[::-1], top_imp[::-1]):
    ax.text(v + max(top_imp)*0.005, b.get_y()+b.get_height()/2,
            f'{v:.4f}', va='center', color=P['sub'], fontsize=8)
style(ax, 'Random Forest — Feature Importance (Top 20)', 'Score', 'Feature')
plt.tight_layout(); save('rf_feature_importance.png')

# Plot 4 — RI distribution: actual vs predicted
fig, ax = plt.subplots(figsize=(9, 5), facecolor=P['bg'])
ax.hist(y_test, bins=60, color=P['green'], edgecolor=P['border'],
        alpha=0.6, density=True, label='Actual RI')
ax.hist(y_pred, bins=60, color=P['blue'],  edgecolor=P['border'],
        alpha=0.6, density=True, label='Predicted RI')
style(ax, 'RI Distribution — Actual vs Predicted', 'RI (mm/h)', 'Density')
ax.legend(fontsize=9)
ax.text(0.97, 0.96,
        f'Actual  mean = {y_test.mean():.2f} mm/h\n'
        f'Predict mean = {y_pred.mean():.2f} mm/h',
        transform=ax.transAxes, ha='right', va='top',
        color=P['text'], fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=P['bg'], edgecolor=P['border']))
plt.tight_layout(); save('rf_ri_distribution.png')

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  CURRENT RI MODEL — FINAL REPORT")
print("=" * 62)
print(f"  Target            : Computed RI (DSD physics formula)")
print(f"  Training samples  : {len(X_train):,}")
print(f"  Test samples      : {len(X_test):,}")
print(f"  Features used     : {len(FEATURES)}")
print(f"")
print(f"  RI Statistics (matches comparism.py):")
print(f"  Max RI            : {df['RI_computed'].max():.3f} mm/h")
print(f"  Mean RI           : {df['RI_computed'].mean():.3f} mm/h")
print(f"")
print(f"  Model Performance:")
print(f"  R² Score          : {r2:.4f}")
print(f"  RMSE              : {rmse:.4f} mm/h")
print(f"  MAE               : {mae:.4f} mm/h")
print(f"  5-Fold CV R²      : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"")
print(f"  Top 5 features:")
for rank, i in enumerate(idx[:5], 1):
    print(f"    {rank}. {FEATURES[i]:<18} → {importances[i]:.4f}")
print(f"\n  Plots saved to: {PLOT_DIR}/")
print("=" * 62)