import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestRegressor
from sklearn.multioutput     import MultiOutputRegressor
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
F      = 0.005
F_cm2  = F * 1e4      # 50 cm²
t_s    = 30

Di  = np.array([0.359,0.455,0.551,0.656,0.771,0.917,1.131,1.331,1.506,
                1.665,1.912,2.259,2.589,2.869,3.205,3.544,3.916,4.350,
                4.859,5.373])
Vi  = np.array([0.87,1.34,1.79,2.17,2.55,2.90,3.20,3.46,3.72,
                4.01,4.60,5.39,6.34,7.06,7.58,8.01,8.35,8.61,
                8.81,8.94])
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

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  ML TASK 2 — NEXT-INTERVAL DSD PARAMETER FORECASTING")
print("  Predict Dm, Z, LWC at t+1 given DSD state at t")
print("  Train: 2010–2014  |  Test: 2015")
print("=" * 62)

print("\n[STEP 1] Loading data...")
df = pd.read_csv("processed_data.csv", low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

print(f"  Loaded rows : {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE DSD PARAMETERS FOR EVERY INTERVAL
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Computing DSD parameters for all intervals...")

N_mat = df[N_COLS].values.astype(float)
ND    = N_mat / (F * t_s * Vi * dDi)       # N(D) [m⁻³ mm⁻¹]

# Rain Rate R [mm/h]
R = (np.pi/6.0) * (3.6e3 / (F_cm2 * t_s)) * np.sum(N_mat * (Di**3), axis=1)

# Radar Reflectivity Z [mm⁶ m⁻³]
Z = np.sum(ND * (Di**6) * dDi, axis=1)

# Mass-weighted Mean Diameter Dm [mm]
num_Dm = np.sum(ND * (Di**4) * dDi, axis=1)
den_Dm = np.sum(ND * (Di**3) * dDi, axis=1)
Dm     = np.where(den_Dm > 0, num_Dm / den_Dm, 0)

# Liquid Water Content LWC [g/m³]
LWC = (np.pi/6) * 1e-3 * np.sum(ND * (Di**3) * dDi, axis=1)

# Log transforms (stabilise heavy-tailed distributions)
log_Z   = np.log1p(Z)
log_LWC = np.log1p(LWC)

df['R']       = R
df['Z']       = Z
df['Dm']      = Dm
df['LWC']     = LWC
df['log_Z']   = log_Z
df['log_LWC'] = log_LWC
df['hour']    = df['timestamp'].dt.hour
df['month']   = df['timestamp'].dt.month

# Keep valid rain intervals only
mask = (R > 0.1) & (Z > 0) & (R < 300) & (N_mat.sum(axis=1) > 0)
df   = df[mask].reset_index(drop=True)
print(f"  Valid rain intervals: {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — BUILD LAG FEATURES
# ─────────────────────────────────────────────────────────────────
# KEY IDEA: The model sees the DSD at time t and predicts
# the DSD parameters at time t+1 (30 seconds ahead).
#
# Features at time t:
#   - Raw drop counts n1-n20 (the DSD shape)
#   - Derived parameters: Dm, Z, LWC, log_Z, log_LWC
#   - Lag-1 values of Dm, Z, LWC (t-1 state, for trend context)
#   - Time context: hour, month
#
# Targets at time t+1:
#   - Dm_next, log_Z_next, log_LWC_next
#
# Why log_Z and log_LWC as targets?
#   Z and LWC are heavy-tailed. Training on raw values would
#   make the model obsess over rare extreme events.
#   Log-transforming stabilises the target distribution.
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Building lag features and next-interval targets...")

# Lag-1 features (previous interval's parameters)
df['Dm_lag1']      = df['Dm'].shift(1)
df['log_Z_lag1']   = df['log_Z'].shift(1)
df['log_LWC_lag1'] = df['log_LWC'].shift(1)

# Next-interval targets (what we want to predict)
df['Dm_next']      = df['Dm'].shift(-1)
df['log_Z_next']   = df['log_Z'].shift(-1)
df['log_LWC_next'] = df['log_LWC'].shift(-1)

# Drop rows with NaN (first and last row after shifting)
df = df.dropna(subset=['Dm_lag1', 'Dm_next']).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────
# CRITICAL: Only keep CONSECUTIVE intervals.
# If there's a gap in time (e.g., no rain for an hour), the
# "next" row is not actually the next physical interval.
# We must not predict across gaps — that would be nonsense.
# ─────────────────────────────────────────────────────────────────
time_diff = df['timestamp'].diff().dt.total_seconds()
consecutive = (time_diff <= 60)          # allow up to 60s (2 intervals)
df = df[consecutive].reset_index(drop=True)
print(f"  Consecutive-interval pairs: {len(df):,}")

FEATURES = (
    N_COLS +
    ['Dm', 'log_Z', 'log_LWC',          # current DSD parameters
     'Dm_lag1', 'log_Z_lag1', 'log_LWC_lag1',  # previous interval (trend)
     'hour', 'month']                    # temporal context
)

TARGETS = ['Dm_next', 'log_Z_next', 'log_LWC_next']

print(f"  Features : {len(FEATURES)}")
print(f"  Targets  : Dm_next, log_Z_next, log_LWC_next")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — YEAR-BASED TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Year-based train/test split (train 2010-2014, test 2015)...")

train_df = df[df['timestamp'].dt.year <= 2014].copy()
test_df  = df[df['timestamp'].dt.year == 2015].copy()

X_train = train_df[FEATURES].values
X_test  = test_df[FEATURES].values

# Separate targets
y_train = train_df[TARGETS].values   # shape (n, 3)
y_test  = test_df[TARGETS].values

print(f"  Train samples : {len(X_train):,}")
print(f"  Test samples  : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN MULTI-OUTPUT RANDOM FOREST
# ─────────────────────────────────────────────────────────────────
# MultiOutputRegressor trains ONE separate forest per target.
# This is better than a single forest because Dm, Z, and LWC
# have different scales and relationships with the features.
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Training Multi-Output Random Forest...")

rf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators      = 200,
        max_depth         = 15,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = 'sqrt',
        n_jobs            = -1,
        random_state      = 42
    ),
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("  Training complete.")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Evaluating on 2015 test set...")

y_pred = rf.predict(X_test)     # shape (n, 3)

target_names = ['Dm_next (mm)', 'log_Z_next', 'log_LWC_next']

print(f"\n  {'Target':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("  " + "-"*50)

metrics = {}
for i, tname in enumerate(target_names):
    r2   = r2_score(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
    metrics[tname] = {'r2': r2, 'rmse': rmse, 'mae': mae}
    print(f"  {tname:<20} {r2:>8.4f} {rmse:>10.4f} {mae:>10.4f}")

# ─────────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 7] Generating plots...")

# ── Plot 1: Actual vs Predicted scatter (3 targets) ──────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=P['bg'])
colors = [P['blue'], P['red'], P['green']]

for i, (ax, tname, col) in enumerate(zip(axes, target_names, colors)):
    act  = y_test[:, i]
    pred = y_pred[:, i]
    # Sample for scatter (max 5000 points)
    idx  = np.random.choice(len(act), min(5000, len(act)), replace=False)
    ax.scatter(act[idx], pred[idx], alpha=0.2, s=8,
               color=col, edgecolors='none')
    lim = [min(act.min(), pred.min())*0.95,
           max(act.max(), pred.max())*1.05]
    ax.plot(lim, lim, '--', color='#555555', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(lim); ax.set_ylim(lim)
    r2 = metrics[tname]['r2']
    rmse = metrics[tname]['rmse']
    style(ax, f'Actual vs Predicted\n{tname}',
          f'Actual {tname}', f'Predicted {tname}')
    ax.text(0.05, 0.93, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
            transform=ax.transAxes, fontsize=9, va='top',
            color=P['text'],
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=P['bg'], edgecolor=P['border']))
    ax.legend(fontsize=8)

plt.suptitle('Next-Interval DSD Parameter Forecasting — 2015 Test Set',
             fontsize=13, fontweight='bold', color=P['text'])
plt.tight_layout()
plt.show()

# ── Plot 2: Time series sample — 200 consecutive intervals ───────
# Pick a monsoon month (July 2015) for a meaningful sequence
test_july = test_df[test_df['timestamp'].dt.month == 7].copy()
test_july['pred_Dm']      = y_pred[test_df['timestamp'].dt.month == 7, 0]
test_july['pred_log_Z']   = y_pred[test_df['timestamp'].dt.month == 7, 1]
test_july['pred_log_LWC'] = y_pred[test_df['timestamp'].dt.month == 7, 2]

n_show = min(300, len(test_july))
ts     = test_july.iloc[:n_show]

fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                         sharex=True, facecolor=P['bg'])

pairs = [
    ('Dm_next',      'pred_Dm',      'Dm (mm)',        P['blue']),
    ('log_Z_next',   'pred_log_Z',   'log Z',          P['red']),
    ('log_LWC_next', 'pred_log_LWC', 'log LWC (g/m³)', P['green']),
]

for ax, (actual_col, pred_col, ylabel, col) in zip(axes, pairs):
    ax.plot(ts['timestamp'], ts[actual_col],
            color=col, linewidth=1.2, label='Actual', alpha=0.9)
    ax.plot(ts['timestamp'], ts[pred_col],
            color=P['orange'], linewidth=1.0, linestyle='--',
            label='Predicted (t+1)', alpha=0.85)
    style(ax, '', 'Time (July 2015)', ylabel)
    ax.legend(fontsize=9)

plt.suptitle('Forecasting Sample — 300 Consecutive Intervals (July 2015)',
             fontsize=13, fontweight='bold', color=P['text'])
plt.tight_layout()
plt.show()

# ── Plot 3: Residuals ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=P['bg'])
colors = [P['blue'], P['red'], P['green']]

for i, (ax, tname, col) in enumerate(zip(axes, target_names, colors)):
    residuals = y_test[:, i] - y_pred[:, i]
    ax.hist(residuals, bins=60, color=col,
            edgecolor=P['border'], alpha=0.85, density=True)
    ax.axvline(0, color=P['orange'], linewidth=1.5, linestyle='--')
    style(ax, f'Residuals — {tname}', 'Residual', 'Density')
    ax.text(0.97, 0.96,
            f'μ = {residuals.mean():.4f}\nσ = {residuals.std():.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color=P['text'],
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=P['bg'], edgecolor=P['border']))

plt.suptitle('Residual Distributions — Next-Interval Forecasting',
             fontsize=13, fontweight='bold', color=P['text'])
plt.tight_layout()
plt.show()

# ── Plot 4: Feature importance per target ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=P['bg'])
colors = [P['blue'], P['red'], P['green']]

for i, (ax, tname, col) in enumerate(zip(axes, target_names, colors)):
    imp     = rf.estimators_[i].feature_importances_
    feat_df = pd.DataFrame({'feature': FEATURES, 'importance': imp})
    feat_df = feat_df.sort_values('importance', ascending=False).head(12)

    bar_colors = [P['orange'] if j < 3 else col
                  for j in range(len(feat_df))]
    ax.barh(range(len(feat_df)),
            feat_df['importance'].values[::-1],
            color=bar_colors[::-1],
            edgecolor=P['border'], alpha=0.85)
    ax.set_yticks(range(len(feat_df)))
    ax.set_yticklabels(feat_df['feature'].values[::-1],
                       color=P['text'], fontsize=8)
    style(ax, f'Feature Importance\n{tname}',
          'Importance', 'Feature')
    ax.text(0.97, 0.05, 'Orange = top 3',
            transform=ax.transAxes, ha='right',
            fontsize=8, color=P['sub'])

plt.suptitle('Feature Importance per Forecasting Target',
             fontsize=13, fontweight='bold', color=P['text'])
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  ML TASK 2 — FINAL REPORT")
print("=" * 62)
print(f"  Task       : Forecast Dm, log_Z, log_LWC at t+1")
print(f"  Train      : 2010–2014  ({len(X_train):,} interval pairs)")
print(f"  Test       : 2015       ({len(X_test):,} interval pairs)")
print(f"  Features   : {len(FEATURES)}")
print()
print(f"  Performance:")
print(f"  {'Target':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("  " + "-"*50)
for tname, m in metrics.items():
    print(f"  {tname:<20} {m['r2']:>8.4f} {m['rmse']:>10.4f} {m['mae']:>10.4f}")
print()
print("  Physical interpretation:")
print("  The model is learning how rain EVOLVES over time.")
print("  Key insight: current Dm and log_Z are the strongest")
print("  predictors of next-interval Dm and log_Z respectively.")
print("  This means rain tends to change gradually —")
print("  sudden jumps in DSD parameters are rare.")
print("  The model captures this temporal persistence.")
print("=" * 62)
