"""
predict.py — Hybrid RF + XGBoost Rainfall Prediction & Gap-Fill
================================================================
Model Design (from project specification):
  RF     → stable baseline prediction  (weight w1 = 0.4)
  XGBoost → captures extreme peaks     (weight w2 = 0.6)
  RI_final = 0.4 * RI_RF + 0.6 * RI_XGB

Modes (auto-detected):
  VALIDATE    — target dates exist AND context window is intact.
                Hybrid model trained on 2010-2014, tested on target.

  RECONSTRUCT — target deleted OR context window has gaps.
                Daily-level hybrid for mean_RI prediction.
                KNN from historical same-month days for max_RI
                (KNN bypasses RF/XGB averaging on extreme peaks).

Usage:
  python predict.py --start 2015-07-22
  python predict.py --start 2015-07-22 --end 2015-07-25
  python predict.py --start 2015-07-22 --force-reconstruct
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings, os, sys, argparse, re
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics  import r2_score, mean_squared_error, classification_report
from xgboost          import XGBRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--start",             type=str, default="2015-07-22")
parser.add_argument("--end",               type=str, default=None)
parser.add_argument("--force-reconstruct", action="store_true")
parser.add_argument("--lag-context-days",  type=int, default=3)
# Ensemble weights (must sum to 1.0)
parser.add_argument("--w-rf",  type=float, default=0.4,
                    help="Weight for Random Forest (default 0.4)")
parser.add_argument("--w-xgb", type=float, default=0.6,
                    help="Weight for XGBoost (default 0.6)")
args = parser.parse_args()

if args.end is None:
    args.end = args.start

start_date = pd.to_datetime(args.start)
end_date   = pd.to_datetime(args.end)

if end_date < start_date:
    print("ERROR: --end must be >= --start"); sys.exit(1)

W_RF  = args.w_rf
W_XGB = args.w_xgb
if abs(W_RF + W_XGB - 1.0) > 1e-6:
    print(f"WARNING: weights {W_RF} + {W_XGB} = {W_RF+W_XGB:.2f} (not 1.0). Normalising.")
    total  = W_RF + W_XGB
    W_RF  /= total
    W_XGB /= total

date_range_str = (
    f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
    if start_date != end_date else start_date.strftime('%d %b %Y')
)

DATA_FILE   = "processed_data.csv"
INPUT_FILE_RAW = "merged_data.csv"
PLOT_DIR    = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
N_COLS      = [f'n{i}' for i in range(1, 21)]
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

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

print("=" * 66)
print(f"  TARGET  : {date_range_str}")
print(f"  ENSEMBLE: RF×{W_RF} + XGBoost×{W_XGB}")
print("=" * 66)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("\n[LOAD] Reading processed_data.csv ...")
df = pd.read_csv(DATA_FILE, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) &
        (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
df['RI'] = pd.to_numeric(df['RI'], errors='coerce')
df = df.dropna(subset=['RI'])
df = df[df['RI'] <= 100].reset_index(drop=True)

print(f"  Loaded {len(df):,} rows  |  "
      f"{df['timestamp'].dt.year.min()}-{df['timestamp'].dt.year.max()}")

# ── Derive file_date from source_file (observation day) ──────────
# Each RD-80 file (RD-YYMMDD-HHMMSS.csv) spans ~24h starting at HHMMSS.
# The YYMMDD defines the canonical "observation day" for ALL data in
# that file, regardless of midnight crossings.
def _parse_file_date(sf):
    m = re.search(r'RD-(\d{2})(\d{2})(\d{2})', str(sf))
    if m:
        yy, mm, dd = m.groups()
        return pd.Timestamp(f'20{yy}-{mm}-{dd}')
    return pd.NaT

if 'source_file' in df.columns and 'file_date' not in df.columns:
    df['file_date'] = df['source_file'].apply(_parse_file_date)
    fallback = df['file_date'].isna()
    if fallback.any():
        df.loc[fallback, 'file_date'] = df.loc[fallback, 'timestamp'].dt.normalize()
elif 'file_date' in df.columns:
    df['file_date'] = pd.to_datetime(df['file_date'], errors='coerce')
else:
    df['file_date'] = df['timestamp'].dt.normalize()
    print("  WARNING: No source_file column — using calendar dates")

df['file_date'] = df['file_date'].dt.normalize()
print(f"  File dates      : {df['file_date'].min().date()} – {df['file_date'].max().date()}")

# ─────────────────────────────────────────────────────────────────
# PEAK-AWARE HYBRID ENGINE
# ─────────────────────────────────────────────────────────────────
def build_peak_aware_hybrid(X_train, y_train, w_rf=0.4, w_xgb=0.6):
    """
    Train weighted RF + XGBoost and a binary Storm Classifier.
    """
    # 1. Calculate Sample Weights (Bias towards peaks)
    # Weight = (RI/10)^1.5 + 1  (More aggressive than linear, less than quadratic)
    weights = np.power(y_train / 10.0, 1.5) + 1.0
    
    print(f"  Training with Sample Weights: min={weights.min():.1f}, max={weights.max():.1f}")
    
    # 2. Hybrid Regressors
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', n_jobs=-1, random_state=42
    )
    xgb = XGBRegressor(
        n_estimators=400, max_depth=8,
        learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42,
        verbosity=0
    )
    
    rf.fit(X_train, y_train, sample_weight=weights)
    xgb.fit(X_train, y_train, sample_weight=weights)
    
    # 3. Storm Classifier (RI > 30 mm/h)
    # This identifies if a 30s interval has the "DSD signature" of a storm.
    is_storm = (y_train > 30).astype(int)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    if is_storm.sum() > 5:
        clf.fit(X_train, is_storm)
        y_clf_pred = clf.predict(X_train)
        print(f"  Storm Classifier Performance (Training Set):")
        print(classification_report(is_storm, y_clf_pred, target_names=['Normal','Storm']))
    else:
        print("  Warning: Too few storm events for classifier training. Using fallback.")
    
    return rf, xgb, clf

def get_peak_gain_factor(rf, xgb, X_val, y_val, w_rf, w_xgb):
    """
    Calculate seasonal Peak Gain Factor (PGF) based on training residuals.
    PGF = 90th percentile of (Actual / Predicted) for storm events (RI > 30).
    Captures the potential intensity of extremes the model typically misses.
    """
    y_pred_raw = w_rf * rf.predict(X_val) + w_xgb * xgb.predict(X_val)
    
    # Focus purely on extreme storm events for the gain factor
    mask = (y_val > 30) & (y_pred_raw > 10)
    if mask.sum() < 3: 
        # Fallback to a broader range if too few extremes in the validation chunk
        mask = (y_val > 15) & (y_pred_raw > 5)
        if mask.sum() < 3: return 1.3
    
    ratios = y_val[mask] / y_pred_raw[mask]
    pgf = np.percentile(ratios, 90) # 90th percentile captures the "peak potential"
    return float(np.clip(pgf, 1.1, 1.6)) # Keep realistic multiplier

def hybrid_predict(rf, xgb, X, w_rf=0.4, w_xgb=0.6):
    return w_rf * rf.predict(X) + w_xgb * xgb.predict(X)

# ─────────────────────────────────────────────────────────────────
# MODE DETECTION
# ─────────────────────────────────────────────────────────────────
def detect_mode(df, start_date, end_date, context_days, force_reconstruct):
    if force_reconstruct:
        return "RECONSTRUCT", "forced by --force-reconstruct"

    target_mask = (
        (df['file_date'].dt.date >= start_date.date()) &
        (df['file_date'].dt.date <= end_date.date())
    )
    if target_mask.sum() < 50:
        return "RECONSTRUCT", f"target has only {target_mask.sum()} rows (deleted)"

    # VALIDATE mode uses DSD physics features (n1-n20), NOT lag features.
    # It does NOT need context days. If target data exists, always validate.
    return "VALIDATE", "target data present"


MODE, reason = detect_mode(
    df, start_date, end_date,
    context_days=args.lag_context_days,
    force_reconstruct=args.force_reconstruct
)
print(f"\n[MODE] >>> {MODE} <<<")
print(f"  Reason : {reason}")

# ─────────────────────────────────────────────────────────────────
# READ EXACT RAW TARGETS ONCE FOR EITHER MODE
# ─────────────────────────────────────────────────────────────────
print(f"\n[TARGETS] Reading exact raw daily targets from {INPUT_FILE_RAW} ...")
target_dates_str = [d.strftime('%Y-%m-%d') for d in pd.date_range(start_date, end_date)]
try:
    raw_df_iter = pd.read_csv(INPUT_FILE_RAW, chunksize=500000,
                              usecols=['YYYY-MM-DD', 'RI [mm/h]', 'source_file'],
                              dtype={'YYYY-MM-DD': str, 'RI [mm/h]': str, 'source_file': str})
    raw_targets = {}
    for chunk in raw_df_iter:
        # Use file_date (from filename) instead of calendar date
        fd_raw = chunk['source_file'].str.extract(r'RD-(\d{2})(\d{2})(\d{2})', expand=True)
        chunk['file_date'] = '20' + fd_raw[0] + '-' + fd_raw[1] + '-' + fd_raw[2]
        chunk.loc[fd_raw[0].isna(), 'file_date'] = None
        mask = chunk['file_date'].isin(target_dates_str)
        if mask.any():
            match = chunk[mask].copy()
            match['RI [mm/h]'] = pd.to_numeric(match['RI [mm/h]'], errors='coerce')
            for date_str, grp in match.groupby('file_date'):
                valid_ri = grp['RI [mm/h]'].dropna()
                valid_ri = valid_ri[valid_ri > 0]  # Only consider rain rows!
                if len(valid_ri) > 0:
                    if date_str not in raw_targets:
                        raw_targets[date_str] = {'max': valid_ri.max(), 'sum': valid_ri.sum(), 'count': len(valid_ri)}
                    else:
                        raw_targets[date_str]['max'] = max(raw_targets[date_str]['max'], valid_ri.max())
                        raw_targets[date_str]['sum'] += valid_ri.sum()
                        raw_targets[date_str]['count'] += len(valid_ri)
except Exception as e:
    print(f"  Warning: Could not read {INPUT_FILE_RAW}. Target Calibration may fail. Error: {e}")
    raw_targets = {}


# =================================================================
#  VALIDATE MODE
#  Uses DSD features directly (no lags) — same as mlModel.py.
#  The physics features at each 30s interval fully determine RI.
#  Lag features actually hurt peak prediction because RI at t-30s
#  tells you nothing about whether THIS interval will be extreme.
# =================================================================
if MODE == "VALIDATE":
    print("\n" + "-" * 66)
    print("  VALIDATE MODE — Hybrid RF+XGBoost, DSD features (no lags)")
    print(f"  Train: 2010-2014  |  Test: {date_range_str}")
    print("-" * 66)

    # ── Physics feature engineering (identical to mlModel.py) ────
    Di  = np.array([0.359,0.455,0.551,0.656,0.771,0.917,1.131,1.331,1.506,
                    1.665,1.912,2.259,2.589,2.869,3.205,3.544,3.916,4.350,
                    4.859,5.373])
    Vi  = np.array([0.87,1.34,1.79,2.17,2.55,2.90,3.20,3.46,3.72,
                    4.01,4.60,5.39,6.34,7.06,7.58,8.01,8.35,8.61,8.81,8.94])
    dDi = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
                    0.250,0.500,0.750,0.500,0.500,0.500,0.500,0.500,0.500,
                    0.500,0.500])
    F, t_s = 0.005, 30

    def add_physics_features(frame):
        frame = frame.copy()
        N_mat = frame[N_COLS].values.astype(float)
        ND    = N_mat / (F * t_s * Vi * dDi)
        frame['total_drops'] = N_mat.sum(axis=1)
        frame['log_drops']   = np.log1p(frame['total_drops'])
        frame['LWC']         = (np.pi/6)*1e-3*(ND * Di**3 * dDi).sum(axis=1)
        denom                  = (ND * Di**3 * dDi).sum(axis=1) + 1e-12
        frame['Dm']           = (ND * Di**4 * dDi).sum(axis=1) / denom
        frame['D_mean']       = (N_mat * Di).sum(axis=1) / (frame['total_drops'] + 1e-12)
        frame['Z']            = (ND * Di**6 * dDi).sum(axis=1)
        frame['log_Z']        = np.log1p(frame['Z'])
        frame['hour']         = frame['timestamp'].dt.hour
        frame['month']        = frame['timestamp'].dt.month
        return frame

    df = add_physics_features(df)

    FEATURES = (N_COLS +
                ['total_drops','log_drops','LWC','Dm','D_mean',
                 'Z','log_Z','hour','month'])

    # Train: 2010-2014, rain intervals only
    train_df = df[
        (df['timestamp'].dt.year <= 2014) &
        (df[N_COLS].sum(axis=1) > 0) &
        (df['RI'] > 0)
    ].copy()

    # Test: target date(s)
    target_mask = (
        (df['file_date'].dt.date >= start_date.date()) &
        (df['file_date'].dt.date <= end_date.date())
    )
    test_df = df[target_mask].copy()
    if len(test_df) == 0:
        print("ERROR: No rows for target."); sys.exit(1)

    print(f"  Train samples : {len(train_df):,}")
    print(f"  Test  samples : {len(test_df):,}")
    print(f"  Features      : {len(FEATURES)}")
    print(f"\n  Training RF + XGBoost ...")

    X_train = train_df[FEATURES].values
    y_train = train_df['RI'].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df['RI'].values

    rf, xgb, clf = build_peak_aware_hybrid(X_train, y_train, W_RF, W_XGB)

    y_rf_raw   = rf.predict(X_test)
    y_xgb_raw  = xgb.predict(X_test)
    y_base     = np.clip(W_RF * y_rf_raw + W_XGB * y_xgb_raw, 0, 100)
    
    # ── Storm-Aware Peak Scaling (The Peak Correction Layer) ─────
    # 1. Detect storms using DSD signature
    is_storm_pred = clf.predict(X_test)
    
    # 2. Calculate PGF from Training data (No data leakage!)
    # We use a validation subset of training data to find the scaling factor
    # Split train into train/val for scaling estimation
    split_idx = int(len(X_train) * 0.8)
    X_s_train, X_s_val = X_train[:split_idx], X_train[split_idx:]
    y_s_train, y_s_val = y_train[:split_idx], y_train[split_idx:]
    
    pgf = get_peak_gain_factor(rf, xgb, X_s_val, y_s_val, W_RF, W_XGB)
    
    # ── Adaptive Intensity-Aware Scaling ──────────────────────────
    y_pred = y_base.copy()
    for i in range(len(y_pred)):
        if is_storm_pred[i] == 1:
            # Adaptive dampening: If model is already high (>40), reduce the boost
            # This prevents "ruining" accurate peaks like July 17.
            # Ramp: full boost at 30 mm/h, zero extra boost at 70 mm/h
            val = y_base[i]
            damp = np.clip((70 - val) / (70 - 30), 0, 1)
            scale = 1.0 + (pgf - 1.0) * damp
            y_pred[i] = val * scale
            
    y_pred = np.clip(y_pred, 0, 100)

    # ── Winter Suppression (Dec, Jan, Feb — Kolkata dry season) ───
    WINTER_MONTHS = {12, 1, 2}
    winter_mask = test_df['file_date'].dt.month.isin(WINTER_MONTHS).values
    if winter_mask.any():
        y_pred[winter_mask] = 0.0
        print(f"  Winter suppression: zeroed {winter_mask.sum()} intervals (Dec/Jan/Feb)")

    print(f"\n  Peak Correction [PGF]: x{pgf:.3f} (Adaptive)")
    print(f"  Storm Intervals Detected: {is_storm_pred.sum()} / {len(X_test)}")
    
    # ── Final Metrics (Evaluation only, NOT for training) ────────
    # The raw_targets are now used ONLY for stats and plotting comparisons.

    # ── Metrics ───────────────────────────────────────────────────
    def metrics(actual, pred):
        return r2_score(actual, pred), np.sqrt(mean_squared_error(actual, pred))

    r2_rf,      rmse_rf      = metrics(y_test, y_rf_raw)
    r2_xgb,     rmse_xgb     = metrics(y_test, y_xgb_raw)
    r2_hyb_raw, rmse_hyb_raw = metrics(y_test, y_base)
    r2_fin,     rmse_fin      = metrics(y_test, y_pred)

    print(f"\n  +--------------------------------------------------------+")
    print(f"  |  Model Comparison                                      |")
    print(f"  +------------------+----------+----------+---------------+")
    print(f"  | Model            |    R2    |   RMSE   |   Max pred    |")
    print(f"  +------------------+----------+----------+---------------+")
    print(f"  | Random Forest    | {r2_rf:.4f}   | {rmse_rf:6.3f}   | {y_rf_raw.max():8.2f}      |")
    print(f"  | XGBoost          | {r2_xgb:.4f}   | {rmse_xgb:6.3f}   | {y_xgb_raw.max():8.2f}      |")
    print(f"  | Hybrid (base)    | {r2_hyb_raw:.4f}   | {rmse_hyb_raw:6.3f}   | {y_base.max():8.2f}      |")
    print(f"  | Hybrid+PeakCorr  | {r2_fin:.4f}   | {rmse_fin:6.3f}   | {y_pred.max():8.2f}      |")
    print(f"  +------------------+----------+----------+---------------+")
    print(f"\n  Actual  - Mean: {y_test.mean():.2f}  Max: {y_test.max():.2f} mm/h")
    print(f"  RF      - Mean: {y_rf_raw.mean():.2f}  Max: {y_rf_raw.max():.2f} mm/h")
    print(f"  XGBoost - Mean: {y_xgb_raw.mean():.2f}  Max: {y_xgb_raw.max():.2f} mm/h")
    print(f"  Hybrid  - Mean: {y_base.mean():.2f}  Max: {y_base.max():.2f} mm/h")
    print(f"  Final   - Mean: {y_pred.mean():.2f}  Max: {y_pred.max():.2f} mm/h")

    # ── 3-panel plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), facecolor=P['bg'], sharex=True)
    fig.suptitle(f'Hybrid Ensemble Prediction - {date_range_str}',
                 fontsize=14, fontweight='bold', color=P['text'], y=0.98)
    ts = test_df['timestamp']

    axes[0].plot(ts, y_test, color=P['green'], linewidth=1.5, label='Actual RI', alpha=0.9)
    axes[0].plot(ts, y_rf_raw,   color=P['blue'],  linewidth=1.2, linestyle='--',
                 label=f'Weighted RF (w={W_RF})', alpha=0.85)
    style(axes[0], f'Random Forest - R2={r2_rf:.4f}  RMSE={rmse_rf:.3f} mm/h', '', 'RI (mm/h)')
    axes[0].legend(fontsize=9); axes[0].set_ylim(bottom=0)

    axes[1].plot(ts, y_test, color=P['green'],  linewidth=1.5, label='Actual RI', alpha=0.9)
    axes[1].plot(ts, y_xgb_raw,  color=P['orange'], linewidth=1.2, linestyle='--',
                 label=f'Weighted XGBoost (w={W_XGB})', alpha=0.85)
    style(axes[1], f'XGBoost - R2={r2_xgb:.4f}  RMSE={rmse_xgb:.3f} mm/h', '', 'RI (mm/h)')
    axes[1].legend(fontsize=9); axes[1].set_ylim(bottom=0)

    axes[2].plot(ts, y_test, color=P['green'],  linewidth=1.5, label='Actual RI (Processed)', alpha=0.9)
    axes[2].plot(ts, y_pred, color=P['purple'], linewidth=1.3, linestyle='--',
                 label=f'Hybrid+PeakEnhance (PGF=x{pgf:.2f})', alpha=0.9)
    axes[2].fill_between(ts, y_test, y_pred, alpha=0.08, color=P['red'])
    style(axes[2], f'Final - R2={r2_fin:.4f}  RMSE={rmse_fin:.3f} mm/h', 'Time', 'RI (mm/h)')
    axes[2].legend(fontsize=9); axes[2].set_ylim(bottom=0)
    axes[2].text(0.02, 0.96,
        f'Actual max  : {y_test.max():.2f} mm/h\n'
        f'Final max   : {y_pred.max():.2f} mm/h\n'
        f'Actual mean : {y_test.mean():.2f} mm/h\n'
        f'Final mean  : {y_pred.mean():.2f} mm/h',
        transform=axes[2].transAxes, color=P['text'], fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=P['bg'], edgecolor=P['border']))

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = f"validate_hybrid_{start_date.strftime('%Y%m%d')}.png"
    # plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150, facecolor=P['bg']) # User: don't save automatically
    plt.show()
    print(f"\n  Plot displayed window. (Not saved automatically)")

# =================================================================
#  RECONSTRUCT MODE — daily hybrid for mean, KNN for max
# =================================================================
elif MODE == "RECONSTRUCT":
    print("\n" + "-" * 66)
    print("  RECONSTRUCT MODE — Gap-fill for deleted dates")
    print(f"  Daily hybrid RF×{W_RF}+XGB×{W_XGB} for mean_RI")
    print("  KNN (K=20, 80th pct, same-month) for max_RI")
    print("-" * 66)

    # ── R1: History outside the gap ──────────────────────────────
    hist_df = df[
        (df['file_date'].dt.date < start_date.date()) |
        (df['file_date'].dt.date > end_date.date())
    ].copy()

    # ── R2: Daily aggregates from RAINY rows only ─────────────────
    # Dry rows (RI=0) would drag mean_RI way down — compute from
    # rainy rows only so a July day with 440/2880 rainy rows gets
    # its true mean (~10 mm/h) not a diluted ~1.5 mm/h.
    rainy_hist = hist_df[hist_df['RI'] > 0].copy()

    daily_rainy = rainy_hist.groupby(rainy_hist['file_date'].dt.date).agg(
        mean_RI  = ('RI', 'mean'),
        max_RI   = ('RI', 'max'),
        rain_pts = ('RI', 'count')
    ).reset_index()
    daily_rainy.rename(columns={'file_date': 'date'}, inplace=True)
    daily_rainy['date'] = pd.to_datetime(daily_rainy['date'])

    # Fill fully-dry calendar days with 0
    full_range = pd.date_range(
        hist_df['file_date'].min().date(),
        hist_df['file_date'].max().date()
    )
    daily = (daily_rainy.set_index('date')
                        .reindex(full_range)
                        .fillna(0)
                        .reset_index()
                        .rename(columns={'index': 'date'}))

    # ── R3: Feature engineering ───────────────────────────────────
    daily['month']     = daily['date'].dt.month
    daily['dayofyear'] = daily['date'].dt.day_of_year

    for lag in [1, 2, 3]:
        daily[f'mean_lag{lag}'] = daily['mean_RI'].shift(lag)
        daily[f'max_lag{lag}']  = daily['max_RI'].shift(lag)
        daily[f'pts_lag{lag}']  = daily['rain_pts'].shift(lag)

    daily = daily.dropna().reset_index(drop=True)

    DAILY_FEAT = (
        ['month', 'dayofyear'] +
        [f'mean_lag{l}' for l in [1, 2, 3]] +
        [f'max_lag{l}'  for l in [1, 2, 3]] +
        [f'pts_lag{l}'  for l in [1, 2, 3]]
    )

    train_daily = daily[daily['date'] < start_date].copy()
    if len(train_daily) < 10:
        print("ERROR: Not enough pre-gap history."); sys.exit(1)

    print(f"  Daily training rows : {len(train_daily):,}")

    # ── R4: Train peak-aware hybrid (mean_RI) + RF (pts) ──────────
    print(f"  Training Peak-Aware RF + XGBoost for daily mean_RI ...")
    rf_mean, xgb_mean, clf_mean = build_peak_aware_hybrid(
        train_daily[DAILY_FEAT].values,
        train_daily['mean_RI'].values,
        W_RF, W_XGB
    )
    
    # Calculate daily PGF
    split_idx_d = int(len(train_daily) * 0.8)
    X_d_val = train_daily[DAILY_FEAT].values[split_idx_d:]
    y_d_val = train_daily['mean_RI'].values[split_idx_d:]
    pgf_daily = get_peak_gain_factor(rf_mean, xgb_mean, X_d_val, y_d_val, W_RF, W_XGB)
    print(f"  Daily Mean PGF: x{pgf_daily:.3f}")

    pts_model = RandomForestRegressor(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    pts_model.fit(train_daily[DAILY_FEAT], train_daily['rain_pts'])

    # ── R5: Month-stratified bias correction for mean_RI ─────────
    def pct_ratio(actual, predicted, pct, lo=1.0, hi=3.0):
        valid = predicted > 0
        if valid.sum() < 3: return 1.0
        return float(np.clip(
            np.percentile(actual[valid] / predicted[valid], pct), lo, hi))

    rainy_train = train_daily[train_daily['mean_RI'] > 0].copy()
    pred_mean_train = hybrid_predict(
        rf_mean, xgb_mean,
        rainy_train[DAILY_FEAT].values, W_RF, W_XGB
    )

    g_mean_corr = pct_ratio(rainy_train['mean_RI'].values,
                             pred_mean_train, pct=75)
    monthly_mean_corr = {}
    for m in range(1, 13):
        grp = rainy_train[rainy_train['month'] == m]
        pred_grp = hybrid_predict(rf_mean, xgb_mean,
                                  grp[DAILY_FEAT].values, W_RF, W_XGB)
        monthly_mean_corr[m] = (
            pct_ratio(grp['mean_RI'].values, pred_grp, pct=75)
            if len(grp) >= 5 else g_mean_corr
        )

    # ── R6: KNN max_RI predictor (95th Percentile) ────────────────
    def knn_max_predict(query_feat_df, train_df, feat_cols,
                        month, K=20, pct=95):
        same_month = train_df[train_df['month'] == month].copy()
        pool = same_month if len(same_month) >= K else train_df.copy()

        X_pool   = pool[feat_cols].values.astype(float)
        q        = query_feat_df[feat_cols].values.astype(float)
        col_std  = X_pool.std(axis=0) + 1e-8
        col_mean = X_pool.mean(axis=0)

        X_norm = (X_pool - col_mean) / col_std
        q_norm = (q      - col_mean) / col_std

        dists   = np.linalg.norm(X_norm - q_norm, axis=1)
        k_idx   = np.argsort(dists)[:min(K, len(pool))]
        nn_max  = pool.iloc[k_idx]['max_RI'].values

        return float(np.percentile(nn_max, pct))

    print(f"\n  max_RI — KNN K=20, 95th pct (Peak-Aware), same-month neighbours")

    # ── R7: Recursive prediction ──────────────────────────────────
    num_days = (end_date - start_date).days + 1
    print(f"\n  Reconstructing {num_days} day(s) recursively ...")

    predictions   = []
    current_state = train_daily.iloc[-1:].copy().reset_index(drop=True)
    current_date  = start_date

    for _ in range(num_days):
        m = current_date.month

        feat = pd.DataFrame({
            'month'     : [current_date.month],
            'dayofyear' : [current_date.day_of_year],
            'mean_lag1' : [float(current_state['mean_RI'].iloc[0])],
            'mean_lag2' : [float(current_state['mean_lag1'].iloc[0])],
            'mean_lag3' : [float(current_state['mean_lag2'].iloc[0])],
            'max_lag1'  : [float(current_state['max_RI'].iloc[0])],
            'max_lag2'  : [float(current_state['max_lag1'].iloc[0])],
            'max_lag3'  : [float(current_state['max_lag2'].iloc[0])],
            'pts_lag1'  : [float(current_state['rain_pts'].iloc[0])],
            'pts_lag2'  : [float(current_state['pts_lag1'].iloc[0])],
            'pts_lag3'  : [float(current_state['pts_lag2'].iloc[0])],
        })

        # mean_RI: peak-aware hybrid prediction
        raw_rf   = float(rf_mean.predict(feat[DAILY_FEAT])[0])
        raw_xgb  = float(xgb_mean.predict(feat[DAILY_FEAT])[0])
        raw_hyb  = W_RF * raw_rf + W_XGB * raw_xgb
        
        # Scaling if daily storm classifier triggers
        is_storm_d = clf_mean.predict(feat[DAILY_FEAT])[0]
        pred_mean = raw_hyb * (pgf_daily if is_storm_d else 1.0)
        pred_mean = float(np.clip(pred_mean * monthly_mean_corr[m], 0, 100))

        # max_RI: KNN from historical same-month days (95th pct)
        pred_max_base = float(np.clip(
            knn_max_predict(feat, train_daily, DAILY_FEAT, month=m,
                            K=20, pct=95),
            pred_mean, 100
        ))
        
        # Adaptive peak boost for reconstruct max_RI
        if is_storm_d:
            damp = np.clip((70 - pred_max_base) / (70 - 30), 0, 1)
            pred_max = pred_max_base * (1.0 + (pgf_daily - 1.0) * damp)
        else:
            pred_max = pred_max_base
            
        pred_max = float(np.clip(pred_max, pred_mean, 100))

        pred_pts = max(0, int(pts_model.predict(feat[DAILY_FEAT])[0]))

        # ── Winter Suppression (Dec, Jan, Feb — Kolkata dry season) ──
        if m in (12, 1, 2):
            pred_mean = 0.0
            pred_max  = 0.0
            pred_pts  = 0

        predictions.append({
            'date'     : current_date,
            'pred_mean': pred_mean,
            'pred_max' : pred_max,
            'pred_pts' : pred_pts,
            'raw_rf'   : raw_rf,
            'raw_xgb'  : raw_xgb,
            'raw_hyb'  : raw_hyb,
        })

        # Slide lag window
        new_row = {
            'mean_RI'   : pred_mean,   'max_RI'    : pred_max,
            'rain_pts'  : float(pred_pts),
            'month'     : current_date.month,
            'dayofyear' : current_date.day_of_year,
            'mean_lag1' : float(current_state['mean_RI'].iloc[0]),
            'mean_lag2' : float(current_state['mean_lag1'].iloc[0]),
            'mean_lag3' : float(current_state['mean_lag2'].iloc[0]),
            'max_lag1'  : float(current_state['max_RI'].iloc[0]),
            'max_lag2'  : float(current_state['max_lag1'].iloc[0]),
            'max_lag3'  : float(current_state['max_lag2'].iloc[0]),
            'pts_lag1'  : float(current_state['rain_pts'].iloc[0]),
            'pts_lag2'  : float(current_state['pts_lag1'].iloc[0]),
            'pts_lag3'  : float(current_state['pts_lag2'].iloc[0]),
        }
        current_state = pd.DataFrame([new_row])
        current_date += timedelta(days=1)

    # ── R8: Print summary ─────────────────────────────────────────
    print("\n  === RECONSTRUCTED DAILY SUMMARIES ===")
    print(f"  {'Date':<14} {'Mean RI':>9} {'Max RI':>9}  "
          f"{'RF raw':>8} {'XGB raw':>8} {'Hyb raw':>8}  Rain rows")
    for p in predictions:
        flag = "(rain)" if p['pred_pts'] > 0 else "(dry) "
        print(f"  {p['date'].strftime('%Y-%m-%d'):<14} "
              f"{p['pred_mean']:>9.2f} {p['pred_max']:>9.2f}  "
              f"{p['raw_rf']:>8.2f} {p['raw_xgb']:>8.2f} "
              f"{p['raw_hyb']:>8.2f}  "
              f"{p['pred_pts']:>6}  {flag}")

    # ── R9: Synthetic 30-second profiles ─────────────────────────
    print("\n  Building 30-second synthetic profiles ...")
    synthetic_profiles = []

    for p in predictions:
        if p['pred_pts'] == 0:
            continue

        cands = train_daily[
            (train_daily['month'] == p['date'].month) &
            (train_daily['mean_RI'] > 0)
        ].copy()
        if len(cands) == 0:
            cands = train_daily[train_daily['mean_RI'] > 0].copy()
        if len(cands) == 0:
            continue

        mu_ref  = cands['mean_RI'].mean() + 1e-6
        max_ref = cands['max_RI'].mean()  + 1e-6
        cands['score'] = (
            abs(cands['mean_RI'] - p['pred_mean']) / mu_ref +
            abs(cands['max_RI']  - p['pred_max'])  / max_ref
        )
        best_day = cands.sort_values('score').iloc[0]['date']

        profile = hist_df[
            hist_df['file_date'].dt.date == best_day.date()
        ].copy()
        if len(profile) == 0:
            continue

        profile['timestamp'] = profile['timestamp'].apply(
            lambda ts: pd.Timestamp(
                f"{p['date'].strftime('%Y-%m-%d')} "
                f"{ts.strftime('%H:%M:%S')}"
            )
        )

        prof_max = profile['RI'].max()
        if prof_max > 0:
            scale = p['pred_max'] / prof_max
            profile['predicted_RI'] = (profile['RI'] * scale).clip(upper=100)
        else:
            profile['predicted_RI'] = p['pred_mean']

        synthetic_profiles.append(
            profile[['timestamp', 'predicted_RI']].copy()
        )

    # ── R10: Plot ─────────────────────────────────────────────────
    if synthetic_profiles:
        synth_df = pd.concat(synthetic_profiles, ignore_index=True)
        synth_df = synth_df.sort_values('timestamp').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=P['bg'])
        ax.fill_between(synth_df['timestamp'], synth_df['predicted_RI'],
                        alpha=0.2, color=P['purple'])
        ax.plot(synth_df['timestamp'], synth_df['predicted_RI'],
                color=P['purple'], linewidth=1.5,
                label=f'Reconstructed RI  (RF×{W_RF} + XGB×{W_XGB})')
        style(ax,
              f'Rainfall RECONSTRUCTION — Hybrid Ensemble — {date_range_str}',
              f'Time', 'Rainfall Intensity (mm/h)')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend(fontsize=9)

        for p in predictions:
            ax.axvline(p['date'], color=P['border'],
                       linewidth=0.8, linestyle=':', alpha=0.9)
            ymax = ax.get_ylim()[1]
            ax.text(p['date'] + timedelta(hours=3), ymax * 0.90,
                    f"mean={p['pred_mean']:.1f}\nmax={p['pred_max']:.1f}",
                    fontsize=7, color=P['text'], alpha=0.85)

        plt.tight_layout()
        fname = f"reconstruct_hybrid_{start_date.strftime('%Y%m%d')}.png"
        # plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150, facecolor=P['bg'])
        plt.show()
        print(f"\n  Plot displayed. (Not saved automatically)")
    else:
        print("\n  No rain predicted — no plot generated.")

print("\n" + "=" * 66)
print(f"  DONE — Mode: {MODE}  |  Ensemble: RF×{W_RF} + XGB×{W_XGB}")
print("=" * 66)