"""
predict.py — Unified Rainfall Prediction & Gap-Fill Script
===========================================================
Replaces forecast.py entirely.

Modes (auto-detected, or forced with --force-reconstruct):

  VALIDATE    — target dates exist AND the context window immediately before
                them is also intact. Trains on 2010–2014, validates on target.

  RECONSTRUCT — triggered automatically when EITHER:
                  (a) target dates are missing/deleted from processed_data.csv
                  (b) the N days before the target window are missing
                      (lag features would be poisoned by the gap)
                Uses daily pattern-matching + recursive prediction to recover
                the deleted window.

Usage examples:
  python predict.py --start 2015-07-22
  python predict.py --start 2015-07-22 --end 2015-07-25
  python predict.py --start 2015-07-22 --force-reconstruct
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, sys, argparse
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Predict or reconstruct RI for any date range.")
parser.add_argument("--start",             type=str, default="2015-07-22",
                    help="Start date YYYY-MM-DD")
parser.add_argument("--end",               type=str, default=None,
                    help="End date YYYY-MM-DD (defaults to start)")
parser.add_argument("--force-reconstruct", action="store_true",
                    help="Force RECONSTRUCT mode even if data exists")
parser.add_argument("--lag-context-days",  type=int, default=3,
                    help="How many days before the window must be intact (default 3)")
args = parser.parse_args()

if args.end is None:
    args.end = args.start

start_date = pd.to_datetime(args.start)
end_date   = pd.to_datetime(args.end)

if end_date < start_date:
    print("ERROR: --end must be >= --start")
    sys.exit(1)

date_range_str = (
    f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
    if start_date != end_date else
    start_date.strftime('%d %b %Y')
)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_FILE = "processed_data.csv"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
N_COLS    = [f'n{i}' for i in range(1, 21)]

# Palette (light theme)
P = {'bg':'#f8f9fa','panel':'#ffffff','border':'#dee2e6','text':'#1a1a2e',
     'sub':'#6c757d','blue':'#4361ee','red':'#f72585','green':'#2dc653',
     'orange':'#ff6b35'}

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(P['panel'])
    ax.tick_params(colors=P['sub'], labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor(P['border'])
    if title:  ax.set_title(title,   color=P['text'], fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=P['sub'],  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=P['sub'],  fontsize=9)
    ax.grid(True, color=P['border'], linewidth=0.5, alpha=0.8)

print("=" * 62)
print(f"  TARGET: {date_range_str}")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("\n[LOAD] Reading processed_data.csv ...")
df = pd.read_csv(DATA_FILE, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['RI'] = pd.to_numeric(df['RI'], errors='coerce')
df = df.dropna(subset=['RI'])
df = df[df['RI'] <= 100].reset_index(drop=True)

print(f"  Loaded {len(df):,} rows  |  "
      f"{df['timestamp'].dt.year.min()}-{df['timestamp'].dt.year.max()}")

# ─────────────────────────────────────────────────────────────────
# AUTO MODE DETECTION
# ─────────────────────────────────────────────────────────────────
def detect_mode(df, start_date, end_date, context_days, force_reconstruct):
    """
    Returns 'RECONSTRUCT' if:
      (a) force flag set, OR
      (b) target window has < 50 rows (data deleted/missing), OR
      (c) the context_days immediately before start_date are missing
          (lag features would be poisoned by the gap)
    Returns 'VALIDATE' otherwise.
    """
    if force_reconstruct:
        return "RECONSTRUCT", "forced by --force-reconstruct flag"

    # (b) Check target window row count
    target_mask = (
        (df['timestamp'].dt.date >= start_date.date()) &
        (df['timestamp'].dt.date <= end_date.date())
    )
    n_target = target_mask.sum()
    if n_target < 50:
        return "RECONSTRUCT", f"target window has only {n_target} rows (data deleted)"

    # (c) Check that context window before the target is intact
    #     A "day" is considered present if >= 10 rows exist for it
    context_start = start_date - timedelta(days=context_days)
    context_end   = start_date - timedelta(days=1)

    context_dates = pd.date_range(context_start, context_end, freq='D')
    daily_counts  = df.groupby(df['timestamp'].dt.date)['RI'].count()

    missing_context = []
    for d in context_dates:
        count = daily_counts.get(d.date(), 0)
        if count < 10:
            missing_context.append(str(d.date()))

    if missing_context:
        return ("RECONSTRUCT",
                f"context window is incomplete — missing or sparse days: "
                f"[{', '.join(missing_context)}]. Lag features would be "
                f"poisoned by the gap.")

    return "VALIDATE", "target data present and context window is intact"


MODE, reason = detect_mode(
    df, start_date, end_date,
    context_days=args.lag_context_days,
    force_reconstruct=args.force_reconstruct
)

print(f"\n[MODE] >>> {MODE} <<<")
print(f"  Reason : {reason}")

# =================================================================
#  VALIDATE MODE
# =================================================================
if MODE == "VALIDATE":
    print("\n" + "-" * 62)
    print("  VALIDATE MODE - Training on 2010-2014, testing on target")
    print("-" * 62)

    df['total_drops'] = df[N_COLS].sum(axis=1)
    df['hour']        = df['timestamp'].dt.hour
    df['month']       = df['timestamp'].dt.month

    # Lag features are safe: context window verified intact above
    for lag in [1, 2, 3]:
        df[f'RI_lag{lag}'] = df['RI'].shift(lag)

    df = df.dropna(subset=['RI_lag1', 'RI_lag2', 'RI_lag3']).reset_index(drop=True)

    train_df = df[
        (df['timestamp'].dt.year <= 2014) &
        (df['RI'] > 0)
    ].copy()

    # Re-apply target mask after index reset
    target_mask = (
        (df['timestamp'].dt.date >= start_date.date()) &
        (df['timestamp'].dt.date <= end_date.date())
    )
    test_df = df[target_mask].copy()

    if len(test_df) == 0:
        print("ERROR: No valid rows for target after lag construction.")
        sys.exit(1)

    FEATURES = N_COLS + ['total_drops', 'hour', 'month',
                         'RI_lag1', 'RI_lag2', 'RI_lag3']
    TARGET   = 'RI'

    print(f"  Train samples : {len(train_df):,}")
    print(f"  Test  samples : {len(test_df):,}")

    model = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', n_jobs=-1, random_state=42
    )
    model.fit(train_df[FEATURES], train_df[TARGET])
    y_pred = model.predict(test_df[FEATURES])
    y_test = test_df[TARGET].values

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  R2   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f} mm/h")
    print(f"\n  Actual Mean RI    : {y_test.mean():.2f} mm/h")
    print(f"  Predicted Mean RI : {y_pred.mean():.2f} mm/h")
    print(f"  Actual Max RI     : {y_test.max():.2f} mm/h")
    print(f"  Predicted Max RI  : {y_pred.max():.2f} mm/h")

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=P['bg'])
    ax.plot(test_df['timestamp'], y_test,
            color=P['green'], linewidth=1.8, label='Actual RI')
    ax.plot(test_df['timestamp'], y_pred,
            color=P['red'], linewidth=1.4, linestyle='--', label='Predicted RI')
    style(ax, f'Rainfall Prediction - {date_range_str}',
          f'Time ({date_range_str})', 'Rainfall Intensity (mm/h)')
    ax.text(0.02, 0.96,
            f'R2 = {r2:.4f}   RMSE = {rmse:.3f} mm/h',
            transform=ax.transAxes, color=P['text'], fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=P['bg'],
                      edgecolor=P['border']))
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = f"validate_{start_date.strftime('%Y%m%d')}.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150, facecolor=P['bg'])
    plt.show()
    print(f"\n  Plot saved: plots/{fname}")

# =================================================================
#  RECONSTRUCT MODE
# =================================================================
elif MODE == "RECONSTRUCT":
    print("\n" + "-" * 62)
    print("  RECONSTRUCT MODE - Gap-fill / imputation for deleted dates")
    print("-" * 62)

    # Step R1: Build daily aggregates from history OUTSIDE the gap
    hist_df = df[
        (df['timestamp'].dt.date < start_date.date()) |
        (df['timestamp'].dt.date > end_date.date())
    ].copy()

    daily = hist_df.groupby(hist_df['timestamp'].dt.date).agg(
        mean_RI  = ('RI', 'mean'),
        max_RI   = ('RI', 'max'),
        rain_pts = ('RI', 'count')
    ).reset_index()
    daily.rename(columns={'timestamp': 'date'}, inplace=True)
    daily['date'] = pd.to_datetime(daily['date'])

    # Fill missing calendar days in history with zeros (dry days)
    full_range = pd.date_range(
        hist_df['timestamp'].min().date(),
        hist_df['timestamp'].max().date()
    )
    daily = (daily.set_index('date')
                  .reindex(full_range)
                  .fillna(0)
                  .reset_index()
                  .rename(columns={'index': 'date'}))

    # Step R2: Feature engineering
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

    # Train only on days strictly before the gap
    train_daily = daily[daily['date'] < start_date].copy()
    if len(train_daily) < 10:
        print("ERROR: Not enough pre-gap history to train the reconstruction model.")
        sys.exit(1)

    print(f"  Daily training rows : {len(train_daily):,}")

    rf_kw = dict(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    mean_model = RandomForestRegressor(**rf_kw)
    max_model  = RandomForestRegressor(**rf_kw)
    pts_model  = RandomForestRegressor(**rf_kw)

    mean_model.fit(train_daily[DAILY_FEAT], train_daily['mean_RI'])
    max_model.fit(train_daily[DAILY_FEAT],  train_daily['max_RI'])
    pts_model.fit(train_daily[DAILY_FEAT],  train_daily['rain_pts'])

    # ── BIAS CORRECTION ──────────────────────────────────────────
    # Random Forest systematically underestimates extremes because
    # it averages across trees. On rainy training days, measure how
    # much the model underpredicts the actual max, then apply that
    # correction ratio to every future prediction.
    #
    # We use the 85th percentile of (actual/predicted) ratios
    # across all rainy training days — this targets peak events
    # without overcorrecting dry days.
    # ─────────────────────────────────────────────────────────────
    rainy_train = train_daily[train_daily['max_RI'] > 0].copy()
    if len(rainy_train) > 10:
        max_pred_on_train = max_model.predict(rainy_train[DAILY_FEAT])
        valid_mask = max_pred_on_train > 0
        if valid_mask.sum() > 5:
            ratios = rainy_train['max_RI'].values[valid_mask] / max_pred_on_train[valid_mask]
            max_bias_correction = float(np.percentile(ratios, 85))
            # Sanity-bound: don't overcorrect more than 3x
            max_bias_correction = np.clip(max_bias_correction, 1.0, 3.0)
        else:
            max_bias_correction = 1.0
    else:
        max_bias_correction = 1.0

    # Same correction for mean RI (less critical, use 70th percentile)
    if len(rainy_train) > 10:
        mean_pred_on_train = mean_model.predict(rainy_train[DAILY_FEAT])
        valid_mask = mean_pred_on_train > 0
        if valid_mask.sum() > 5:
            ratios_mean = rainy_train['mean_RI'].values[valid_mask] / mean_pred_on_train[valid_mask]
            mean_bias_correction = float(np.clip(np.percentile(ratios_mean, 70), 1.0, 2.5))
        else:
            mean_bias_correction = 1.0
    else:
        mean_bias_correction = 1.0

    print(f"\n  Bias correction factors (RF underestimation fix):")
    print(f"    max_RI  correction : x{max_bias_correction:.3f}  (85th pct of actual/predicted on training)")
    print(f"    mean_RI correction : x{mean_bias_correction:.3f}  (70th pct of actual/predicted on training)")

    # Step R3: Recursive daily prediction
    num_days = (end_date - start_date).days + 1
    print(f"  Reconstructing {num_days} day(s) recursively ...")

    predictions   = []
    current_state = train_daily.iloc[-1:].copy().reset_index(drop=True)
    current_date  = start_date

    for _ in range(num_days):
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

        pred_mean = float(np.clip(
            mean_model.predict(feat)[0] * mean_bias_correction, 0, 100))
        pred_max  = float(np.clip(
            max_model.predict(feat)[0]  * max_bias_correction,  0, 100))
        pred_pts  = max(0, int(pts_model.predict(feat)[0]))

        # Physical constraint: max must be >= mean
        if pred_max < pred_mean:
            pred_max = pred_mean * 1.5

        predictions.append({
            'date'      : current_date,
            'pred_mean' : pred_mean,
            'pred_max'  : pred_max,
            'pred_pts'  : pred_pts
        })

        # Slide the lag window
        cs = current_state
        new_row = {
            'mean_RI'   : pred_mean,
            'max_RI'    : pred_max,
            'rain_pts'  : float(pred_pts),
            'month'     : current_date.month,
            'dayofyear' : current_date.day_of_year,
            'mean_lag1' : float(cs['mean_RI'].iloc[0]),
            'mean_lag2' : float(cs['mean_lag1'].iloc[0]),
            'mean_lag3' : float(cs['mean_lag2'].iloc[0]),
            'max_lag1'  : float(cs['max_RI'].iloc[0]),
            'max_lag2'  : float(cs['max_lag1'].iloc[0]),
            'max_lag3'  : float(cs['max_lag2'].iloc[0]),
            'pts_lag1'  : float(cs['rain_pts'].iloc[0]),
            'pts_lag2'  : float(cs['pts_lag1'].iloc[0]),
            'pts_lag3'  : float(cs['pts_lag2'].iloc[0]),
        }
        current_state = pd.DataFrame([new_row])
        current_date += timedelta(days=1)

    # Step R4: Print daily summary
    print("\n  === RECONSTRUCTED DAILY SUMMARIES ===")
    for p in predictions:
        rain_flag = "(rain)" if p['pred_pts'] > 0 else "(dry)"
        print(f"  {p['date'].strftime('%Y-%m-%d')}  |  "
              f"Mean RI: {p['pred_mean']:5.2f} mm/h  |  "
              f"Max RI: {p['pred_max']:5.2f} mm/h  |  "
              f"Rain rows: {p['pred_pts']:4d}  {rain_flag}")

    # Step R5: Synthetic 30-second profiles via pattern matching
    print("\n  Building 30-second synthetic profiles via pattern matching ...")
    synthetic_profiles = []

    for p in predictions:
        if p['pred_pts'] == 0:
            continue

        # Find best matching historical day: same month, closest mean+max RI
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
            hist_df['timestamp'].dt.date == best_day.date()
        ].copy()

        if len(profile) == 0:
            continue

        # Re-stamp onto target date
        profile['timestamp'] = profile['timestamp'].apply(
            lambda ts: pd.Timestamp(
                f"{p['date'].strftime('%Y-%m-%d')} {ts.strftime('%H:%M:%S')}"
            )
        )

        # Scale proportionally to predicted max
        if profile['RI'].max() > 0:
            scale = p['pred_max'] / profile['RI'].max()
            profile['predicted_RI'] = (profile['RI'] * scale).clip(upper=100)
        else:
            profile['predicted_RI'] = p['pred_mean']

        synthetic_profiles.append(
            profile[['timestamp', 'predicted_RI']].copy()
        )

    # Step R6: Plot
    if synthetic_profiles:
        synth_df = pd.concat(synthetic_profiles, ignore_index=True)
        synth_df = synth_df.sort_values('timestamp').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=P['bg'])
        ax.fill_between(synth_df['timestamp'], synth_df['predicted_RI'],
                        alpha=0.25, color=P['orange'])
        ax.plot(synth_df['timestamp'], synth_df['predicted_RI'],
                color=P['orange'], linewidth=1.5, label='Reconstructed RI')
        style(ax,
              f'Rainfall RECONSTRUCTION (Gap-Fill) - {date_range_str}',
              f'Time ({date_range_str})',
              'Rainfall Intensity (mm/h)')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)

        # Mark day boundaries and annotate predicted means
        for p in predictions:
            ax.axvline(p['date'], color=P['border'], linewidth=0.8,
                       linestyle=':', alpha=0.9)
            ymax = ax.get_ylim()[1]
            ax.text(p['date'] + timedelta(hours=3), ymax * 0.90,
                    f"u={p['pred_mean']:.1f}", fontsize=7,
                    color=P['text'], alpha=0.85)

        plt.tight_layout()
        fname = f"reconstruct_{start_date.strftime('%Y%m%d')}.png"
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150, facecolor=P['bg'])
        plt.show()
        print(f"\n  Plot saved: plots/{fname}")
    else:
        print("\n  No rain predicted for the gap period - no profile plot generated.")

print("\n" + "=" * 62)
print(f"  DONE - Mode: {MODE}")
print("=" * 62)