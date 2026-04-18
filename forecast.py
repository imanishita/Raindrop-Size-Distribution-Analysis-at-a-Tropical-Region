import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, sys
import argparse
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# -----------------------------
# ARG PARSING
# -----------------------------
parser = argparse.ArgumentParser(description="Predict or Reconstruct RI for any date range.")
parser.add_argument("--start", type=str, default="2015-07-23", help="Start date in YYYY-MM-DD")
parser.add_argument("--end", type=str, default=None, help="End date in YYYY-MM-DD (defaults to start date)")
parser.add_argument("--force-reconstruct", action="store_true", help="Force reconstruct mode even if partial data exists")
args = parser.parse_args()

if args.end is None:
    args.end = args.start

start_date = pd.to_datetime(args.start)
end_date = pd.to_datetime(args.end)

date_range_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}" if start_date != end_date else start_date.strftime('%d %b %Y')

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "processed_data.csv"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

N_COLS = [f'n{i}' for i in range(1, 21)]

print("=" * 60)
print(f"FORECAST FOR: {date_range_str}")
print("=" * 60)

# -----------------------------
# LOAD DATA
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

# Keep rain only for validation, cap at 100 max
df = df[df['RI'] <= 100]
df = df[(df[N_COLS].sum(axis=1) > 0) & (df['RI'] > 0)].reset_index(drop=True)

# -----------------------------
# DETERMINE OPERATING MODE
# -----------------------------
test_mask = (df['timestamp'].dt.date >= start_date.date()) & (df['timestamp'].dt.date <= end_date.date())
test_df = df[test_mask].copy()

# If we have less than 50 rows for the requested period, we assume data was deleted and switch to Reconstruct mode
if args.force_reconstruct:
    MODE = "RECONSTRUCT"
else:
    MODE = "VALIDATE" if len(test_df) >= 50 else "RECONSTRUCT"

if MODE == "VALIDATE":
    print(">>> MODE: VALIDATE (Data found for target dates)")
    # -----------------------------
    # VALIDATE MODE (Standard DSD RF)
    # -----------------------------
    df['total_drops'] = df[N_COLS].sum(axis=1)
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    for lag in [1, 2, 3]:
        df[f'RI_lag{lag}'] = df['RI'].shift(lag)

    df = df.dropna(subset=['RI_lag1','RI_lag2','RI_lag3']).reset_index(drop=True)

    FEATURES = N_COLS + ['total_drops', 'hour', 'month', 'RI_lag1', 'RI_lag2', 'RI_lag3']
    TARGET = 'RI'

    train_df = df[df['timestamp'].dt.year <= 2014]
    
    # Re-apply mask because indices changed after dropna
    test_mask = (df['timestamp'].dt.date >= start_date.date()) & (df['timestamp'].dt.date <= end_date.date())
    test_df = df[test_mask]

    if len(test_df) == 0:
        raise ValueError(f"No valid lag data for validation mode on {date_range_str}!")

    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples : {len(test_df):,} ({date_range_str})")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

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
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n=== RESULTS FOR {date_range_str} ===")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f} mm/h")

    print(f"\n=== SUMMARY ({date_range_str}) ===")
    print(f"Actual Mean RI    : {y_test.mean():.2f} mm/h")
    print(f"Predicted Mean RI : {y_pred.mean():.2f} mm/h")
    print(f"Actual Max RI     : {y_test.max():.2f} mm/h")
    print(f"Predicted Max RI  : {y_pred.max():.2f} mm/h")

    # Full Day Plot
    plt.figure(figsize=(14,6))
    plt.plot(test_df['timestamp'], y_test, label="Actual RI", color='green', linewidth=2)
    plt.plot(test_df['timestamp'], y_pred, label="Predicted RI", color='red', linestyle='--', linewidth=2)
    plt.xlabel(f"Time ({date_range_str})")
    plt.ylabel("Rainfall Intensity (mm/h)")
    plt.title(f"Rainfall Prediction — {date_range_str}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename_safe = date_range_str.replace(' ', '_').lower()
    plot_path = os.path.join(PLOT_DIR, f"forecast_{filename_safe}.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\nPlot saved at: {plot_path}")

elif MODE == "RECONSTRUCT":
    print(">>> MODE: RECONSTRUCT (Missing or deleted data detected)")
    print("Building Daily Time-Series Model to recover missing days...")

    # -----------------------------
    # RECONSTRUCT MODE (Gap filling)
    # -----------------------------
    # 1. Build a daily aggregate of the entire historical dataset
    daily = df.groupby(df['timestamp'].dt.date).agg(
        mean_RI=('RI', 'mean'),
        max_RI=('RI', 'max'),
        rain_pts=('RI', 'count')
    ).reset_index()
    daily.rename(columns={'timestamp': 'date'}, inplace=True)
    daily['date'] = pd.to_datetime(daily['date'])
    
    # 2. Fill in completely missing historical days with 0 (no rain recorded)
    full_range = pd.date_range(df['timestamp'].min().date(), df['timestamp'].max().date())
    daily = daily.set_index('date').reindex(full_range).fillna(0).reset_index()
    daily.rename(columns={'index': 'date'}, inplace=True)
    
    # 3. Feature engineering for time-series model
    daily['month'] = daily['date'].dt.month
    daily['dayofyear'] = daily['date'].dt.dayofyear
    
    for lag in [1, 2, 3]:
        daily[f'mean_lag{lag}'] = daily['mean_RI'].shift(lag)
        daily[f'max_lag{lag}'] = daily['max_RI'].shift(lag)
        daily[f'pts_lag{lag}'] = daily['rain_pts'].shift(lag)
        
    daily = daily.dropna().reset_index(drop=True)
    
    FEATURES = ['month', 'dayofyear'] + \
               [f'mean_lag{l}' for l in [1,2,3]] + \
               [f'max_lag{l}' for l in [1,2,3]] + \
               [f'pts_lag{l}' for l in [1,2,3]]
               
    # Train data: all historical days before the gap
    train_daily = daily[daily['date'] < start_date]
    if len(train_daily) == 0:
        print("Not enough historical data before start date to train the gap model.")
        sys.exit(1)
        
    # Model configuration
    mean_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    mean_model.fit(train_daily[FEATURES], train_daily['mean_RI'])
    
    max_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    max_model.fit(train_daily[FEATURES], train_daily['max_RI'])

    pts_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    pts_model.fit(train_daily[FEATURES], train_daily['rain_pts'])
    
    # RECURSIVE PREDICTION
    print(f"\nPredicting missing values recursively from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    predictions = []
    
    # Base state before the gap (last known row)
    current_state = train_daily.iloc[-1:].copy()
    num_days = (end_date - start_date).days + 1
    
    current_date = start_date
    for i in range(num_days):
        feat = pd.DataFrame({
            'month': [current_date.month],
            'dayofyear': [current_date.dayofyear],
            'mean_lag1': [current_state['mean_RI'].values[0]],
            'mean_lag2': [current_state['mean_lag1'].values[0]],
            'mean_lag3': [current_state['mean_lag2'].values[0]],
            'max_lag1': [current_state['max_RI'].values[0]],
            'max_lag2': [current_state['max_lag1'].values[0]],
            'max_lag3': [current_state['max_lag2'].values[0]],
            'pts_lag1': [current_state['rain_pts'].values[0]],
            'pts_lag2': [current_state['pts_lag1'].values[0]],
            'pts_lag3': [current_state['pts_lag2'].values[0]]
        })
        
        pred_mean = mean_model.predict(feat)[0]
        pred_max = max_model.predict(feat)[0]
        pred_pts = max(0, int(pts_model.predict(feat)[0]))
        
        if pred_max < pred_mean:
            pred_max = pred_mean * 1.5
            
        predictions.append({
            'date': current_date,
            'pred_mean': pred_mean,
            'pred_max': pred_max,
            'pred_pts': pred_pts
        })
        
        # Update current state for the next recursive step
        current_state['mean_lag3'] = current_state['mean_lag2']
        current_state['mean_lag2'] = current_state['mean_lag1']
        current_state['mean_lag1'] = current_state['mean_RI']
        current_state['mean_RI'] = pred_mean
        
        current_state['max_lag3'] = current_state['max_lag2']
        current_state['max_lag2'] = current_state['max_lag1']
        current_state['max_lag1'] = current_state['max_RI']
        current_state['max_RI'] = pred_max
        
        current_state['pts_lag3'] = current_state['pts_lag2']
        current_state['pts_lag2'] = current_state['pts_lag1']
        current_state['pts_lag1'] = current_state['rain_pts']
        current_state['rain_pts'] = pred_pts
        
        current_date += timedelta(days=1)
        
    print("\n=== RECONSTRUCTED DAILY SUMMARIES ===")
    for p in predictions:
        print(f"Date: {p['date'].strftime('%Y-%m-%d')} | Mean RI: {p['pred_mean']:.2f} mm/h | Max RI: {p['pred_max']:.2f} mm/h | Rain Rows: {p['pred_pts']}")
        
    print("\nGenerating synthetic 30-second profiles via historical Pattern Matching...")
    synthetic_profiles = []
    
    for p in predictions:
        if p['pred_pts'] == 0:
            continue
            
        # Find closest historical day in the same month
        hist_candidates = train_daily[(train_daily['month'] == p['date'].month) & (train_daily['mean_RI'] > 0)]
        if len(hist_candidates) == 0:
            hist_candidates = train_daily[train_daily['mean_RI'] > 0]
            
        hist_candidates['score'] = abs(hist_candidates['mean_RI'] - p['pred_mean'])/(hist_candidates['mean_RI'].mean() + 1e-6) + \
                                   abs(hist_candidates['max_RI'] - p['pred_max'])/(hist_candidates['max_RI'].mean() + 1e-6)
                                   
        best_day = hist_candidates.sort_values('score').iloc[0]['date']
        
        # Extract the profile
        profile = df[df['timestamp'].dt.date == best_day.date()].copy()
        
        if len(profile) == 0:
            continue
            
        # Adjust timestamp
        time_offsets = profile['timestamp'].dt.time
        new_timestamps = [pd.Timestamp(f"{p['date'].strftime('%Y-%m-%d')} {t}") for t in time_offsets]
        profile['timestamp'] = new_timestamps
        
        # Scale values
        if profile['RI'].max() > 0:
            scale_factor_max = p['pred_max'] / profile['RI'].max()
            profile['predicted_RI'] = profile['RI'] * scale_factor_max
        else:
            profile['predicted_RI'] = p['pred_mean']
            
        synthetic_profiles.append(profile[['timestamp', 'predicted_RI']])
        
    if len(synthetic_profiles) > 0:
        synth_df = pd.concat(synthetic_profiles, ignore_index=True)
        
        plt.figure(figsize=(14,6))
        plt.plot(synth_df['timestamp'], synth_df['predicted_RI'], label="Reconstructed RI", color='orange', linewidth=2)
        plt.xlabel(f"Time ({date_range_str})")
        plt.ylabel("Rainfall Intensity (mm/h)")
        plt.title(f"Rainfall RECONSTRUCTION (Missing Data) — {date_range_str}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename_safe = date_range_str.replace(' ', '_').lower()
        plot_path = os.path.join(PLOT_DIR, f"reconstruct_{filename_safe}.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()

        print(f"\nPlot saved at: {plot_path}")
    else:
        print("\nNo rain predicted for the gap period. Plot skipped.")