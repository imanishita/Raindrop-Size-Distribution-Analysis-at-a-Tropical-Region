import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
warnings_imported = False
try:
    import warnings
    warnings.filterwarnings('ignore')
    warnings_imported = True
except:
    pass

# ─────────────────────────────────────────────────────────────────
# CONSTANTS (same as rest of pipeline)
# ─────────────────────────────────────────────────────────────────
F      = 0.005       # sensor area [m²]
F_cm2  = F * 1e4     # → 50 cm²
t_s    = 30          # sampling interval [s]

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
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

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
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  Z-R RELATIONSHIP — KOLKATA (2010-2015)")
print("=" * 62)

df = pd.read_csv("processed_data.csv", low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# ─────────────────────────────────────────────────────────────────
# COMPUTE R and Z from DSD
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 1] Computing R and Z from DSD for every interval...")

N_mat = df[N_COLS].values.astype(float)

# RI (mm/h) using F in cm² — same formula as the rest of the pipeline
factor = (np.pi / 6.0) * (3.6e3 / (F_cm2 * t_s))
R = factor * np.sum(N_mat * (Di ** 3), axis=1)

# N(D) — number concentration [m⁻³ mm⁻¹]
ND = N_mat / (F * t_s * Vi * dDi)

# Radar Reflectivity Z [mm⁶ m⁻³]
Z = np.sum(ND * (Di ** 6) * dDi, axis=1)

df['R'] = R
df['Z'] = Z

# ─────────────────────────────────────────────────────────────────
# CLEAN — keep rain intervals with valid R and Z
# ─────────────────────────────────────────────────────────────────
mask = (R > 0.1) & (Z > 1) & (R < 300) & (Z < 1e8)
df_rain = df[mask].copy()
print(f"  Rain intervals used : {len(df_rain):,}")

# ─────────────────────────────────────────────────────────────────
# FIT Z = a * R^b  (log-linear regression)
# log(Z) = log(a) + b * log(R)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Fitting Z = a * R^b ...")

logR = np.log10(df_rain['R'].values)
logZ = np.log10(df_rain['Z'].values)

slope, intercept, r_value, p_value, std_err = stats.linregress(logR, logZ)
b_kolkata = slope
a_kolkata  = 10 ** intercept

print(f"\n  ┌─────────────────────────────────────────────┐")
print(f"  │  KOLKATA Z-R:  Z = {a_kolkata:.1f} * R^{b_kolkata:.3f}          │")
print(f"  │  Marshall-Palmer: Z = 200 * R^1.600         │")
print(f"  │  R² of fit       : {r_value**2:.4f}                    │")
print(f"  └─────────────────────────────────────────────┘")

# ─────────────────────────────────────────────────────────────────
# COMPARE WITH MARSHALL-PALMER
# ─────────────────────────────────────────────────────────────────
# Marshall-Palmer: Z = 200 * R^1.6  (universal standard)
# Stratiform:      Z = 200 * R^1.6
# Convective:      Z = 300 * R^1.4  (Battan 1973 range)

R_fit = np.linspace(0.1, 150, 500)
Z_kolkata = a_kolkata * R_fit ** b_kolkata
Z_mp      = 200    * R_fit ** 1.6     # Marshall-Palmer
Z_conv    = 300    * R_fit ** 1.4     # Convective reference

# ─────────────────────────────────────────────────────────────────
# SEASONAL Z-R FITS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Season-wise Z-R fits...")

seasons = {
    'monsoon':      [6, 7, 8, 9],
    'post-monsoon': [10, 11],
    'pre-monsoon':  [3, 4, 5],
    'winter':       [12, 1, 2]
}

season_colors = {
    'monsoon':      '#4361ee',
    'post-monsoon': '#f72585',
    'pre-monsoon':  '#2dc653',
    'winter':       '#ff6b35'
}

season_fits = {}
for sname, months in seasons.items():
    sub = df_rain[df_rain['timestamp'].dt.month.isin(months)]
    if len(sub) < 100:
        continue
    sl, ic, rv, _, _ = stats.linregress(
        np.log10(sub['R'].values),
        np.log10(sub['Z'].values)
    )
    a_s = 10 ** ic
    b_s = sl
    season_fits[sname] = (a_s, b_s, rv**2, len(sub))
    print(f"  {sname:<15} → Z = {a_s:.1f} * R^{b_s:.3f}  (R²={rv**2:.3f}, n={len(sub):,})")

# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Plotting...")

# Sample for scatter (max 8000 points)
sample = df_rain.sample(min(8000, len(df_rain)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=P['bg'])

# ── Plot 1: Z-R scatter + fits ──────────────────────────────────
ax = axes[0]
ax.scatter(sample['R'], sample['Z'], alpha=0.12, s=8,
           color=P['blue'], label='Observations (sampled)')
ax.plot(R_fit, Z_kolkata, color=P['red'],    linewidth=2.2,
        label=f'Kolkata fit: Z = {a_kolkata:.0f}R^{b_kolkata:.2f}')
ax.plot(R_fit, Z_mp,      color='#555555',   linewidth=1.8, linestyle='--',
        label='Marshall-Palmer: Z = 200R^1.6')
ax.plot(R_fit, Z_conv,    color=P['orange'], linewidth=1.5, linestyle=':',
        label='Convective ref: Z = 300R^1.4')

ax.set_xscale('log'); ax.set_yscale('log')
style(ax, 'Z-R Relationship — Kolkata vs Marshall-Palmer',
      'Rain Rate R (mm/h)', 'Radar Reflectivity Z (mm⁶ m⁻³)')
ax.legend(fontsize=8.5, loc='upper left')
ax.text(0.97, 0.08,
        f'n = {len(df_rain):,}\nR² = {r_value**2:.4f}',
        transform=ax.transAxes, ha='right', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=P['bg'], edgecolor=P['border']))

# ── Plot 2: Seasonal Z-R fits ───────────────────────────────────
ax2 = axes[1]
ax2.plot(R_fit, Z_mp, color='#aaaaaa', linewidth=1.5, linestyle='--',
         label='Marshall-Palmer (ref)', zorder=5)
for sname, (a_s, b_s, r2_s, n_s) in season_fits.items():
    Z_s = a_s * R_fit ** b_s
    ax2.plot(R_fit, Z_s, linewidth=2,
             color=season_colors.get(sname, '#333333'),
             label=f'{sname}: Z={a_s:.0f}R^{b_s:.2f}')

ax2.set_xscale('log'); ax2.set_yscale('log')
style(ax2, 'Seasonal Z-R Relationships — Kolkata',
      'Rain Rate R (mm/h)', 'Radar Reflectivity Z (mm⁶ m⁻³)')
ax2.legend(fontsize=8.5)

plt.tight_layout()
path = os.path.join(PLOT_DIR, 'zr_relationship.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Z-R RELATIONSHIP — FINAL REPORT")
print("=" * 62)
print(f"  Kolkata Z-R :  Z = {a_kolkata:.2f} * R^{b_kolkata:.4f}")
print(f"  Marshall-Palmer: Z = 200 * R^1.6000")
print(f"  Intercept diff : {a_kolkata - 200:+.2f}")
print(f"  Exponent  diff : {b_kolkata - 1.6:+.4f}")
if a_kolkata > 200:
    print(f"  → Kolkata intercept HIGHER than MP → larger drops per rain rate")
else:
    print(f"  → Kolkata intercept LOWER than MP → smaller drops per rain rate")
if b_kolkata > 1.6:
    print(f"  → Kolkata exponent HIGHER → reflectivity grows faster with rain rate")
else:
    print(f"  → Kolkata exponent LOWER → typical tropical monsoon DSD profile")
print("=" * 62)
