import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
F      = 0.005
F_cm2  = F * 1e4
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

N_COLS   = [f'n{i}' for i in range(1, 21)]
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

P = {'bg':'#f8f9fa','panel':'#ffffff','border':'#dee2e6','text':'#1a1a2e',
     'sub':'#6c757d','blue':'#4361ee','red':'#f72585','green':'#2dc653',
     'orange':'#ff6b35','purple':'#7209b7','teal':'#0d9488'}

SEASONS = {
    'monsoon':      [6, 7, 8, 9],
    'post-monsoon': [10, 11],
    'pre-monsoon':  [3, 4, 5],
    'winter':       [12, 1, 2]
}
SEASON_COLORS = {
    'monsoon': P['blue'], 'post-monsoon': P['red'],
    'pre-monsoon': P['green'], 'winter': P['orange']
}
SEASON_MARKERS = {
    'monsoon': 'o', 'post-monsoon': 's',
    'pre-monsoon': '^', 'winter': 'D'
}

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(P['panel'])
    ax.tick_params(colors=P['sub'], labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor(P['border'])
    if title:  ax.set_title(title,   color=P['text'], fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=P['sub'],  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=P['sub'],  fontsize=9)
    ax.grid(True, color=P['border'], linewidth=0.5, alpha=0.8)

# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────
print("=" * 62)
print("  SEASONAL DSD ANALYSIS — KOLKATA (2010-2015)")
print("=" * 62)

df = pd.read_csv("processed_data.csv", low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df = df[(df['timestamp'].dt.year >= 2010) & (df['timestamp'].dt.year <= 2015)]
df = df.sort_values('timestamp').reset_index(drop=True)

for c in N_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

print(f"  Loaded rows: {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# COMPUTE DSD PARAMETERS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 1] Computing DSD parameters...")

N_mat = df[N_COLS].values.astype(float)
ND    = N_mat / (F * t_s * Vi * dDi)

R   = (np.pi/6.0) * (3.6e3/(F_cm2*t_s)) * np.sum(N_mat*(Di**3), axis=1)
Z   = np.sum(ND*(Di**6)*dDi, axis=1)
LWC = (np.pi/6)*1e-3 * np.sum(ND*(Di**3)*dDi, axis=1)

num = np.sum(ND*(Di**4)*dDi, axis=1)
den = np.sum(ND*(Di**3)*dDi, axis=1)
Dm  = np.where(den > 0, num/den, 0)

# Slope parameter Λ (from exponential DSD fit  N(D)=N0*exp(-Λ*D))
# Estimated from Dm: Λ ≈ 4/Dm  (Marshall-Palmer relationship)
Lambda = np.where(Dm > 0, 4.0 / Dm, 0)

# Intercept N0 from N(D) at first bin
N0 = ND[:, 0] * np.exp(Lambda * Di[0])

df['R']      = R
df['Z']      = Z
df['LWC']    = LWC
df['Dm']     = Dm
df['Lambda'] = Lambda
df['N0']     = N0
df['month']  = df['timestamp'].dt.month

# Season label
def get_season(m):
    if m in [6,7,8,9]:   return 'monsoon'
    if m in [10,11]:      return 'post-monsoon'
    if m in [3,4,5]:      return 'pre-monsoon'
    return 'winter'

df['season'] = df['month'].apply(get_season)

# Keep rain-only intervals
mask = (R > 0.1) & (Z > 0) & (R < 300)
df   = df[mask].reset_index(drop=True)
ND   = N_mat[mask] / (F * t_s * Vi * dDi)
print(f"  Rain intervals: {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# SEASONAL STATISTICS TABLE
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Seasonal statistics...")

season_order = ['monsoon', 'post-monsoon', 'pre-monsoon', 'winter']

print(f"\n  {'Season':<15} {'N':>7} {'R_mean':>8} {'Z_mean':>10} "
      f"{'LWC_mean':>10} {'Dm_mean':>9} {'Λ_mean':>9}")
print(f"  {'-'*72}")

season_stats = {}
for s in season_order:
    sub = df[df['season'] == s]
    if len(sub) < 50:
        continue
    stats_row = {
        'n':       len(sub),
        'R_mean':  sub['R'].mean(),
        'R_std':   sub['R'].std(),
        'Z_mean':  sub['Z'].mean(),
        'LWC_mean':sub['LWC'].mean(),
        'Dm_mean': sub['Dm'].mean(),
        'Dm_std':  sub['Dm'].std(),
        'Lam_mean':sub['Lambda'].mean(),
    }
    season_stats[s] = stats_row
    print(f"  {s:<15} {len(sub):>7,} {sub['R'].mean():>8.2f} "
          f"{sub['Z'].mean():>10.1f} {sub['LWC'].mean():>10.4f} "
          f"{sub['Dm'].mean():>9.3f} {sub['Lambda'].mean():>9.3f}")

# ─────────────────────────────────────────────────────────────────
# MEAN N(D) PER SEASON
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Computing seasonal N(D) curves...")

ND_season = {}
for s in season_order:
    idx = df['season'] == s
    if idx.sum() < 50:
        continue
    N_sub  = df.loc[idx, N_COLS].values.astype(float)
    ND_sub = N_sub / (F * t_s * Vi * dDi)
    ND_season[s] = ND_sub.mean(axis=0)

# Marshall-Palmer fit for reference
def mp_nd(R_ref, Di):
    lam = 4.1 * R_ref**(-0.21)
    N0  = 8000
    return N0 * np.exp(-lam * Di)

# ─────────────────────────────────────────────────────────────────
# MONTHLY MEAN DSD PARAMETERS (for seasonal trend plot)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Monthly aggregates...")

monthly = df.groupby('month').agg(
    R_mean  =('R',   'mean'),
    Z_mean  =('Z',   'mean'),
    LWC_mean=('LWC', 'mean'),
    Dm_mean =('Dm',  'mean'),
    Lam_mean=('Lambda', 'mean'),
    count   =('R',   'count')
).reset_index()

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Plotting...")

# ── FIGURE 1: N(D) curves + MP comparison ───────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=P['bg'])

ax = axes[0]
for s in season_order:
    if s not in ND_season:
        continue
    n_sub = (df['season']==s).sum()
    ax.semilogy(Di, ND_season[s],
                marker=SEASON_MARKERS[s], linewidth=2, markersize=5,
                color=SEASON_COLORS[s],
                label=f'{s.capitalize()} (n={n_sub:,})')

# Marshall-Palmer for mean monsoon R
R_mon = season_stats.get('monsoon', {}).get('R_mean', 10)
ax.semilogy(Di, mp_nd(R_mon, Di), 'k--', linewidth=1.5,
            label=f'Marshall-Palmer (R={R_mon:.1f} mm/h)')

style(ax, 'Mean N(D) by Season — Kolkata',
      'Drop Diameter D (mm)', 'N(D)  [m⁻³ mm⁻¹]')
ax.legend(fontsize=9)

# Ratio monsoon/post-monsoon if both exist
ax2 = axes[1]
if 'monsoon' in ND_season and 'post-monsoon' in ND_season:
    nd_m = ND_season['monsoon']
    nd_p = ND_season['post-monsoon']

    # Only plot ratio where BOTH seasons have reliable data
    # Threshold: N(D) must be > 1% of that season's peak value
    min_threshold = max(nd_m.max(), nd_p.max()) * 0.01
    valid = (nd_m > min_threshold) & (nd_p > min_threshold)

    ratio = np.where(valid, nd_m / (nd_p + 1e-6), np.nan)

    ax2.bar(Di[valid], ratio[valid], width=dDi[valid] * 0.6,
            color=P['blue'], alpha=0.75,
            label='Monsoon / Post-monsoon N(D) ratio')
    ax2.axhline(1.0, color='#555555', linestyle='--', linewidth=1.2)
    ax2.set_xlim(0, Di[valid].max() + 0.3)
    style(ax2, 'N(D) Ratio: Monsoon vs Post-monsoon',
          'Drop Diameter D (mm)', 'N(D) ratio')
    ax2.legend(fontsize=9)
    ax2.text(0.97, 0.95,
             'Ratio > 1: More drops in monsoon\nRatio < 1: More in post-monsoon',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8.5,
             color=P['sub'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor=P['bg'],
                       edgecolor=P['border']))
else:
    ax2.text(0.5, 0.5, 'Insufficient data for ratio',
             transform=ax2.transAxes, ha='center')

fig1.suptitle('Seasonal Drop Size Distribution — Kolkata (2010-2015)',
              fontsize=13, fontweight='bold', color=P['text'])
plt.tight_layout()
path1 = os.path.join(PLOT_DIR, 'seasonal_nd_curves.png')
plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor=P['bg'])
plt.close()
print(f"  Saved: {path1}")

# ── FIGURE 2: Monthly parameter trends ──────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), facecolor=P['bg'])

# Colour each month by season
def month_color(m):
    if m in [6,7,8,9]:   return P['blue']
    if m in [10,11]:      return P['red']
    if m in [3,4,5]:      return P['green']
    return P['orange']

m_colors = [month_color(m) for m in monthly['month']]
x_ticks  = monthly['month'].values
x_labels = [month_names[m-1] for m in x_ticks]

for ax, col, ytitle in zip(
    axes2.flatten(),
    ['R_mean', 'Dm_mean', 'LWC_mean', 'Lam_mean'],
    ['Mean Rain Rate R (mm/h)',
     'Mean Mass-weighted Dm (mm)',
     'Mean Liquid Water Content (g/m³)',
     'Mean Slope Λ (mm⁻¹)']
):
    bars = ax.bar(monthly['month'], monthly[col],
                  color=m_colors, edgecolor=P['border'],
                  linewidth=0.5, alpha=0.85)
    ax.plot(monthly['month'], monthly[col], 'o-',
            color=P['text'], linewidth=1.2, markersize=4, alpha=0.6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)

    # Season shade bands
    season_spans = [(6,9,P['blue']), (10,11,P['red']),
                    (3,5,P['green']), (12,12,P['orange']),
                    (1,2,P['orange'])]
    for s_start, s_end, sc in season_spans:
        ax.axvspan(s_start-0.5, s_end+0.5, alpha=0.06, color=sc, zorder=0)

    style(ax, f'Monthly {ytitle}', 'Month', ytitle)

# Legend for seasons
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=P['blue'],   label='Monsoon (Jun-Sep)'),
    Patch(facecolor=P['red'],    label='Post-monsoon (Oct-Nov)'),
    Patch(facecolor=P['green'],  label='Pre-monsoon (Mar-May)'),
    Patch(facecolor=P['orange'], label='Winter (Dec-Feb)'),
]
fig2.legend(handles=legend_handles, loc='lower center',
            ncol=4, fontsize=9, framealpha=0.8,
            bbox_to_anchor=(0.5, -0.02))

fig2.suptitle('Monthly DSD Parameter Trends — Kolkata (2010-2015)',
              fontsize=13, fontweight='bold', color=P['text'])
plt.tight_layout(rect=[0, 0.05, 1, 1])
path2 = os.path.join(PLOT_DIR, 'seasonal_dsd_parameters.png')
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor=P['bg'])
plt.close()
print(f"  Saved: {path2}")

# ── FIGURE 3: Dm vs log10(Z) scatter by season ──────────────────
fig3, ax3 = plt.subplots(figsize=(10, 7), facecolor=P['bg'])

for s in season_order:
    sub = df[df['season'] == s]
    if len(sub) < 50:
        continue
    samp = sub.sample(min(2000, len(sub)), random_state=42)
    ax3.scatter(samp['Dm'], np.log10(samp['Z']+1),
                alpha=0.18, s=8, color=SEASON_COLORS[s], label=s.capitalize())

    # Season centroid
    ax3.scatter(sub['Dm'].mean(), np.log10(sub['Z']+1).mean(),
                marker='*', s=260, color=SEASON_COLORS[s],
                edgecolors='white', linewidth=0.8, zorder=5)

ax3.axvline(2.0, color='#888888', linestyle='--', linewidth=1.2,
            label='Dm=2mm (strat/conv threshold)')
style(ax3, 'Dm vs log₁₀(Z) Scatter by Season — Kolkata',
      'Mass-weighted Dm (mm)', 'log₁₀(Z)  [mm⁶ m⁻³]')
ax3.text(0.97, 0.05,
         'Stars = seasonal centroids',
         transform=ax3.transAxes, ha='right', fontsize=9, color=P['sub'])
ax3.legend(fontsize=9, markerscale=1.5)
path3 = os.path.join(PLOT_DIR, 'dm_vs_logZ_seasonal.png')
plt.savefig(path3, dpi=150, bbox_inches='tight', facecolor=P['bg'])
plt.close()
print(f"  Saved: {path3}")

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SEASONAL DSD ANALYSIS — FINAL REPORT")
print("=" * 62)
for s in season_order:
    if s not in season_stats:
        continue
    st = season_stats[s]
    print(f"  {s.upper()}")
    print(f"    Intervals : {st['n']:,}")
    print(f"    R mean    : {st['R_mean']:.2f} mm/h")
    print(f"    Z mean    : {st['Z_mean']:.1f} mm⁶/m³")
    print(f"    LWC mean  : {st['LWC_mean']:.4f} g/m³")
    print(f"    Dm mean   : {st['Dm_mean']:.3f} mm  (σ={st['Dm_std']:.3f})")
    print(f"    Λ mean    : {st['Lam_mean']:.3f} mm⁻¹")
    print()
print("  Key finding:")
if 'monsoon' in season_stats and 'post-monsoon' in season_stats:
    dm_m = season_stats['monsoon']['Dm_mean']
    dm_p = season_stats['post-monsoon']['Dm_mean']
    r_m  = season_stats['monsoon']['R_mean']
    r_p  = season_stats['post-monsoon']['R_mean']
    print(f"  Monsoon drops are {'larger' if dm_m > dm_p else 'smaller'} on average")
    print(f"  (Dm monsoon={dm_m:.2f}mm vs post-monsoon={dm_p:.2f}mm)")
    print(f"  Rain intensity: monsoon={r_m:.2f} vs post-monsoon={r_p:.2f} mm/h")
print("=" * 62)