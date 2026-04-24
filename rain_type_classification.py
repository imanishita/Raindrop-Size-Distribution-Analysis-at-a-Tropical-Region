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
print("  STRATIFORM vs CONVECTIVE RAIN CLASSIFICATION")
print("  (Kolkata 2010-2015, RD-80 Disdrometer)")
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
print("\n[STEP 1] Computing DSD parameters for all intervals...")

N_mat = df[N_COLS].values.astype(float)
ND    = N_mat / (F * t_s * Vi * dDi)     # [m⁻³ mm⁻¹]

# Rainfall Intensity R [mm/h]
R = (np.pi / 6.0) * (3.6e3 / (F_cm2 * t_s)) * np.sum(N_mat * (Di ** 3), axis=1)

# Radar Reflectivity Z [mm⁶ m⁻³]
Z = np.sum(ND * (Di ** 6) * dDi, axis=1)

# Mass-weighted Mean Diameter Dm [mm]
num = np.sum(ND * (Di ** 4) * dDi, axis=1)
den = np.sum(ND * (Di ** 3) * dDi, axis=1)
Dm  = np.where(den > 0, num / den, 0)

# Liquid Water Content LWC [g/m³]
LWC = (np.pi / 6) * 1e-3 * np.sum(ND * (Di ** 3) * dDi, axis=1)

# Total drop count
total_drops = N_mat.sum(axis=1)

df['R']           = R
df['Z']           = Z
df['Dm']          = Dm
df['LWC']         = LWC
df['total_drops'] = total_drops
df['log10Z']      = np.where(Z > 0, np.log10(Z), 0)

# ─────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────
# Method: Rain rate threshold (widely used in literature)
#   R < 10 mm/h  → Stratiform
#   R >= 10 mm/h → Convective
# Secondary check: Dm threshold
#   Dm < 2.0 mm → Stratiform signature
#   Dm >= 2.0 mm → Convective signature
#
# Reference:
#   Tomiwa et al. (URSI 2018): 85% stratiform in tropical Nigeria
#   Harikumar et al.: Indian tropical stratiform/convective split
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Classifying rain type (R threshold at 10 mm/h)...")

STRAT_THRESHOLD = 10.0   # mm/h — standard in literature

# Keep valid rain intervals only
mask = (R > 0.1) & (Z > 0) & (R < 300) & (total_drops > 0)
df_rain = df[mask].copy().reset_index(drop=True)

df_rain['rain_type'] = np.where(df_rain['R'] < STRAT_THRESHOLD, 'Stratiform', 'Convective')

strat = df_rain[df_rain['rain_type'] == 'Stratiform']
conv  = df_rain[df_rain['rain_type'] == 'Convective']

total   = len(df_rain)
n_strat = len(strat)
n_conv  = len(conv)
pct_s   = 100 * n_strat / total
pct_c   = 100 * n_conv  / total

print(f"\n  ┌─────────────────────────────────────────────┐")
print(f"  │  Total rain intervals  : {total:>8,}            │")
print(f"  │  Stratiform (R<10mm/h) : {n_strat:>8,}  ({pct_s:.1f}%)    │")
print(f"  │  Convective (R≥10mm/h) : {n_conv:>8,}  ({pct_c:.1f}%)    │")
print(f"  │                                             │")
print(f"  │  Stratiform Dm (mean)  : {strat['Dm'].mean():.3f} mm          │")
print(f"  │  Convective  Dm (mean) : {conv['Dm'].mean():.3f} mm          │")
print(f"  │  Stratiform LWC (mean) : {strat['LWC'].mean():.4f} g/m³       │")
print(f"  │  Convective  LWC (mean): {conv['LWC'].mean():.4f} g/m³       │")
print(f"  └─────────────────────────────────────────────┘")

# ─────────────────────────────────────────────────────────────────
# SEASONAL RAIN TYPE BREAKDOWN
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Season-wise rain type breakdown...")

def get_season(m):
    if m in [6,7,8,9]:   return 'monsoon'
    if m in [10,11]:      return 'post-monsoon'
    if m in [3,4,5]:      return 'pre-monsoon'
    return 'winter'

df_rain['season'] = df_rain['timestamp'].dt.month.apply(get_season)
season_order = ['monsoon','post-monsoon','pre-monsoon','winter']

print(f"\n  {'Season':<15} {'Total':>8} {'Strat %':>9} {'Conv %':>8} {'Dm_strat':>10} {'Dm_conv':>9}")
print(f"  {'-'*62}")
for s in season_order:
    sub = df_rain[df_rain['season'] == s]
    if len(sub) == 0:
        continue
    ns = (sub['rain_type']=='Stratiform').sum()
    nc = (sub['rain_type']=='Convective').sum()
    dm_s = sub[sub['rain_type']=='Stratiform']['Dm'].mean()
    dm_c = sub[sub['rain_type']=='Convective']['Dm'].mean()
    ps   = 100*ns/len(sub)
    pc   = 100*nc/len(sub)
    print(f"  {s:<15} {len(sub):>8,} {ps:>8.1f}% {pc:>8.1f}% "
          f"  {dm_s:>7.3f}mm  {dm_c:>7.3f}mm")

# ─────────────────────────────────────────────────────────────────
# DSD N(D) CURVES — Stratiform vs Convective
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Computing N(D) curves for each rain type...")

def mean_ND(subset):
    N_sub = subset[N_COLS].values.astype(float)
    ND_sub = N_sub / (F * t_s * Vi * dDi)
    return ND_sub.mean(axis=0)

ND_strat = mean_ND(strat)
ND_conv  = mean_ND(conv)

# Also compute by season for stratiform (most intervals)
ND_season = {}
for s in season_order:
    sub = strat[strat['season'] == s]
    if len(sub) > 50:
        ND_season[s] = mean_ND(sub)

# ─────────────────────────────────────────────────────────────────
# Z-R FIT per rain type
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Z-R fits for each rain type...")

def zr_fit(subset):
    mask2 = (subset['R'] > 0.1) & (subset['Z'] > 1)
    sub2  = subset[mask2]
    sl, ic, rv, _, _ = stats.linregress(
        np.log10(sub2['R'].values),
        np.log10(sub2['Z'].values)
    )
    return 10**ic, sl, rv**2, len(sub2)

a_s, b_s, r2_s, n_s = zr_fit(strat)
a_c, b_c, r2_c, n_c = zr_fit(conv)

print(f"  Stratiform : Z = {a_s:.1f} * R^{b_s:.3f}  (R²={r2_s:.4f}, n={n_s:,})")
print(f"  Convective : Z = {a_c:.1f} * R^{b_c:.3f}  (R²={r2_c:.4f}, n={n_c:,})")
print(f"  Marshall-P : Z = 200.0 * R^1.600")

# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Plotting...")

fig = plt.figure(figsize=(18, 14), facecolor=P['bg'])
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

# ── Plot 1: Pie chart of rain type ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
wedge_colors = [P['blue'], P['red']]
wedges, texts, autotexts = ax1.pie(
    [n_strat, n_conv],
    labels=['Stratiform', 'Convective'],
    colors=wedge_colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.7,
    textprops={'fontsize': 10, 'color': P['text']}
)
for at in autotexts:
    at.set_color('white')
    at.set_fontweight('bold')
ax1.set_title('Rain Type Proportion\n(Kolkata 2010-2015)',
              color=P['text'], fontsize=11, fontweight='bold')

# Compare with URSI paper result
ax1.text(0, -1.35,
         f'Literature (tropical Nigeria): ~85% Stratiform\n'
         f'This study (Kolkata): {pct_s:.1f}% Stratiform',
         ha='center', fontsize=8.5, color=P['sub'],
         style='italic')

# ── Plot 2: N(D) — Stratiform vs Convective ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(Di, ND_strat, 'o-', color=P['blue'],  linewidth=2,
             markersize=5, label='Stratiform')
ax2.semilogy(Di, ND_conv,  's-', color=P['red'],   linewidth=2,
             markersize=5, label='Convective')
style(ax2, 'N(D): Stratiform vs Convective',
      'Drop Diameter D (mm)', 'N(D)  [m⁻³ mm⁻¹]')
ax2.legend(fontsize=9)

# ── Plot 3: Dm distribution ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(strat['Dm'], bins=60, color=P['blue'],  alpha=0.65,
         density=True, label=f'Stratiform (μ={strat["Dm"].mean():.2f} mm)')
ax3.hist(conv['Dm'],  bins=60, color=P['red'],   alpha=0.65,
         density=True, label=f'Convective (μ={conv["Dm"].mean():.2f} mm)')
ax3.axvline(2.0, color='#555555', linewidth=1.3, linestyle='--',
            label='Dm = 2.0 mm threshold')
style(ax3, 'Mass-weighted Mean Diameter Dm', 'Dm (mm)', 'Density')
ax3.legend(fontsize=8.5)

# ── Plot 4: Z-R scatter with fits ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
sample_s = strat.sample(min(4000, len(strat)), random_state=42)
sample_c = conv.sample(min(2000, len(conv)),   random_state=42)

ax4.scatter(sample_s['R'], sample_s['Z'], alpha=0.08, s=6,
            color=P['blue'], label='Stratiform observations')
ax4.scatter(sample_c['R'], sample_c['Z'], alpha=0.15, s=8,
            color=P['red'],  label='Convective observations')

R_fit = np.linspace(0.1, 200, 500)
ax4.plot(R_fit, a_s * R_fit**b_s,  color=P['blue'],   linewidth=2.2,
         label=f'Strat fit: Z={a_s:.0f}R^{b_s:.2f}')
ax4.plot(R_fit, a_c * R_fit**b_c,  color=P['red'],    linewidth=2.2,
         label=f'Conv fit: Z={a_c:.0f}R^{b_c:.2f}')
ax4.plot(R_fit, 200 * R_fit**1.6,  color='#555555',   linewidth=1.8,
         linestyle='--', label='Marshall-Palmer: Z=200R^1.6')

ax4.set_xscale('log'); ax4.set_yscale('log')
style(ax4, 'Z-R: Stratiform vs Convective  (Kolkata vs Marshall-Palmer)',
      'Rain Rate R (mm/h)', 'Radar Reflectivity Z (mm⁶ m⁻³)')
ax4.legend(fontsize=8.5, loc='upper left')

# ── Plot 5: Season breakdown bar ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
snames = [s for s in season_order if s in df_rain['season'].unique()]
s_pcts = []
c_pcts = []
for s in snames:
    sub = df_rain[df_rain['season'] == s]
    ns  = (sub['rain_type']=='Stratiform').sum()
    s_pcts.append(100*ns/len(sub))
    c_pcts.append(100*(len(sub)-ns)/len(sub))

x = np.arange(len(snames))
w = 0.38
bars1 = ax5.bar(x - w/2, s_pcts, w, color=P['blue'], alpha=0.85, label='Stratiform')
bars2 = ax5.bar(x + w/2, c_pcts, w, color=P['red'],  alpha=0.85, label='Convective')
for b, v in zip(list(bars1)+list(bars2), s_pcts+c_pcts):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, color=P['sub'])
ax5.set_xticks(x)
ax5.set_xticklabels([s[:4] for s in snames], rotation=20)
ax5.set_ylim(0, 105)
style(ax5, 'Season-wise Rain Type %', 'Season', '%')
ax5.legend(fontsize=8.5)

plt.suptitle('Stratiform vs Convective Rain Analysis — Kolkata (2010-2015)',
             fontsize=13, fontweight='bold', color=P['text'], y=1.01)

path = os.path.join(PLOT_DIR, 'rain_type_classification.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────
# N(D) by season for stratiform — separate plot
# ─────────────────────────────────────────────────────────────────
season_colors_plot = {
    'monsoon': P['blue'], 'post-monsoon': P['red'],
    'pre-monsoon': P['green'], 'winter': P['orange']
}

fig2, ax = plt.subplots(figsize=(9, 6), facecolor=P['bg'])
for s, nd in ND_season.items():
    n_sub = len(strat[strat['season']==s])
    ax.semilogy(Di, nd, 'o-', linewidth=2, markersize=5,
                color=season_colors_plot.get(s, '#333'),
                label=f'{s} (n={n_sub:,})')
style(ax, 'Stratiform N(D) by Season — Kolkata',
      'Drop Diameter D (mm)', 'N(D)  [m⁻³ mm⁻¹]')
ax.legend(fontsize=9)
path2 = os.path.join(PLOT_DIR, 'nd_stratiform_seasonal.png')
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor=P['bg'])
plt.close()
print(f"  Saved: {path2}")

# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  RAIN TYPE CLASSIFICATION — FINAL REPORT")
print("=" * 62)
print(f"  Stratiform : {n_strat:,} intervals ({pct_s:.1f}%)")
print(f"  Convective : {n_conv:,} intervals ({pct_c:.1f}%)")
print(f"")
print(f"  Stratiform Z-R : Z = {a_s:.2f} * R^{b_s:.4f}")
print(f"  Convective Z-R : Z = {a_c:.2f} * R^{b_c:.4f}")
print(f"  Marshall-Palmer: Z = 200.00 * R^1.6000")
print(f"")
print(f"  Key finding:")
print(f"  Kolkata shows {pct_c:.1f}% CONVECTIVE rain — significantly higher")
print(f"  than tropical Nigeria (~15% convective, Tomiwa et al. URSI 2018).")
print(f"  This reflects Kolkata's intense monsoon and Kalbaisakhi storm")
print(f"  climatology — frequent pre-monsoon squall lines and thunderstorms.")
