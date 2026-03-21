# 🌧️ Rain Drop Size Distribution Analysis & ML-Based Rainfall Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-Data-green?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-purple?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Final Year Project**
**B.Tech Electronics & Communication Engineering**
**University of Calcutta**

*Manishita Biswas · Anik Khajanchi*

</div>

---

## 📌 Overview

This project presents a complete pipeline for **Rain Drop Size Distribution (DSD) analysis** and **Machine Learning-based Rainfall Intensity prediction** using six years (2010–2015) of real-world data collected from an **RD-80 Piezoelectric Disdrometer** in a tropical region (Kolkata, India).

The project bridges **atmospheric physics** and **data science** — computing rainfall parameters from first principles and then training a Random Forest model to learn the DSD → Rainfall Intensity mapping directly from data.

---

## 🎯 Objectives

- ✅ Parse and merge raw RD-80 `.txt` files across multiple years (2010–2015)
- ✅ Preprocess 313,856 intervals with timestamp alignment and noise removal
- ✅ Compute physics-based Rainfall Intensity (RI) and Drop Size Distribution N(D)
- ✅ Compare Instrument RI vs Physics-computed RI
- ✅ Engineer DSD-derived features: LWC, Dm, Z, log_Z
- ✅ Train a Random Forest model achieving **R² = 0.9995**
- ✅ Analyse feature importance — confirming LWC and radar reflectivity Z as dominant predictors
- ⏳ Short-term rainfall forecasting *(see Future Work)*

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| Dataset size | 313,856 intervals (2010–2015) |
| Rain intervals | 268,528 (85.5%) |
| RI range | 0.12 – 300 mm/h |
| RI mean (computed) | 41.89 mm/h |
| **ML R² Score** | **0.9995** |
| **RMSE** | **1.44 mm/h** |
| **MAE** | **0.46 mm/h** |
| 5-Fold CV R² | 0.9995 ± 0.0001 |
| Test samples | 53,706 |

> The model explains **99.95% of variance** in rainfall intensity with an average prediction error of only **1.44 mm/h** across 53,706 held-out test samples.

---

## 🔬 Background: RD-80 Disdrometer

The **RD-80 is a piezoelectric acoustic disdrometer**. Raindrops fall onto a 50 cm² membrane, generating voltage signals whose amplitude and duration are used to classify drops into **20 diameter bins** ranging from 0.313 mm to 6.0 mm at **30-second sampling intervals**.

> **Note:** The RD-80 is an acoustic (impact-based) sensor — not an optical sensor. The piezoelectric membrane detects drop impact force to estimate drop size and count.

---

## ⚙️ Methodology

### 1. Data Preprocessing

```
RD-80 .txt files (2010–2015)
         ↓
   Parse & merge all year folders
         ↓
   Build unified timestamp
         ↓
   Remove zero-drop intervals
         ↓
   313,856 clean records
```

- Handles multiple yearly folder structures automatically
- Removes sensor noise spikes (RI > 300 mm/h capped)
- Deduplicates overlapping timestamps

---

### 2. Physics-Based Rainfall Intensity

$$RI = \frac{\pi}{6} \cdot \frac{3.6 \times 10^3}{F_{cm^2} \cdot t} \sum_{i=1}^{20} n_i D_i^3$$

Where:
- $n_i$ = drop count in bin $i$
- $D_i$ = drop diameter (mm)
- $F$ = sensor area (50 cm²)
- $t$ = sampling interval (30 s)

**Key finding:** The instrument-recorded RI (mean 27.35 mm/h) differs from the physics-computed RI (mean 41.41 mm/h) due to known RD-80 piezoelectric membrane dead-time effects and internal smoothing — a documented sensor characteristic.

---

### 3. Drop Size Distribution

$$N(D_i) = \frac{n_i}{F \cdot t \cdot v(D_i) \cdot \Delta D_i}$$

Where $v(D_i)$ is the terminal fall velocity and $\Delta D_i$ is the bin width. This gives the number concentration of drops [m⁻³ mm⁻¹] — the fundamental quantity in rainfall microphysics.

---

### 4. Feature Engineering

Beyond raw drop counts, the following physics-derived features were engineered:

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `LWC` | $(π/6) \cdot 10^{-3} \sum N(D) \cdot D^3 \cdot \Delta D$ | Liquid Water Content [g/m³] |
| `Dm` | $\sum N D^4 \Delta D / \sum N D^3 \Delta D$ | Mass-weighted mean diameter [mm] |
| `Z` | $\sum N(D) \cdot D^6 \cdot \Delta D$ | Radar reflectivity factor [mm⁶ m⁻³] |
| `log_Z` | $\log(1 + Z)$ | Log-transform for heavy-tailed Z |
| `total_drops` | $\sum n_i$ | Total drop count |

---

### 5. Machine Learning Model

**Algorithm:** Random Forest Regression (Scikit-learn)

```
Input  (29 features):  n1–n20 drop counts + LWC + Dm + D_mean + Z + log_Z + hour + month
                                          ↓
                               Random Forest (200 trees)
                               max_depth = 15, max_features = sqrt
                                          ↓
Output:  Rainfall Intensity (mm/h)
```

**Why Random Forest?**
- Handles the nonlinear DSD → RI relationship naturally
- Robust to sensor noise through ensemble averaging
- Provides interpretable feature importance rankings
- No scaling required for tree-based methods

**Top 5 Features by Importance:**

| Rank | Feature | Importance | Physical Reason |
|------|---------|-----------|----------------|
| 1 | LWC | 27.85% | Directly proportional to rainfall mass |
| 2 | log_Z | 16.56% | Radar reflectivity — basis of Z-R relations |
| 3 | Z | 14.18% | Radar reflectivity (raw) |
| 4 | n7 | 10.07% | Mid-size drops (1.13mm) — most numerous in tropical rain |
| 5 | n8 | 6.97% | Mid-size drops (1.33mm) — carry most rainfall mass |

> The model independently rediscovered that **LWC and Z are the dominant rainfall predictors** — validating decades of radar meteorology and disdrometer research.

---

## 📈 Visualisations

| Plot | Description |
|------|-------------|
| `plots/rf_actual_vs_predicted.png` | Scatter: actual vs predicted RI (R²=0.9995) |
| `plots/rf_feature_importance.png` | Top 20 features ranked by importance |
| `plots/rf_residuals.png` | Residual distribution (μ=0.023, σ=1.437) |
| `plots/rf_ri_distribution.png` | Actual vs predicted RI histogram overlay |
| `plots/comparism.png` | Instrument RI vs Computed RI comparison |
| `plots/DistributionCurve.png` | DSD N(D) curves |
| `plots/RainIntensity.png` | Rainfall intensity time series 2010–2015 |
| `plots/dataset_overview.png` | Multi-year dataset diagnostic overview |

---

## 📂 Repository Structure

```
RAINDROP ANALYSIS/
│
├── data/
│   ├── 2010-12/               # Raw RD-80 data files
│   ├── 2013/
│   ├── 2014/
│   └── 2015/
│
├── plots/                     # All output visualisations
│
├── mergeData.py               # Step 1: Merge all yearly RD-80 .txt files
├── preprocess.py              # Step 2: Clean, timestamp, feature extraction
├── rainIntensity.py           # Step 3: Physics-based RI + DSD computation
├── comparism.py               # Step 4: Instrument vs Computed RI comparison
├── plotDistributionCurve.py   # Step 5: DSD N(D) curve visualisation
├── rf_current_ri.py           # Step 6: Random Forest ML model (R²=0.9995)
├── main.py                    # Full pipeline runner
│
├── merged_data.csv            # Combined dataset (all years)
├── processed_data.csv         # Cleaned & feature-engineered dataset
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run the Full Pipeline
```bash
# Step 1: Merge all yearly RD-80 .txt files into one dataset
python mergeData.py

# Step 2: Clean data and extract features
python preprocess.py

# Step 3: Compute physics-based RI
python rainIntensity.py

# Step 4: Compare Instrument RI vs Computed RI
python comparism.py

# Step 5: Plot DSD curves
python plotDistributionCurve.py

# Step 6: Train Random Forest ML model
python rf_current_ri.py

# Or run the entire pipeline at once
python main.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib |
| Machine Learning | Scikit-learn |
| IDE | VS Code |

---

## 🔭 Future Work

### Short-Term Rainfall Forecasting
The next planned extension is a **next-interval RI forecasting model** using temporal lag features:

- **Model:** Random Forest / XGBoost / LightGBM
- **Input:** DSD features at time *t* + lag values from *t-1, t-2, t-3*
- **Output:** Predicted RI at time *t + 30s* or *t + 1 hour*
- **Approach:** Year-based train/test split (train 2010–2014, test on 2015) to simulate real deployment on unseen future data
- **Application:** Real-time flood early warning system integration

### Additional Planned Extensions
- Z-R relationship derivation and comparison with Marshall-Palmer (Z = 200R^1.6)
- Convective vs Stratiform rainfall classification using DSD parameters
- Seasonal and inter-annual DSD variability analysis
- Integration with NASA POWER satellite precipitation data for regional validation

---

## 📖 Key Equations Summary

| Parameter | Formula |
|-----------|---------|
| Rainfall Intensity | $RI = \frac{\pi}{6} \cdot \frac{3.6 \times 10^3}{F \cdot t} \sum n_i D_i^3$ |
| Drop Size Distribution | $N(D_i) = \frac{n_i}{F \cdot t \cdot v(D_i) \cdot \Delta D_i}$ |
| Liquid Water Content | $LWC = \frac{\pi}{6} \times 10^{-3} \sum N(D_i) D_i^3 \Delta D_i$ |
| Mass-weighted Diameter | $D_m = \frac{\sum N(D_i) D_i^4 \Delta D_i}{\sum N(D_i) D_i^3 \Delta D_i}$ |
| Radar Reflectivity | $Z = \sum N(D_i) D_i^6 \Delta D_i$ |

---

## 🧠 Key Scientific Contributions

1. **Multi-year tropical DSD analysis** across 313,856 rain intervals (2010–2015) — one of the largest disdrometer datasets analysed in the region
2. **Physics-ML integration** — features engineered from first-principle DSD equations fed into Random Forest, achieving R² = 0.9995
3. **Instrument characterisation** — quantified the 14 mm/h systematic offset between RD-80 instrument RI and physics-computed RI, attributed to piezoelectric dead-time effects
4. **Independent validation of Z-R theory** — feature importance analysis revealed Z and LWC as dominant predictors, consistent with 70 years of radar meteorology literature
5. **Reproducible pipeline** — end-to-end Python pipeline from raw `.txt` sensor files to trained ML model

---

## 👩‍💻 Authors

**Manishita Biswas** · **Anik Khajanchi**

B.Tech — Electronics & Communication Engineering
University of Calcutta

---

## ⭐ Acknowledgements

- RD-80 disdrometer data collected at the tropical monitoring station, Kolkata
- NASA POWER MERRA-2 for regional precipitation validation data
- Scikit-learn, NumPy, Pandas open-source communities

---

<div align="center">
If you found this project useful, please consider giving it a ⭐
</div>
