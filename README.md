# 🌧️ Rain Drop Size Distribution Analysis and Rainfall Prediction

### **Final Year Project — B.Tech (Electronics & Communication Engineering)**
### **Using Python & RD-80 Disdrometer Data**

---

## 📌 Project Overview

This project analyzes **Rain Drop Size Distribution (DSD)** in a tropical region using multi-year real-world data (2010–2015) collected from an **RD-80 Optical Disdrometer**.

The work integrates:
- 🌧️ **Rainfall microphysics (DSD analysis)**
- 📊 **Classical rainfall computation**
- 🤖 **Machine Learning-based prediction**
- ⏱️ **Short-term rainfall forecasting (30-second ahead)**

---

## 🎯 Objectives

- Convert raw RD-80 data (`.txt → .csv`)
- Merge multi-year datasets (2010–2015)
- Perform **data preprocessing & cleaning**
- Compute Rainfall Intensity (RI) and Drop Size Distribution N(D)
- Compare **Instrument RI vs Computed RI**
- Generate DSD curves and rainfall intensity time series
- Develop an **ML model for rainfall prediction**
- Build a **30-second ahead forecasting model**

---

## 📂 Dataset Description

The dataset consists of RD-80 disdrometer measurements sampled every **30 seconds**:

| Field | Description |
|-------|-------------|
| `n1 … n20` | Drop count per size class (20 classes) |
| `Di` | Drop diameters |
| `v(Di)` | Terminal fall velocities |
| `ΔDi` | Diameter bin width |
| `RI` | Rainfall Intensity |
| `RA` | Rain Amount |
| `RAT` | Accumulated Rainfall |
| Timestamp | `YYYY-MM-DD hh:mm:ss` |

---

## ⚙️ Methodology

### 🔹 1. Data Preprocessing
- Convert `.txt` → `.csv` and merge all yearly files
- Create unified timestamp column
- Remove **no-rain intervals** (where `Σnᵢ = 0`)
- Extract temporal features: hour, month, season

### 🔹 2. Rainfall Intensity (Physics-Based)

$$
RI = \frac{\pi}{6} \cdot \frac{3.6 \times 10^3}{F \cdot t} \sum_{i=1}^{20} n_i D_i^3
$$

Based on drop volume and fall velocity; captures nonlinear rainfall behavior.

### 🔹 3. Drop Size Distribution (DSD)

$$
N(D_i) = \frac{n_i}{F \cdot t \cdot v(D_i) \cdot \Delta D_i}
$$

Represents the number density of raindrops — key to understanding rainfall microphysics.

### 🔹 4. Comparison Analysis
- Instrument RI vs Computed RI
- Log-scale comparison and temporal pattern validation

### 🔹 5. Machine Learning Model

| Item | Detail |
|------|--------|
| **Model** | Random Forest Regression |
| **Input Features** | Drop counts (`n1…n20`), total drop count, hour, month |
| **Target** | Rainfall Intensity (RI) |
| **Goal** | Map DSD → Rainfall Intensity |

### 🔹 6. Rainfall Forecasting *(Novel Contribution)*
- **Forecast horizon:** 30 seconds ahead
- Time-series regression approach on sequential DSD windows

---

## 📊 Results & Visualizations

| Plot | File |
|------|------|
| Drop Size Distribution Curve | `plots/DistributionCurve.png` |
| Rainfall Intensity vs Time | `plots/RainIntensity.png` |
| Instrument RI vs Computed RI | `plots/comparism.png` |
| RI Overlay Diagnostics | `ri_overlay_diagnostics.png` |

---

## 📁 Repository Structure

```
RAINDROP ANALYSIS/
│
├── data/
│   ├── 2010-12/          # Raw RD-80 data (2010–2012)
│   ├── 2013/             # Raw RD-80 data (2013)
│   ├── 2014/             # Raw RD-80 data (2014)
│   └── 2015/             # Raw RD-80 data (2015)
│
├── plots/
│   ├── comparism.png           # Instrument RI vs Computed RI
│   ├── DistributionCurve.png   # DSD curve
│   └── RainIntensity.png       # Rainfall intensity time series
│
├── main.py                     # Main pipeline entry point
├── mergeData.py                # Merges multi-year data files
├── preprocess.py               # Data cleaning & feature extraction
├── rainIntensity.py            # Physics-based RI computation
├── plotDistributionCurve.py    # DSD visualization
├── comparism.py                # RI comparison & overlay plots
│
├── merged_data.csv             # Combined dataset (all years)
├── processed_data.csv          # Cleaned & feature-engineered dataset
├── ri_overlay_diagnostics.png  # Diagnostic overlay plot
│
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Programming | Python 3 |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| IDE | VS Code / Jupyter Notebook |
| Hardware | RD-80 Optical Disdrometer |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/raindrop-analysis.git
cd raindrop-analysis

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Run the pipeline
python mergeData.py        # Step 1: Merge raw data
python preprocess.py       # Step 2: Clean & preprocess
python rainIntensity.py    # Step 3: Compute RI
python comparism.py        # Step 4: Compare RI values
python plotDistributionCurve.py  # Step 5: Plot DSD
python main.py             # Or run the full pipeline at once
```

---

## 🧠 Key Contributions

- Multi-year tropical rainfall analysis (2010–2015)
- Integration of **physics-based computation + machine learning**
- Real-world RD-80 disdrometer data processing pipeline
- Short-term (30-second) rainfall forecasting model
- Feature importance analysis of raindrop size classes

---

## 👩‍💻 Authors

**Manishita Biswas** · **Anik Khajanchi**

B.Tech – Electronics & Communication Engineering
University of Calcutta

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub!
