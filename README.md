# PHQ-9 Temporal Change-point Detection


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Temporal clustering and change point detection framework based on PHQ-9 scores to analyze mental health trajectories over time. Includes synthetic data generation, EDA, and robust visual/statistical interpretation.

This repository presents an end-to-end system that:

- Generates synthetic PHQ-9 data resembling real-world clinical trends.
- Performs exploratory data analysis (EDA) on patient mental health assessments.
- Detects statistically significant **change points** using the **PELT** algorithm with L1 regularization.
- Visualizes the progression of depression severity using robust statistical and graphical tools.

---

## 🧩 Key Features

- 🔁 **Synthetic Data Generator**: Clinically structured PHQ-9 progression  
- 📊 **EDA Module**: Clustering, daily averages, summary stats  
- 🧠 **Change Point Detection**: PELT with LASSO regularization and statistical validation  
- 📉 **Advanced Visualization**: Scatter plots, segmentation diagrams, validation plots  
- 📁 **Well-Organized Results**: Auto-generated plots, CSVs, JSON summaries  

---

## 📐 Model Architecture: PELT + LASSO for Change Point Detection

We leverage the **Pruned Exact Linear Time (PELT)** algorithm for exact and efficient change point detection, combined with a **LASSO penalty** to prevent over-segmentation by penalizing the number of change points.

- **Why PELT ?** It’s one of the only exact, linear-time algorithms for multiple change point detection.
- **Why LASSO Penalty ?** To sparsify segment detection and maintain interpretability while handling multivariate shifts.

This method robustly detects both abrupt and subtle distributional changes in temporal PHQ-9 trajectories.

---

## 📊 Temporal Clustering via Coefficient of Variation (CV)

After change point detection, we compute the **Coefficient of Variation (CV)** for each segment:

- CV = (Standard Deviation) / (Mean)
- A higher CV implies instability or increased fluctuation in depression scores.

We perform clustering of these CV segments using **unsupervised methods** (e.g., KMeans) and visualize distinct psychological states over time.

This approach provides interpretable behavioral phases (stable vs volatile mental states).

---


## 🚀 Getting Started

### 🛠 Installation

```bash
git clone https://github.com/satyaki-mitra/phq9-temporal-changepoint-detection.git
cd phq9-temporal-changepoint-detection
pip install -r requirements.txt
```

## ▶️ Run the Pipeline

### 🔹 Step 1: Generate Synthetic Data

```bash
python generate_data.py
```
**Outputs:**

- `data/synthetic_phq9_data.csv`

**Logs:**

- `logs/phq9_data_generator.log`

## 🔹 Step 2: Perform EDA

```bash
python eda_performer.py
```

**Outputs (saved in `results/eda_results/`):**

- `cluster_results.png`, `cluster_optimization.png`, `scatter_plot.png`, `daily_averages.png`
- `summary_statistics.csv`, `cluster_characteristics.csv`, `clustering_method_comparison.csv`
- `analysis_summary.json`

**Logs:**

- `logs/phq9_exploratory_data_analysis.log`

## 🔹 Step 3: Detect Change Points

```bash
python run.py
```

**Outputs:**

- `aggregated_data_plot.png`
- `aggregated_cv_data.csv`
- `change_point_analysis_results.json`
- `change_point_detected_scatter_diagram.png`
- `statistical_test_results.csv`
- `validation_plot.png`
- `cluster_boundaries.csv`

**Logs:**

- `logs/change_point_detection.log`

## 📂 Project Structure

```plaintext
phq9-temporal-changepoint-detection/
├── config.py
├── generate_data.py
├── eda_performer.py
├── run.py
├── src/
│   ├── change_point_detector.py
│   ├── phq9_data_analyzer.py
│   └── synthetic_phq9_data_generator.py
├── data/
│   └── synthetic_phq9_data.csv
├── results/
│   ├── aggregated_cv_data.csv
│   ├── aggregated_data_plot.png
│   ├── change_point_analysis_results.json
│   ├── change_point_detected_scatter_diagram.png
│   ├── cluster_boundaries.csv
│   ├── statistical_test_results.csv
│   ├── validation_plot.png
│   └── eda_results/
│       ├── analysis_summary.json
│       ├── cluster_characteristics.csv
│       ├── cluster_optimization.png
│       ├── cluster_results.png
│       ├── clustering_method_comparison.csv
│       ├── daily_averages.png
│       ├── scatter_plot.png
│       └── summary_statistics.csv
├── logs/
│   └── *.log
├── notebooks/
│   └── *.ipynb
├── docs/
│   └── *.pdf
├── requirements.txt
└── README.md
```

## 🔬 Methodology Highlights

### 📌 Data Processing

- Aggregation Metric: **Coefficient of Variation (CV)**
- Temporal Aggregation: **Day-wise PHQ-9 consolidation**
- Sparse Sampling Support

### 📌 Change Point Detection

- Algorithm: **PELT**
- Cost Function: **L1 norm**
- Regularization: **LASSO**
- Significance Tests: **Wilcoxon rank-sum**, fallback to **T-test**

### 📌 Clustering (EDA)

- **KMeans** and other clustering strategies
- Multiple cluster evaluation metrics
- Daily score trends and visual diagnostics

---

## 📈 Example Outputs

- **Change Point Visualization**: `change_point_detected_scatter_diagram.png`, `aggregated_data_plot.png`
- **Clustered Patterns**: `cluster_results.png`, `scatter_plot.png`
- **Daily Averages**: `daily_averages.png`, `summary_statistics.csv`

![Aggregated CV Plot](results/aggregated_data_plot.png)

**Interpretation:**  
This graph illustrates the daily coefficient of variation in PHQ-9 scores across patients. Elevated CV values may reflect more heterogeneous mental health responses or periods of instability in mood among patients.

---

### 🔹 Change Point Detection Results

![Change Point Detection](results/change_point_detected_scatter_diagram.png)

**Interpretation:**  
Using the PELT algorithm with LASSO regularization, this scatter plot marks statistically significant temporal change points. These points represent major shifts in population-level depression patterns.

---

### 🔹 Model Validation – Predicted vs Observed

![Validation Plot](results/validation_plot.png)

**Interpretation:**  
This plot shows the model’s fit quality, helping validate the detected change points by overlaying predicted vs actual score deviations. Well-aligned points indicate strong model robustness.

---

## 🧪 Core Libraries

- `ruptures` – for change point detection  
- `pandas`, `numpy` – data manipulation  
- `matplotlib`, `seaborn` – data visualization  
- `scikit-learn` – clustering  
- `scipy.stats` – statistical validation  

---

## 📘 Notebooks for Exploration

- `notebooks/PHQ-9_Synthetic_data_Generation_and_EDA.ipynb`
- `notebooks/Temporal Clustering  of PHQ-9 Scores of a Specified Set of Patients.ipynb`

---

## ⚕️ Clinical and Research Applications

- 📉 **Treatment Response Monitoring**
- 🛑 **Relapse Detection**
- 📊 **Clinical Trial Analytics**
- 🎯 **Personalized Intervention Strategies**

---

## 🙋 Author

**Satyaki Mitra**  
_Data Scientist | ML Enthusiast | Clinical AI Research_

---

⭐ *If you found this useful, consider starring the repository!*
