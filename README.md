# PHQ-9 Temporal Change-point Detection


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Advanced temporal clustering system for detecting significant shifts in depression severity patterns using PHQ-9 questionnaire data.

## 🎯 Overview

This project implements a sophisticated change point detection system to identify critical moments in depression treatment trajectories. By analyzing Patient Health Questionnaire-9 (PHQ-9) scores over time, the system can detect when patients experience significant improvements or deteriorations in their mental health status.

### Key Features

- **🔍 Change Point Detection**: PELT algorithm with L1 regularization for optimal segmentation
- **📊 Statistical Validation**: Wilcoxon rank-sum and t-tests for cluster significance
- **📈 Comprehensive Visualization**: Multi-layered plots for clinical interpretation
- **🏥 Healthcare-Ready**: Handles sparse, real-world clinical data patterns
- **🧪 Synthetic Data Generation**: Clinically realistic PHQ-9 progression simulation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/satyaki-mitra/phq9-temporal-changepoint-detection.git
cd phq9-temporal-changepoint-detection
pip install -r requirements.txt
```

### Basic Usage

```python
from src.run import change_point_detection_run

# Analyze PHQ-9 data for change points
result = change_point_detection_run("data/PHQ_9_sample_dataset.csv")
print(result)  # "Successful Run"
```

### Generate Synthetic Data

```python
# Open and run the Jupyter notebook
jupyter notebook notebooks/PHQ-9_Synthetic_data_Generation_and_EDA.ipynb
```

## 📋 Methodology

### 1. Data Preprocessing
- **Aggregation Metric**: Coefficient of Variation (CV) across daily PHQ-9 scores
- **Missing Data Handling**: Sparse survey completion patterns
- **Temporal Alignment**: 365-day treatment timeline analysis

### 2. Change Point Detection
- **Algorithm**: Pruned Exact Linear Time (PELT)
- **Cost Function**: L1 norm (Least Absolute Deviations)
- **Regularization**: LASSO for optimal segmentation
- **Parameters**: Configurable penalty, minimum cluster size, jump constraints

### 3. Statistical Validation
- **Significance Testing**: Wilcoxon rank-sum test for non-parametric data
- **Fallback Method**: T-test for constant value groups
- **Alpha Level**: 0.05 significance threshold
- **Interpretation**: Automated p-value analysis

## 📊 Sample Results

The system typically identifies **6 distinct phases** in depression treatment:

1. **Initial Assessment** (Days 2-10): High variability, baseline establishment
2. **Early Treatment** (Days 11-19): Moderate improvement, stabilization begins
3. **Treatment Response** (Days 21-60): Significant variation, potential setbacks
4. **Stabilization Period** (Days 64-71): Reduced variability, consistent improvement
5. **Recovery Phase** (Days 73-84): Temporary fluctuations, overall progress
6. **Maintenance** (Days 92-364): Long-term stability, sustained recovery

## 🗂️ Project Structure

```
depression-trajectory-clustering/
├── 📊 data/
│   └── PHQ_9_sample_dataset.csv
├── 📓 notebooks/
│   └── PHQ-9_Synthetic_data_Generation_and_EDA.ipynb
├── 🔧 src/
│   ├── run.py
│   ├── config.py
│   ├── preprocessing/
│   │   └── preprocess_data.py
│   ├── change_detection/
│   │   └── detect_change_points.py
│   ├── visualization/
│   │   └── visualize_change_points.py
│   └── tests/
│       └── test_cluster_validity.py
├── 📈 plots/
├── 📚 docs/
├── requirements.txt
└── README.md
```

## 🛠️ Technical Implementation

### Core Technologies

- **Python 3.8+**: Primary development language
- **Ruptures**: Change point detection library
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib**: Visualization and plotting
- **SciPy**: Statistical testing and validation
- **Scikit-learn**: Additional clustering methods

### Algorithm Details

```python
# PELT Configuration
model = "l1"                    # L1 regularization
cost_function = rpt.costs.CostL1
min_size = 2                    # Minimum cluster size
jump = 1                        # Search granularity
penalty = 0.5                   # Regularization strength
```

## 📈 Clinical Applications

### Mental Health Treatment
- **Treatment Efficacy Monitoring**: Identify when interventions are working
- **Relapse Detection**: Early warning system for symptom deterioration
- **Personalized Care**: Tailor treatment plans based on trajectory patterns

### Research Applications
- **Clinical Trials**: Objective measurement of treatment response timelines
- **Population Health**: Understanding depression recovery patterns at scale
- **Intervention Timing**: Optimal moments for treatment adjustments

## 🔬 Validation & Testing

### Statistical Rigor
- **Hypothesis Testing**: Automated significance testing between clusters
- **Cross-Validation**: Robust parameter selection
- **Sensitivity Analysis**: Penalty parameter optimization

### Quality Metrics
- **Cluster Separation**: Statistical significance between segments
- **Temporal Consistency**: Logical progression of depression scores
- **Clinical Validity**: Alignment with known treatment timelines

## 📊 Example Outputs

### Change Point Visualization
![Change Points](plots/validated_clusters_plot.png)

### Segmentation Analysis
![Segmentation](plots/final_segmentation_plot.png)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Clinical Validation**: Based on established PHQ-9 depression screening protocols
- **Statistical Methods**: PELT algorithm implementation using the Ruptures library
- **Healthcare Standards**: Aligned with clinical depression assessment guidelines

## 🙋‍♂️ Author

**Satyaki Mitra**  

---

⭐ **If this project helped you, please consider starring it!**
