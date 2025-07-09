# Smart Maintenance for ICS: AI-Powered Centrifugal Pump Monitoring

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

## 🧠 Project Overview

**Title**: Smart Maintenance for Industrial Centrifugal Pumps Using AI  
**Goal**: Simulate real-time ICS pump data and perform predictive failure detection using machine learning.

---

## ⚙️ Features

- 🏭 Realistic **centrifugal pump simulation**
- 📈 Generation of rich **sensor datasets**
- 🔧 Configurable **failure types** and thresholds
- 🎥 **Live animations** of sensor trends (export as .mp4)
- 🤖 **Random Forest**-based predictive model
- 🔍 **Gradio interface** for upload, training, inference, and visualization
- 📊 Full maintenance analysis with urgency scoring and recommendations

---

## 📁 Project Structure

```
smart-maintenance-ics/
├── data/
│   ├── fail.csv                  # Sensor data under failure conditions
│   ├── non_fail.csv              # Sensor data under normal operation
│   └── merged.csv                # Combined dataset for training
│
├── inference_results/
│   ├── failure/                  # Output visualizations for failure predictions
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── learning_curves.png
│   │   ├── maintenance_timeline.png
│   │   └── ...
│   └── non-failure/              # Output visualizations for non-failure predictions
│       └── ...
│
├── models/
│   ├── centrifuge.py             # Virtual pump simulator + Gradio UI
│   └── prediction_algo.py        # Random Forest ML model + inference + Gradio UI
│
├── visualizations/
│   ├── failure/                  # Sensor animations (fail state)
│   └── non-failure/              # Sensor animations (non-fail state)
│       └── *.mp4
```

---

## 🚀 How to Run

### Option 1: Google Colab (recommended)
Upload `centrifuge.py` and `prediction_algo.py` and run each section interactively.

### Option 2: Local
```bash
pip install gradio pandas numpy scikit-learn matplotlib seaborn joblib
python models/centrifuge.py     # For simulation + animation + dataset gen
python models/prediction_algo.py  # For training + prediction + maintenance analysis
```

---

## 🎯 Predictive Maintenance Logic

- Uses predicted failure transition labels
- Computes:
  - Failure proportions (critical/moderate/safe)
  - Operating time span
  - Urgency levels: **Urgent**, **High**, **Moderate**, **Low**
  - Maintenance timeframes (in hours/days)
- Suggests recommended actions:
  - Replace impeller / bearing
  - Lubrication / seal inspection
  - Flow/electrical adjustments

---

## 📊 Model Output Samples

- `confusion_matrix.png`
- `feature_importance.png`
- `learning_curves.png`
- `inference_label_distribution.png`
- `maintenance_timeline.png`
- `inference_results.csv`

---

## 📦 Requirements

- Python 3.8+
- Gradio
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib

---

## 📬 Author

**Adityan Balasubramanian**  
🔗 [badi-sh.com](https://badi-sh.com)  
📧 badishsec@gmail.com
