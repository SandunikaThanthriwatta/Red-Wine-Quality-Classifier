# 🍷 Red Wine Quality Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Predicting red wine quality as **Good** or **Bad** using **Random Forest** and **Support Vector Machine** classifiers, with hyperparameter tuning via GridSearchCV.

## 📖 About the Project

This project classifies red wine samples as either **Good** (quality > 6.5) or **Bad** (quality ≤ 6.5) based on 11 physicochemical features. Two ML models are compared and evaluated.

## 📊 Dataset

**Source:** UCI Red Wine Quality — [Download from Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

- 1,599 red wine samples
- 11 physicochemical features
- Binary target: Good (1) vs Bad (0)

**Quality Distribution:** 5→681, 6→638, 7→199, 4→53, 8→18, 3→10

## 🗂️ Project Structure

```
Red-Wine-Quality-Classifier/
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # loading, splitting, scaling
│   ├── model.py          # RF, SVC, tuned SVC
│   └── evaluate.py       # metrics, confusion matrix
├── Red_Wine_Quality_RF_and_SVC.ipynb
├── main.py
├── requirements.txt
└── README.md
```

## 📈 Exploratory Data Analysis

Key feature trends observed vs wine quality:

| Feature | Trend |
|---|---|
| Volatile Acidity | ⬇️ Downward — lower = better quality |
| Alcohol | ⬆️ Upward — higher = better quality |
| Sulphates | ⬆️ Upward — more = better quality |
| Citric Acid | ⬆️ Upward — more = better quality |

> All EDA plots are available in the [original notebook](Red_Wine_Quality_RF_and_SVC.ipynb).

## 🤖 Models & Results

Data split: **80% train / 20% test** | Scaling: `StandardScaler`

### Accuracy Comparison

| Model | Accuracy | Notes |
|---|---|---|
| 🌲 Random Forest | **88.13%** | n_estimators=100 |
| 🔷 SVC (Default) | **87.50%** | Default RBF kernel |
| ✅ SVC (Tuned) | **89.69%** | C=1.2, gamma=0.9, kernel=rbf |

🏆 **Best Model:** Tuned SVC — **89.69% accuracy**

RF Cross-Validation (10-fold) Mean: **91.17%**

### Confusion Matrix — Random Forest

```
Predicted →     0      1
Actual 0:      263     10
Actual 1:       28     19
```
Correct: 282 / 320 | Accuracy: **88.13%**

### Confusion Matrix — Tuned SVC

```
Predicted →     0      1
Actual 0:      271      2
Actual 1:       31     16
```
Correct: 287 / 320 | Accuracy: **89.69%**

## 🚀 How to Run

```bash
# 1. Clone
git clone https://github.com/SandunikaThanthriwatta/Red-Wine-Quality-Classifier.git
cd Red-Wine-Quality-Classifier

# 2. Install
pip install -r requirements.txt

# 3. Add dataset (winequality-red.csv) to root folder

# 4. Run
python main.py
```

Or open the notebook:
```bash
jupyter notebook Red_Wine_Quality_RF_and_SVC.ipynb
```

## 📦 Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn
```

## 👩‍💻 Author

**Sandunika Thanthriwatta** — [@SandunikaThanthriwatta](https://github.com/SandunikaThanthriwatta)

---
*P. Cortez et al. Modeling wine preferences by data mining from physicochemical properties. DSS, 2009.*
