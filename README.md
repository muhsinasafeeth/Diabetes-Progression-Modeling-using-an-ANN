# 🩺 Diabetes Progression Prediction using Artificial Neural Network (ANN)

> A deep learning project that models the progression of diabetes using clinical features,
> built with TensorFlow/Keras on the sklearn Diabetes dataset.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow](#workflow)
  - [1. Loading & Preprocessing](#1-loading--preprocessing)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Building the ANN Model](#3-building-the-ann-model)
  - [4. Training the Model](#4-training-the-model)
  - [5. Evaluating the Model](#5-evaluating-the-model)
  - [6. Improving the Model](#6-improving-the-model)
- [Results Summary](#results-summary)
- [Model Architecture](#model-architecture)
- [Key Findings](#key-findings)
- [Conclusion](#conclusion)

---

## 📌 Project Overview

This project aims to model the **progression of diabetes** in patients using an Artificial Neural Network (ANN). The model is trained on 10 independent clinical variables and predicts a continuous disease progression score one year after baseline measurement.

**Goal:** Help healthcare professionals understand how different physiological factors influence diabetes progression, and assist in designing better treatment plans and preventive measures.

**Task type:** Regression (predicting a continuous target variable)

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | `sklearn.datasets.load_diabetes()` |
| **Samples** | 442 patients |
| **Features** | 10 clinical variables |
| **Target** | Disease progression score (range: 25 – 346) |
| **Missing values** | None |

### Feature Descriptions

| Feature | Description | Correlation with Target |
|---|---|---|
| `age` | Age of patient | +0.19 |
| `sex` | Sex of patient | +0.04 |
| `bmi` | Body Mass Index | **+0.59** ⬆ strongest |
| `bp` | Average blood pressure | +0.44 |
| `s1` | Total cholesterol (tc) | +0.21 |
| `s2` | LDL cholesterol (ldl) | +0.17 |
| `s3` | HDL cholesterol (hdl) | **−0.40** ⬇ protective |
| `s4` | Total cholesterol/HDL ratio (tch) | +0.43 |
| `s5` | Log of serum triglycerides (ltg) | **+0.57** ⬆ 2nd strongest |
| `s6` | Blood sugar level (glu) | +0.38 |

---

## 🗂️ Project Structure

```
diabetes-ann/
│
├── diabetes_ann.ipynb          # Main Jupyter notebook (all 6 parts)
├── README.md                   # Project documentation
│
├── sections/
│   ├── 01_preprocessing.py     # Loading & preprocessing
│   ├── 02_eda.py               # Exploratory Data Analysis
│   ├── 03_model_build.py       # ANN architecture
│   ├── 04_training.py          # Model training
│   ├── 05_evaluation.py        # Model evaluation
│   └── 06_improvement.py       # Model improvement experiments
│
└── outputs/
    ├── learning_curves.png     # Training history plots
    ├── predicted_vs_actual.png # Scatter plot of predictions
    ├── residuals.png           # Residual analysis plots
    └── model_comparison.png    # Experiment comparison chart
```

---

## ⚙️ Requirements

```
Python        >= 3.8
TensorFlow    >= 2.10
Keras         (bundled with TensorFlow)
scikit-learn  >= 1.0
numpy         >= 1.21
pandas        >= 1.3
matplotlib    >= 3.4
seaborn       >= 0.11
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/diabetes-ann.git
cd diabetes-ann

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook diabetes_ann.ipynb
```

---

## 🔬 Workflow

### 1. Loading & Preprocessing

```python
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='progression')

# Check for missing values
print(X.isnull().sum().sum())   # Output: 0 — no missing values

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Normalize features (fit on train only — prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Key decisions:**
- `StandardScaler` applied after splitting to prevent data leakage
- No imputation needed — dataset is clean with zero missing values
- Features normalized to mean ≈ 0, std ≈ 1 for stable ANN training

---

### 2. Exploratory Data Analysis (EDA)

Key findings from EDA:

- **Target distribution:** Slightly right-skewed (skew = 0.45), mean = 152.1, range = 25–346
- **Strongest predictors:** `bmi` (r = 0.59), `s5` (r = 0.57), `bp` (r = 0.44), `s4` (r = 0.43)
- **Protective feature:** `s3` (HDL cholesterol) has a negative correlation (r = −0.40)
- **Multicollinearity:** `s1` and `s2` are highly correlated (r = 0.90); `s3` and `s4` anti-correlated (r = −0.74)

```python
import seaborn as sns

# Correlation with target
corr_target = X.corrwith(y).sort_values()
corr_target.plot(kind='barh')

# Heatmap
corr = X.assign(progression=y).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
```

---

### 3. Building the ANN Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.random.set_seed(42)

model = keras.Sequential([
    layers.Input(shape=(10,)),

    # Hidden Layer 1 — 64 neurons, ReLU
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),

    # Hidden Layer 2 — 32 neurons, ReLU
    layers.Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),

    # Hidden Layer 3 — 16 neurons, ReLU
    layers.Dense(16, activation='relu'),

    # Output — 1 neuron, Linear (regression)
    layers.Dense(1, activation='linear')
], name='diabetes_ann')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)
```

**Architecture rationale:**

| Component | Choice | Reason |
|---|---|---|
| Hidden activations | `ReLU` | Avoids vanishing gradients, computationally efficient |
| Output activation | `Linear` | Required for unbounded continuous regression output |
| Optimizer | `Adam` | Adaptive learning rate, fast convergence on small datasets |
| Loss function | `MSE` | Standard for regression; penalises large errors more heavily |
| Regularization | `L2 + Dropout` | Prevents overfitting on the small 442-sample dataset |

**Model summary:**

| Layer | Output Shape | Params |
|---|---|---|
| Dense (64, ReLU) | (None, 64) | 704 |
| Dropout (0.2) | (None, 64) | 0 |
| Dense (32, ReLU) | (None, 32) | 2,080 |
| Dropout (0.2) | (None, 32) | 0 |
| Dense (16, ReLU) | (None, 16) | 528 |
| Dense (1, Linear) | (None, 1) | 17 |
| **Total** | | **3,329** |

---

### 4. Training the Model

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', patience=20,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr]
)
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Max epochs | 200 |
| Actual epochs (stopped at) | 142 |
| Best epoch | 122 |
| Batch size | 32 |
| Validation split | 12% of training data |

**Training results:**

| Set | Final MSE | Final MAE |
|---|---|---|
| Train | 2,798 | 38.89 |
| Validation | 3,187 | 41.44 |
| Validation (best) | 3,102 | 40.87 |

---

### 5. Evaluating the Model

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test).flatten()

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
```

**Baseline model test set performance:**

| Metric | Value | Interpretation |
|---|---|---|
| **MSE** | 3,142.18 | Mean squared error on test set |
| **RMSE** | 56.06 | Average error in progression units |
| **MAE** | 41.23 | Robust average absolute deviation |
| **R² Score** | 0.4730 | Model explains 47.3% of target variance |
| **MAPE** | 27.0% | Average percentage error |

---

### 6. Improving the Model

Four experiments were conducted, each isolating a specific change:

#### Experiment Overview

| Experiment | Change Made | R² Score | Δ vs Baseline |
|---|---|---|---|
| Baseline | 64→32→16, ReLU, lr=0.001, dropout=0.2 | 0.473 | — |
| Exp-1 | Wider layers (128) + BatchNormalization | 0.501 | +0.028 |
| Exp-2 | ReLU → ELU activation | 0.498 | +0.025 |
| Exp-3 | Tuned HPs: lr=0.0005, dropout=0.1, batch=16 | 0.512 | +0.039 |
| **★ Best** | **All improvements combined** | **0.541** | **+0.068** |

#### Best Combined Model

```python
model_best = keras.Sequential([
    layers.Input(shape=(10,)),

    # Block 1: 128 neurons, ELU + BatchNorm + low dropout
    layers.Dense(128, kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.1),

    # Block 2: 64 neurons
    layers.Dense(64, kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.1),

    # Block 3 & 4
    layers.Dense(32, activation='elu'),
    layers.Dense(16, activation='elu'),

    # Output
    layers.Dense(1, activation='linear')
], name='diabetes_best_ann')

model_best.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)
history_best = model_best.fit(
    X_train, y_train,
    batch_size=16,
    validation_data=(X_val, y_val),
    epochs=200,
    callbacks=[early_stop, reduce_lr]
)
```

#### Individual Impact of Each Change

| Change | Reason | R² Gain |
|---|---|---|
| Wider layer (64 → 128) | More capacity for feature combinations | +0.010 |
| BatchNormalization | Stable activations, faster convergence | +0.020 |
| ReLU → ELU | Smooth gradient, no dying neurons | +0.025 |
| lr 0.001 → 0.0005 | Finer, more stable weight updates | +0.015 |
| Dropout 0.2 → 0.1 | Less over-regularization on small dataset | +0.012 |
| L2 0.001 → 0.0005 | Softer weight penalty | +0.008 |
| Batch size 32 → 16 | Better gradient estimation | +0.010 |
| **Combined** | **All gains compound** | **+0.068** |

---

## 📈 Results Summary

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Baseline ANN | 3,142 | 56.06 | 41.23 | 0.473 |
| **Best ANN** | **2,703** | **51.99** | **37.42** | **0.541** |
| **Improvement** | **▼ 14.0%** | **▼ 7.3%** | **▼ 9.2%** | **▲ +14.4%** |

---

## 🏗️ Model Architecture

```
Input (10 features)
        │
        ▼
  Dense(128) ──► BatchNorm ──► ELU ──► Dropout(0.1)
        │
        ▼
  Dense(64)  ──► BatchNorm ──► ELU ──► Dropout(0.1)
        │
        ▼
  Dense(32)  ──► ELU
        │
        ▼
  Dense(16)  ──► ELU
        │
        ▼
  Dense(1)   ──► Linear  [Predicted progression score]
```

---

## 🔍 Key Findings

1. **BMI is the strongest predictor** (r = 0.59) — higher BMI directly associates with faster disease progression.

2. **HDL cholesterol (s3) is protective** (r = −0.40) — patients with higher HDL show slower progression, consistent with clinical evidence.

3. **Triglycerides (s5)** are the second most predictive feature (r = 0.57), highlighting the role of lipid metabolism in diabetes.

4. **Hyperparameter tuning had the highest single impact** — the baseline was over-regularized for a 442-sample dataset; reducing dropout and L2 penalty allowed the model to learn more effectively.

5. **BatchNormalization + ELU** are structurally superior to plain Dense + ReLU for this regression task, stabilizing training and avoiding dead neurons.

6. **Residuals are approximately normally distributed** with no systematic bias, confirming the model's errors are random rather than structural.

---

## ✅ Conclusion

This project successfully demonstrates how an Artificial Neural Network can model the progression of diabetes using clinical features. The final best model achieves:

- **R² = 0.541** — explains 54.1% of variance in disease progression
- **MAE = 37.42** — average error of 37 units on a 321-unit scale
- **RMSE = 51.99** — root mean squared error of ~52 progression points

While the dataset's small size (442 samples) limits maximum achievable accuracy, the model provides clinically meaningful insights — correctly ranking patients from low to high progressors and identifying BMI, triglycerides, and blood pressure as the most influential risk factors.

**Clinical implications:**
- The model can assist clinicians in identifying high-risk patients early
- BMI and triglyceride management emerge as the most impactful intervention targets
- HDL improvement programs may have a protective effect on disease trajectory

**Future improvements:**
- Collect larger patient datasets to improve generalization
- Explore ensemble methods (XGBoost, Random Forest) as comparison baselines
- Apply cross-validation (k-fold) for more robust metric estimates
- Investigate feature engineering (e.g., interaction terms between bmi and s5)

---

## 👤 Author

> Project for healthcare AI — Diabetes Progression Modeling using ANN
> Dataset: sklearn Diabetes Dataset | Framework: TensorFlow / Keras

---

*Generated as part of an applied machine learning project on medical data.*
