# 🩺 Medical Insurance Cost Predictor

An end-to-end Machine Learning web application that predicts individual annual medical insurance charges using tuned **XGBoost Regression** and custom **Feature Engineering**.

---

## 📌 Problem Statement
Medical insurance costs vary significantly based on patient demographics and health factors. Standard models often struggle with high variance due to extreme cost jumps associated with combined risk factors (e.g., smoking combined with high BMI). 

The goal of this project is to build a robust regression pipeline that accurately estimates medical bills while mitigating severe overfitting between training and validation stages.

---

## 📊 Dataset Overview
The dataset contains **2,772 records** of health insurance beneficiaries with the following features:

* **`age`**: Primary beneficiary age (numerical)
* **`sex`**: Insurance contractor gender (`female`, `male`)
* **`bmi`**: Body Mass Index (numerical)
* **`children`**: Number of children covered by health insurance (numerical)
* **`smoker`**: Smoking status (`yes`, `no`)
* **`region`**: Residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`)
* **`charges`** *(Target)*: Individual medical costs billed by health insurance

---

## 🛠️ Preprocessing & Feature Engineering
Standard linear correlations missed crucial non-linear interactions. To capture key medical cost drivers without deep, overfitted tree branches, the following features were engineered:

1. **`is_obese`**: Binary flag indicating BMI ≥ 30.
2. **`obese_smoker`**: Interaction term (`is_obese` × `smoker`) capturing the primary exponential cost jump.
3. **`age_smoker`**: Interaction term (`age` × `smoker`).
4. **`is_parent`**: Binary indicator derived from `children > 0`.

### Data Pipeline Configuration
* **Numeric Features**: Scaled using `StandardScaler` (`age`, `bmi`, `children`, `age_smoker`).
* **Categorical Features**: One-hot encoded using `OneHotEncoder(drop='first')` (`smoker`, `region`, `sex`).
* **Binary Interactions**: Passed through directly (`is_obese`, `obese_smoker`, `is_parent`).

---

## ⚙️ Model Architecture & Hyperparameter Tuning
We evaluated **Ridge Regression**, **Random Forest**, and **XGBoost Regressor**. Default XGBoost exhibited severe overfitting ($R^2$ Train: ~0.99 vs. Val: ~0.75).

To resolve this, we constrained tree complexity using regularization parameters:
* `max_depth=3`: Restricts tree depth to force general pattern learning.
* `learning_rate=0.03`: Smooths step size updates across boosting iterations.
* `subsample=0.8` & `colsample_bytree=0.8`: Introduces row and feature sampling variance reduction.
* `reg_alpha=1.0` ($L_1$) & `reg_lambda=5.0` ($L_2$): Suppresses leaf weight complexity.

---

## 📈 Evaluation Results

| Evaluation Metric | Training Set | Validation Set | 5-Fold Cross-Validation | Held-Out Test Set |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | \$2,234.25 | \$3,240.69 | — | **\$1,909.11** |
| **$R^2$ Score** | **0.8791** | **0.7935** | **0.8565 (±0.030)** | **0.9318** |

*Cross-Validation across 5 folds confirmed a stable mean $R^2$ score of **85.65%**, confirming strong generalization without overfitting.*

---

## ⚠️ Limitations & Future Work
1. **Regional Representation**: Dataset regions are restricted to US geographical blocks and do not reflect international pricing structures.
2. **Medical History Constraints**: Lacks granular clinical attributes (e.g., pre-existing conditions, specific prescriptions) that heavily influence insurance underwriting.
3. **Price Inflation**: Medical charges reflect historic baseline costs and require real-time index adjustments for operational deployment.

---

## 🚀 Local Setup & Run

1. **Clone Repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/medical-insurance-cost-predictor.git](https://github.com/YOUR_USERNAME/medical-insurance-cost-predictor.git)
   cd medical-insurance-cost-predictor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```
