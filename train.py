import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

#Setting seed for reproducibility
np.random.seed(193)

# LOADING DATA 
print("Loading dataset...")
df = pd.read_csv('data/medical_insurance.csv')
df.drop_duplicates(inplace=True)

#FEATURE ENGINEERING
def create_engineered_features(data):
    data = data.copy()
    data['is_obese'] = (data['bmi'] >= 30).astype(int)
    smoker_binary = (data['smoker'] == 'yes').astype(int)
    
    # Key Interaction Terms
    data['obese_smoker'] = data['is_obese'] * smoker_binary
    data['age_smoker'] = data['age'] * smoker_binary
    data['is_parent'] = (data['children'] > 0).astype(int)
    return data

# Spliting columns into Features & Target
X = df.drop(columns=['charges'])
y = df['charges']


#Splitting dataset into train, validation, and test sets (80% train, 10% validation, 10% test)
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=193)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.5, random_state=193)

# Applying Feature Engineering
X_train_eng = create_engineered_features(X_train)
X_val_eng = create_engineered_features(X_val)
X_test_eng = create_engineered_features(X_test)

# DATA PREPROCESSING
num_cols = ['age', 'bmi', 'children', 'age_smoker']
cat_cols = ['smoker', 'region', 'sex']
binary_cols = ['is_obese', 'obese_smoker', 'is_parent']

#preprocessor
preprocessor_eng = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('passthrough', 'passthrough', binary_cols)
    ]
)

# Fit preprocessor on training data only
X_train_pr = preprocessor_eng.fit_transform(X_train_eng)
X_val_pr = preprocessor_eng.transform(X_val_eng)
X_test_pr = preprocessor_eng.transform(X_test_eng)


#TRAINING THE TUNED XGBOOST MODEL

print("Training tuned XGBoost Regressor...")
model = XGBRegressor(
    random_state=193,
    max_depth=3,
    learning_rate=0.03,
    n_estimators=150,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0
)

model.fit(X_train_pr, y_train)

# Evaluation on Validation & Test sets
val_preds = model.predict(X_val_pr)
test_preds = model.predict(X_test_pr)

print("\n--- Model Performance ---")
print(f"Training MAE   : ${mean_absolute_error(y_train, model.predict(X_train_pr)):,.2f}")
print(f"Training R2    : {r2_score(y_train, model.predict(X_train_pr)):,.4f}")
print(f"Validation MAE : ${mean_absolute_error(y_val, val_preds):,.2f}")
print(f"Validation R2  : {r2_score(y_val, val_preds):.4f}")
print(f"Test MAE       : ${mean_absolute_error(y_test, test_preds):,.2f}")
print(f"Test R2        : {r2_score(y_test, test_preds):.4f}")


#SAVING PIPELINE FOR STREAMLIT DEPLOYMENT

joblib.dump(preprocessor_eng, 'models/preprocessor.joblib')
joblib.dump(model, 'models/xgb_model.joblib')
print("\nSaved 'preprocessor.joblib' and 'xgb_model.joblib' for Streamlit!")


# DATA VISUALIZATIONS

# Reconstruct transformed feature names for plotting
cat_encoder = preprocessor_eng.named_transformers_['cat']
encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + encoded_cat_names + binary_cols


# Plot 1: Feature Importances

plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
importances.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('XGBoost Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('Screenshots/feature_importance.png', dpi=300)
plt.show()


# Plot 2: Actual vs. Predicted Charges

plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_preds, alpha=0.7, color='teal', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction (45°)')
plt.title('Actual vs. Predicted Medical Charges (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Screenshots/actual_vs_predicted.png', dpi=300)
plt.show()


# Plot 3: Residuals Distribution

residuals = y_test - test_preds
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple', bins=25)
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals Distribution (Errors = Actual - Predicted)', fontsize=14, fontweight='bold')
plt.xlabel('Prediction Error ($)')
plt.tight_layout()
plt.savefig('Screenshots/residuals_distribution.png', dpi=300)
plt.show()