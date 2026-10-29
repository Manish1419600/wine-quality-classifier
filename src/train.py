import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# ==========================================
# 1. DATA ACQUISITION & TARGET DEFINITION
# ==========================================
# Fetching the Wine Quality dataset directly from the UCI Machine Learning Repository
url = "https://uci.edu"
df = pd.read_csv(url, sep=';')

print("--- Initial Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# Transform into a binary classification problem: High Quality (1) vs Low Quality (0)
# Standard wine quality scores range from 3 to 8. We set the threshold at 6.
df['quality_label'] = (df['quality'] >= 6).astype(int)

# Drop the original continuous quality column to avoid data leakage
df = df.drop(columns=['quality'])

# ==========================================
# 2. FEATURE ENGINEERING & PREPROCESSING
# ==========================================
# Interaction feature: Ratio of volatile acidity to fixed acidity
# High volatile acidity often ruins taste, while fixed acidity balances it.
df['acid_ratio'] = df['volatile_acidity'] / (df['fixed_acidity'] + 1e-5)

# Separate independent features (X) and target label (y)
X = df.drop(columns=['quality_label'])
y = df['quality_label']

# Split data: 80% Training, 20% Testing (Using stratification to balance class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize features using standard normal distribution (Mean=0, Var=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING (RANDOM FOREST ENGINE)
# ==========================================
# Initialising the Random Forest Classifier with ensemble tuning parameters
rf_model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=12, 
    random_state=42, 
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)

# ==========================================
# 4. MODEL VALIDATION & PERFORMANCE EVALUATION
# ==========================================
# 5-Fold Stratified Cross-Validation on the training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

# Test set predictions
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Output evaluation metrics to terminal
print("--- Validation & Performance Metrics ---")
print(f"5-Fold CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")

print("--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ==========================================
importance = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print("--- Feature Importance Ranking ---")
print(feature_imp_df.to_string(index=False))
