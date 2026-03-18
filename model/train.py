"""
FraudWatch Model Training
Trains a fraud detection model on synthetic data that mirrors the
statistical properties of the Kaggle Credit Card Fraud dataset
(284,807 transactions, 0.172% fraud rate).

When you download the real Kaggle dataset (creditcard.csv), replace
the data generation block with:
    df = pd.read_csv('creditcard.csv')
and the model will train on real data automatically.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline

np.random.seed(42)

# ── DATASET ──────────────────────────────────────────────────────────────────
# Mirrors Kaggle creditcard.csv structure: 284,807 rows, 0.172% fraud
# Features V1-V28 are PCA components (as in real dataset), plus Amount and Time

N_TOTAL = 284807
N_FRAUD = 492   # exact count from real Kaggle dataset
N_LEGIT = N_TOTAL - N_FRAUD

print(f"Generating {N_TOTAL:,} transactions ({N_FRAUD} fraud, {N_LEGIT:,} legitimate)...")

# Legitimate transactions
legit_v = np.random.randn(N_LEGIT, 28)
legit_v[:, 0] *= 2.0   # V1 has higher variance in real data
legit_v[:, 3] *= 1.5
legit_amount = np.abs(np.random.lognormal(4.0, 1.5, N_LEGIT))   # skewed amounts
legit_time   = np.sort(np.random.uniform(0, 172792, N_LEGIT))

# Fraud transactions — shifted means on key features (matches real dataset patterns)
fraud_v = np.random.randn(N_FRAUD, 28)
fraud_v[:, 0]  += -4.8   # V1: strongly negative for fraud
fraud_v[:, 1]  +=  3.2   # V2: positive
fraud_v[:, 2]  += -3.1
fraud_v[:, 3]  +=  4.0
fraud_v[:, 4]  += -3.5
fraud_v[:, 9]  += -5.0   # V10: strongly predictive
fraud_v[:, 10] += -5.5   # V11
fraud_v[:, 11] +=  4.8   # V12
fraud_v[:, 13] += -7.5   # V14: most predictive feature
fraud_v[:, 15] += -3.2
fraud_v[:, 16] += -5.8
fraud_v[:, 17] += -8.0   # V18: very predictive
fraud_fraud_amount = np.abs(np.random.lognormal(3.5, 1.2, N_FRAUD))
fraud_time = np.random.uniform(0, 172792, N_FRAUD)

# Build DataFrames
v_cols = [f'V{i}' for i in range(1, 29)]

df_legit = pd.DataFrame(legit_v, columns=v_cols)
df_legit['Amount'] = legit_amount
df_legit['Time']   = legit_time
df_legit['Class']  = 0

df_fraud = pd.DataFrame(fraud_v, columns=v_cols)
df_fraud['Amount'] = fraud_fraud_amount
df_fraud['Time']   = fraud_time
df_fraud['Class']  = 1

df = pd.concat([df_legit, df_fraud], ignore_index=True).sample(frac=1, random_state=42)
df['Amount_log'] = np.log1p(df['Amount'])
df['Hour'] = (df['Time'] / 3600).astype(int) % 24

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")

# ── FEATURES ─────────────────────────────────────────────────────────────────
FEATURES = v_cols + ['Amount_log', 'Hour']
X = df[FEATURES]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
# Random Forest with class_weight to handle extreme imbalance
print("\nTraining Random Forest...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 100},   # penalise missing fraud heavily
        n_jobs=-1,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# ── EVALUATION ───────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_proba     = model.predict_proba(X_test)[:, 1]
roc_auc     = roc_auc_score(y_test, y_proba)
avg_prec    = average_precision_score(y_test, y_proba)
precision   = precision_score(y_test, y_pred)
recall      = recall_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
cm          = confusion_matrix(y_test, y_pred)

print(f"\n{'='*50}")
print(f"ROC-AUC Score       : {roc_auc:.4f}")
print(f"Avg Precision Score : {avg_prec:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Feature importances
rf = model.named_steps['clf']
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 Features:")
print(feat_imp.head(10))

# ── SAVE ──────────────────────────────────────────────────────────────────────
OUT = os.path.dirname(__file__)
with open(os.path.join(OUT, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save metadata for dashboard
meta = {
    'features': FEATURES,
    'roc_auc': round(roc_auc, 4),
    'avg_precision': round(avg_prec, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1': round(f1, 4),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'fraud_rate': round(df['Class'].mean() * 100, 3),
    'top_features': feat_imp.head(10).index.tolist(),
    'top_importances': [round(v, 4) for v in feat_imp.head(10).values.tolist()],
    'confusion_matrix': cm.tolist()
}
with open(os.path.join(OUT, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nModel saved to {OUT}/model.pkl")
print(f"Metadata saved to {OUT}/meta.pkl")
