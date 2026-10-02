import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


DATASET_PATH = "data/uploads/ott_dataset.csv"
MODEL_PATH = "models/ott_model.pkl"


print("Loading OTT dataset...")
df = pd.read_csv(DATASET_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))


target_column = "cancelled"

if target_column not in df.columns:
    print("ERROR: cancelled column not found.")
    exit()


print("\nLoading trained OTT model...")
model = joblib.load(MODEL_PATH)


X = df.drop(columns=[target_column])
y = df[target_column]


# Remove customer ID because it was removed during training
if "customer_id" in X.columns:
    X = X.drop(columns=["customer_id"])


print("\nGenerating predictions...")

y_pred = model.predict(X)
y_probability = model.predict_proba(X)[:, 1]


accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, y_probability)

cm = confusion_matrix(y, y_pred)


print("\n========================================")
print("OTT MODEL EVALUATION")
print("========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y,
    y_pred,
    target_names=["Active", "Cancelled"],
    zero_division=0
))

print("========================================")