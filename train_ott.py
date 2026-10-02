import os
import pandas as pd
import joblib

from core.ml_engine import train_model


DATASET_PATH = "data/uploads/ott_dataset.csv"
MODEL_PATH = "models/ott_model.pkl"


if not os.path.exists(DATASET_PATH):
    print("OTT dataset not found.")
    print("Expected location:")
    print(DATASET_PATH)
    exit()


df = pd.read_csv(DATASET_PATH)

print("OTT Dataset Loaded")
print("-------------------------")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("Columns:")
print(df.columns.tolist())
print()


target_column = "cancelled"

if target_column not in df.columns:
    print("ERROR: 'cancelled' column not found.")
    exit()


print("Training OTT Churn Model...")
print("-------------------------")

model, accuracy, report, X_test, y_test, predictions = train_model(
    df,
    target_column
)


os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)


print()
print("-------------------------")
print("OTT MODEL TRAINING COMPLETE")
print("-------------------------")

print("Accuracy:", accuracy)

print()
print("Classification Report:")
print(report)

print()
print("Model saved to:")
print(MODEL_PATH)