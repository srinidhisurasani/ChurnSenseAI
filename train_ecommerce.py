import pandas as pd
import joblib
from core.ml_engine import train_model

DATASET_PATH = "data/uploads/ecommerce_dataset.csv"
MODEL_PATH = "models/ecommerce_model.pkl"

print("Loading E-Commerce dataset...")

df = pd.read_csv(DATASET_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

# Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Convert numeric columns
numeric_columns = [
    "tenure",
    "citytier",
    "warehousetohome",
    "hourspendonapp",
    "numberofdeviceregistered",
    "satisfactionscore",
    "numberofaddress",
    "complain",
    "orderamounthikefromlastyear",
    "couponused",
    "ordercount",
    "daysincelastorder",
    "cashbackamount"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# Check target
if "churn" not in df.columns:
    raise ValueError(
        "Churn column not found in the E-Commerce dataset."
    )

print("\nTraining E-Commerce Churn model...")

result = train_model(
    df,
    "churn"
)

print("\nModel Accuracy:")
print(f"{result['accuracy'] * 100:.2f}%")

print("\nClassification Report:")
print(result["report"])

joblib.dump(
    result["model"],
    MODEL_PATH
)

print("\nE-Commerce model saved successfully:")
print(MODEL_PATH)