import pandas as pd
import joblib

from core.ml_engine import train_model


DATASET_PATH = "data/uploads/telecom_dataset.csv"
MODEL_PATH = "models/telecom_model.pkl"


# Load dataset
df = pd.read_csv(DATASET_PATH)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nTelecom Dataset Loaded")
print("----------------------")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())


# Handle common Telecom dataset column names
column_mapping = {
    "monthlycharges": "monthly_charges",
    "monthly_charge": "monthly_charges",
    "monthlycharges_": "monthly_charges",
    "churn": "churn"
}

df = df.rename(columns=column_mapping)


# Make sure required columns exist
required_columns = [
    "tenure",
    "monthly_charges",
    "churn"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    print("\nERROR")
    print("Missing required columns:", missing)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise ValueError("Telecom dataset does not have the required columns.")


# Train model
print("\nTraining Telecom model...")
print("--------------------------")

result = train_model(
    df,
    "churn"
)


# Display accuracy
accuracy = result["accuracy"] * 100

print("\nTelecom Model Training Completed")
print("--------------------------------")
print(f"Accuracy: {accuracy:.2f}%")


# Save model
joblib.dump(
    result["model"],
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)