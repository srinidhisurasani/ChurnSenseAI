import pandas as pd
import joblib

from core.ml_engine import train_model


# Banking dataset
file_path = "data/uploads/telecom_dataset.csv"

# Read dataset
df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(list(df.columns))


# Train model
result = train_model(
    df,
    "churn"
)


# Save trained model
model_path = "models/banking_model.pkl"

joblib.dump(
    result["model"],
    model_path
)


# Display results
print("\n--------------------------------")
print("MODEL RESULTS")
print("--------------------------------")

print(
    "Accuracy:",
    round(result["accuracy"] * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    pd.DataFrame(
        result["report"]
    ).transpose()
)

print("\nModel saved successfully!")
print("Location:", model_path)