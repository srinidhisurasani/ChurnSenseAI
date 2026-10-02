import pandas as pd
import joblib

from core.prediction import predict_customers


# Dataset
file_path = "data/uploads/telecom_dataset.csv"

df = pd.read_csv(file_path)


# Load trained model
model = joblib.load(
    "models/banking_model.pkl"
)


# Generate predictions
results = predict_customers(
    model,
    df,
    "churn"
)


print("\n--------------------------------")
print("CUSTOMER CHURN PREDICTIONS")
print("--------------------------------")

print(
    results[
        [
            "customer_id",
            "churn_probability",
            "risk"
        ]
    ].head(20).to_string(index=False)
)


print("\n--------------------------------")
print("RISK SUMMARY")
print("--------------------------------")

print(
    results["risk"].value_counts()
)