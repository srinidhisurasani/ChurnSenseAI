import pandas as pd
import joblib

from core.explainer import explain_customer


# Load dataset
df = pd.read_csv(
    "data/uploads/banking_dataset.csv"
)

# Load trained model
model = joblib.load(
    "models/banking_model.pkl"
)

# Select one customer
customer = df.iloc[[0]]

# Generate explanation
explanation = explain_customer(
    model,
    customer,
    "churn"
)

print("\n--------------------------------")
print("CUSTOMER EXPLANATION")
print("--------------------------------")

print(
    explanation[
        [
            "feature",
            "shap_value"
        ]
    ].head(10).to_string(index=False)
)