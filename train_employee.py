import pandas as pd
import joblib

from core.ml_engine import train_model


DATASET_PATH = "data/uploads/employee_dataset.csv"
MODEL_PATH = "models/employee_model.pkl"


print("Loading employee dataset...")

df = pd.read_csv(DATASET_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# Check target column
if "attrition" not in df.columns:
    raise ValueError(
        "Attrition column not found in the dataset."
    )


# Convert numeric columns
numeric_columns = [
    "age",
    "job_satisfaction",
    "monthly_income",
    "job_level",
    "years_at_company",
    "years_in_current_role",
    "years_since_last_promotion",
    "work_life_balance",
    "distance_from_home",
    "environment_satisfaction",
    "performance_rating",
    "employee_number"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


print("\nTraining Employee Resignation model...")

result = train_model(
    df,
    "attrition"
)


print("\nModel Accuracy:")
print(f"{result['accuracy'] * 100:.2f}%")


print("\nClassification Report:")
print(result["report"])


# Save model
joblib.dump(
    result["model"],
    MODEL_PATH
)


print("\nEmployee model saved successfully:")
print(MODEL_PATH)