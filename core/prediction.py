import pandas as pd


def predict_customers(model, df, target_column):

    # Keep original data
    data = df.copy()

    # Remove target column
    if target_column in data.columns:
        X = data.drop(columns=[target_column])
    else:
        X = data.copy()

    # Remove target-related columns that should not be used as features
    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # Probability of class 1
    churn_probability = probabilities[:, 1] * 100

    # Add predictions
    data["prediction"] = predictions

    data["churn_probability"] = churn_probability.round(2)

    # Risk level
    data["risk"] = data["churn_probability"].apply(
        calculate_risk
    )

    return data


def calculate_risk(probability):

    if probability >= 70:
        return "HIGH"

    elif probability >= 40:
        return "MEDIUM"

    else:
        return "LOW"