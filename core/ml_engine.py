import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def train_model(df, target_column):

    df = clean_column_names(df)

    target_column = target_column.lower()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    # Remove rows where target is missing
    df = df.dropna(subset=[target_column]).copy()

    # Convert target values
    target_mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "churned": 1,
        "not_churned": 0,
        "cancelled": 1,
        "active": 0,
        "attrition": 1,
        "no_attrition": 0
    }

    if df[target_column].dtype == "object":

        df[target_column] = (
            df[target_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

    else:
        df[target_column] = pd.to_numeric(
            df[target_column],
            errors="coerce"
        )

    df = df.dropna(subset=[target_column]).copy()

    df[target_column] = df[target_column].astype(int)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remove common ID columns
    id_columns = [
        "customer_id",
        "customerid",
        "employee_id",
        "employeeid",
        "employeenumber",
        "id"
    ]

    columns_to_drop = [
        column for column in id_columns
        if column in X.columns
    ]

    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

    # Convert common numeric columns
    numeric_candidates = [
        "age",
        "tenure",
        "balance",
        "credit_score",
        "monthly_charges",
        "monthlycharges",
        "monthly_fee",
        "watch_hours",
        "avg_watch_hours_per_day",
        "last_login_days",
        "monthly_spend",
        "satisfaction_score",
        "support_tickets",
        "num_profiles",
        "num_ott_platforms",
        "netflix",
        "prime_video",
        "disney_hotstar",
        "jiocinema",
        "sonyliv",
        "zee5",
        "auto_renewal",
        "active_member",
        "products_number",
        "credit_card",
        "estimated_salary",
        "order_count",
        "ordercount",
        "days_since_last_order",
        "daysincelastorder",
        "complain",
        "cashback_amount",
        "cashbackamount",
        "warehouse_to_home",
        "warehousetohome",
        "hour_spend_on_app",
        "hourspendonapp"
    ]

    for column in numeric_candidates:

        if column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # Random Forest model
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Complete ML pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return (
        model,
        accuracy,
        report,
        X_test,
        y_test,
        predictions
    )