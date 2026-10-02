DOMAIN_RULES = {
    "telecom": {
        "target": "churn",
        "required_columns": [
            "tenure",
            "monthlycharges"
        ]
    },

    "banking": {
        "target": "churn",
        "required_columns": [
            "credit_score",
            "balance",
            "active_member"
        ]
    },

    "employee": {
        "target": "attrition",
        "required_columns": [
            "age",
            "attrition",
            "employeenumber",
            "jobsatisfaction",
            "monthlyincome"
        ]
    },

    "ecommerce": {
        "target": "churn",
        "required_columns": [
            "tenure",
            "ordercount",
            "daysincelastorder",
            "satisfactionscore"
        ]
    },

    "ott": {
        "target": "cancelled",
        "required_columns": [
            "watch_hours",
            "subscription_type"
        ]
    }
}


def normalize_column(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("*", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def validate_dataset(domain, columns):

    print("\n==============================")
    print("VALIDATOR DOMAIN:", domain)
    print("VALIDATOR COLUMNS:", list(columns))
    print("EMPLOYEE RULE:", DOMAIN_RULES.get("employee"))
    print("==============================\n")

    if domain not in DOMAIN_RULES:
        return False, "Invalid domain"

    rules = DOMAIN_RULES[domain]

    uploaded_columns = {
        normalize_column(column)
        for column in columns
    }

    missing_columns = []

    for required_column in rules["required_columns"]:

        required_normalized = normalize_column(
            required_column
        )

        if required_normalized not in uploaded_columns:
            missing_columns.append(required_column)
            
    print("NORMALIZED UPLOADED COLUMNS:", uploaded_columns)
    print("MISSING COLUMNS:", missing_columns)

    if missing_columns:

        return False, (
            "Dataset does not match the selected application. "
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    return True, "Dataset is compatible"