import pandas as pd
import shap


def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def explain_customer(model, customer_data, target_column="churn"):

    customer = customer_data.copy()

    # Remove target column
    if target_column in customer.columns:
        X = customer.drop(columns=[target_column]).copy()
    else:
        X = customer.copy()

    # Remove common ID columns
    id_columns = []

    for column in X.columns:

        normalized = normalize_name(column)

        if normalized in [
            "customerid",
            "customer_id",
            "employeeid",
            "employee_id",
            "userid",
            "user_id",
            "accountid",
            "account_id"
        ]:
            id_columns.append(column)

    if id_columns:
        X = X.drop(columns=id_columns)

    # Telecom column correction
    if "monthlycharges" in X.columns:
        X = X.rename(
            columns={
                "monthlycharges": "monthly_charges"
            }
        )

    # Convert known numeric columns
    numeric_columns = [

        # Telecom
        "tenure",
        "monthly_charges",
        "monthlycharges",
        "totalcharges",
        "total_charges",
        "seniorcitizen",

        # Banking
        "credit_score",
        "age",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary",

        # Employee
        "employee_count",
        "age",
        "daily_rate",
        "distance_from_home",
        "education",
        "employee_number",
        "environment_satisfaction",
        "hourly_rate",
        "job_involvement",
        "job_level",
        "job_satisfaction",
        "monthly_income",
        "monthly_rate",
        "num_companies_worked",
        "percent_salary_hike",
        "performance_rating",
        "relationship_satisfaction",
        "standard_hours",
        "stock_option_level",
        "total_working_years",
        "training_times_last_year",
        "work_life_balance",
        "years_at_company",
        "years_in_current_role",
        "years_since_last_promotion",
        "years_with_curr_manager",

        # E-Commerce
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
        "cashbackamount",

        # OTT
        "netflix",
        "prime_video",
        "disney_hotstar",
        "jiocinema",
        "sonyliv",
        "zee5",
        "num_ott_platforms",
        "watch_hours",
        "avg_watch_hours_per_day",
        "last_login_days",
        "monthly_spend",
        "satisfaction_score",
        "support_tickets",
        "auto_renewal",
        "num_profiles"
    ]

    for column in numeric_columns:

        if column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # --------------------------------------------------
    # GET MODEL COMPONENTS
    # --------------------------------------------------

    if "preprocessor" not in model.named_steps:
        raise ValueError(
            "The loaded model does not contain a preprocessor."
        )

    preprocessor = model.named_steps["preprocessor"]

    if "classifier" in model.named_steps:
       classifier = model.named_steps["classifier"]

    elif "model" in model.named_steps:
     classifier = model.named_steps["model"]

    else:
      raise ValueError(
        "The loaded model does not contain a classifier step."
    )

    # --------------------------------------------------
    # TRANSFORM CUSTOMER
    # --------------------------------------------------

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # --------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------
    # SHAP EXPLAINER
    # --------------------------------------------------

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(
        X_transformed,
        check_additivity=False
    )

    # --------------------------------------------------
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # --------------------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            values = shap_values[1][0]
        else:
            values = shap_values[0][0]

    else:

        if len(shap_values.shape) == 3:

            # Example:
            # (samples, features, classes)

            if shap_values.shape[2] > 1:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0, :, 0]

        elif len(shap_values.shape) == 2:

            values = shap_values[0]

        else:

            values = shap_values

    # --------------------------------------------------
    # CREATE SHAP DATAFRAME
    # --------------------------------------------------

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values
    })

    results = []

    # --------------------------------------------------
    # PROCESS FEATURES
    # --------------------------------------------------

    for _, row in explanation.iterrows():

        raw_feature = str(row["feature"])

        shap_value = float(
            row["shap_value"]
        )

        clean_name, original_column, category = (
            parse_feature_name(raw_feature)
        )

        if original_column is None:
            continue

        # Find matching original dataframe column
        matching_column = None

        normalized_original = normalize_name(
            original_column
        )

        for column in X.columns:

            normalized_column = normalize_name(
                column
            )

            if normalized_column == normalized_original:
                matching_column = column
                break

        if matching_column is None:
            continue

        # --------------------------------------------------
        # CATEGORICAL FEATURE
        # --------------------------------------------------

        if category is not None:

            customer_value = X.iloc[0][
                matching_column
            ]

            if pd.isna(customer_value):
                continue

            customer_value = str(
                customer_value
            ).strip()

            if (
                customer_value.lower()
                != category.lower()
            ):
                continue

            display_value = customer_value

        # --------------------------------------------------
        # NUMERIC FEATURE
        # --------------------------------------------------

        else:

            customer_value = X.iloc[0][
                matching_column
            ]

            display_value = format_value(
                customer_value
            )

        results.append({

            "feature": clean_name,

            "value": display_value,

            "shap_value": shap_value,

            "importance": abs(shap_value)

        })

    # --------------------------------------------------
    # REMOVE DUPLICATE BUSINESS FEATURES
    # --------------------------------------------------

    final_results = {}

    for item in results:

        key = item["feature"]

        if (
            key not in final_results
            or
            abs(item["shap_value"])
            >
            abs(
                final_results[key]["shap_value"]
            )
        ):

            final_results[key] = item

    explanation = pd.DataFrame(
        list(final_results.values())
    )

    if explanation.empty:
        return explanation

    # Sort by importance
    explanation = explanation.sort_values(
        by="importance",
        ascending=False
    )

    return explanation.reset_index(
        drop=True
    )


# ==========================================================
# FEATURE NAME PARSER
# ==========================================================

def parse_feature_name(feature_name):

    feature_name = str(feature_name)

    # Remove sklearn ColumnTransformer prefixes
    feature_name = (
        feature_name
        .replace("numerical__", "")
        .replace("categorical__", "")
    )

    # ------------------------------------------------------
    # DISPLAY NAME MAP
    # ------------------------------------------------------

    feature_map = {

        # -------------------------------
        # Telecom
        # -------------------------------

        "tenure": "Tenure",

        "monthly_charges":
            "Monthly Charges",

        "monthlycharges":
            "Monthly Charges",

        "totalcharges":
            "Total Charges",

        "total_charges":
            "Total Charges",

        "seniorcitizen":
            "Senior Citizen",

        "contract":
            "Contract",

        "internetservice":
            "Internet Service",

        "phoneservice":
            "Phone Service",

        "multiplelines":
            "Multiple Lines",

        "onlinesecurity":
            "Online Security",

        "onlinebackup":
            "Online Backup",

        "deviceprotection":
            "Device Protection",

        "techsupport":
            "Technical Support",

        "streamingtv":
            "Streaming TV",

        "streamingmovies":
            "Streaming Movies",

        "paperlessbilling":
            "Paperless Billing",

        "paymentmethod":
            "Payment Method",

        "partner":
            "Partner",

        "dependents":
            "Dependents",

        "gender":
            "Gender",

        # -------------------------------
        # Banking
        # -------------------------------

        "credit_score":
            "Credit Score",

        "balance":
            "Balance",

        "age":
            "Age",

        "products_number":
            "Number of Products",

        "credit_card":
            "Credit Card",

        "active_member":
            "Active Member",

        "estimated_salary":
            "Estimated Salary",

        "country":
            "Country",

        # -------------------------------
        # Employee
        # -------------------------------

        "employeesatisfaction":
            "Employee Satisfaction",

        "jobsatisfaction":
            "Job Satisfaction",

        "monthlyincome":
            "Monthly Income",

        "job_satisfaction":
            "Job Satisfaction",

        "monthly_income":
            "Monthly Income",

        "environment_satisfaction":
            "Environment Satisfaction",

        "work_life_balance":
            "Work Life Balance",

        "job_involvement":
            "Job Involvement",

        "job_level":
            "Job Level",

        "years_at_company":
            "Years At Company",

        "years_in_current_role":
            "Years In Current Role",

        "years_since_last_promotion":
            "Years Since Last Promotion",

        "years_with_curr_manager":
            "Years With Current Manager",

        "total_working_years":
            "Total Working Years",

        "num_companies_worked":
            "Number of Companies Worked",

        "distance_from_home":
            "Distance From Home",

        "percent_salary_hike":
            "Percent Salary Hike",

        "performance_rating":
            "Performance Rating",

        "relationship_satisfaction":
            "Relationship Satisfaction",

        "stock_option_level":
            "Stock Option Level",

        "training_times_last_year":
            "Training Times Last Year",

        # -------------------------------
        # E-Commerce
        # -------------------------------

        "citytier":
            "City Tier",

        "warehousetohome":
            "Warehouse To Home",

        "hourspendonapp":
            "Hours Spent On App",

        "numberofdeviceregistered":
            "Number Of Devices Registered",

        "satisfactionscore":
            "Satisfaction Score",

        "numberofaddress":
            "Number Of Addresses",

        "complain":
            "Complaint",

        "orderamounthikefromlastyear":
            "Order Amount Hike From Last Year",

        "couponused":
            "Coupons Used",

        "ordercount":
            "Order Count",

        "daysincelastorder":
            "Days Since Last Order",

        "cashbackamount":
            "Cashback Amount",

        # -------------------------------
        # OTT
        # -------------------------------

        "netflix":
            "Netflix",

        "prime_video":
            "Prime Video",

        "disney_hotstar":
            "Disney+ Hotstar",

        "jiocinema":
            "JioCinema",

        "sonyliv":
            "SonyLIV",

        "zee5":
            "ZEE5",

        "num_ott_platforms":
            "Number Of OTT Platforms",

        "watch_hours":
            "Watch Hours",

        "avg_watch_hours_per_day":
            "Average Watch Hours Per Day",

        "last_login_days":
            "Days Since Last Login",

        "monthly_spend":
            "Monthly Spend",

        "satisfaction_score":
            "Satisfaction Score",

        "support_tickets":
            "Support Tickets",

        "auto_renewal":
            "Auto Renewal",

        "num_profiles":
            "Number Of Profiles",

        "subscription_type":
            "Subscription Type",

        "payment_method":
            "Payment Method",

        "device":
            "Device",

        "region":
            "Region"
    }

    # ------------------------------------------------------
    # CHECK CATEGORICAL FEATURES
    # ------------------------------------------------------

    # Sort longest keys first so that names like
    # avg_watch_hours_per_day are matched correctly.

    sorted_columns = sorted(
        feature_map.keys(),
        key=len,
        reverse=True
    )

    for column in sorted_columns:

        prefix = column + "_"

        if feature_name.startswith(prefix):

            category = feature_name[
                len(prefix):
            ]

            return (
                feature_map[column],
                column,
                category
            )

    # ------------------------------------------------------
    # CHECK NUMERIC FEATURES
    # ------------------------------------------------------

    if feature_name in feature_map:

        return (
            feature_map[feature_name],
            feature_name,
            None
        )

    # ------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------

    clean_name = (
        feature_name
        .replace("_", " ")
        .title()
    )

    return (
        clean_name,
        feature_name,
        None
    )


# ==========================================================
# FORMAT VALUES
# ==========================================================

def format_value(value):

    if pd.isna(value):
        return "Not available"

    if isinstance(value, float):

        if value.is_integer():
            return str(
                int(value)
            )

        return f"{value:.2f}"

    return str(value)