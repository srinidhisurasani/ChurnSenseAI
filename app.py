from flask import Flask, render_template, request
import os
import pandas as pd
import joblib

from core.dataset_validator import validate_dataset
from core.prediction import predict_customers
from core.explainer import explain_customer
from core.recommender import generate_recommendations


app = Flask(__name__)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = "data/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# DOMAINS
# ============================================================

DOMAINS = {

    "telecom": "Telecom Customer Churn",

    "banking": "Banking Customer Churn",

    "employee": "Software Employee Resignation",

    "ecommerce": "E-Commerce Customer Churn",

    "ott": "OTT / Subscription Cancellation"
}


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

DOMAIN_CONFIG = {

    "telecom": {
        "dataset": "data/uploads/telecom_dataset.csv",
        "model": "models/telecom_model.pkl",
        "target": "churn",
        "id_column": "customerid"
    },

    "banking": {
        "dataset": "data/uploads/banking_dataset.csv",
        "model": "models/banking_model.pkl",
        "target": "churn",
        "id_column": "customer_id"
    },

    "employee": {
        "dataset": "data/uploads/employee_dataset.csv",
        "model": "models/employee_model.pkl",
        "target": "attrition",
        "id_column": "employeenumber"
    },

    "ecommerce": {
        "dataset": "data/uploads/ecommerce_dataset.csv",
        "model": "models/ecommerce_model.pkl",
        "target": "churn",
        "id_column": "customerid"
    },

    "ott": {
    "dataset": "data/uploads/ott_dataset.csv",
    "model": "models/ott_model.pkl",
    "target": "cancelled",
    "id_column": "customer_id"
}
}


# ============================================================
# NUMERIC COLUMN DEFINITIONS
# ============================================================

NUMERIC_COLUMNS = {

    "telecom": [
        "tenure",
        "monthlycharges",
        "monthly_charges",
        "totalcharges",
        "total_charges",
        "seniorcitizen"
    ],

    "banking": [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary"
    ],

    "employee": [
        "age",
        "dailyrate",
        "daily_rate",
        "distancefromhome",
        "distance_from_home",
        "employeenumber",
        "employee_number",
        "environmentsatisfaction",
        "environment_satisfaction",
        "hourlyrate",
        "hourly_rate",
        "jobinvolvement",
        "job_involvement",
        "joblevel",
        "job_level",
        "jobsatisfaction",
        "job_satisfaction",
        "monthlyincome",
        "monthly_income",
        "monthlyrate",
        "monthly_rate",
        "numcompaniesworked",
        "num_companies_worked",
        "percentsalaryhike",
        "percent_salary_hike",
        "performancerating",
        "performance_rating",
        "relationshipsatisfaction",
        "relationship_satisfaction",
        "stockoptionlevel",
        "stock_option_level",
        "totalworkingyears",
        "total_working_years",
        "trainingtimeslastyear",
        "training_times_last_year",
        "worklifebalance",
        "work_life_balance",
        "yearsatcompany",
        "years_at_company",
        "yearsincurrentrole",
        "years_in_current_role",
        "yearssincelastpromotion",
        "years_since_last_promotion",
        "yearswithcurrmanager",
        "years_with_curr_manager"
    ],

    "ecommerce": [
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
    ],

    "ott": [
        "watch_hours"
    ]
}


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def clean_dataframe(df, domain):

    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("*", "", regex=False)
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Clean string values
    for column in df.select_dtypes(
        include=["object"]
    ).columns:

        df[column] = df[column].map(
            lambda value:
            value.replace("*", "").strip()
            if isinstance(value, str)
            else value
        )

    # Convert numeric columns
    for column in NUMERIC_COLUMNS.get(
        domain,
        []
    ):

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_model_data(
    df,
    domain,
    target
):

    model_df = df.copy()

    # Remove target
    if target in model_df.columns:

        model_df = model_df.drop(
            columns=[target]
        )

    # Get ID column
    config = DOMAIN_CONFIG.get(
        domain,
        {}
    )

    id_column = config.get(
        "id_column"
    )

    # Remove ID for all domains except employee
    if domain != "employee":

        if (
            id_column
            and
            id_column in model_df.columns
        ):

            model_df = model_df.drop(
                columns=[id_column]
            )

    # Telecom column correction
    if domain == "telecom":

        if (
            "monthlycharges" in model_df.columns
            and
            "monthly_charges" not in model_df.columns
        ):

            model_df = model_df.rename(
                columns={
                    "monthlycharges":
                    "monthly_charges"
                }
            )

    return model_df


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(probability):

    probability = float(probability)

    if probability >= 70:

        return "HIGH"

    elif probability >= 40:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# SAFE FORM VALUE
# ============================================================

def get_form_value(
    form,
    name,
    default=None
):

    value = form.get(
        name,
        ""
    )

    if value is None:

        return default

    value = value.strip()

    if value == "":

        return default

    return value


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html",
        domains=DOMAINS
    )


# ============================================================
# DOMAIN PAGE
# ============================================================

@app.route("/domain/<domain>")
def domain_page(domain):

    if domain not in DOMAINS:

        return "Invalid domain", 404

    return render_template(
        "domain.html",
        domain=domain,
        domain_name=DOMAINS[domain]
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload/<domain>",
    methods=["GET", "POST"]
)
def upload_dataset(domain):

    if domain not in DOMAINS:

        return "Invalid domain", 404

    if request.method == "GET":

        return render_template(
            "upload.html",
            domain=domain
        )

    file = request.files.get(
        "file"
    )

    if file is None:

        return render_template(
            "upload.html",
            domain=domain,
            error="Please select a CSV file."
        )

    if file.filename == "":

        return render_template(
            "upload.html",
            domain=domain,
            error="Please select a CSV file."
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        return render_template(
            "upload.html",
            domain=domain,
            error="Only CSV files are supported."
        )

    try:

        df = pd.read_csv(file)

        df = clean_dataframe(
            df,
            domain
        )

        valid, message = validate_dataset(
            domain,
            df.columns
        )

        if not valid:

            return render_template(
                "upload.html",
                domain=domain,
                error=message
            )

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{domain}_dataset.csv"
        )

        df.to_csv(
            file_path,
            index=False
        )

        config = DOMAIN_CONFIG.get(
            domain
        )

        if config is None:

            return render_template(
                "domain.html",
                domain=domain,
                domain_name=DOMAINS[domain]
            )

        model_path = config["model"]

        target = config["target"]

        if not os.path.exists(
            model_path
        ):

            return render_template(
                "upload.html",
                domain=domain,
                error=(
                    "Dataset uploaded successfully, "
                    "but the trained model was not found."
                )
            )

        model = joblib.load(
            model_path
        )

        model_df = prepare_model_data(
            df,
            domain,
            target
        )

        predictions = predict_customers(
            model,
            model_df,
            target
        )

        predictions = predictions.sort_values(
            by="churn_probability",
            ascending=False
        )

        id_column = config.get(
            "id_column"
        )

        if (
            id_column
            and
            id_column in df.columns
        ):

            predictions["customer_id"] = (
                df.loc[
                    predictions.index,
                    id_column
                ].values
            )

        else:

            predictions["customer_id"] = (
                predictions.index
            )

        customers = predictions.to_dict(
            orient="records"
        )

        total_customers = len(
            customers
        )

        high_risk = len(
            predictions[
                predictions["risk"] == "HIGH"
            ]
        )

        medium_risk = len(
            predictions[
                predictions["risk"] == "MEDIUM"
            ]
        )

        low_risk = len(
            predictions[
                predictions["risk"] == "LOW"
            ]
        )

        return render_template(
            "predictions.html",

            domain=domain,

            predictions=customers,

            total_customers=total_customers,

            high_risk=high_risk,

            medium_risk=medium_risk,

            low_risk=low_risk
        )

    except Exception as e:

        return render_template(
            "upload.html",
            domain=domain,
            error=str(e)
        )


# ============================================================
# PREDICT SAVED DATASET
# ============================================================

@app.route(
    "/predict/<domain>"
)
def predict(domain):

    if domain not in DOMAIN_CONFIG:

        return (
            "This domain is not configured yet.",
            404
        )

    config = DOMAIN_CONFIG[
        domain
    ]

    dataset_path = config[
        "dataset"
    ]

    model_path = config[
        "model"
    ]

    target = config[
        "target"
    ]

    if not os.path.exists(
        dataset_path
    ):

        return (
            f"""
            <h2>Dataset Not Found</h2>
            <p>{dataset_path}</p>
            """,
            404
        )

    if not os.path.exists(
        model_path
    ):

        return (
            f"""
            <h2>Model Not Found</h2>
            <p>{model_path}</p>
            """,
            404
        )

    try:

        df = pd.read_csv(
            dataset_path
        )

        df = clean_dataframe(
            df,
            domain
        )

        model = joblib.load(
            model_path
        )

        model_df = prepare_model_data(
            df,
            domain,
            target
        )

        predictions = predict_customers(
            model,
            model_df,
            target
        )

        predictions = predictions.sort_values(
            by="churn_probability",
            ascending=False
        )

        id_column = config.get(
            "id_column"
        )

        if (
            id_column
            and
            id_column in df.columns
        ):

            predictions["customer_id"] = (
                df.loc[
                    predictions.index,
                    id_column
                ].values
            )

        else:

            predictions["customer_id"] = (
                predictions.index
            )

        customers = predictions.to_dict(
            orient="records"
        )

        total_customers = len(
            customers
        )

        high_risk = len(
            predictions[
                predictions["risk"] == "HIGH"
            ]
        )

        medium_risk = len(
            predictions[
                predictions["risk"] == "MEDIUM"
            ]
        )

        low_risk = len(
            predictions[
                predictions["risk"] == "LOW"
            ]
        )

        return render_template(
            "predictions.html",

            domain=domain,

            predictions=customers,

            total_customers=total_customers,

            high_risk=high_risk,

            medium_risk=medium_risk,

            low_risk=low_risk
        )

    except Exception as e:

        return (
            f"""
            <h2>Prediction Error</h2>
            <pre>{str(e)}</pre>
            """,
            500
        )


# ============================================================
# EXPLANATION
# ============================================================

@app.route(
    "/explain/<domain>/<customer_id>"
)
def explain(
    domain,
    customer_id
):

    if domain not in DOMAIN_CONFIG:

        return (
            "This domain is not configured yet.",
            404
        )

    config = DOMAIN_CONFIG[
        domain
    ]

    dataset_path = config[
        "dataset"
    ]

    model_path = config[
        "model"
    ]

    target = config[
        "target"
    ]

    id_column = config[
        "id_column"
    ]

    if not os.path.exists(
        dataset_path
    ):

        return (
            "Dataset not found.",
            404
        )

    if not os.path.exists(
        model_path
    ):

        return (
            "Model not found.",
            404
        )

    try:

        df = pd.read_csv(
            dataset_path
        )

        df = clean_dataframe(
            df,
            domain
        )

        if id_column not in df.columns:

            return (
                f"ID column '{id_column}' not found.",
                404
            )

        model = joblib.load(
            model_path
        )

        customer_rows = df.loc[
            df[id_column]
            .astype(str)
            .str.strip()
            .eq(str(customer_id).strip())
        ]

        if customer_rows.empty:

            return (
                f"Customer {customer_id} not found.",
                404
            )

        customer = (
            customer_rows.iloc[0].copy()
        )

        model_customer = (
            customer.to_frame().T
        )

        model_customer = prepare_model_data(
            model_customer,
            domain,
            target
        )

        prediction_result = predict_customers(
            model,
            model_customer,
            target
        )

        churn_probability = float(
            prediction_result.iloc[0][
                "churn_probability"
            ]
        )

        risk = str(
            prediction_result.iloc[0][
                "risk"
            ]
        )

        explanation = explain_customer(
            model,
            customer.to_frame().T,
            target
        )

        recommendations = (
            generate_recommendations(
                customer,
                explanation,
                domain
            )
        )

        return render_template(
            "explanation.html",

            domain=domain,

            customer_id=customer_id,

            churn_probability=churn_probability,

            risk=risk,

            explanation=(
                explanation.to_dict(
                    orient="records"
                )
            ),

            recommendations=recommendations
        )

    except Exception as e:

        return (
            f"""
            <h2>Explanation Error</h2>
            <pre>{str(e)}</pre>
            """,
            500
        )


# ============================================================
# TELECOM SIMULATION
# ============================================================

@app.route(
    "/simulate/telecom/<customer_id>",
    methods=["GET", "POST"]
)
def simulate_telecom(customer_id):

    try:

        model_path = (
            "models/telecom_model.pkl"
        )

        dataset_path = (
            "data/uploads/telecom_dataset.csv"
        )

        if not os.path.exists(
            model_path
        ):

            return (
                "Telecom model not found.",
                404
            )

        if not os.path.exists(
            dataset_path
        ):

            return (
                "Telecom dataset not found.",
                404
            )

        model = joblib.load(
            model_path
        )

        df = pd.read_csv(
            dataset_path
        )

        df = clean_dataframe(
            df,
            "telecom"
        )

        id_column = "customerid"

        if id_column not in df.columns:

            return (
                "Customer ID column not found.",
                404
            )

        rows = df.loc[
            df[id_column]
            .astype(str)
            .str.strip()
            .eq(str(customer_id).strip())
        ]

        if rows.empty:

            return (
                f"Customer {customer_id} not found.",
                404
            )

        customer = rows.iloc[0].copy()

        model_customer = prepare_model_data(
            customer.to_frame().T,
            "telecom",
            "churn"
        )

        original_probability = round(
            float(
                model.predict_proba(
                    model_customer
                )[0, 1]
            ) * 100,
            2
        )

        original_risk = calculate_risk(
            original_probability
        )

        simulation_result = None

        if request.method == "POST":

            simulated_customer = (
                model_customer.copy()
            )

            # Contract
            value = get_form_value(
                request.form,
                "contract"
            )

            if (
                value is not None
                and
                "contract"
                in simulated_customer.columns
            ):

                simulated_customer.loc[
                    simulated_customer.index[0],
                    "contract"
                ] = value

            # Tenure
            value = get_form_value(
                request.form,
                "tenure"
            )

            if (
                value is not None
                and
                "tenure"
                in simulated_customer.columns
            ):

                simulated_customer.loc[
                    simulated_customer.index[0],
                    "tenure"
                ] = float(value)

            # Online Security
            value = get_form_value(
                request.form,
                "onlinesecurity"
            )

            if (
                value is not None
                and
                "onlinesecurity"
                in simulated_customer.columns
            ):

                simulated_customer.loc[
                    simulated_customer.index[0],
                    "onlinesecurity"
                ] = value

            # Technical Support
            value = get_form_value(
                request.form,
                "techsupport"
            )

            if (
                value is not None
                and
                "techsupport"
                in simulated_customer.columns
            ):

                simulated_customer.loc[
                    simulated_customer.index[0],
                    "techsupport"
                ] = value

            # Payment Method
            value = get_form_value(
                request.form,
                "paymentmethod"
            )

            if (
                value is not None
                and
                "paymentmethod"
                in simulated_customer.columns
            ):

                simulated_customer.loc[
                    simulated_customer.index[0],
                    "paymentmethod"
                ] = value

            # Monthly Charges
            value = get_form_value(
                request.form,
                "monthly_charges"
            )

            if (
                value is not None
                and
                "monthly_charges"
                in simulated_customer.columns
            ):

                simulated_customer.loc[
                    simulated_customer.index[0],
                    "monthly_charges"
                ] = float(value)

            simulated_probability = round(
                float(
                    model.predict_proba(
                        simulated_customer
                    )[0, 1]
                ) * 100,
                2
            )

            change = round(
                simulated_probability
                - original_probability,
                2
            )

            simulation_result = {

                "original_probability":
                    original_probability,

                "simulated_probability":
                    simulated_probability,

                "change":
                    change,

                "simulated_risk":
                    calculate_risk(
                        simulated_probability
                    )
            }

        return render_template(
            "simulation_telecom.html",

            customer_id=customer_id,

            customer=customer,

            original_probability=
                original_probability,

            original_risk=
                original_risk,

            simulation_result=
                simulation_result
        )

    except Exception as e:

        return (
            f"""
            <h2>Telecom Simulation Error</h2>
            <p>{str(e)}</p>
            """,
            500
        )


# ============================================================
# BANKING SIMULATION
# ============================================================

@app.route(
    "/simulate/banking/<customer_id>",
    methods=["GET", "POST"]
)
def simulate_banking(customer_id):

    try:

        model_path = (
            "models/banking_model.pkl"
        )

        dataset_path = (
            "data/uploads/banking_dataset.csv"
        )

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        if not os.path.exists(
            dataset_path
        ):

            return (
                "Banking dataset not found.",
                404
            )

        if not os.path.exists(
            model_path
        ):

            return (
                "Banking model not found.",
                404
            )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = joblib.load(
            model_path
        )

        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        df = pd.read_csv(
            dataset_path
        )

        df = clean_dataframe(
            df,
            "banking"
        )

        # ----------------------------------------------------
        # CUSTOMER ID
        # ----------------------------------------------------

        id_column = "customer_id"

        if id_column not in df.columns:

            return (
                "Customer ID column not found.",
                404
            )

        # ----------------------------------------------------
        # FIND CUSTOMER
        # ----------------------------------------------------

        customer_rows = df.loc[
            df[id_column]
            .astype(str)
            .str.strip()
            .eq(str(customer_id).strip())
        ]

        if customer_rows.empty:

            return (
                f"Customer {customer_id} not found.",
                404
            )

        customer = (
            customer_rows.iloc[0].copy()
        )

        # ----------------------------------------------------
        # PREPARE MODEL DATA
        # ----------------------------------------------------

        model_customer = prepare_model_data(
            customer.to_frame().T,
            "banking",
            "churn"
        )

        # ----------------------------------------------------
        # ORIGINAL PROBABILITY
        # ----------------------------------------------------

        original_probability = round(
            float(
                model.predict_proba(
                    model_customer
                )[0, 1]
            ) * 100,
            2
        )

        original_risk = calculate_risk(
            original_probability
        )

        result = None

        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

        if request.method == "POST":

            simulated_customer = (
                model_customer.copy()
            )

            # -----------------------------------------------
            # ACTIVE MEMBER
            # -----------------------------------------------

            value = get_form_value(
                request.form,
                "active_member"
            )

            if value is not None:

                try:

                    numeric_value = float(
                        value
                    )

                    if (
                        "active_member"
                        in simulated_customer.columns
                    ):

                        simulated_customer.loc[
                            simulated_customer.index[0],
                            "active_member"
                        ] = numeric_value

                except (
                    ValueError,
                    TypeError
                ):

                    pass

            # -----------------------------------------------
            # NUMBER OF PRODUCTS
            # -----------------------------------------------

            value = get_form_value(
                request.form,
                "products_number"
            )

            if value is not None:

                try:

                    numeric_value = float(
                        value
                    )

                    if (
                        "products_number"
                        in simulated_customer.columns
                    ):

                        simulated_customer.loc[
                            simulated_customer.index[0],
                            "products_number"
                        ] = numeric_value

                except (
                    ValueError,
                    TypeError
                ):

                    pass

            # -----------------------------------------------
            # TENURE
            # -----------------------------------------------

            value = get_form_value(
                request.form,
                "tenure"
            )

            if value is not None:

                try:

                    numeric_value = float(
                        value
                    )

                    if (
                        "tenure"
                        in simulated_customer.columns
                    ):

                        simulated_customer.loc[
                            simulated_customer.index[0],
                            "tenure"
                        ] = numeric_value

                except (
                    ValueError,
                    TypeError
                ):

                    pass

            # -----------------------------------------------
            # BALANCE
            # -----------------------------------------------

            value = get_form_value(
                request.form,
                "balance"
            )

            if value is not None:

                try:

                    numeric_value = float(
                        value
                    )

                    if (
                        "balance"
                        in simulated_customer.columns
                    ):

                        simulated_customer.loc[
                            simulated_customer.index[0],
                            "balance"
                        ] = numeric_value

                except (
                    ValueError,
                    TypeError
                ):

                    pass

            # -----------------------------------------------
            # SIMULATED PROBABILITY
            # -----------------------------------------------

            simulated_probability = round(
                float(
                    model.predict_proba(
                        simulated_customer
                    )[0, 1]
                ) * 100,
                2
            )

            # -----------------------------------------------
            # CHANGE
            # -----------------------------------------------

            change = round(
                simulated_probability
                - original_probability,
                2
            )

            # -----------------------------------------------
            # RESULT
            # -----------------------------------------------

            result = {

                "original_probability":
                    original_probability,

                "simulated_probability":
                    simulated_probability,

                "change":
                    change,

                "original_risk":
                    original_risk,

                "simulated_risk":
                    calculate_risk(
                        simulated_probability
                    )
            }

        # ----------------------------------------------------
        # CUSTOMER DICTIONARY
        # ----------------------------------------------------

        selected_customer = (
            customer.to_dict()
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        return render_template(

            "simulation.html",

            selected_customer=
                selected_customer,

            customer=
                selected_customer,

            customer_id=
                str(customer_id),

            original_probability=
                original_probability,

            original_risk=
                original_risk,

            result=
                result
        )

    except Exception as e:

        return (
            f"""
            <h2>Banking Simulation Error</h2>

            <p>{str(e)}</p>

            <p>
                Customer ID:
                <strong>{customer_id}</strong>
            </p>
            """,
            500
        )


# ============================================================
# EMPLOYEE SIMULATION
# ============================================================

@app.route(
    "/simulate/employee/<employee_id>",
    methods=["GET", "POST"]
)
def simulate_employee(employee_id):

    try:

        model_path = "models/employee_model.pkl"
        dataset_path = "data/uploads/employee_dataset.csv"

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        if not os.path.exists(model_path):
            return "Employee model not found.", 404

        if not os.path.exists(dataset_path):
            return "Employee dataset not found.", 404

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = joblib.load(model_path)

        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        df = pd.read_csv(dataset_path)

        df = clean_dataframe(
            df,
            "employee"
        )

        # ----------------------------------------------------
        # EMPLOYEE NUMBER
        # ----------------------------------------------------

        id_column = "employeenumber"

        if id_column not in df.columns:
            return "Employee number column not found.", 404

        employee_number = pd.to_numeric(
            employee_id,
            errors="coerce"
        )

        rows = df.loc[
            df[id_column] == employee_number
        ]

        if rows.empty:
            return (
                f"Employee {employee_id} not found.",
                404
            )

        employee = rows.iloc[0].copy()

        # ----------------------------------------------------
        # PREPARE MODEL DATA
        # ----------------------------------------------------

        model_employee = employee.to_frame().T

        model_employee = prepare_model_data(
            model_employee,
            "employee",
            "attrition"
        )

        # ----------------------------------------------------
        # ORIGINAL PROBABILITY
        # ----------------------------------------------------

        original_probability = round(
            float(
                model.predict_proba(
                    model_employee
                )[0, 1]
            ) * 100,
            2
        )

        original_risk = calculate_risk(
            original_probability
        )

        result = None

        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

        if request.method == "POST":

            simulated_employee = model_employee.copy()

            # ------------------------------------------------
            # OVERTIME
            # ------------------------------------------------

            overtime = get_form_value(
                request.form,
                "overtime"
            )

            if overtime is not None:

                if "overtime" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "overtime"
                    ] = overtime

            # ------------------------------------------------
            # JOB SATISFACTION
            # ------------------------------------------------

            job_satisfaction = get_form_value(
                request.form,
                "job_satisfaction"
            )

            if job_satisfaction is None:

                job_satisfaction = get_form_value(
                    request.form,
                    "jobsatisfaction"
                )

            if job_satisfaction is not None:

                job_satisfaction = float(
                    job_satisfaction
                )

                if "job_satisfaction" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "job_satisfaction"
                    ] = job_satisfaction

                elif "jobsatisfaction" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "jobsatisfaction"
                    ] = job_satisfaction

            # ------------------------------------------------
            # MONTHLY INCOME
            # ------------------------------------------------

            monthly_income = get_form_value(
                request.form,
                "monthly_income"
            )

            if monthly_income is None:

                monthly_income = get_form_value(
                    request.form,
                    "monthlyincome"
                )

            if monthly_income is not None:

                monthly_income = float(
                    monthly_income
                )

                if "monthly_income" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "monthly_income"
                    ] = monthly_income

                elif "monthlyincome" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "monthlyincome"
                    ] = monthly_income

            # ------------------------------------------------
            # WORK LIFE BALANCE
            # ------------------------------------------------

            work_life_balance = get_form_value(
                request.form,
                "work_life_balance"
            )

            if work_life_balance is None:

                work_life_balance = get_form_value(
                    request.form,
                    "worklifebalance"
                )

            if work_life_balance is not None:

                work_life_balance = float(
                    work_life_balance
                )

                if "work_life_balance" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "work_life_balance"
                    ] = work_life_balance

                elif "worklifebalance" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "worklifebalance"
                    ] = work_life_balance

            # ------------------------------------------------
            # YEARS AT COMPANY
            # ------------------------------------------------

            years_at_company = get_form_value(
                request.form,
                "years_at_company"
            )

            if years_at_company is None:

                years_at_company = get_form_value(
                    request.form,
                    "yearsatcompany"
                )

            if years_at_company is not None:

                years_at_company = float(
                    years_at_company
                )

                if "years_at_company" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "years_at_company"
                    ] = years_at_company

                elif "yearsatcompany" in simulated_employee.columns:

                    simulated_employee.loc[
                        simulated_employee.index[0],
                        "yearsatcompany"
                    ] = years_at_company

            # ------------------------------------------------
            # SIMULATED PROBABILITY
            # ------------------------------------------------

            simulated_probability = round(
                float(
                    model.predict_proba(
                        simulated_employee
                    )[0, 1]
                ) * 100,
                2
            )

            # ------------------------------------------------
            # CHANGE
            # ------------------------------------------------

            change = round(
                simulated_probability
                - original_probability,
                2
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result = {

                "original_probability":
                    original_probability,

                "simulated_probability":
                    simulated_probability,

                "change":
                    change,

                "original_risk":
                    original_risk,

                "simulated_risk":
                    calculate_risk(
                        simulated_probability
                    ),

                # Scenario values
                "overtime":
                    overtime
                    if overtime is not None
                    else employee.get(
                        "overtime",
                        "Not available"
                    ),

                "job_satisfaction":
                    job_satisfaction
                    if job_satisfaction is not None
                    else employee.get(
                        "job_satisfaction",
                        employee.get(
                            "jobsatisfaction",
                            "Not available"
                        )
                    ),

                "monthly_income":
                    monthly_income
                    if monthly_income is not None
                    else employee.get(
                        "monthly_income",
                        employee.get(
                            "monthlyincome",
                            "Not available"
                        )
                    ),

                "work_life_balance":
                    work_life_balance
                    if work_life_balance is not None
                    else employee.get(
                        "work_life_balance",
                        employee.get(
                            "worklifebalance",
                            "Not available"
                        )
                    ),

                "years_at_company":
                    years_at_company
                    if years_at_company is not None
                    else employee.get(
                        "years_at_company",
                        employee.get(
                            "yearsatcompany",
                            "Not available"
                        )
                    )
            }

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        return render_template(
            "simulation_employee.html",

            employee=employee.to_dict(),

            employee_id=employee_id,

            original_probability=original_probability,

            original_risk=original_risk,

            result=result
        )

    except Exception as e:

        return (
            f"""
            <h2>Employee Simulation Error</h2>
            <p>{str(e)}</p>
            """,
            500
        )


# ============================================================
# E-COMMERCE SIMULATION
# ============================================================

@app.route(
    "/simulate/ecommerce/<customer_id>",
    methods=["GET", "POST"]
)
def simulate_ecommerce(customer_id):

    try:

        model_path = (
            "models/ecommerce_model.pkl"
        )

        dataset_path = (
            "data/uploads/ecommerce_dataset.csv"
        )

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        if not os.path.exists(
            model_path
        ):

            return (
                "E-Commerce model not found. "
                "Please train the model first.",
                404
            )

        if not os.path.exists(
            dataset_path
        ):

            return (
                "E-Commerce dataset not found.",
                404
            )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = joblib.load(
            model_path
        )

        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        df = pd.read_csv(
            dataset_path
        )

        df = clean_dataframe(
            df,
            "ecommerce"
        )

        # ----------------------------------------------------
        # CUSTOMER ID
        # ----------------------------------------------------

        id_column = "customerid"

        if id_column not in df.columns:

            return (
                "Customer ID column not found "
                "in E-Commerce dataset.",
                404
            )

        customer_rows = df.loc[
            df[id_column]
            .astype(str)
            .str.strip()
            .eq(str(customer_id).strip())
        ]

        if customer_rows.empty:

            return (
                f"Customer {customer_id} not found.",
                404
            )

        customer = (
            customer_rows.iloc[0].copy()
        )

        # ----------------------------------------------------
        # PREPARE MODEL DATA
        # ----------------------------------------------------

        model_customer = prepare_model_data(
            customer.to_frame().T,
            "ecommerce",
            "churn"
        )

        # ----------------------------------------------------
        # ORIGINAL PREDICTION
        # ----------------------------------------------------

        original_probability = round(
            float(
                model.predict_proba(
                    model_customer
                )[0, 1]
            ) * 100,
            2
        )

        original_risk = calculate_risk(
            original_probability
        )

        simulation_result = None

        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

        if request.method == "POST":

            simulated_customer = (
                model_customer.copy()
            )

            numeric_fields = [

                "tenure",

                "satisfactionscore",

                "ordercount",

                "daysincelastorder",

                "cashbackamount",

                "complain",

                "warehousetohome",

                "hourspendonapp"
            ]

            for field in numeric_fields:

                value = get_form_value(
                    request.form,
                    field
                )

                if value is None:

                    continue

                try:

                    numeric_value = float(
                        value
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

                if field in simulated_customer.columns:

                    simulated_customer.loc[
                        simulated_customer.index[0],
                        field
                    ] = numeric_value

            # ------------------------------------------------
            # SIMULATED PREDICTION
            # ------------------------------------------------

            simulated_probability = round(
                float(
                    model.predict_proba(
                        simulated_customer
                    )[0, 1]
                ) * 100,
                2
            )

            # ------------------------------------------------
            # CHANGE
            # ------------------------------------------------

            change = round(
                simulated_probability
                - original_probability,
                2
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            simulation_result = {

                "original_probability":
                    original_probability,

                "simulated_probability":
                    simulated_probability,

                "change":
                    change,

                "original_risk":
                    original_risk,

                "simulated_risk":
                    calculate_risk(
                        simulated_probability
                    )
            }

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        return render_template(

            "simulation_ecommerce.html",

            customer_id=
                customer_id,

            customer=
                customer.to_dict(),

            original_probability=
                original_probability,

            original_risk=
                original_risk,

            simulation_result=
                simulation_result
        )

    except Exception as e:

        return (
            f"""
            <h2>E-Commerce Simulation Error</h2>

            <p>{str(e)}</p>

            <p>
                Customer ID:
                <strong>{customer_id}</strong>
            </p>
            """,
            500
        )


# ============================================================
# START APPLICATION
# ============================================================


@app.route("/simulate/ott/<customer_id>", methods=["GET", "POST"])
def simulate_ott(customer_id):

    domain = "ott"
    config = DOMAIN_CONFIG[domain]

    dataset_path = config["dataset"]
    model_path = config["model"]

    simulation_result = None
    error = None

    try:

        # -----------------------------------------
        # Load dataset
        # -----------------------------------------

        df = pd.read_csv(dataset_path)

        df = clean_dataframe(
            df,
            domain
        )

        id_column = config["id_column"]
        target = config["target"]

        # -----------------------------------------
        # Find customer
        # -----------------------------------------

        customer_rows = df[
            df[id_column]
            .astype(str)
            .str.strip()
            ==
            str(customer_id).strip()
        ]

        if customer_rows.empty:

            raise ValueError(
                f"Customer {customer_id} was not found."
            )

        original_customer = (
            customer_rows.iloc[0].copy()
        )

        # -----------------------------------------
        # Load model
        # -----------------------------------------

        model = joblib.load(
            model_path
        )

        # -----------------------------------------
        # ORIGINAL PREDICTION
        # -----------------------------------------

        original_df = (
            original_customer
            .to_frame()
            .T
        )

        original_model_df = (
            prepare_model_data(
                original_df,
                domain,
                target
            )
        )

        original_prediction = model.predict_proba(
            original_model_df
        )

        original_probability = (
            float(
                original_prediction[0][1]
            ) * 100
        )

        original_probability = round(
            original_probability,
            2
        )

        original_risk = calculate_risk(
            original_probability
        )

        # -----------------------------------------
        # POST SIMULATION
        # -----------------------------------------

        if request.method == "POST":

            simulated_customer = (
                original_customer.copy()
            )

            # -------------------------------------
            # Subscription Type
            # -------------------------------------

            value = request.form.get(
                "subscription_type"
            )

            if value and value.strip():

                simulated_customer[
                    "subscription_type"
                ] = value.strip()

            # -------------------------------------
            # Watch Hours
            # -------------------------------------

            value = request.form.get(
                "watch_hours"
            )

            if value and value.strip():

                simulated_customer[
                    "watch_hours"
                ] = float(value)

            # -------------------------------------
            # Average Watch Hours
            # -------------------------------------

            value = request.form.get(
                "avg_watch_hours_per_day"
            )

            if value and value.strip():

                simulated_customer[
                    "avg_watch_hours_per_day"
                ] = float(value)

            # -------------------------------------
            # Last Login
            # -------------------------------------

            value = request.form.get(
                "last_login_days"
            )

            if value and value.strip():

                simulated_customer[
                    "last_login_days"
                ] = float(value)

            # -------------------------------------
            # Monthly Spend
            # -------------------------------------

            value = request.form.get(
                "monthly_spend"
            )

            if value and value.strip():

                simulated_customer[
                    "monthly_spend"
                ] = float(value)

            # -------------------------------------
            # Satisfaction Score
            # -------------------------------------

            value = request.form.get(
                "satisfaction_score"
            )

            if value and value.strip():

                simulated_customer[
                    "satisfaction_score"
                ] = float(value)

            # -------------------------------------
            # Support Tickets
            # -------------------------------------

            value = request.form.get(
                "support_tickets"
            )

            if value and value.strip():

                simulated_customer[
                    "support_tickets"
                ] = float(value)

            # -------------------------------------
            # Auto Renewal
            # -------------------------------------

            value = request.form.get(
                "auto_renewal"
            )

            if value and value.strip():

                simulated_customer[
                    "auto_renewal"
                ] = int(value)

            # -------------------------------------
            # Number Of Profiles
            # -------------------------------------

            value = request.form.get(
                "num_profiles"
            )

            if value and value.strip():

                simulated_customer[
                    "num_profiles"
                ] = float(value)

            # -------------------------------------
            # Number Of OTT Platforms
            # -------------------------------------

            value = request.form.get(
                "num_ott_platforms"
            )

            if value and value.strip():

                simulated_customer[
                    "num_ott_platforms"
                ] = float(value)

            # -------------------------------------
            # Simulated dataframe
            # -------------------------------------

            simulated_df = (
                simulated_customer
                .to_frame()
                .T
            )

            simulated_model_df = (
                prepare_model_data(
                    simulated_df,
                    domain,
                    target
                )
            )

            # -------------------------------------
            # Simulated prediction
            # -------------------------------------

            simulated_prediction = (
                model.predict_proba(
                    simulated_model_df
                )
            )

            simulated_probability = (
                float(
                    simulated_prediction[0][1]
                ) * 100
            )

            simulated_probability = round(
                simulated_probability,
                2
            )

            simulated_risk = calculate_risk(
                simulated_probability
            )

            # -------------------------------------
            # Probability change
            # -------------------------------------

            change = round(
                simulated_probability
                -
                original_probability,
                2
            )

            # -------------------------------------
            # Scenario summary
            # -------------------------------------

            changes = []

            comparison_fields = [

                (
                    "subscription_type",
                    "Subscription Type"
                ),

                (
                    "watch_hours",
                    "Watch Hours"
                ),

                (
                    "avg_watch_hours_per_day",
                    "Average Watch Hours Per Day"
                ),

                (
                    "last_login_days",
                    "Days Since Last Login"
                ),

                (
                    "monthly_spend",
                    "Monthly Spend"
                ),

                (
                    "satisfaction_score",
                    "Satisfaction Score"
                ),

                (
                    "support_tickets",
                    "Support Tickets"
                ),

                (
                    "auto_renewal",
                    "Auto Renewal"
                ),

                (
                    "num_profiles",
                    "Number Of Profiles"
                ),

                (
                    "num_ott_platforms",
                    "Number Of OTT Platforms"
                )
            ]

            for field, label in comparison_fields:

                old_value = original_customer.get(
                    field
                )

                new_value = simulated_customer.get(
                    field
                )

                if str(old_value) != str(
                    new_value
                ):

                    changes.append(
                        f"{label}: "
                        f"{old_value} → "
                        f"{new_value}"
                    )

            if changes:

                scenario_summary = "; ".join(
                    changes
                )

            else:

                scenario_summary = (
                    "No changes were made to "
                    "the customer scenario."
                )

            # -------------------------------------
            # Store result
            # -------------------------------------

            simulation_result = {

                "original_probability":
                    original_probability,

                "simulated_probability":
                    simulated_probability,

                "change":
                    change,

                "original_risk":
                    original_risk,

                "simulated_risk":
                    simulated_risk,

                "scenario_summary":
                    scenario_summary
            }

    except Exception as e:

        error = str(e)

        print(
            "\nOTT SIMULATION ERROR:"
        )

        print(error)

    # -----------------------------------------
    # Render page
    # -----------------------------------------

    return render_template(
        "simulation_ott.html",

        domain=domain,

        customer_id=customer_id,

        original_probability=locals().get(
            "original_probability",
            0
        ),

        original_risk=locals().get(
            "original_risk",
            "UNKNOWN"
        ),

        simulation_result=simulation_result,

        error=error
    )

if __name__ == "__main__":

    app.run(
        debug=True
    )