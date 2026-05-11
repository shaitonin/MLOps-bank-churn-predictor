"""
src/features.py — Shared feature engineering for inference interfaces.

Converts raw customer inputs (as collected from a form or CSV) into the
engineered feature vector expected by the trained pipeline.

Used by:
    app/streamlit_app.py  — interactive web form
    app/main.py           — REST API (optional helper for clients)
"""


def build_features(
    credit_score: float,
    age: float,
    tenure: float,
    balance: float,
    num_products: float,
    has_cr_card: bool,
    is_active: bool,
    salary: float,
    geography: str,
    gender: str,
) -> dict:
    """
    Build the engineered feature dict from raw customer attributes.

    Returns a flat dict ready to pass to ChurnPredictor.predict_single().
    All keys match the feature columns in outputs/models/model_schema.json.
    """
    age_group_young  = int(age < 30)
    age_group_middle = int(30 <= age < 55)
    age_group_senior = int(age >= 55)
    engagement       = int(is_active) + int(has_cr_card) + min(num_products, 2) - 1
    balance_ratio    = round(balance / max(salary, 1), 4)
    products_year    = round(num_products / max(tenure, 1), 4)
    age_inactivity   = age * int(not is_active)

    return {
        'CreditScore':        float(credit_score),
        'Age':                float(age),
        'Tenure':             float(tenure),
        'Balance':            float(balance),
        'NumOfProducts':      float(num_products),
        'HasCrCard':          int(has_cr_card),
        'IsActiveMember':     int(is_active),
        'EstimatedSalary':    float(salary),
        'Geography_France':   int(geography == 'France'),
        'Geography_Germany':  int(geography == 'Germany'),
        'Geography_Spain':    int(geography == 'Spain'),
        'Gender_Female':      int(gender == 'Female'),
        'Gender_Male':        int(gender == 'Male'),
        'AgeGroup_Middle':    age_group_middle,
        'AgeGroup_Senior':    age_group_senior,
        'AgeGroup_Young':     age_group_young,
        'AgeInactivity':      age_inactivity,
        'EngagementScore':    engagement,
        'BalanceSalaryRatio': balance_ratio,
        'ProductsPerYear':    products_year,
    }
