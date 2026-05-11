"""Tests for src/features.py — feature engineering logic."""
import pytest
from src.features import build_features


VALID_KWARGS = dict(
    credit_score=650,
    age=42,
    tenure=5,
    balance=125000.0,
    num_products=2,
    has_cr_card=True,
    is_active=True,
    salary=80000.0,
    geography='France',
    gender='Male',
)

EXPECTED_KEYS = {
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male',
    'AgeGroup_Middle', 'AgeGroup_Senior', 'AgeGroup_Young',
    'AgeInactivity', 'EngagementScore', 'BalanceSalaryRatio', 'ProductsPerYear',
}


def test_output_keys():
    result = build_features(**VALID_KWARGS)
    assert set(result.keys()) == EXPECTED_KEYS


def test_geography_one_hot_france():
    result = build_features(**{**VALID_KWARGS, 'geography': 'France'})
    assert result['Geography_France'] == 1
    assert result['Geography_Germany'] == 0
    assert result['Geography_Spain'] == 0


def test_geography_one_hot_germany():
    result = build_features(**{**VALID_KWARGS, 'geography': 'Germany'})
    assert result['Geography_France'] == 0
    assert result['Geography_Germany'] == 1
    assert result['Geography_Spain'] == 0


def test_geography_one_hot_spain():
    result = build_features(**{**VALID_KWARGS, 'geography': 'Spain'})
    assert result['Geography_France'] == 0
    assert result['Geography_Germany'] == 0
    assert result['Geography_Spain'] == 1


def test_gender_one_hot_male():
    result = build_features(**{**VALID_KWARGS, 'gender': 'Male'})
    assert result['Gender_Male'] == 1
    assert result['Gender_Female'] == 0


def test_gender_one_hot_female():
    result = build_features(**{**VALID_KWARGS, 'gender': 'Female'})
    assert result['Gender_Male'] == 0
    assert result['Gender_Female'] == 1


def test_age_group_young():
    result = build_features(**{**VALID_KWARGS, 'age': 25})
    assert result['AgeGroup_Young'] == 1
    assert result['AgeGroup_Middle'] == 0
    assert result['AgeGroup_Senior'] == 0


def test_age_group_middle():
    result = build_features(**{**VALID_KWARGS, 'age': 42})
    assert result['AgeGroup_Young'] == 0
    assert result['AgeGroup_Middle'] == 1
    assert result['AgeGroup_Senior'] == 0


def test_age_group_senior():
    result = build_features(**{**VALID_KWARGS, 'age': 60})
    assert result['AgeGroup_Young'] == 0
    assert result['AgeGroup_Middle'] == 0
    assert result['AgeGroup_Senior'] == 1


def test_age_inactivity_active_customer():
    result = build_features(**{**VALID_KWARGS, 'is_active': True, 'age': 50})
    assert result['AgeInactivity'] == 0


def test_age_inactivity_inactive_customer():
    result = build_features(**{**VALID_KWARGS, 'is_active': False, 'age': 50})
    assert result['AgeInactivity'] == 50


def test_balance_salary_ratio():
    result = build_features(**{**VALID_KWARGS, 'balance': 80000, 'salary': 40000})
    assert result['BalanceSalaryRatio'] == pytest.approx(2.0, rel=1e-4)


def test_balance_salary_ratio_zero_salary():
    # salary=0 should not raise ZeroDivisionError
    result = build_features(**{**VALID_KWARGS, 'salary': 0, 'balance': 100})
    assert result['BalanceSalaryRatio'] == pytest.approx(100.0, rel=1e-4)


def test_products_per_year():
    result = build_features(**{**VALID_KWARGS, 'num_products': 2, 'tenure': 4})
    assert result['ProductsPerYear'] == pytest.approx(0.5, rel=1e-4)


def test_products_per_year_zero_tenure():
    # tenure=0 should not raise ZeroDivisionError
    result = build_features(**{**VALID_KWARGS, 'tenure': 0, 'num_products': 3})
    assert result['ProductsPerYear'] == pytest.approx(3.0, rel=1e-4)


def test_engagement_score_full():
    # active=True, has_cr_card=True, num_products=2 → 1+1+2-1 = 3
    result = build_features(**{**VALID_KWARGS, 'is_active': True, 'has_cr_card': True, 'num_products': 2})
    assert result['EngagementScore'] == 3


def test_engagement_score_inactive_no_card():
    # active=False, has_cr_card=False, num_products=1 → 0+0+1-1 = 0
    result = build_features(**{**VALID_KWARGS, 'is_active': False, 'has_cr_card': False, 'num_products': 1})
    assert result['EngagementScore'] == 0


def test_passthrough_fields():
    result = build_features(**VALID_KWARGS)
    assert result['CreditScore'] == 650.0
    assert result['Age'] == 42.0
    assert result['Tenure'] == 5.0
    assert result['Balance'] == 125000.0
    assert result['NumOfProducts'] == 2.0
    assert result['HasCrCard'] == 1
    assert result['IsActiveMember'] == 1
    assert result['EstimatedSalary'] == 80000.0
