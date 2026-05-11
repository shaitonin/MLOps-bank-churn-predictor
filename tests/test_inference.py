"""Tests for src/inference.py — ChurnPredictor."""
import pytest
import pandas as pd
from pathlib import Path
from src.inference import ChurnPredictor
from src.features import build_features

MODEL_PATH = Path(__file__).resolve().parent.parent / 'outputs' / 'models' / 'pipeline_final.joblib'

# Skip all tests in this file if the model file is not present
# (e.g. in a clean CI environment without the trained model)
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason='pipeline_final.joblib not found — run notebooks/deploy.py first',
)


@pytest.fixture(scope='module')
def predictor():
    return ChurnPredictor()


@pytest.fixture
def sample_features():
    return build_features(
        credit_score=650, age=42, tenure=5, balance=125000,
        num_products=2, has_cr_card=True, is_active=True,
        salary=80000, geography='France', gender='Male',
    )


@pytest.fixture
def high_risk_features():
    return build_features(
        credit_score=400, age=55, tenure=2, balance=180000,
        num_products=1, has_cr_card=False, is_active=False,
        salary=60000, geography='Germany', gender='Female',
    )


# ── Predictor initialisation ──────────────────────────────────────────────────

def test_predictor_loads(predictor):
    assert predictor is not None


def test_predictor_has_source(predictor):
    assert predictor.source.startswith(('mlflow:', 'local:'))


def test_predictor_threshold_default(predictor):
    assert predictor.threshold == pytest.approx(0.40, abs=0.01)


# ── predict_single ────────────────────────────────────────────────────────────

def test_predict_single_returns_dict(predictor, sample_features):
    result = predictor.predict_single(sample_features)
    assert isinstance(result, dict)


def test_predict_single_required_keys(predictor, sample_features):
    result = predictor.predict_single(sample_features)
    assert 'churn_probability' in result
    assert 'churn_flag' in result
    assert 'risk_level' in result
    assert 'threshold_used' in result
    assert 'model_source' in result


def test_predict_single_probability_range(predictor, sample_features):
    result = predictor.predict_single(sample_features)
    assert 0.0 <= result['churn_probability'] <= 1.0


def test_predict_single_flag_is_binary(predictor, sample_features):
    result = predictor.predict_single(sample_features)
    assert result['churn_flag'] in (0, 1)


def test_predict_single_risk_level_valid(predictor, sample_features):
    result = predictor.predict_single(sample_features)
    assert result['risk_level'] in ('low', 'medium', 'high')


def test_predict_single_custom_threshold(predictor, sample_features):
    result_low  = predictor.predict_single(sample_features, threshold=0.01)
    result_high = predictor.predict_single(sample_features, threshold=0.99)
    # With threshold=0.01 almost everyone is flagged, with 0.99 almost no one
    assert result_low['churn_flag'] == 1
    assert result_high['churn_flag'] == 0


def test_predict_single_high_risk_customer(predictor, high_risk_features):
    result = predictor.predict_single(high_risk_features)
    # Germany, age 55, inactive, 1 product — should produce elevated probability
    assert result['churn_probability'] > 0.3


# ── predict (batch) ───────────────────────────────────────────────────────────

def test_predict_batch_returns_dataframe(predictor, sample_features, high_risk_features):
    df = pd.DataFrame([sample_features, high_risk_features])
    result = predictor.predict(df)
    assert isinstance(result, pd.DataFrame)


def test_predict_batch_columns(predictor, sample_features):
    df = pd.DataFrame([sample_features])
    result = predictor.predict(df)
    assert set(result.columns) == {'churn_probability', 'churn_flag', 'risk_level'}


def test_predict_batch_row_count(predictor, sample_features, high_risk_features):
    df = pd.DataFrame([sample_features, high_risk_features])
    result = predictor.predict(df)
    assert len(result) == 2


def test_predict_batch_probabilities_in_range(predictor, sample_features, high_risk_features):
    df = pd.DataFrame([sample_features, high_risk_features])
    result = predictor.predict(df)
    assert (result['churn_probability'] >= 0).all()
    assert (result['churn_probability'] <= 1).all()
