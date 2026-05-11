"""
inference.py — Loads the production MLflow model and runs predictions.

Direct usage:
    from src.inference import ChurnPredictor
    predictor = ChurnPredictor()
    result = predictor.predict(df)        # DataFrame with features
    result = predictor.predict_single({   # dict with a single customer
        'Age': 42, 'Balance': 125000, ...
    })
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class ChurnPredictor:
    """
    Inference wrapper for the churn pipeline.

    Loads the model in two ways (automatic fallback):
      1. MLflow Model Registry (name + stage 'Production')
      2. Local file outputs/models/pipeline_final.joblib
    """

    def __init__(
        self,
        model_name: str = 'bank-churn-lgbm',
        stage: str = 'Production',
        threshold: float | None = None,
    ):
        self.model_name = model_name
        self.stage      = stage
        self._pipeline  = None
        self._schema    = self._load_schema()
        self._threshold = threshold if threshold is not None else self._schema.get('threshold', 0.40)
        self._load_model()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_schema(self) -> dict:
        schema_path = ROOT_DIR / 'outputs' / 'models' / 'model_schema.json'
        if schema_path.exists():
            return json.loads(schema_path.read_text())
        return {'threshold': 0.40, 'feature_columns': [], 'n_features': 20}

    def _load_model(self):
        """Try MLflow first, fall back to local file."""
        try:
            import mlflow
            import mlflow.sklearn
            from src.utils.config_loader import load_yaml
            config       = load_yaml(ROOT_DIR / 'config' / 'modeling.yaml')
            tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
            _known       = ('sqlite://', 'postgresql://', 'mysql://', 'http://', 'https://', 'file://')
            resolved     = tracking_uri if any(tracking_uri.startswith(s) for s in _known) \
                           else (ROOT_DIR / tracking_uri).as_uri()
            mlflow.set_tracking_uri(resolved)
            model_uri        = f'models:/{self.model_name}/{self.stage}'
            self._pipeline   = mlflow.sklearn.load_model(model_uri)
            self._source     = f'mlflow:{model_uri}'
        except Exception:
            local_path = ROOT_DIR / 'outputs' / 'models' / 'pipeline_final.joblib'
            if not local_path.exists():
                raise FileNotFoundError(
                    'Model not found. Run notebooks/deploy.py first.'
                )
            self._pipeline = joblib.load(local_path)
            self._source   = f'local:{local_path}'

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Receives a feature DataFrame and returns a DataFrame with:
          churn_probability  : probability [0, 1]
          churn_flag         : 1 if >= threshold, 0 otherwise
          risk_level         : 'high' | 'medium' | 'low'
        """
        thr   = threshold if threshold is not None else self._threshold
        probs = self._pipeline.predict_proba(X)[:, 1]
        flags = (probs >= thr).astype(int)

        risk = pd.cut(
            probs,
            bins=[-0.001, 0.30, 0.55, 1.001],
            labels=['low', 'medium', 'high'],
        )

        return pd.DataFrame({
            'churn_probability': probs.round(4),
            'churn_flag':        flags,
            'risk_level':        risk,
        }, index=X.index)

    def predict_single(
        self,
        customer: dict,
        threshold: float | None = None,
    ) -> dict:
        """Receives a dictionary with a single customer and returns a prediction."""
        df  = pd.DataFrame([customer])
        out = self.predict(df, threshold=threshold)
        row = out.iloc[0]
        return {
            'churn_probability': float(row['churn_probability']),
            'churn_flag':        int(row['churn_flag']),
            'risk_level':        str(row['risk_level']),
            'threshold_used':    threshold or self._threshold,
            'model_source':      self._source,
        }

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def source(self) -> str:
        return self._source
