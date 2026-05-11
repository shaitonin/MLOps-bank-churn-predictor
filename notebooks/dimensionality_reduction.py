# %%
# Controlled experiment: hyperparameters fixed at the best values
# found by Optuna, varying only the DR technique.
#
# Techniques compared:
#   none   — no reduction  (20 features, baseline)
#   PCA    — principal components (unsupervised, 18 components)
#   LDA    — linear discriminant analysis (supervised, 1 binary component)
#
# Pipeline: DataImputer → [StandardScaler] → FeatureReducer → SMOTE → LGBMClassifier
#   • StandardScaler added before PCA and LDA (both are scale-sensitive)
#   • LightGBM does not require scaling — kept consistent with modelagem.py
#
# Output: MLflow experiment "bank-churn-classification"
#         runs named: dr_lgbm_none | dr_lgbm_pca18 | dr_lgbm_lda1

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import mlflow

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
for _p in [str(ROOT_DIR), str(CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import DataImputer
from src.feature_reducer import FeatureReducer

warnings.filterwarnings('ignore')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# %%
SEED      = 42
THRESHOLD = 0.40        # same threshold as Part 3
N_SPLITS  = 5           # StratifiedKFold

config      = load_yaml(CONFIG_DIR / 'modeling.yaml')
pipe_cfg    = config.get('pipeline_config', {})
smote_cfg   = config.get('smote', {})
holdout_cfg = config.get('holdout', {})
cv_cfg      = config.get('cv', {})

logger = get_logger('dr_experiment', config.get('logging', {}))

# Best LightGBM hyperparameters found by Optuna (Part 3)
BEST_LGBM_PARAMS = {
    'n_estimators':       528,
    'num_leaves':         30,
    'learning_rate':      0.014374142761540865,
    'subsample':          0.6871832428932195,
    'colsample_bytree':   0.48032682353647654,
    'min_child_samples':  69,
    'reg_alpha':          0.008049603836309057,
    'reg_lambda':         0.013759155275447583,
    'is_unbalance':       True,
    'random_state':       SEED,
    'verbose':            -1,
    'n_jobs':             -1,
}

# %%
# MLflow — same experiment as Part 3
tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
_known_schemes = ('sqlite://', 'postgresql://', 'mysql://', 'http://', 'https://', 'file://')
if any(tracking_uri.startswith(s) for s in _known_schemes):
    _resolved_uri = tracking_uri
else:
    _resolved_uri = (ROOT_DIR / tracking_uri).as_uri()

mlflow.set_tracking_uri(_resolved_uri)
experiment_name = config.get('mlflow', {}).get('experiment_name', 'bank-churn-classification')
mlflow.set_experiment(experiment_name)
logger.info('MLflow URI: %s | Experiment: %s', _resolved_uri, experiment_name)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────

# %%
features_file = ROOT_DIR / 'data' / 'features' / 'bank_churn_features.parquet'
if not features_file.exists():
    raise FileNotFoundError(f"Run preprocessamento.py first: {features_file}")

df         = pq.read_table(str(features_file)).to_pandas()
target_col = config.get('feature_selection', {}).get('target', 'Exited')
feat_cols  = [c for c in df.columns if c != target_col]
X          = df[feat_cols].copy()
y          = df[target_col]

# sanitize column names (XGBoost / LightGBM reject special characters)
_rename = {c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
           for c in X.columns if any(ch in c for ch in ('<', '[', ']'))}
if _rename:
    X        = X.rename(columns=_rename)
    feat_cols = list(X.columns)

logger.info('Dataset: %d samples, %d features | churn=%.1f%%',
            len(df), len(feat_cols), y.mean() * 100)

# %%
# Train / Holdout split — same parameters as Part 3
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=holdout_cfg.get('test_size', 0.20),
    random_state=SEED,
    stratify=y if holdout_cfg.get('stratify', True) else None,
)
logger.info('Train: %d | Holdout: %d', len(X_train), len(X_holdout))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

# %%
def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        'roc_auc':       float(roc_auc_score(y_true, y_prob)),
        'f1':            float(f1_score(y_true, y_pred, zero_division=0)),
        'recall':        float(recall_score(y_true, y_pred, zero_division=0)),
        'precision':     float(precision_score(y_true, y_pred, zero_division=0)),
        'avg_precision': float(average_precision_score(y_true, y_prob)),
    }


def _run_cv(pipeline, X, y, n_splits, threshold) -> dict:
    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = []
    t0    = time.time()
    for train_idx, val_idx in cv.split(X, y):
        m = clone(pipeline)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = m.predict_proba(X.iloc[val_idx])[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        folds.append(_metrics(y.iloc[val_idx].values, y_pred, y_prob))
    elapsed = time.time() - t0

    df_folds = pd.DataFrame(folds)
    agg = {}
    for col in ['roc_auc', 'f1', 'recall', 'precision', 'avg_precision']:
        agg[f'cv_{col}_mean'] = float(df_folds[col].mean())
        agg[f'cv_{col}_std']  = float(df_folds[col].std())
    agg['training_time_s'] = elapsed
    return agg


def _build_pipeline(reducer_params: dict) -> ImbPipeline:
    """Build pipeline: DataImputer → [StandardScaler] → FeatureReducer → SMOTE → LightGBM."""
    imp_specs = []
    for spec in pipe_cfg.get('imputation', []):
        normalized = dict(spec)
        if 'group_by' in normalized and 'group_column' not in normalized:
            normalized['group_column'] = normalized.pop('group_by')
        imp_specs.append(normalized)

    steps = []
    if imp_specs:
        steps.append(('imputer', DataImputer(imputation_config=imp_specs)))

    # StandardScaler before PCA and LDA: both are sensitive to feature scale
    if reducer_params.get('method') in ('pca', 'lda'):
        steps.append(('scaler', StandardScaler()))

    steps.append(('reducer', FeatureReducer(**reducer_params)))

    if smote_cfg.get('enabled', False):
        steps.append(('smote', SMOTE(
            k_neighbors=smote_cfg.get('k_neighbors', 5),
            random_state=smote_cfg.get('random_state', SEED),
        )))

    steps.append(('lgbm', LGBMClassifier(**BEST_LGBM_PARAMS)))
    return ImbPipeline(steps)


def _count_output_features(pipeline, X_sample, y_sample) -> int:
    """Return the number of features after reduction (useful for MLflow logging)."""
    try:
        p = clone(pipeline)
        # Fit only up to the reducer to inspect the output
        steps_no_smote_model = [(n, s) for n, s in p.steps
                                if n not in ('smote', 'lgbm')]
        temp = ImbPipeline(steps_no_smote_model)
        Xt = temp.fit_transform(X_sample.head(200), y_sample.head(200))
        return Xt.shape[1]
    except Exception:
        return -1

# %%
# ─────────────────────────────────────────────────────────────────────────────
# DR Experiment Definitions
# ─────────────────────────────────────────────────────────────────────────────

# %%
DR_CONFIGS = [
    {
        'run_name':      'dr_lgbm_none',
        'reducer_params': {'method': 'none'},
        'description':   'No dimensionality reduction — 20 original features',
    },
    {
        'run_name':      'dr_lgbm_pca18',
        'reducer_params': {'method': 'pca', 'n_components': 18, 'random_state': SEED},
        'description':   'PCA — 18 principal components (unsupervised)',
    },
    {
        'run_name':      'dr_lgbm_lda1',
        'reducer_params': {'method': 'lda', 'n_components': 1},
        'description':   'LDA — 1 discriminant component (supervised, binary)',
    },
]

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Run Experiments
# ─────────────────────────────────────────────────────────────────────────────

# %%
results_summary = []

for exp in DR_CONFIGS:
    run_name      = exp['run_name']
    reducer_params = exp['reducer_params']
    description   = exp['description']
    method        = reducer_params['method']

    logger.info('─' * 60)
    logger.info('Starting: %s — %s', run_name, description)

    pipeline = _build_pipeline(reducer_params)

    # Cross-validation
    cv_metrics = _run_cv(pipeline, X_train, y_train, N_SPLITS, THRESHOLD)

    # Train on full training set for holdout evaluation
    final_model = clone(pipeline)
    final_model.fit(X_train, y_train)
    y_prob_ho = final_model.predict_proba(X_holdout)[:, 1]
    y_pred_ho = (y_prob_ho >= THRESHOLD).astype(int)
    holdout_metrics = _metrics(y_holdout.values, y_pred_ho, y_prob_ho)

    n_out = _count_output_features(pipeline, X_train, y_train)

    logger.info('  CV ROC-AUC : %.4f ± %.4f', cv_metrics['cv_roc_auc_mean'], cv_metrics['cv_roc_auc_std'])
    logger.info('  CV F1      : %.4f | Recall: %.4f', cv_metrics['cv_f1_mean'], cv_metrics['cv_recall_mean'])
    logger.info('  Holdout AUC: %.4f | F1: %.4f | Recall: %.4f',
                holdout_metrics['roc_auc'], holdout_metrics['f1'], holdout_metrics['recall'])
    logger.info('  Features after DR: %d → %d | CV Time: %.1fs',
                len(feat_cols), n_out, cv_metrics['training_time_s'])

    # Log MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            'model':       'lightgbm',
            'reducer':     method,
            'stage':       'dr_comparison',
            'description': description,
        })
        mlflow.log_param('reducer_method',    method)
        mlflow.log_param('n_features_in',     len(feat_cols))
        mlflow.log_param('n_features_out',    n_out)
        mlflow.log_param('threshold',         THRESHOLD)
        mlflow.log_param('cv_n_splits',       N_SPLITS)
        if method == 'pca':
            mlflow.log_param('pca_n_components', reducer_params.get('n_components'))
        if method == 'lda':
            mlflow.log_param('lda_n_components', 1)
        for k, v in BEST_LGBM_PARAMS.items():
            mlflow.log_param(f'lgbm_{k}', v)

        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics({f'holdout_{k}': v for k, v in holdout_metrics.items()})

    results_summary.append({
        'method':        method.upper() if method != 'none' else 'No DR',
        'features':      n_out,
        'cv_auc':        cv_metrics['cv_roc_auc_mean'],
        'cv_auc_std':    cv_metrics['cv_roc_auc_std'],
        'cv_f1':         cv_metrics['cv_f1_mean'],
        'cv_recall':     cv_metrics['cv_recall_mean'],
        'ho_auc':        holdout_metrics['roc_auc'],
        'ho_f1':         holdout_metrics['f1'],
        'ho_recall':     holdout_metrics['recall'],
        'ho_precision':  holdout_metrics['precision'],
        'cv_time_s':     round(cv_metrics['training_time_s'], 1),
    })

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Comparative Summary
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('FINAL RESULTS — DR Technique Comparison')

df_res = pd.DataFrame(results_summary)
df_res = df_res.sort_values('ho_auc', ascending=False).reset_index(drop=True)

cols_display = {
    'method':       'DR Technique',
    'features':     'Features',
    'cv_auc':       'CV AUC',
    'cv_auc_std':   '±std',
    'cv_f1':        'CV F1',
    'cv_recall':    'CV Recall',
    'ho_auc':       'Holdout AUC',
    'ho_f1':        'Holdout F1',
    'ho_recall':    'Holdout Recall',
    'ho_precision': 'Holdout Prec.',
    'cv_time_s':    'CV Time (s)',
}

print('\n' + '=' * 90)
print('DIMENSIONALITY REDUCTION TECHNIQUE COMPARISON — LightGBM (threshold=0.40)')
print('=' * 90)
print(df_res.rename(columns=cols_display).to_string(index=False, float_format='%.4f'))
print('=' * 90)

logger.info('Experiments logged to MLflow under experiment: %s', experiment_name)
logger.info('Filter in UI by tag stage=dr_comparison')
