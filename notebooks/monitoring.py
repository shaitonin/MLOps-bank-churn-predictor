# %%
# Simulates a production monitoring cycle:
#   1. "Production batch" = second half of holdout (never-seen data)
#   2. Data drift: KS test feature by feature (train vs. production)
#   3. Model drift: compares production metrics against deploy baseline
#   4. PSI (Population Stability Index) for continuous features
#   5. Logs everything to MLflow — separate run per "monitoring cycle"

# %%
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import mlflow
from scipy import stats

ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
for _p in [str(ROOT_DIR), str(CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.inference import ChurnPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

warnings.filterwarnings('ignore')

# %%
config  = load_yaml(CONFIG_DIR / 'modeling.yaml')
logger  = get_logger('monitoring', config.get('logging', {}))

SEED       = 42
THRESHOLD  = 0.40

# Alert thresholds for drift
KS_PVALUE_ALERT   = 0.05   # p < 0.05 → distribution shifted
PSI_ALERT         = 0.20   # PSI > 0.20 → severe drift
PSI_WARNING       = 0.10   # PSI > 0.10 → moderate drift
AUC_DROP_ALERT    = 0.03   # drop > 3pp → model drift alert
RECALL_DROP_ALERT = 0.05   # drop > 5pp → model drift alert

# %%
# MLflow
tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
_known = ('sqlite://', 'postgresql://', 'mysql://', 'http://', 'https://', 'file://')
_resolved_uri = tracking_uri if any(tracking_uri.startswith(s) for s in _known) \
                else (ROOT_DIR / tracking_uri).as_uri()
mlflow.set_tracking_uri(_resolved_uri)
experiment_name = config.get('mlflow', {}).get('experiment_name', 'bank-churn-classification')
mlflow.set_experiment(experiment_name)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────

# %%
features_file = ROOT_DIR / 'data' / 'features' / 'bank_churn_features.parquet'
df         = pq.read_table(str(features_file)).to_pandas()
target_col = config.get('feature_selection', {}).get('target', 'Exited')
feat_cols  = [c for c in df.columns if c != target_col]
X          = df[feat_cols].copy()
y          = df[target_col]

_rename = {c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
           for c in X.columns if any(ch in c for ch in ('<', '[', ']'))}
if _rename:
    X        = X.rename(columns=_rename)
    feat_cols = list(X.columns)

# Replicate the split from deploy.py (same seed → same holdout)
holdout_cfg = config.get('holdout', {})
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=holdout_cfg.get('test_size', 0.20),
    random_state=SEED,
    stratify=y,
)

# "Production" = half of holdout (simulates a monthly batch)
half = len(X_holdout) // 2
X_prod = X_holdout.iloc[:half].copy()
y_prod = y_holdout.iloc[:half].copy()
X_ref  = X_holdout.iloc[half:].copy()   # deploy reference

logger.info('Train (ref): %d | Prod batch: %d', len(X_train), len(X_prod))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Drift Functions
# ─────────────────────────────────────────────────────────────────────────────

# %%
def ks_drift(X_train: pd.DataFrame, X_prod: pd.DataFrame) -> pd.DataFrame:
    """KS test for each numeric feature. p < 0.05 → drift detected."""
    results = []
    for col in X_train.select_dtypes(include=np.number).columns:
        stat, pval = stats.ks_2samp(X_train[col].dropna(), X_prod[col].dropna())
        results.append({
            'feature':   col,
            'ks_stat':   round(stat, 4),
            'p_value':   round(pval, 4),
            'drift':     pval < KS_PVALUE_ALERT,
        })
    return pd.DataFrame(results).sort_values('p_value')


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between probability distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints  = np.unique(breakpoints)
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual,   bins=breakpoints)
    exp_pct = np.where(exp_counts == 0, 1e-6, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / len(actual))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# %%
# ─────────────────────────────────────────────────────────────────────────────
# Run Monitoring
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('Loading production model...')
predictor = ChurnPredictor()
logger.info('Model loaded: %s', predictor.source)

# %%
# Metrics on reference batch (deploy)
out_ref  = predictor.predict(X_ref)
y_pred_r = out_ref['churn_flag'].values
y_prob_r = out_ref['churn_probability'].values
ref_metrics = {
    'ref_roc_auc': float(roc_auc_score(y_holdout.iloc[half:], y_prob_r)),
    'ref_recall':  float(recall_score(y_holdout.iloc[half:], y_pred_r, zero_division=0)),
    'ref_f1':      float(f1_score(y_holdout.iloc[half:], y_pred_r, zero_division=0)),
}

# Metrics on production batch
out_prod  = predictor.predict(X_prod)
y_pred_p  = out_prod['churn_flag'].values
y_prob_p  = out_prod['churn_probability'].values
prod_metrics = {
    'prod_roc_auc': float(roc_auc_score(y_prod, y_prob_p)),
    'prod_recall':  float(recall_score(y_prod, y_pred_p, zero_division=0)),
    'prod_f1':      float(f1_score(y_prod, y_pred_p, zero_division=0)),
    'prod_precision': float(precision_score(y_prod, y_pred_p, zero_division=0)),
}

auc_drop    = ref_metrics['ref_roc_auc'] - prod_metrics['prod_roc_auc']
recall_drop = ref_metrics['ref_recall']  - prod_metrics['prod_recall']

logger.info('Ref  — AUC: %.4f | Recall: %.4f | F1: %.4f',
            ref_metrics['ref_roc_auc'], ref_metrics['ref_recall'], ref_metrics['ref_f1'])
logger.info('Prod — AUC: %.4f | Recall: %.4f | F1: %.4f',
            prod_metrics['prod_roc_auc'], prod_metrics['prod_recall'], prod_metrics['prod_f1'])

# %%
# Data drift — KS test
logger.info('─' * 60)
logger.info('Calculating data drift (KS test)...')
ks_results  = ks_drift(X_train, X_prod)
drifted     = ks_results[ks_results['drift']]
n_drifted   = len(drifted)

logger.info('Features with drift detected (p < %.2f): %d/%d',
            KS_PVALUE_ALERT, n_drifted, len(feat_cols))
if n_drifted > 0:
    for _, row in drifted.iterrows():
        logger.warning('  DRIFT: %-30s  KS=%.4f  p=%.4f',
                       row['feature'], row['ks_stat'], row['p_value'])

# %%
# PSI on predicted probabilities
psi_score = psi(y_prob_r, y_prob_p)
logger.info('PSI on predicted probabilities: %.4f', psi_score)
if psi_score > PSI_ALERT:
    logger.warning('PSI SEVERE (%.3f > %.2f) — consider retraining', psi_score, PSI_ALERT)
elif psi_score > PSI_WARNING:
    logger.warning('PSI MODERATE (%.3f > %.2f) — monitor closely', psi_score, PSI_WARNING)
else:
    logger.info('PSI stable (%.3f)', psi_score)

# %%
# Model drift
model_drift_auc    = auc_drop    > AUC_DROP_ALERT
model_drift_recall = recall_drop > RECALL_DROP_ALERT

if model_drift_auc or model_drift_recall:
    logger.warning('MODEL DRIFT DETECTED — AUC drop=%.4f | Recall drop=%.4f',
                   auc_drop, recall_drop)
else:
    logger.info('No significant model drift — AUC drop=%.4f | Recall drop=%.4f',
                auc_drop, recall_drop)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Log to MLflow
# ─────────────────────────────────────────────────────────────────────────────

# %%
drift_summary = {
    'n_features_drifted':    n_drifted,
    'psi_score':             round(psi_score, 4),
    'auc_drop':              round(auc_drop, 4),
    'recall_drop':           round(recall_drop, 4),
    'data_drift_detected':   int(n_drifted > 0),
    'model_drift_detected':  int(model_drift_auc or model_drift_recall),
}

with mlflow.start_run(run_name='monitoring_cycle_01'):
    mlflow.set_tags({
        'stage':          'monitoring',
        'model_source':   predictor.source,
        'drift_detected': str(drift_summary['data_drift_detected'] or drift_summary['model_drift_detected']),
    })

    mlflow.log_metrics({**ref_metrics, **prod_metrics, **drift_summary})

    # Log per-feature drift table as CSV artifact
    ks_path = ROOT_DIR / 'outputs' / 'models' / 'ks_drift_results.csv'
    ks_path.parent.mkdir(parents=True, exist_ok=True)
    ks_results.to_csv(ks_path, index=False)
    mlflow.log_artifact(str(ks_path), artifact_path='monitoring')

logger.info('─' * 60)
logger.info('Monitoring logged to MLflow.')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Business Impact Report
# ─────────────────────────────────────────────────────────────────────────────

# %%
n_customers        = len(X_prod)
n_actual_churn     = int(y_prod.sum())
n_detected         = int(y_pred_p[y_prod == 1].sum())   # TP
n_false_positives  = int((y_pred_p == 1).sum()) - n_detected  # FP

revenue_per_customer   = 500     # € average annual revenue per retained customer
campaign_cost          = 40      # € cost of a single retention action
retention_success_rate = 0.30    # 30% of approached customers are retained

preserved_revenue   = n_detected * retention_success_rate * revenue_per_customer
total_campaign_cost = (n_detected + n_false_positives) * campaign_cost
roi                 = (preserved_revenue - total_campaign_cost) / max(total_campaign_cost, 1)

print('\n' + '=' * 60)
print('BUSINESS IMPACT REPORT — Production Batch')
print('=' * 60)
print(f'Customers in batch       : {n_customers}')
print(f'Actual churners          : {n_actual_churn}')
print(f'Detected churners (TP)   : {n_detected}  ({n_detected/max(n_actual_churn,1)*100:.1f}%)')
print(f'False positives (FP)     : {n_false_positives}')
print(f'Preserved revenue (est.) : €{preserved_revenue:,.0f}')
print(f'Campaign cost            : €{total_campaign_cost:,.0f}')
print(f'Estimated ROI            : {roi:.1f}x')
print('=' * 60)
