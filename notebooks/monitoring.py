# %%
# Simula um ciclo de monitoramento em produção:
#   1. "Batch de produção" = segunda metade do holdout (dados nunca vistos)
#   2. Data drift: teste KS feature a feature (treino vs. produção)
#   3. Model drift: compara métricas de produção com baseline do deploy
#   4. PSI (Population Stability Index) para features contínuas
#   5. Loga tudo no MLflow — run separado por "ciclo de monitoramento"

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

# Thresholds de alerta para drift
KS_PVALUE_ALERT   = 0.05   # p < 0.05 → distribuição mudou
PSI_ALERT         = 0.20   # PSI > 0.20 → drift severo
PSI_WARNING       = 0.10   # PSI > 0.10 → drift moderado
AUC_DROP_ALERT    = 0.03   # queda > 3pp → alerta de model drift
RECALL_DROP_ALERT = 0.05   # queda > 5pp → alerta de model drift

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
# Carregar Dados
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

# Replica o split do deploy.py (mesmo seed → mesmo holdout)
holdout_cfg = config.get('holdout', {})
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=holdout_cfg.get('test_size', 0.20),
    random_state=SEED,
    stratify=y,
)

# "Produção" = metade do holdout (simulação de um batch mensal)
half = len(X_holdout) // 2
X_prod = X_holdout.iloc[:half].copy()
y_prod = y_holdout.iloc[:half].copy()
X_ref  = X_holdout.iloc[half:].copy()   # referência de deploy

logger.info('Treino (ref): %d | Prod batch: %d', len(X_train), len(X_prod))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Funções de Drift
# ─────────────────────────────────────────────────────────────────────────────

# %%
def ks_drift(X_train: pd.DataFrame, X_prod: pd.DataFrame) -> pd.DataFrame:
    """Teste KS para cada feature numérica. p < 0.05 → drift detectado."""
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
    """Population Stability Index entre distribuições de probabilidade."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints  = np.unique(breakpoints)
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual,   bins=breakpoints)
    exp_pct = np.where(exp_counts == 0, 1e-6, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / len(actual))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# %%
# ─────────────────────────────────────────────────────────────────────────────
# Executar Monitoramento
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('Carregando modelo de produção...')
predictor = ChurnPredictor()
logger.info('Modelo carregado: %s', predictor.source)

# %%
# Métricas no batch de referência (deploy)
out_ref  = predictor.predict(X_ref)
y_pred_r = out_ref['churn_flag'].values
y_prob_r = out_ref['churn_probability'].values
ref_metrics = {
    'ref_roc_auc': float(roc_auc_score(y_holdout.iloc[half:], y_prob_r)),
    'ref_recall':  float(recall_score(y_holdout.iloc[half:], y_pred_r, zero_division=0)),
    'ref_f1':      float(f1_score(y_holdout.iloc[half:], y_pred_r, zero_division=0)),
}

# Métricas no batch de produção
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
logger.info('Calculando data drift (KS test)...')
ks_results  = ks_drift(X_train, X_prod)
drifted     = ks_results[ks_results['drift']]
n_drifted   = len(drifted)

logger.info('Features com drift detectado (p < %.2f): %d/%d',
            KS_PVALUE_ALERT, n_drifted, len(feat_cols))
if n_drifted > 0:
    for _, row in drifted.iterrows():
        logger.warning('  DRIFT: %-30s  KS=%.4f  p=%.4f',
                       row['feature'], row['ks_stat'], row['p_value'])

# %%
# PSI nas probabilidades preditas
psi_score = psi(y_prob_r, y_prob_p)
logger.info('PSI nas probabilidades preditas: %.4f', psi_score)
if psi_score > PSI_ALERT:
    logger.warning('PSI SEVERO (%.3f > %.2f) — considerar re-treino', psi_score, PSI_ALERT)
elif psi_score > PSI_WARNING:
    logger.warning('PSI MODERADO (%.3f > %.2f) — monitorar', psi_score, PSI_WARNING)
else:
    logger.info('PSI estável (%.3f)', psi_score)

# %%
# Model drift
model_drift_auc    = auc_drop    > AUC_DROP_ALERT
model_drift_recall = recall_drop > RECALL_DROP_ALERT

if model_drift_auc or model_drift_recall:
    logger.warning('MODEL DRIFT DETECTADO — AUC drop=%.4f | Recall drop=%.4f',
                   auc_drop, recall_drop)
else:
    logger.info('Sem model drift significativo — AUC drop=%.4f | Recall drop=%.4f',
                auc_drop, recall_drop)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Log no MLflow
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

    # Loga tabela de drift por feature como artefato CSV
    ks_path = ROOT_DIR / 'outputs' / 'models' / 'ks_drift_results.csv'
    ks_path.parent.mkdir(parents=True, exist_ok=True)
    ks_results.to_csv(ks_path, index=False)
    mlflow.log_artifact(str(ks_path), artifact_path='monitoring')

logger.info('─' * 60)
logger.info('Monitoramento logado no MLflow.')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Relatório de Impacto de Negócio
# ─────────────────────────────────────────────────────────────────────────────

# %%
n_clientes         = len(X_prod)
n_churn_real       = int(y_prod.sum())
n_detectados       = int(y_pred_p[y_prod == 1].sum())   # TP
n_falsos_positivos = int((y_pred_p == 1).sum()) - n_detectados  # FP

receita_por_cliente   = 500     # € receita anual média por cliente retido
custo_campanha        = 40      # € custo de uma ação de retenção
taxa_retencao_sucesso = 0.30    # 30% dos clientes abordados são retidos

receita_preservada  = n_detectados * taxa_retencao_sucesso * receita_por_cliente
custo_total_campanha = (n_detectados + n_falsos_positivos) * custo_campanha
roi                 = (receita_preservada - custo_total_campanha) / max(custo_total_campanha, 1)

print('\n' + '=' * 60)
print('RELATÓRIO DE IMPACTO DE NEGÓCIO — Batch de Produção')
print('=' * 60)
print(f'Clientes no batch        : {n_clientes}')
print(f'Churners reais           : {n_churn_real}')
print(f'Churners detectados (TP) : {n_detectados}  ({n_detectados/max(n_churn_real,1)*100:.1f}%)')
print(f'Falsos positivos (FP)    : {n_falsos_positivos}')
print(f'Receita preservada (est.): €{receita_preservada:,.0f}')
print(f'Custo campanhas          : €{custo_total_campanha:,.0f}')
print(f'ROI estimado             : {roi:.1f}x')
print('=' * 60)
