# %%
# Treina o modelo final (LightGBM sem DR, hiperparâmetros Optuna),
# persiste o pipeline completo como artefato MLflow e registra no
# MLflow Model Registry com versionamento automático.
#
# Fluxo:
#   1. Treino no conjunto completo de treino (sem holdout)
#   2. Avaliação final no holdout
#   3. Log do pipeline serializado como artefato MLflow
#   4. Registro no Model Registry → estágio "Staging"
#   5. Promoção para "Production" se métricas atendem thresholds mínimos

# %%
import sys
import time
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
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
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

# %%
SEED           = 42
THRESHOLD      = 0.40
MODEL_NAME     = 'bank-churn-lgbm'   # nome no MLflow Model Registry

# thresholds mínimos para promoção automática Staging → Production
PROMOTE_MIN_AUC    = 0.85
PROMOTE_MIN_RECALL = 0.70

config      = load_yaml(CONFIG_DIR / 'modeling.yaml')
pipe_cfg    = config.get('pipeline_config', {})
smote_cfg   = config.get('smote', {})
holdout_cfg = config.get('holdout', {})
logger      = get_logger('deploy', config.get('logging', {}))

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
# MLflow
tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
_known_schemes = ('sqlite://', 'postgresql://', 'mysql://', 'http://', 'https://', 'file://')
_resolved_uri = tracking_uri if any(tracking_uri.startswith(s) for s in _known_schemes) \
                else (ROOT_DIR / tracking_uri).as_uri()

mlflow.set_tracking_uri(_resolved_uri)
experiment_name = config.get('mlflow', {}).get('experiment_name', 'bank-churn-classification')
mlflow.set_experiment(experiment_name)
client = MlflowClient()

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

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=holdout_cfg.get('test_size', 0.20),
    random_state=SEED,
    stratify=y,
)
logger.info('Treino: %d | Holdout: %d | Features: %d', len(X_train), len(X_holdout), len(feat_cols))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Construir e Treinar Pipeline Final
# ─────────────────────────────────────────────────────────────────────────────

# %%
def _build_final_pipeline() -> ImbPipeline:
    imp_specs = []
    for spec in pipe_cfg.get('imputation', []):
        normalized = dict(spec)
        if 'group_by' in normalized and 'group_column' not in normalized:
            normalized['group_column'] = normalized.pop('group_by')
        imp_specs.append(normalized)

    steps = []
    if imp_specs:
        steps.append(('imputer', DataImputer(imputation_config=imp_specs)))
    steps.append(('reducer', FeatureReducer(method='none')))
    steps.append(('smote', SMOTE(
        k_neighbors=smote_cfg.get('k_neighbors', 5),
        random_state=smote_cfg.get('random_state', SEED),
    )))
    steps.append(('lgbm', LGBMClassifier(**BEST_LGBM_PARAMS)))
    return ImbPipeline(steps)

# %%
logger.info('─' * 60)
logger.info('Treinando pipeline final no conjunto de treino completo...')
t0       = time.time()
pipeline = _build_final_pipeline()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0
logger.info('Treino concluído em %.1fs', train_time)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Avaliação no Holdout
# ─────────────────────────────────────────────────────────────────────────────

# %%
y_prob = pipeline.predict_proba(X_holdout)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

holdout_metrics = {
    'holdout_roc_auc':       float(roc_auc_score(y_holdout, y_prob)),
    'holdout_f1':            float(f1_score(y_holdout, y_pred, zero_division=0)),
    'holdout_recall':        float(recall_score(y_holdout, y_pred, zero_division=0)),
    'holdout_precision':     float(precision_score(y_holdout, y_pred, zero_division=0)),
    'holdout_avg_precision': float(average_precision_score(y_holdout, y_prob)),
    'training_time_s':       train_time,
}

logger.info('Holdout — AUC: %.4f | Recall: %.4f | F1: %.4f | Prec: %.4f',
            holdout_metrics['holdout_roc_auc'],
            holdout_metrics['holdout_recall'],
            holdout_metrics['holdout_f1'],
            holdout_metrics['holdout_precision'])

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Persistir Modelo e Registrar no MLflow
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Salva pipeline serializado localmente (para uso pela API sem MLflow)
models_dir = ROOT_DIR / 'outputs' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
local_model_path = models_dir / 'pipeline_final.joblib'
joblib.dump(pipeline, local_model_path)
logger.info('Pipeline serializado em: %s', local_model_path)

# Salva metadados de input (schema esperado)
schema = {
    'feature_columns': feat_cols,
    'target_column':   target_col,
    'threshold':       THRESHOLD,
    'n_features':      len(feat_cols),
}
import json
schema_path = models_dir / 'model_schema.json'
schema_path.write_text(json.dumps(schema, indent=2))

# %%
logger.info('Registrando no MLflow...')
with mlflow.start_run(run_name='deploy_final_model') as run:
    mlflow.set_tags({
        'model':   'lightgbm',
        'stage':   'deploy',
        'reducer': 'none',
    })

    # Parâmetros
    mlflow.log_param('threshold',    THRESHOLD)
    mlflow.log_param('reducer',      'none')
    mlflow.log_param('n_features',   len(feat_cols))
    for k, v in BEST_LGBM_PARAMS.items():
        mlflow.log_param(f'lgbm_{k}', v)

    # Métricas do holdout
    mlflow.log_metrics(holdout_metrics)

    # Artefatos
    mlflow.log_artifact(str(local_model_path), artifact_path='model')
    mlflow.log_artifact(str(schema_path),      artifact_path='model')

    # Registra pipeline completo como modelo MLflow (formato sklearn)
    mlflow.sklearn.log_model(
        sk_model       = pipeline,
        artifact_path  = 'pipeline',
        registered_model_name = MODEL_NAME,
        input_example  = X_holdout.head(3),
    )

    run_id = run.info.run_id
    logger.info('Run ID: %s', run_id)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Promoção Automática: Staging → Production
# ─────────────────────────────────────────────────────────────────────────────

# %%
auc    = holdout_metrics['holdout_roc_auc']
recall = holdout_metrics['holdout_recall']

# Obtém a versão recém-registrada
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest   = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
version  = latest.version

logger.info('─' * 60)
logger.info('Modelo registrado: %s v%s', MODEL_NAME, version)
logger.info('AUC=%.4f (mín=%.2f) | Recall=%.4f (mín=%.2f)',
            auc, PROMOTE_MIN_AUC, recall, PROMOTE_MIN_RECALL)

if auc >= PROMOTE_MIN_AUC and recall >= PROMOTE_MIN_RECALL:
    client.transition_model_version_stage(
        name    = MODEL_NAME,
        version = version,
        stage   = 'Production',
        archive_existing_versions=True,
    )
    logger.info('✓ Promovido para PRODUCTION (v%s)', version)
    promotion_status = 'promoted_to_production'
else:
    client.transition_model_version_stage(
        name    = MODEL_NAME,
        version = version,
        stage   = 'Staging',
    )
    logger.warning('⚠ Mantido em STAGING — métricas abaixo do mínimo')
    promotion_status = 'kept_in_staging'

# Atualiza tag do run com resultado da promoção
client.set_tag(run_id, 'promotion_status', promotion_status)
client.set_tag(run_id, 'model_version',    version)

# %%
logger.info('─' * 60)
logger.info('Deploy concluído.')
logger.info('  Modelo  : %s v%s → %s', MODEL_NAME, version, promotion_status)
logger.info('  Artefato: %s', local_model_path)
logger.info('  Schema  : %s', schema_path)
logger.info('  Run ID  : %s', run_id)
