# %%
# Input  : data/features/bank_churn_features.parquet  ← preprocessing_pipeline.py
# Output : mlruns/           (MLFlow server — experiments, runs, artifacts)
#          outputs/modeling/ (PNG plots saved locally before logging)
#
# Configured models:
#   Linear    : LogisticRegression   (interpretable baseline; requires scaling)
#   Bagging   : RandomForestClassifier
#   Boosting  : XGBClassifier        (scale_pos_weight=3.91)
#   Boosting  : LGBMClassifier       (is_unbalance=True; leaf-wise growth)
#
# Leak-free pipeline (ImbPipeline):
#   DataImputer → (StandardScaler + RobustScaler if LogisticRegression) →
#   FeatureReducer (none|rfe|pca|kpca) → SMOTE → Estimator
#   Each step is fitted ONLY on the training indices of each CV fold.
#
# Dimensionality reduction (src/feature_reducer.py):
#   RFE  → preserves interpretability (original features selected)
#   PCA  → orthogonal components — incompatible with original SHAP
#   KPCA → non-linear version via kernel trick (rbf | poly | cosine)
#   Optuna co-optimizes: reduction method + parameters + estimator.
# ─────────────────────────────────────────────────────────────────────────────

# %%
import sys
import time
import warnings
import importlib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')
from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

import optuna
import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    average_precision_score, confusion_matrix as sk_confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
for _p in [str(ROOT_DIR), str(CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import DataImputer, StandardScalerTransformer, RobustScalerTransformer
from src.feature_reducer import FeatureReducer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

# %%
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        'roc_auc':       float(roc_auc_score(y_true, y_prob)),
        'f1':            float(f1_score(y_true, y_pred, zero_division=0)),
        'recall':        float(recall_score(y_true, y_pred, zero_division=0)),
        'precision':     float(precision_score(y_true, y_pred, zero_division=0)),
        'avg_precision': float(average_precision_score(y_true, y_prob)),
    }

# %%
def _run_cv(pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, threshold: float) -> list[dict]:
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = clone(pipeline)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = m.predict_proba(X.iloc[val_idx])[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = _compute_metrics(y.iloc[val_idx].values, y_pred, y_prob)
        metrics['fold'] = fold_i + 1
        fold_metrics.append(metrics)
    return fold_metrics

# %%
def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    df = pd.DataFrame(fold_metrics)
    result = {}
    for col in ['roc_auc', 'f1', 'recall', 'precision', 'avg_precision']:
        result[f'cv_{col}_mean'] = float(df[col].mean())
        result[f'cv_{col}_std']  = float(df[col].std())
    return result

# %%
def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    ptype = spec['type']
    if ptype == 'log_float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
    elif ptype == 'float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']))
    elif ptype == 'int':
        return trial.suggest_int(name, int(spec['low']), int(spec['high']))
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    raise ValueError(f'Unknown search_space type: {ptype!r}')

# %%
def _build_model(model_cfg: dict, extra_params: dict | None = None):
    module = importlib.import_module(model_cfg['module'])
    cls    = getattr(module, model_cfg['class'])
    params = dict(model_cfg.get('default_params') or {})
    if extra_params:
        params.update(extra_params)
    return cls(**params)

# %%
def _build_pipeline(
    model_cfg: dict,
    model_params: dict | None,
    reducer_params: dict | None,
    pipe_cfg: dict,
    smote_cfg: dict,
    seed: int,
    feature_cols: list[str] | None = None,
) -> ImbPipeline:
    """
    Builds an ImbPipeline:
        DataImputer → (scalers if LogisticRegression) → FeatureReducer → SMOTE → estimator

    SMOTE is applied only in fit() (training) — never in transform() (validation/holdout).
    """
    steps = []

    # Imputation
    imp_specs = []
    for spec in pipe_cfg.get('imputation', []):
        normalized = dict(spec)
        # normalize key group_by → group_column (DataImputer standard)
        if 'group_by' in normalized and 'group_column' not in normalized:
            normalized['group_column'] = normalized.pop('group_by')
        imp_specs.append(normalized)
    if imp_specs:
        steps.append(('imputer', DataImputer(imputation_config=imp_specs)))

    # Scaling — only for Logistic Regression
    if model_cfg.get('class') == 'LogisticRegression':
        scale_cfg = pipe_cfg.get('scaling', {})
        avail = set(feature_cols or [])
        std_cols = [c for c in scale_cfg.get('standard_columns', []) if not avail or c in avail]
        rob_cols = [c for c in scale_cfg.get('robust_columns', []) if not avail or c in avail]
        log_cols = scale_cfg.get('log_transform_before_robust', [])
        if std_cols:
            steps.append(('std_scaler', StandardScalerTransformer(columns=std_cols)))
        if rob_cols:
            steps.append(('rob_scaler', RobustScalerTransformer(columns=rob_cols, log_columns=log_cols)))

    # Feature reduction
    rp = reducer_params or {'method': 'none'}
    steps.append(('reducer', FeatureReducer(**rp)))

    # SMOTE
    if smote_cfg.get('enabled', False):
        steps.append(('smote', SMOTE(
            k_neighbors=smote_cfg.get('k_neighbors', 5),
            random_state=smote_cfg.get('random_state', seed),
        )))

    steps.append(('estimator', _build_model(model_cfg, model_params)))
    return ImbPipeline(steps)

# %%
def _default_reducer_params() -> dict:
    params = {'method': _red_method}
    cfg = feat_red_cfg.get(_red_method, {})
    if _red_method == 'rfe':
        params['n_features_to_select'] = cfg.get('n_features_to_select', 15)
        params['rfe_estimator']        = cfg.get('rfe_estimator', 'random_forest')
    elif _red_method == 'pca':
        params['n_components'] = cfg.get('n_components', 15)
    elif _red_method == 'kpca':
        params['n_components'] = cfg.get('n_components', 12)
        params['kernel']       = cfg.get('kernel', 'rbf')
        params['gamma']        = cfg.get('gamma', None)
        params['degree']       = cfg.get('degree', 3)
        params['coef0']        = cfg.get('coef0', 1.0)
    return params

# %%
def _get_feature_importance(pipeline: ImbPipeline, feature_names: list[str],
                             X_val: pd.DataFrame, y_val: pd.Series) -> pd.Series:
    estimator = pipeline.named_steps['estimator']
    reducer   = pipeline.named_steps.get('reducer')
    imp_features = (
        reducer.selected_features
        if (reducer is not None and hasattr(reducer, 'selected_features'))
        else feature_names
    )
    if hasattr(estimator, 'feature_importances_'):
        return pd.Series(estimator.feature_importances_, index=imp_features)
    elif hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_).flatten()
        return pd.Series(coef, index=imp_features)
    else:
        sample = min(2000, len(X_val))
        idx = np.random.default_rng(42).choice(len(X_val), sample, replace=False)
        r = permutation_importance(pipeline, X_val.iloc[idx], y_val.iloc[idx],
                                   n_repeats=5, random_state=42, n_jobs=-1)
        return pd.Series(r.importances_mean, index=feature_names)

# %%
# ─── Plot Functions ─────────────────────────────────────────────────────────

def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)

# %%
def _plot_roc_curve(y_true, y_prob, model_name: str, out_dir: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC Curve — {model_name}')
    ax.legend()
    path = out_dir / f'roc_curve_{model_name}.png'
    _save_fig(fig, path)
    return path

# %%
def _plot_pr_curve(y_true, y_prob, model_name: str, out_dir: Path) -> Path:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f'AP = {ap:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {model_name}')
    ax.legend()
    path = out_dir / f'pr_curve_{model_name}.png'
    _save_fig(fig, path)
    return path

# %%
def _plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: Path) -> Path:
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained', 'Churn'], yticklabels=['Retained', 'Churn'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    path = out_dir / f'confusion_matrix_{model_name}.png'
    _save_fig(fig, path)
    return path

# %%
def _plot_threshold_analysis(y_true, y_prob, model_name: str, opt_threshold: float,
                              out_dir: Path) -> Path:
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s, recs, precs = [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
        precs.append(precision_score(y_true, yp, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, f1s,   label='F1')
    ax.plot(thresholds, recs,  label='Recall')
    ax.plot(thresholds, precs, label='Precision')
    ax.axvline(opt_threshold, color='red', linestyle='--', label=f'threshold={opt_threshold}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold Analysis — {model_name}')
    ax.legend()
    path = out_dir / f'threshold_analysis_{model_name}.png'
    _save_fig(fig, path)
    return path

# %%
def _plot_feature_importance(importance: pd.Series, model_name: str, out_dir: Path,
                              top_n: int = 20) -> Path:
    top = importance.abs().nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, top_n // 2)))
    top.plot(kind='barh', ax=ax)
    ax.set_title(f'Feature Importance (top {top_n}) — {model_name}')
    ax.set_xlabel('Importance')
    path = out_dir / f'feature_importance_{model_name}.png'
    _save_fig(fig, path)
    return path

# %%
def _plot_cv_fold_comparison(fold_metrics: list[dict], model_name: str, out_dir: Path) -> Path:
    df = pd.DataFrame(fold_metrics).set_index('fold')
    fig, ax = plt.subplots(figsize=(7, 4))
    df[['roc_auc', 'f1', 'recall', 'precision']].plot(ax=ax, marker='o')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title(f'CV Fold Comparison — {model_name}')
    ax.legend()
    path = out_dir / f'cv_fold_comparison_{model_name}.png'
    _save_fig(fig, path)
    return path

# %%
def _plot_class_distribution(y_before, y_after, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, y, title in zip(axes, [y_before, y_after], ['Before SMOTE', 'After SMOTE']):
        vals = pd.Series(y).value_counts().sort_index()
        vals.plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['Retained (0)', 'Churn (1)'], rotation=0)
    fig.tight_layout()
    path = out_dir / 'class_distribution.png'
    _save_fig(fig, path)
    return path

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Configuration Loading
# ─────────────────────────────────────────────────────────────────────────────

# %%
config       = load_yaml(CONFIG_DIR / 'pipeline.yaml')
modeling_cfg = load_yaml(CONFIG_DIR / 'modeling.yaml')
config.update(modeling_cfg)

logger = get_logger('modelagem', config.get('logging'))

logger.info('=== Bank Churn — Modeling with MLFlow + Optuna ===')

# %%
_mod_cfg        = config.get('modeling', {})
tracking_uri    = _mod_cfg.get('tracking_uri', 'mlruns')
experiment_name = _mod_cfg.get('experiment_name', 'bank-churn-classification')
SEED            = _mod_cfg.get('random_seed', 42)
pipe_cfg        = config.get('pipeline', {})
smote_cfg       = pipe_cfg.get('smote', {})
feat_red_cfg    = config.get('feature_reduction', {})
optuna_cfg      = config.get('optuna', {})
_global_n_trials = optuna_cfg.get('default_trials', 30)
threshold_cfg   = config.get('threshold_tuning', {})
DEFAULT_THRESHOLD = threshold_cfg.get('default_threshold', 0.50)
OPT_THRESHOLD     = threshold_cfg.get('optimized_threshold', 0.40)

# Explicit scheme URI (sqlite://, http://) → use as-is
# Relative path (e.g. "mlruns") → convert to file://
_known_schemes = ('sqlite://', 'postgresql://', 'mysql://', 'http://', 'https://', 'file://')
if any(tracking_uri.startswith(s) for s in _known_schemes):
    _resolved_uri = tracking_uri
else:
    _resolved_uri = (ROOT_DIR / tracking_uri).as_uri()

mlflow.set_tracking_uri(_resolved_uri)
mlflow.set_experiment(experiment_name)

logger.info('MLFlow URI       : %s', _resolved_uri)
logger.info('Experiment      : %s', experiment_name)
logger.info('Random seed     : %d', SEED)
logger.info('Default threshold : %.2f | optimized: %.2f', DEFAULT_THRESHOLD, OPT_THRESHOLD)

# %%
cv_cfg        = config.get('cv', {})
holdout_cfg   = config.get('holdout', {})
models_cfg    = config.get('models', {})
artifacts_cfg = config.get('artifacts', {})

enabled_models = [k for k, v in models_cfg.items() if v.get('enabled', True)]
logger.info('Enabled models   : %s', enabled_models)
logger.info('SMOTE enabled     : %s', smote_cfg.get('enabled', False))
logger.info('Feature reducer   : method=%s', feat_red_cfg.get('method', 'none'))

# %%
_red_method = feat_red_cfg.get('method', 'none')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load Features
# ─────────────────────────────────────────────────────────────────────────────

# %%
features_dir  = ROOT_DIR / 'data' / 'features'
features_file = features_dir / 'bank_churn_features.parquet'

logger.info('─' * 60)
logger.info('SECTION 1: Load Features')
logger.info('Reading: %s', features_file)

if not features_file.exists():
    raise FileNotFoundError(
            f"File not found: {features_file}\n"
            "Run preprocessing_pipeline.py before this script."
    )

# %%
schema = pq.read_schema(str(features_file))
logger.info('Schema (%d columns):', len(schema))
for field in schema:
    logger.info('  %-30s %s', field.name, field.type)

# %%
df = pq.read_table(str(features_file)).to_pandas()
logger.info('Shape: %s', df.shape)

# %%
target_col   = config.get('feature_selection', {}).get('target', 'Exited')
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col]

# XGBoost rejects columns with special characters — sanitize once
_rename_map = {
    c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
    for c in X.columns if any(ch in c for ch in ('<', '[', ']'))
}
if _rename_map:
    X = X.rename(columns=_rename_map)
    feature_cols = list(X.columns)
    logger.info('Renamed columns: %s', _rename_map)

logger.info('Features : %d', len(feature_cols))
logger.info('Target   : %s  (churn=%.1f%%)', target_col, y.mean() * 100)

n_nulls = X.isna().sum().sum()
if n_nulls > 0:
    logger.warning('WARNING: %d null values found in features!', n_nulls)
else:
    logger.info('No null values ✓')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Train / Holdout Split
# ─────────────────────────────────────────────────────────────────────────────

# %%
test_size = holdout_cfg.get('test_size', 0.20)
stratify  = holdout_cfg.get('stratify', True)

logger.info('─' * 60)
logger.info('SECTION 2: Train / Holdout Split')
logger.info('Test size: %.0f%%  |  Stratify: %s', test_size * 100, stratify)

# %%
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=test_size,
    random_state=SEED,
    stratify=y if stratify else None,
)

logger.info('Train   : %d samples — churn=%.1f%%', len(X_train), y_train.mean() * 100)
logger.info('Holdout : %d samples — churn=%.1f%%', len(X_holdout), y_holdout.mean() * 100)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Model Registration and Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SECTION 3: Model Registration')

rows = []
for name, cfg in models_cfg.items():
    rows.append({
        'model'          : name,
        'enabled'        : '✓' if cfg.get('enabled', True) else '✗',
        'class'          : f"{cfg['module']}.{cfg['class']}",
        'optuna_trials'  : cfg.get('optuna_trials', _global_n_trials),
        'search_space'   : len(cfg.get('search_space') or {}),
    })

models_table = pd.DataFrame(rows)
logger.info('\n%s', models_table.to_string(index=False))

logger.info('Feature reducer: method=%s | params=%s', _red_method, _default_reducer_params())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Baseline: Cross-Validation with Default Parameters
# ─────────────────────────────────────────────────────────────────────────────

# %%
n_splits = cv_cfg.get('n_splits', 5)
shuffle  = cv_cfg.get('shuffle', True)
cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=SEED)

logger.info('─' * 60)
logger.info('SECTION 4: Baseline CV  (StratifiedKFold, %d folds, threshold=%.2f)',
            n_splits, DEFAULT_THRESHOLD)

# %%
all_results: dict[str, dict] = {}

for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get('enabled', True):
        logger.info('[SKIP] %s', model_name)
        continue

    logger.info('  [BASELINE] %-25s ...', model_name)
    pipeline = _build_pipeline(
        model_cfg=model_cfg,
        model_params=None,
        reducer_params=_default_reducer_params(),
        pipe_cfg=pipe_cfg,
        smote_cfg=smote_cfg,
        seed=SEED,
        feature_cols=feature_cols,
    )
    t0 = time.time()

    with mlflow.start_run(run_name=f'baseline_{model_name}',
                          tags={'stage': 'baseline', 'model': model_name}):
        default_params = {
            str(k): (str(v) if v is None else v)
            for k, v in (model_cfg.get('default_params') or {}).items()
        }
        default_params['reducer_method'] = _red_method
        default_params['threshold']      = DEFAULT_THRESHOLD
        mlflow.log_params(default_params)
        mlflow.set_tag('model_class', f"{model_cfg['module']}.{model_cfg['class']}")

        fold_metrics = _run_cv(pipeline, X_train, y_train, cv, DEFAULT_THRESHOLD)
        for fm in fold_metrics:
            step = fm['fold']
            mlflow.log_metric('fold_roc_auc',       fm['roc_auc'],       step=step)
            mlflow.log_metric('fold_f1',            fm['f1'],            step=step)
            mlflow.log_metric('fold_recall',        fm['recall'],        step=step)
            mlflow.log_metric('fold_precision',     fm['precision'],     step=step)
            mlflow.log_metric('fold_avg_precision', fm['avg_precision'], step=step)

        agg = _aggregate_fold_metrics(fold_metrics)
        mlflow.log_metrics(agg)
        mlflow.log_metric('training_time_s', time.time() - t0)

    all_results[model_name] = {
        **agg,
        'fold_metrics'   : fold_metrics,
        'model_cfg'      : model_cfg,
        'best_params'    : dict(model_cfg.get('default_params') or {}),
        'reducer_params' : _default_reducer_params(),
        'tuned'          : False,
    }
    logger.info(
        '    ROC-AUC: %.4f ± %.4f  |  F1: %.4f  |  Recall: %.4f  |  %.1fs',
        agg['cv_roc_auc_mean'], agg['cv_roc_auc_std'],
        agg['cv_f1_mean'], agg['cv_recall_mean'], time.time() - t0,
    )

# %%
baseline_df = pd.DataFrame([
    {
        'model'             : k,
        'cv_roc_auc_mean'   : v['cv_roc_auc_mean'],
        'cv_roc_auc_std'    : v['cv_roc_auc_std'],
        'cv_f1_mean'        : v['cv_f1_mean'],
        'cv_recall_mean'    : v['cv_recall_mean'],
        'cv_avg_precision_mean': v['cv_avg_precision_mean'],
    }
    for k, v in all_results.items()
]).sort_values('cv_roc_auc_mean', ascending=False)

logger.info('\n%s', baseline_df.round(4).to_string(index=False))
baseline_df

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Hyperparameter Optimization with Optuna
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SECTION 5: Optuna — Hyperparameter Optimization')

# %%
def _make_objective(model_name: str, model_cfg: dict):
    """Creates the Optuna objective function for a model."""
    red_ss = feat_red_cfg.get('search_space', {})

    def objective(trial: optuna.Trial) -> float:
        # Model hyperparameters
        model_params = {}
        for pname, spec in (model_cfg.get('search_space') or {}).items():
            model_params[pname] = _suggest_param(trial, pname, spec)

        # Reduction method (Optuna can choose between none/rfe/pca/kpca)
        if 'method' in red_ss:
            red_method = _suggest_param(trial, 'reducer_method', red_ss['method'])
        else:
            red_method = _red_method

        reducer_params = {'method': red_method}
        if red_method != 'none':
            method_cfg = feat_red_cfg.get(red_method, {})
            for pname, spec in (method_cfg.get('search_space') or {}).items():
                reducer_params[pname] = _suggest_param(trial, f'red_{pname}', spec)

        pipeline = _build_pipeline(
            model_cfg=model_cfg,
            model_params=model_params,
            reducer_params=reducer_params,
            pipe_cfg=pipe_cfg,
            smote_cfg=smote_cfg,
            seed=SEED,
            feature_cols=feature_cols,
        )
        fold_metrics = _run_cv(pipeline, X_train, y_train, cv, DEFAULT_THRESHOLD)
        agg = _aggregate_fold_metrics(fold_metrics)
        return agg['cv_roc_auc_mean']

    return objective

# %%
for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get('enabled', True):
        continue
    if not model_cfg.get('search_space'):
        logger.info('[SKIP Optuna] %s — no search_space', model_name)
        continue

    n_trials = model_cfg.get('optuna_trials', _global_n_trials)
    logger.info('  [OPTUNA] %-25s — %d trials', model_name, n_trials)
    t0 = time.time()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))

    with mlflow.start_run(run_name=f'optuna_{model_name}',
                          tags={'stage': 'optuna', 'model': model_name}):

        def _log_trial(study, trial):
            if trial.value is None:   # trial failed (invalid combination) — skip logging
                return
            with mlflow.start_run(run_name=f'trial_{trial.number}', nested=True,
                                  tags={'trial': str(trial.number), 'model': model_name}):
                mlflow.log_params({str(k): (str(v) if v is None else v)
                                   for k, v in trial.params.items()})
                mlflow.log_metric('cv_roc_auc', trial.value)

        study.optimize(
            _make_objective(model_name, model_cfg),
            n_trials=n_trials,
            callbacks=[_log_trial],
            show_progress_bar=False,
            catch=(ValueError,),   # ignore invalid combinations (e.g. l1 + lbfgs)
        )

        best = study.best_trial
        logger.info(
            '    Best ROC-AUC: %.4f  |  Params: %s  |  %.1fs',
            best.value, best.params, time.time() - t0,
        )

        # Extract model and reducer params separately
        best_model_params   = {k: v for k, v in best.params.items()
                               if not k.startswith('red_') and k != 'reducer_method'}
        best_reducer_method = best.params.get('reducer_method', _red_method)
        best_reducer_params = {'method': best_reducer_method}
        best_reducer_params.update({
            k[4:]: v for k, v in best.params.items() if k.startswith('red_')
        })

        # Log best configuration
        mlflow.log_params({f'best_{k}': (str(v) if v is None else v)
                           for k, v in best.params.items()})
        mlflow.log_metric('best_cv_roc_auc', best.value)
        mlflow.log_metric('optuna_time_s', time.time() - t0)

        # Run full CV with the best params to aggregate all metrics
        best_pipeline = _build_pipeline(
            model_cfg=model_cfg,
            model_params=best_model_params,
            reducer_params=best_reducer_params,
            pipe_cfg=pipe_cfg,
            smote_cfg=smote_cfg,
            seed=SEED,
            feature_cols=feature_cols,
        )
        best_fold_metrics = _run_cv(best_pipeline, X_train, y_train, cv, DEFAULT_THRESHOLD)
        best_agg = _aggregate_fold_metrics(best_fold_metrics)
        mlflow.log_metrics(best_agg)

    # Update all_results if Optuna improved
    prev_auc = all_results[model_name]['cv_roc_auc_mean']
    if best.value > prev_auc:
        all_results[model_name].update({
            **best_agg,
            'fold_metrics'   : best_fold_metrics,
            'best_params'    : best_model_params,
            'reducer_params' : best_reducer_params,
            'tuned'          : True,
        })
        logger.info('    ✓ Optuna improved: %.4f → %.4f', prev_auc, best.value)
    else:
        logger.info('    — Optuna did not improve (%.4f ≤ %.4f)', best.value, prev_auc)

# %%
# Post-Optuna table
tuned_df = pd.DataFrame([
    {
        'model'           : k,
        'cv_roc_auc_mean' : v['cv_roc_auc_mean'],
        'cv_f1_mean'      : v['cv_f1_mean'],
        'cv_recall_mean'  : v['cv_recall_mean'],
        'tuned'           : '✓' if v['tuned'] else '—',
    }
    for k, v in all_results.items()
]).sort_values('cv_roc_auc_mean', ascending=False)

logger.info('\n%s', tuned_df.round(4).to_string(index=False))
tuned_df

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Best Model: Artifacts and Threshold Tuning
# ─────────────────────────────────────────────────────────────────────────────

# %%
out_dir = ROOT_DIR / artifacts_cfg.get('output_dir', 'outputs/modeling')
out_dir.mkdir(parents=True, exist_ok=True)

# Select the best model by average CV ROC-AUC
best_model_name = max(all_results, key=lambda k: all_results[k]['cv_roc_auc_mean'])
best_result     = all_results[best_model_name]
best_model_cfg  = best_result['model_cfg']

logger.info('─' * 60)
logger.info('SECTION 6: Artifacts — Best Model: %s', best_model_name)
logger.info('  CV ROC-AUC : %.4f ± %.4f', best_result['cv_roc_auc_mean'], best_result['cv_roc_auc_std'])
logger.info('  CV F1      : %.4f', best_result['cv_f1_mean'])
logger.info('  CV Recall  : %.4f', best_result['cv_recall_mean'])

# %%
# Train the final pipeline on the full training dataset
final_pipeline = _build_pipeline(
    model_cfg=best_model_cfg,
    model_params=best_result['best_params'],
    reducer_params=best_result['reducer_params'],
    pipe_cfg=pipe_cfg,
    smote_cfg=smote_cfg,
    seed=SEED,
    feature_cols=feature_cols,
)
final_pipeline.fit(X_train, y_train)

y_train_prob = final_pipeline.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_prob >= OPT_THRESHOLD).astype(int)

# %%
with mlflow.start_run(run_name=f'best_model_{best_model_name}',
                      tags={'stage': 'best_model', 'model': best_model_name}):

    mlflow.log_params({str(k): (str(v) if v is None else v)
                       for k, v in best_result['best_params'].items()})
    mlflow.log_params({
        'threshold_used'   : OPT_THRESHOLD,
        'reducer_method'   : best_result['reducer_params'].get('method', 'none'),
    })
    mlflow.log_metrics({f'train_{k}': v
                        for k, v in _compute_metrics(y_train.values, y_train_pred, y_train_prob).items()})
    mlflow.log_metrics({k: v for k, v in best_result.items()
                        if isinstance(v, float)})

    # ── Plots ────────────────────────────────────────────────────────────────
    if artifacts_cfg.get('save_plots', True):
        plots = artifacts_cfg.get('plots', [])

        if 'roc_curve' in plots:
            p = _plot_roc_curve(y_train.values, y_train_prob, best_model_name, out_dir)
            mlflow.log_artifact(str(p), artifact_path='plots')

        if 'precision_recall_curve' in plots:
            p = _plot_pr_curve(y_train.values, y_train_prob, best_model_name, out_dir)
            mlflow.log_artifact(str(p), artifact_path='plots')

        if 'confusion_matrix' in plots:
            p = _plot_confusion_matrix(y_train.values, y_train_pred, best_model_name, out_dir)
            mlflow.log_artifact(str(p), artifact_path='plots')

        if 'threshold_analysis' in plots:
            p = _plot_threshold_analysis(y_train.values, y_train_prob, best_model_name,
                                          OPT_THRESHOLD, out_dir)
            mlflow.log_artifact(str(p), artifact_path='plots')

        if 'feature_importance' in plots:
            importance = _get_feature_importance(final_pipeline, feature_cols, X_train, y_train)
            p = _plot_feature_importance(importance, best_model_name, out_dir)
            mlflow.log_artifact(str(p), artifact_path='plots')

        if 'cv_fold_comparison' in plots:
            p = _plot_cv_fold_comparison(best_result['fold_metrics'], best_model_name, out_dir)
            mlflow.log_artifact(str(p), artifact_path='plots')

        if 'class_distribution' in plots and smote_cfg.get('enabled', False):
            # Apply the partial pipeline (without estimator) to obtain X after SMOTE
            try:
                pipe_no_est = ImbPipeline(
                    [(n, s) for n, s in final_pipeline.steps if n != 'estimator']
                )
                _, y_after_smote = pipe_no_est.fit_resample(X_train, y_train)
                p = _plot_class_distribution(y_train.values, y_after_smote, out_dir)
                mlflow.log_artifact(str(p), artifact_path='plots')
            except Exception:
                pass

        if 'shap_summary' in plots:
            try:
                import shap
                estimator = final_pipeline.named_steps['estimator']
                reducer   = final_pipeline.named_steps.get('reducer')
                imp_features = (
                    reducer.selected_features
                    if (reducer and hasattr(reducer, 'selected_features'))
                    else feature_cols
                )
                # Transform X_train through the pipeline (without estimator)
                pipe_transform = ImbPipeline(
                    [(n, s) for n, s in final_pipeline.steps if n not in ('estimator', 'smote')]
                )
                X_transformed = pipe_transform.fit_transform(X_train, y_train)
                if not isinstance(X_transformed, pd.DataFrame):
                    X_transformed = pd.DataFrame(X_transformed, columns=imp_features)

                sample = min(500, len(X_transformed))
                X_sample = X_transformed.sample(sample, random_state=SEED)

                if hasattr(estimator, 'feature_importances_'):
                    explainer = shap.TreeExplainer(estimator)
                else:
                    explainer = shap.LinearExplainer(estimator, X_sample)
                shap_values = explainer.shap_values(X_sample)

                # For binary classification, shap_values may be a list [neg, pos]
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(sv, X_sample, feature_names=imp_features,
                                  show=False, plot_size=None)
                p = out_dir / f'shap_summary_{best_model_name}.png'
                _save_fig(plt.gcf(), p)
                mlflow.log_artifact(str(p), artifact_path='plots')
            except Exception as e:
                logger.warning('SHAP unavailable or failed: %s', e)

    # Save pipeline in MLflow
    mlflow.sklearn.log_model(final_pipeline, artifact_path='model')
    logger.info('Pipeline saved to MLflow ✓')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Holdout Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SECTION 7: Holdout Evaluation — %s', best_model_name)

# %%
y_holdout_prob = final_pipeline.predict_proba(X_holdout)[:, 1]
y_holdout_pred = (y_holdout_prob >= OPT_THRESHOLD).astype(int)

holdout_metrics = _compute_metrics(y_holdout.values, y_holdout_pred, y_holdout_prob)

logger.info('  Threshold   : %.2f', OPT_THRESHOLD)
for k, v in holdout_metrics.items():
    logger.info('  %-20s: %.4f', k, v)

# %%
with mlflow.start_run(run_name=f'holdout_{best_model_name}',
                      tags={'stage': 'holdout', 'model': best_model_name}):
    mlflow.log_params({
        'model'         : best_model_name,
        'threshold'     : OPT_THRESHOLD,
        'holdout_size'  : len(y_holdout),
        'churn_rate'    : float(y_holdout.mean()),
    })
    mlflow.log_metrics({f'holdout_{k}': v for k, v in holdout_metrics.items()})

    if artifacts_cfg.get('save_plots', True):
        p = _plot_roc_curve(y_holdout.values, y_holdout_prob,
                            f'{best_model_name}_holdout', out_dir)
        mlflow.log_artifact(str(p), artifact_path='plots/holdout')

        p = _plot_confusion_matrix(y_holdout.values, y_holdout_pred,
                                   f'{best_model_name}_holdout', out_dir)
        mlflow.log_artifact(str(p), artifact_path='plots/holdout')

        p = _plot_pr_curve(y_holdout.values, y_holdout_prob,
                           f'{best_model_name}_holdout', out_dir)
        mlflow.log_artifact(str(p), artifact_path='plots/holdout')

# %%
# CV vs Holdout comparison
comparison = pd.DataFrame({
    'metric'    : ['roc_auc', 'f1', 'recall', 'precision', 'avg_precision'],
    'cv_mean'   : [best_result['cv_roc_auc_mean'], best_result['cv_f1_mean'],
                   best_result['cv_recall_mean'], best_result['cv_precision_mean'],
                   best_result['cv_avg_precision_mean']],
    'holdout'   : [holdout_metrics['roc_auc'], holdout_metrics['f1'],
                   holdout_metrics['recall'], holdout_metrics['precision'],
                   holdout_metrics['avg_precision']],
})
comparison['gap'] = comparison['holdout'] - comparison['cv_mean']
logger.info('\n%s', comparison.round(4).to_string(index=False))
comparison

# %%
# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('═' * 60)
logger.info('FINAL SUMMARY — Bank Churn Classification')
logger.info('Best model      : %s', best_model_name)
logger.info('CV ROC-AUC     : %.4f ± %.4f', best_result['cv_roc_auc_mean'], best_result['cv_roc_auc_std'])
logger.info('Holdout ROC-AUC: %.4f', holdout_metrics['roc_auc'])
logger.info('Holdout F1     : %.4f', holdout_metrics['f1'])
logger.info('Holdout Recall : %.4f', holdout_metrics['recall'])
logger.info('Holdout Prec.  : %.4f', holdout_metrics['precision'])
logger.info('Threshold used  : %.2f', OPT_THRESHOLD)

final_df = pd.DataFrame([
    {
        'model'           : k,
        'cv_roc_auc_mean' : v['cv_roc_auc_mean'],
        'cv_roc_auc_std'  : v['cv_roc_auc_std'],
        'cv_f1_mean'      : v['cv_f1_mean'],
        'cv_recall_mean'  : v['cv_recall_mean'],
        'tuned'           : '✓' if v.get('tuned') else '—',
    }
    for k, v in all_results.items()
]).sort_values('cv_roc_auc_mean', ascending=False)

logger.info('\n%s', final_df.round(4).to_string(index=False))
final_df
