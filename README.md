# Bank Customer Churn Prediction — MLOps

MLOps project for predicting bank customer churn. Developed as a practical assignment for the MLOps course at Instituto INFNET.

The objective is to identify, in advance, customers with a high probability of ending their relationship with the bank — before explicit signs become available — allowing proactive retention actions.

---

## Project Structure

```
├── app
│   ├── __init__.py
│   ├── main.py
│   └── streamlit_app.py
├── config
│   ├── data.yaml
│   ├── modeling.yaml
│   ├── pipeline.yaml
│   ├── preprocessing.yaml
│   └── quality.yaml
├── data
│   ├── features
│   │   └── bank_churn_features.parquet
│   ├── processed
│   │   └── bank_churn.parquet
│   └── raw
│       └── Customer-Churn-Records.csv
├── docs
│   └── eda_report.md
├── eda
│   ├── __init__.py
│   └── eda_analysis.py
├── notebooks
│   ├── deploy.py
│   ├── dimensionality_reduction.py
│   ├── ingestao.py
│   ├── modelagem.py
│   ├── monitoring.py
│   ├── preprocessamento.py
│   └── qualidade.py
├── outputs
│   ├── eda
│   │   ├── plots
│   │   │   ├── 01_target_distribution.png
│   │   │   ├── 02_numeric_distributions.png
│   │   │   ├── 03_numeric_by_target_boxplot.png
│   │   │   ├── 04_correlation_heatmap.png
│   │   │   ├── 05_categorical_by_target.png
│   │   │   ├── 06_age_distribution_by_churn.png
│   │   │   ├── 07_balance_distribution.png
│   │   │   ├── 08_creditscore_by_geography.png
│   │   │   ├── 09_satisfaction_and_points.png
│   │   │   ├── 10_products_and_tenure_churn.png
│   │   │   ├── 11_churn_by_geography.png
│   │   │   ├── 12_churn_by_gender.png
│   │   │   ├── 13_churn_by_age_bins.png
│   │   │   ├── 14_cumulative_churn_by_tenure.png
│   │   │   └── 15_active_member_complain_churn.png
│   │   └── eda_report_20260418_101339.json
│   ├── modeling
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix_lightgbm.png
│   │   ├── confusion_matrix_lightgbm_holdout.png
│   │   ├── cv_fold_comparison_lightgbm.png
│   │   ├── feature_importance_lightgbm.png
│   │   ├── pr_curve_lightgbm.png
│   │   ├── pr_curve_lightgbm_holdout.png
│   │   ├── roc_curve_lightgbm.png
│   │   ├── roc_curve_lightgbm_holdout.png
│   │   ├── shap_summary_lightgbm.png
│   │   └── threshold_analysis_lightgbm.png
│   ├── models
│   │   ├── ks_drift_results.csv
│   │   ├── model_schema.json
│   │   └── pipeline_final.joblib
│   └── quality
│       ├── quality_report_20260408_200934.json
│       └── quality_report_20260413_200544.json
├── scripts
│   └── ci_cd.sh
├── src
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   └── logger.py
│   ├── __init__.py
│   ├── dowloader.py
│   ├── feature_reducer.py
│   ├── inference.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   └── quality_checks.py
├── README.md
├── mlflow.db
├── relatorio_tecnico.md
└── requirements.txt
```

---

## Dataset

**Bank Customer Churn** — Kaggle  
10,000 customers, 18 features, target variable `Exited` (churn = 1).  
Imbalance: 79.6% non-churn / 20.4% churn (ratio 3.91:1).

---

## Installation

```bash
git clone <repo>
cd Projeto_MLOps_Bank_Churn

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

For XGBoost on Mac:
```bash
brew install libomp
```

---

## Running the Complete Pipeline

```bash
# 1. Ingestion
python notebooks/ingestao.py

# 2. Data quality
python notebooks/qualidade.py

# 3. Preprocessing and feature engineering
python notebooks/preprocessamento.py

# 4. Experimentation (Optuna + MLflow — ~6h for 30 trials × 4 models)
python notebooks/modelagem.py

# 5. Dimensionality reduction experiment
python notebooks/dimensionality_reduction.py

# 6. Deploy final model
python notebooks/deploy.py

# 7. Post-deploy monitoring
python notebooks/monitoring.py
```

Or via simulated CI/CD pipeline:
```bash
chmod +x scripts/ci_cd.sh
./scripts/ci_cd.sh
```

---

## Services

### MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:////absolute/path/mlflow.db
# http://localhost:5000
```

### REST API (FastAPI)
```bash
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs
```

### Visual Interface (Streamlit)
```bash
streamlit run app/streamlit_app.py
# http://localhost:8501
```

---

## Final Model

**LightGBM** with hyperparameters optimized via Optuna (30 trials, TPE Sampler), without dimensionality reduction, with SMOTE for imbalance handling.

| Metric | Value |
|---------|-------|
| ROC-AUC (holdout) | 0.873 |
| Churn recall (holdout) | 0.816 |
| F1-score (holdout) | 0.597 |
| Decision threshold | 0.40 |

**Pipeline:**
```
DataImputer → FeatureReducer(none) → SMOTE → LGBMClassifier
```

---

## Technologies

| Category | Technologies |
|-----------|------------|
| Modeling | scikit-learn, LightGBM, XGBoost, imbalanced-learn |
| Optimization | Optuna (TPE Sampler) |
| Tracking | MLflow (SQLite backend) |
| Data Quality | Great Expectations |
| Deploy | FastAPI, Streamlit, joblib |
| Monitoring | SciPy (KS test), PSI |

---

## EDA Report

Automated report generated at `outputs/eda/eda_report_*.json` with results of statistical tests (Mann-Whitney U, ANOVA, Kruskal-Wallis, Tukey HSD, chi-squared, Cramér's V), K-Means segmentation, and 15 visualizations saved at `outputs/eda/plots/`.

---

## Technical Report

The full report is at `relatorio_tecnico.md` and covers:

- **Part 1** — Problem definition, prior experiments, and success criteria
- **Part 2** — Data pipeline, quality, and feature engineering
- **Part 3** — Systematic experimentation with 4 models and Optuna
- **Part 4** — Dimensionality reduction analysis (PCA, LDA, t-SNE)
- **Part 5** — Final model selection and justification
- **Part 6** — Deploy, monitoring, CI/CD, and retraining strategy
