# Bank Customer Churn Prediction — MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![CI](https://img.shields.io/github/actions/workflow/status/shaitonin/bank-churn-mlops-english/ci.yml?label=CI&logo=github)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end MLOps project for predicting bank customer churn. Developed as a practical assignment for the MLOps course at Instituto INFNET.

The objective is to identify, in advance, customers with a high probability of ending their relationship with the bank — before explicit signs become available — allowing proactive retention actions.

> **Live demo →** *(coming soon — link will be added after Streamlit Cloud deploy)*

---

## Final Model Results

**LightGBM** with hyperparameters optimized via Optuna (30 trials, TPE Sampler), without dimensionality reduction, with SMOTE for class imbalance handling.

| Metric | Value |
|--------|-------|
| ROC-AUC (holdout) | **0.873** |
| Churn Recall (holdout) | **0.816** |
| F1-score (holdout) | 0.597 |
| Decision threshold | 0.40 |

**Pipeline:**
```
DataImputer → FeatureReducer(none) → SMOTE → LGBMClassifier
```

---

## Project Structure

```
├── app/
│   ├── main.py                   # FastAPI REST API
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── config/
│   ├── data.yaml
│   ├── modeling.yaml
│   ├── pipeline.yaml
│   ├── preprocessing.yaml
│   └── quality.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── docs/
│   └── eda_report.md
├── eda/
│   └── eda_analysis.py           # Full EDA engine (1 100+ lines)
├── notebooks/                    # Pipeline stages (run in order)
│   ├── ingestion_pipeline.py
│   ├── quality_pipeline.py
│   ├── preprocessing_pipeline.py
│   ├── modeling.py
│   ├── dimensionality_reduction.py
│   ├── deploy.py
│   └── monitoring.py
├── outputs/
│   ├── eda/plots/
│   ├── modeling/
│   └── models/
│       ├── pipeline_final.joblib
│       └── model_schema.json
├── scripts/
│   └── ci_cd.sh
├── src/
│   ├── downloader.py
│   ├── features.py               # Shared feature engineering
│   ├── feature_reducer.py
│   ├── inference.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── quality_checks.py
│   └── utils/
├── tests/
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_preprocessing.py
├── .github/workflows/ci.yml      # GitHub Actions CI
├── .streamlit/config.toml
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
└── technical_report.md
```

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
git clone <repo>
cd bank-churn-mlops-english
docker compose up
```

- Streamlit dashboard: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

### Option 2 — Local

```bash
git clone <repo>
cd bank-churn-mlops-english

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

For XGBoost on macOS:
```bash
brew install libomp
```

---

## Running the Complete Pipeline

```bash
# 1. Data ingestion (requires Kaggle credentials in secrets.env)
python notebooks/ingestion_pipeline.py

# 2. Data quality checks
python notebooks/quality_pipeline.py

# 3. Preprocessing and feature engineering
python notebooks/preprocessing_pipeline.py

# 4. Experimentation (Optuna + MLflow — ~6h for 30 trials × 4 models)
python notebooks/modeling.py

# 5. Dimensionality reduction experiment
python notebooks/dimensionality_reduction.py

# 6. Deploy final model to MLflow registry
python notebooks/deploy.py

# 7. Post-deploy monitoring
python notebooks/monitoring.py
```

Or via the CI/CD pipeline script:
```bash
chmod +x scripts/ci_cd.sh
./scripts/ci_cd.sh
```

---

## Running the Tests

```bash
pytest tests/ -v --tb=short
```

---

## Services

### MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
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

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app/streamlit_app.py`
4. Click **Deploy** — no secrets needed (the model file is included in the repo)

---

## Dataset

**Bank Customer Churn** — Kaggle
10,000 customers · 18 features · target variable `Exited` (churn = 1)
Class imbalance: 79.6% non-churn / 20.4% churn (ratio 3.91:1)

---

## Technologies

| Category | Technologies |
|----------|-------------|
| Modeling | scikit-learn, LightGBM, XGBoost, imbalanced-learn (SMOTE) |
| Optimization | Optuna (TPE Sampler, 30 trials) |
| Tracking | MLflow (SQLite backend) |
| Interpretability | SHAP |
| Data Quality | Great Expectations |
| Deploy | FastAPI, Streamlit, joblib |
| Monitoring | SciPy (KS test), PSI |
| Containers | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## EDA Report

Automated statistical report at `outputs/eda/eda_report_*.json` covering:
Mann-Whitney U, ANOVA, Kruskal-Wallis, Tukey HSD, chi-squared, Cramér's V, K-Means segmentation, and 15 visualizations at `outputs/eda/plots/`.

---

## Technical Report

Full report at `technical_report.md`:

- **Part 1** — Problem definition, prior experiments, and success criteria
- **Part 2** — Data pipeline, quality diagnosis, and feature engineering
- **Part 3** — Systematic experimentation with 4 models and Optuna
- **Part 4** — Dimensionality reduction analysis (PCA, LDA)
- **Part 5** — Final model selection and justification
- **Part 6** — Deploy, monitoring, CI/CD, and retraining strategy
