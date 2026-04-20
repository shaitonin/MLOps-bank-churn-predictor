# Bank Customer Churn Prediction — MLOps

Projeto de MLOps para predição de churn de clientes bancários. Desenvolvido como trabalho prático da disciplina de MLOps do Instituto INFNET.

O objetivo é identificar, com antecedência, clientes com alta probabilidade de encerrar seu relacionamento com o banco — antes que sinais explícitos se tornem disponíveis — permitindo ações proativas de retenção.

---

## Estrutura do Projeto

```
├── app/
│   ├── main.py               # API REST de inferência (FastAPI)
│   └── streamlit_app.py      # Interface visual interativa (Streamlit)
├── config/
│   ├── modeling.yaml         # Configuração de modelos, Optuna, MLflow, pipeline
│   └── pipeline.yaml         # Configuração de ingestão de dados
├── data/
│   ├── raw/                  # Dados brutos (CSV do Kaggle)
│   ├── processed/            # Dados processados (Parquet)
│   └── features/             # Features engineered (Parquet)
├── docs/
│   └── relatorio_tecnico.md  # Relatório técnico completo (Partes 1–6)
├── notebooks/
│   ├── ingestao.py           # Ingestão e conversão para Parquet
│   ├── qualidade.py          # Validação de qualidade (Great Expectations)
│   ├── preprocessamento.py   # Feature engineering
│   ├── modelagem.py          # Experimentação com Optuna + MLflow
│   ├── dimensionality_reduction.py  # Experimento de redução de dimensionalidade
│   ├── deploy.py             # Treino final + registro no MLflow Registry
│   └── monitoring.py         # Detecção de drift e monitoramento pós-deploy
├── scripts/
│   └── ci_cd.sh              # Pipeline CI/CD simulado
├── src/
│   ├── preprocessing.py      # DataImputer, scalers
│   ├── feature_reducer.py    # FeatureReducer (none | pca | lda | rfe | kpca)
│   ├── inference.py          # ChurnPredictor — carrega modelo e executa predições
│   └── utils/                # Logger e config loader
├── outputs/
│   ├── models/               # Pipeline serializado (joblib) + schema JSON
│   └── eda/                  # Relatório EDA em JSON
└── requirements.txt
```

---

## Dataset

**Bank Customer Churn** — Kaggle  
10.000 clientes, 18 features, variável-alvo `Exited` (churn = 1).  
Desbalanceamento: 79,6% não-churn / 20,4% churn (razão 3,91:1).

---

## Instalação

```bash
git clone <repo>
cd Projeto_MLOps_Bank_Churn

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Para XGBoost no Mac:
```bash
brew install libomp
```

---

## Execução do Pipeline Completo

```bash
# 1. Ingestão
python notebooks/ingestao.py

# 2. Qualidade dos dados
python notebooks/qualidade.py

# 3. Pré-processamento e feature engineering
python notebooks/preprocessamento.py

# 4. Experimentação (Optuna + MLflow — ~6h para 30 trials × 4 modelos)
python notebooks/modelagem.py

# 5. Experimento de redução de dimensionalidade
python notebooks/dimensionality_reduction.py

# 6. Deploy do modelo final
python notebooks/deploy.py

# 7. Monitoramento pós-deploy
python notebooks/monitoring.py
```

Ou via pipeline CI/CD simulado:
```bash
chmod +x scripts/ci_cd.sh
./scripts/ci_cd.sh
```

---

## Serviços

### MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:////caminho/absoluto/mlflow.db
# http://localhost:5000
```

### API REST (FastAPI)
```bash
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs
```

### Interface Visual (Streamlit)
```bash
streamlit run app/streamlit_app.py
# http://localhost:8501
```

---

## Modelo Final

**LightGBM** com hiperparâmetros otimizados via Optuna (30 trials, TPE Sampler), sem redução de dimensionalidade, com SMOTE para tratamento do desbalanceamento.

| Métrica | Valor |
|---------|-------|
| ROC-AUC (holdout) | 0,873 |
| Recall churn (holdout) | 0,816 |
| F1-score (holdout) | 0,597 |
| Threshold de decisão | 0,40 |

**Pipeline:**
```
DataImputer → FeatureReducer(none) → SMOTE → LGBMClassifier
```

---

## Tecnologias

| Categoria | Tecnologias |
|-----------|------------|
| Modelagem | scikit-learn, LightGBM, XGBoost, imbalanced-learn |
| Otimização | Optuna (TPE Sampler) |
| Rastreamento | MLflow (SQLite backend) |
| Qualidade de dados | Great Expectations |
| Deploy | FastAPI, Streamlit, joblib |
| Monitoramento | SciPy (KS test), PSI |

---

## EDA Report

Relatório automatizado gerado em `outputs/eda/eda_report_*.json` com resultados de testes estatísticos (Mann-Whitney U, ANOVA, Kruskal-Wallis, Tukey HSD, chi-quadrado, Cramér's V), segmentação via K-Means e 15 visualizações salvas em `outputs/eda/plots/`.

---

## Relatório Técnico

O relatório completo está em `docs/relatorio_tecnico.md` e cobre:

- **Parte 1** — Definição do problema, experimentos anteriores e critérios de sucesso
- **Parte 2** — Pipeline de dados, qualidade e feature engineering
- **Parte 3** — Experimentação sistemática com 4 modelos e Optuna
- **Parte 4** — Análise de redução de dimensionalidade (PCA, LDA, t-SNE)
- **Parte 5** — Seleção e justificativa do modelo final
- **Parte 6** — Deploy, monitoramento, CI/CD e estratégia de re-treinamento
