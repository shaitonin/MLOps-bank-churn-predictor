## Parte 1 — Estruturação do Projeto de Machine Learning

### 1.1 Definição do Objetivo Técnico

O objetivo central deste projeto é construir um sistema de classificação binária capaz de identificar, com antecedência, clientes com alta probabilidade de encerrar seu relacionamento com o banco (*churn*), antes que sinais explícitos — como reclamações formais — se tornem disponíveis.

A variável-alvo é `Exited` (0 = cliente retido, 1 = cliente que saiu), com distribuição observada de 79,62% negativos e 20,38% positivos, caracterizando um problema de classes desbalanceadas com razão aproximada de 3,91:1.


### 1.2 Mapeamento dos Experimentos

Cinco experimentos foram conduzidos na fase exploratória (projeto Scikit-Learn), variando arquitetura do modelo, presença da coluna `Complain` e estratégias de regularização e otimização de hiperparâmetros:

| Experimento | Modelo                               | Complain | Principais configurações                                                      | Resultado principal (teste)                        |
|-------------|--------------------------------------|----------|-------------------------------------------------------------------------------|----------------------------------------------------|
| E1          | Perceptron                           | Sim      | StandardScaler, Yeo-Johnson em Balance, OHE em Geography/Gender               | Accuracy ≈ 1,00 — inflacionado por data leakage    |
| E2          | Perceptron                           | Não      | Mesmas transformações; Complain removida                                      | Accuracy ≈ 0,76; F1 classe churn = 0,36            |
| E3          | Árvore de Decisão sem regularização  | Não      | Parâmetros padrão do DecisionTreeClassifier; sem restrição de profundidade    | Accuracy treino 1,00 vs. teste 0,80 — overfitting severo (profundidade 26, 928 folhas) |
| E4          | Árvore de Decisão com regularização  | Não      | Grid Search + StratifiedKFold (10 folds); critério gini, `max_depth = 5`      | Accuracy treino e teste ≈ 0,86; 30 folhas; precisão média CV = 0,7696 (IC 95%: [0,7267–0,8126]) |
| E5          | Random Forest (ensemble)             | Não      | Grid Search; `n_estimators = 100`, `max_depth = 5`, `min_samples_split = 5`, `min_samples_leaf = 1` | Accuracy treino e teste ≈ 0,86; F1 classe churn = 0,51; sem overfitting |

**E1 — Perceptron com Complain:** serviu como controle negativo. A accuracy próxima de 1,00 — incomum para um classificador linear simples — evidenciou que o modelo estava aprendendo a correlação quase perfeita entre `Complain` e `Exited` (Pearson = 0,9957), e não padrões genuínos de comportamento do cliente.

**E2 — Perceptron sem Complain:** ao remover a variável contaminada, o desempenho caiu significativamente. A accuracy no conjunto de teste foi de 0,76, com F1-score de 0,36 para a classe churn — linha de base honesta que reflete a dificuldade real do problema.

**E3 — Árvore de Decisão sem regularização:** treinada com os parâmetros padrão do `DecisionTreeClassifier`, sem nenhuma restrição de profundidade ou número de folhas. A árvore atingiu profundidade 26 e 928 folhas, memorizando completamente os dados de treino (accuracy = 1,00) sem generalizar para o conjunto de teste (accuracy ≈ 0,80). Este experimento demonstrou empiricamente o risco de overfitting em árvores sem controle de complexidade.

**E4 — Árvore de Decisão com regularização:** aplicação de Grid Search com validação cruzada estratificada de 10 folds para busca sistemática de hiperparâmetros. A melhor configuração encontrada utilizou critério gini e profundidade máxima de 5 níveis, resultando em apenas 30 folhas. A regularização por controle de profundidade eliminou o overfitting: treino e teste convergem para accuracy ≈ 0,86, com precisão média na validação cruzada de 0,7696 e coeficiente de variação de 9%.

**E5 — Random Forest:** método ensemble que combina 100 árvores com profundidade máxima 5, treinadas sobre subamostras diferentes dos dados. A combinação das previsões reduz a variância em relação a uma árvore individual, mantendo accuracy ≈ 0,86 tanto em treino quanto em teste — sem sinais de overfitting. A análise de importância das features confirmou `Age` (0,385) e `NumOfProducts` (0,318) como variáveis mais influentes, alinhando-se com os achados da EDA. A limitação persistente foi o recall baixo da classe churn (0,36 no teste), indicando que o desbalanceamento de classes (3,91:1) não foi tratado nesta fase.

### 1.3 Critérios de Sucesso

A escolha das métricas parte de uma assimetria de custo de negócio: um falso negativo (cliente que vai sair e não foi identificado) tem custo maior do que um falso positivo (estratégia de retenção para um cliente que não vai sair). Favorecendo recall sobre precisão.

**Critérios quantitativos definidos:**

- **ROC-AUC ≥ 0,85** — discriminação geral entre classes
- **Recall (classe churn) ≥ 0,70** — detectar ao menos 70% dos churns reais
- **F1-Score ≥ 0,60** — equilíbrio mínimo entre precisão e recall
- **Threshold ajustado para 0,40** (padrão 0,50) — desloca a fronteira de decisão para maximizar recall da classe minoritária

**Métricas de negócio derivadas:**

- Taxa de retenção incremental: clientes identificados corretamente e abordados com ação de retenção
- Custo de intervenção por cliente abordado vs. receita preservada por cliente retido

### 1.4 Análise Exploratória — Principais Achados de Engenharia

A EDA identificou padrões que guiaram diretamente as decisões de feature engineering implementadas no pipeline:

**Variáveis com maior poder discriminante (confirmadas por testes estatísticos):**

- `Age` — Cohen's d = 0,747; clientes que saíram têm média de 44,8 anos vs. 37,4 dos que ficaram
- `IsActiveMember` — clientes inativos com churn 2,6× maior que ativos
- `NumOfProducts` — relação não-linear: 3–4 produtos → churn de 82,7% a 100%
- `Balance` — distribuição bimodal; 36,17% com saldo zero (valor legítimo, não erro)
- `Geography` — Germany com churn 2× maior (Cramér's V = 0,17)
- `Gender` — mulheres com 25,1% de churn vs. 16,5% dos homens (Cramér's V = 0,11)

**Variáveis sem associação linear significativa (p > 0,05):**

- `HasCrCard` (p = 0,485), `EstimatedSalary` (p = 0,211), `Satisfaction Score` (p = 0,559), `Point Earned` (p = 0,644) — mantidas no pipeline pois modelos não-lineares podem extrair valor via interações; SHAP values validarão antes de exclusão definitiva.

---

## Parte 2 — Fundação de Dados e Diagnóstico Inicial

### 2.1 Pipeline de Ingestão

O pipeline de ingestão foi construído seguindo o princípio MLOps de separação entre política (configuração em YAML) e mecanismo (código Python genérico). O arquivo `config/data.yaml` controla todas as decisões de ingestão; o módulo `src/ingest.py` executa sem nenhuma lógica hardcoded.

**Fluxo de dados:**

```
Kaggle API → CSV bruto → data/raw/
           → validação de schema → blocos de 5.000 linhas
           → conversão de tipos (convert_dtypes=True)
           → data/processed/bank_churn.parquet (compressão Snappy)
```

A escolha do formato Parquet — em vez de CSV — reduz o tempo de leitura em leituras subsequentes, preserva os tipos de dados corretamente e comprime o arquivo em aproximadamente 60% do tamanho original. O tamanho de bloco de 5.000 linhas foi calibrado para acomodar datasets maiores (até 50.000 linhas) sem pressão de memória.

A validação de schema verifica a presença obrigatória de todas as 18 colunas originais, lançando erro imediatamente se qualquer coluna estiver ausente — comportamento *fail-fast* que evita erros silenciosos no restante do pipeline.

### 2.2 Diagnóstico de Qualidade (Great Expectations)

O módulo `src/quality_checks.py` executa validações configuradas em `config/quality.yaml`, seguindo o padrão Great Expectations com *expectations* de tabela e de coluna. Todos os parâmetros foram derivados dos dados reais (EDA Seção 9) usando percentis robustos (p01–p99) com tolerâncias de ±2σ.

**Anomalias identificadas no dataset:**

**Outliers em `CreditScore`:** 15 casos com valores abaixo de 400, fora do range operacional de qualquer score de crédito convencional (350–850). Foram mantidos no dataset após investigação — a EDA confirmou que não afetam a distribuição global (std = 96,7).

**Distribuição bimodal em `Balance`:** 36,17% dos clientes possuem saldo exatamente zero. Isso representa um comportamento legítimo enqunanto existem clientes que usam o banco apenas para outros serviços.

**Concentração etária elevada em `Age`:** 359 clientes com idade acima de 60 anos, na cauda direita da distribuição assimétrica (skew = +1,01). Estes clientes coincidem com o grupo de maior taxa de churn, tornando-os relevantes para o modelo — não foram removidos como outliers.

**`NumOfProducts` com valor 4:** 60 clientes contrataram 4 produtos, categoria com churn de 100% na amostra. O range válido foi confirmado como 1–4; estes casos não são erros mas sim o sinal preditivo mais forte do dataset.

### 2.3 Inconsistências Estruturais do Dataset

**Distribuição artificial de `Card Type`:** A coluna apresenta distribuição perfeitamente uniforme entre as quatro categorias (DIAMOND, GOLD, SILVER, PLATINUM), com χ² p = 0,168 e Cramér's V = 0,02 — evidência de dados sintéticos sem associação real com churn. Adicionalmente, `Card Type` contradiz `HasCrCard`: 1.151 clientes sem cartão de crédito (`HasCrCard = 0`) possuem um tipo de cartão atribuído — inconsistência estrutural que confirma a origem sintética. `Card Type` foi removida do pipeline.

**Data leakage em `Complain`:** A correlação de Pearson de 0,9957 entre `Complain` e `Exited` indica que a reclamação formal foi registrada *após* ou *simultaneamente* ao churn, não antes. Em produção, esta informação não estaria disponível no momento da predição. A coluna foi excluída via `ColumnDropper` na primeira etapa do pipeline de pré-processamento, antes de qualquer transformação.

### 2.4 Desbalanceamento de Classes

A razão de 3,91:1 entre clientes retidos e churned exige tratamento explícito para evitar que o modelo aprenda a classificar tudo como negativo (acurácia aparente de 79,62% sem nenhum poder preditivo real). As estratégias de mitigação adotadas estão detalhadas na Seção 3.2.

### 2.5 Impacto na Generalização e Limitações Estruturais

**Origem sintética dos dados:** O dataset foi gerado artificialmente para fins educacionais. Isso implica ausência de correlações espúrias reais presentes em dados de produção (sazonalidade, efeitos macroeconômicos, comportamento de coorte), distribuições perfeitamente uniformes em algumas colunas (`Card Type`, `Tenure`, `EstimatedSalary`) e ausência de ruído de coleta. Modelos treinados neste dataset tendem a superestimar performance quando comparados a datasets reais.

**Ausência de dimensão temporal:** O dataset é um corte transversal (*cross-section*), sem timestamps. Em produção, churn é um fenômeno temporal — a taxa de churn em dezembro pode ser diferente de março. O pipeline atual não implementa features temporais (rolling averages, time-since-last-transaction); esta é a principal limitação para generalização em ambiente real.

**Recomendação:** Para uso em produção, o pipeline deve ser re-calibrado com dados reais, incluindo monitoramento de *data drift* nas distribuições de entrada e *concept drift* na relação entre features e target.

---

### 2.6 Transformações e Técnicas Aplicadas

Esta seção descreve todas as transformações aplicadas ao longo do projeto, organizadas em duas camadas: **pré-processamento** (stateless — executado sobre o dataset completo, antes do split) e **modelagem** (stateful — executado apenas sobre os dados de treino, após o split).

#### 2.6.1 Pré-processamento Stateless

O pré-processamento stateless reúne todas as transformações que não aprendem parâmetros dos dados — ou seja, podem ser aplicadas com segurança sobre o dataset inteiro sem risco de data leakage. Cada transformação é controlada por `config/preprocessing.yaml` e implementada como um transformador Scikit-Learn (`BaseEstimator + TransformerMixin`) em `src/preprocessing.py`.

#### Remoção de Colunas (`ColumnDropper`)

As colunas `RowNumber`, `CustomerId`, `Surname` e `Complain` são removidas na primeira etapa do pipeline. As três primeiras não possuem valor preditivo — são apenas identificadores de registro. `Complain` é removida por data leakage severo (correlação de Pearson = 0,9957 com `Exited`): em produção, a reclamação formal não estaria disponível antes do churn ocorrer.

#### Flags Binárias (`BinaryFlagTransformer`)

Criação de duas novas colunas binárias derivadas de variáveis existentes, com base em padrões identificados na EDA:

- **`HasZeroBalance`** (`Balance == 0`): sinaliza os 36,17% de clientes com saldo zero — grupo com comportamento distinto do restante.
- **`HighRiskProducts`** (`NumOfProducts >= 3`): sinaliza clientes com 3 ou mais produtos, faixa em que o churn atinge 82,7% a 100%.

#### Features de Interação (`InteractionFeatureTransformer`)

Criação de features compostas que capturam sinais combinados identificados na EDA:

- **`AgeInactivity`** = `Age × (1 − IsActiveMember)`: amplifica o sinal de desengajamento em clientes mais velhos. Como inatividade e idade avançada são os dois maiores preditores isolados de churn, sua combinação cria um preditor composto mais forte.
- **`EngagementScore`** = `IsActiveMember + (NumOfProducts == 2) + HasCrCard − HasZeroBalance`: score de engajamento no intervalo −1 a 3. Valores altos indicam clientes mais engajados e com menor risco de churn.

#### Feature de Razão (`RatioFeatureTransformer`)

- **`BalanceSalaryRatio`** = `Balance / (EstimatedSalary + 1)`: razão entre saldo e renda estimada. O `+1` no denominador protege contra divisão por zero. Clientes com saldo alto relativo ao salário são os mais disputados pela concorrência bancária e apresentam maior risco de churn.

#### Discretização Etária (`AgeBinTransformer`)

- **`AgeGroup`**: a variável contínua `Age` é discretizada em 5 faixas etárias ([< 30, 30–40, 40–50, 50–60, 60+]) e convertida para encoding ordinal (0 a 4). Captura a relação não-linear entre idade e churn identificada na EDA: clientes entre 40 e 60 anos concentram a maior propensão à saída.

#### Encoding de Variáveis Categóricas (`CategoricalEncoder`)

- **`Geography`** → One-Hot Encoding com prefixo `geo`, gerando `geo_France`, `geo_Germany` e `geo_Spain`. Cada país é tratado como preditor independente, preservando a diferença de churn 2× maior em Germany.
- **`Gender`** → Binary encoding: Male = 0, Female = 1. Método compacto para variável binária.
- **`Card Type`** → removida do pipeline MLOps (χ² p = 0,168, Cramér's V = 0,02 — sem associação com churn). No projeto Scikit-Learn havia sido codificada ordinalmente (Silver=0, Gold=1, Platinum=2, Diamond=3), mas a análise do Random Forest confirmou importância de 0,0021 — irrelevante.

#### Seleção de Features (`FeatureSelector`)

Etapa final do pré-processamento: seleciona o conjunto definitivo de 21 features mais a variável-alvo `Exited`, descartando colunas intermediárias e originais que foram substituídas por suas versões codificadas (ex: `Geography` e `Gender` originais são descartadas após o encoding).

---

#### 2.6.2 Técnicas de Modelagem Stateful

As técnicas stateful aprendem parâmetros a partir dos dados de treino e só podem ser aplicadas após o split treino/teste. Aplicá-las antes causaria data leakage.

#### Imputação de Dados Ausentes (`DataImputer`)

Estratégias definidas por coluna para tratar valores nulos que possam surgir em produção:

| Coluna            | Estratégia              | Justificativa                                                  |
|-------------------|-------------------------|----------------------------------------------------------------|
| `CreditScore`     | Mediana por Geography   | Score de crédito varia por região geográfica                   |
| `Age`             | Mediana global (37)     | Distribuição assimétrica — mediana mais robusta que média      |
| `Balance`         | Constante zero          | 36% dos clientes têm saldo zero — valor legítimo               |
| `EstimatedSalary` | Mediana global (100.194)| Distribuição uniforme                                          |
| `NumOfProducts`   | Moda (1)                | Valor mais comum                                               |
| `Tenure`          | Mediana global (5)      | Distribuição uniforme                                          |
| `Geography`       | Moda                    | Valor mais frequente                                           |
| `Gender`          | Moda                    | Valor mais frequente                                           |

#### Escalonamento (`StandardScaler` e `RobustScaler`)

O escalonamento é aplicado **somente dentro do pipeline de modelagem, após o split treino/teste**, e **exclusivamente para a Regressão Logística**. Random Forest e XGBoost são baseados em árvore — fazem divisões binárias nos dados e são invariantes à escala das features. Aplicar escalonamento antes do split causaria data leakage: os parâmetros (média, mediana, IQR) seriam calculados sobre o dataset inteiro, incluindo os dados de teste.

A EDA recomendou técnicas diferentes por coluna com base na distribuição de cada variável (Seção 8.1):

| Variável | Técnica | Justificativa EDA |
|----------|---------|-------------------|
| `CreditScore` | `StandardScaler` | Distribuição quase normal (skew = −0,07), sem outliers extremos |
| `EstimatedSalary` | `StandardScaler` | Distribuição uniforme, sem outliers |
| `Point Earned` | `StandardScaler` | Distribuição uniforme, sem outliers |
| `Satisfaction Score` | `StandardScaler` | Discreta 1–5, sem outliers |
| `Age` | `RobustScaler` + log transform | Assimétrica (skew = +1,01), outliers válidos de clientes idosos |
| `Balance` | `RobustScaler` | Bimodal com 36% de zeros — média e std seriam distorcidos |

**Por que `RobustScaler` para `Age` e `Balance`?**

O `StandardScaler` calcula média e desvio padrão, que são sensíveis a outliers e distribuições assimétricas. O `RobustScaler` usa **mediana e IQR** (intervalo interquartil = Q75 − Q25), que são estatísticas resistentes a valores extremos. Para `Age`, a cauda longa de clientes idosos distorceria o desvio padrão; para `Balance`, a concentração de zeros distorceria a média.

**Log transform em `Age`:**

Antes do `RobustScaler`, aplica-se `np.log1p(Age)` — transformação logarítmica que comprime a cauda direita e aproxima a distribuição da normalidade.

**Variáveis não escalonadas** (binárias, ordinais e encodings — já em escala reduzida por natureza): `HasCrCard`, `IsActiveMember`, `HasZeroBalance`, `HighRiskProducts`, `AgeGroup`, `NumOfProducts`, `Tenure`, `AgeInactivity`, `EngagementScore`, `BalanceSalaryRatio`, `geo_France`, `geo_Germany`, `geo_Spain`, `Gender_encoded`.

O pipeline completo — imputer + scalers + modelo — é registrado como um único artefato no MLflow, garantindo que em produção o escalonamento seja aplicado automaticamente sem etapas manuais separadas.

#### Transformação Yeo-Johnson (`Balance`) — projeto anterior

Aplicada no projeto Scikit-Learn em complemento ao StandardScaler para tratar a assimetria de `Balance`. No pipeline MLOps atual foi substituída por duas abordagens mais adequadas: a flag `HasZeroBalance` (stateless, no pré-processamento) e o `RobustScaler` (stateful, no pipeline de modelagem da Regressão Logística).

#### Tratamento de Desbalanceamento de Classes

Três estratégias configuradas para lidar com a razão de 3,91:1 entre classes:

- **`scale_pos_weight = 3,91`**: parâmetro nativo do XGBoost e LightGBM que atribui peso maior aos erros na classe churn durante o treinamento. Sem custo computacional adicional.
- **SMOTE** (`k_neighbors = 5`): gera amostras sintéticas da classe minoritária interpolando entre exemplos reais de churn. Aplicado exclusivamente nos dados de treino, após o split.
- **Ajuste de threshold para 0,40**: reduz o limiar de decisão de 0,50 para 0,40, fazendo o modelo classificar como churn casos com probabilidade acima de 40%. Aumenta o recall da classe churn ao custo de mais falsos positivos — tradeoff justificado pela assimetria de custos do negócio.

#### Otimização de Hiperparâmetros (Grid Search + Validação Cruzada)

Busca sistemática de hiperparâmetros avaliando todas as combinações de um grid pré-definido. Combinada com `StratifiedKFold` de 10 folds, que garante que cada subconjunto de validação mantém a proporção original de classes (80/20). Aplicada nos experimentos E4 e E5:

- **E4 (Árvore de Decisão):** grid sobre `criterion` (gini, entropy) e `max_depth` (3, 5, 7, 10). Melhor configuração: gini + max_depth = 5.
- **E5 (Random Forest):** grid sobre `n_estimators`, `max_depth`, `min_samples_split` e `min_samples_leaf`. Melhor configuração: 100 árvores, max_depth = 5, min_samples_split = 5, min_samples_leaf = 1.

---

## Parte 3 — Experimentação Sistemática de Modelos

### 3.1 Abordagem Experimental

Os experimentos do projeto anterior (E1–E5) estabeleceram uma linha de base com accuracy de 0,86 e F1 churn de 0,51 (Random Forest, sem tratamento de desbalanceamento). Esta fase amplia a experimentação com quatro candidatos, tratamento explícito do desbalanceamento via SMOTE e otimização bayesiana de hiperparâmetros com Optuna, com todos os experimentos rastreados no MLflow.

**Modelos candidatos selecionados:**

| Modelo | Paradigma | Justificativa |
|--------|-----------|---------------|
| `LogisticRegression` | Linear | Baseline interpretável; único que requer escalonamento |
| `RandomForestClassifier` | Bagging | Já testado no projeto anterior (E5); referência de comparação |
| `XGBClassifier` | Boosting depth-wise | Padrão da indústria; `scale_pos_weight=3.91` para desbalanceamento |
| `LGBMClassifier` | Boosting leaf-wise | Crescimento diferente do XGBoost; `is_unbalance=True` nativo |

**Critério de seleção:** ROC-AUC como métrica primária (robusta a desbalanceamento), complementada por F1 e Recall da classe churn.

---

### 3.2 Pipeline End-to-End (scikit-learn + imbalanced-learn)

O pipeline foi construído com `imblearn.Pipeline`, garantindo que cada transformação stateful seja ajustada exclusivamente nos dados de treino de cada fold de CV — sem data leakage:

```
DataImputer → (StandardScaler + RobustScaler¹) → FeatureReducer → SMOTE → Estimador
```

¹ Escalonamento aplicado apenas à Regressão Logística. Random Forest, XGBoost e LightGBM são invariantes à escala das features (baseados em divisões binárias).

**Componentes do pipeline:**

- **`DataImputer`**: imputa valores ausentes por coluna — mediana por grupo (`CreditScore` por `Geography`), constante zero (`Balance`), mediana global (`Age`, `Tenure`, `EstimatedSalary`) e moda (`NumOfProducts`).
- **`StandardScalerTransformer`**: z-score em `CreditScore`, `EstimatedSalary`, `Point Earned`, `Satisfaction Score` (distribuições normais/uniformes).
- **`RobustScalerTransformer`**: escalonamento por mediana/IQR em `Age` (skew=+1,01, com log1p prévio) e `Balance` (bimodal com 36% de zeros).
- **`FeatureReducer`**: suporta `none | rfe | pca | kpca` — configurável via `modeling.yaml`.
- **`SMOTE`**: oversampling sintético da classe minoritária (`k_neighbors=5`) aplicado apenas ao fold de treino.

---

### 3.3 Validação Cruzada e Busca de Hiperparâmetros

**Divisão dos dados:**
- Holdout: 20% separado antes de qualquer treino (`stratify=True` — mantém proporção 20,4% churn)
- CV: `StratifiedKFold` com 5 folds sobre o conjunto de treino (8.000 amostras)

**Threshold ajustado:** 0,40 (padrão sklearn: 0,50). A redução favorece recall da classe churn.

**Optuna — otimização bayesiana (TPE Sampler):**
- 10 trials por modelo nesta execução (configurável em `modeling.yaml → optuna.default_trials`)
- O Optuna co-otimiza simultaneamente os hiperparâmetros do estimador e o método de redução de dimensionalidade (`none | rfe | pca | kpca`)
- Trials com combinações inválidas (ex: `penalty=l1` + `solver=lbfgs`) são ignorados automaticamente

---

### 3.4 Resultados Experimentais

#### Baseline — Parâmetros Padrão

| Modelo | ROC-AUC | ± std | F1 | Recall | Avg Precision |
|--------|---------|-------|-----|--------|---------------|
| LightGBM | **0,8499** | 0,0058 | 0,5937 | 0,5448 | 0,6784 |
| XGBoost | 0,8465 | 0,0089 | 0,5730 | 0,7380 | 0,6700 |
| Random Forest | 0,8456 | 0,0080 | 0,5971 | 0,5969 | 0,6464 |
| Logistic Regression | 0,8252 | 0,0075 | 0,5761 | 0,6380 | 0,6304 |

#### Após Optuna (30 trials/modelo, 5-fold CV)

| Modelo | ROC-AUC baseline | ROC-AUC otimizado | ± std | Melhora | F1 | Recall | Melhor configuração (seleção) |
|--------|-----------------|-------------------|-------|---------|-----|--------|-------------------------------|
| **LightGBM** | 0,8499 | **0,8623** | 0,0076 | +0,0124 | 0,5882 | 0,4877 | RFE (16 feat.), `n_est=528`, `num_leaves=30`, `lr=0,014` |
| XGBoost | 0,8465 | 0,8591 | 0,0092 | +0,0126 | 0,5931 | 0,7693 | RFE (16 feat.), `n_est=164`, `max_depth=6`, `lr=0,023` |
| Random Forest | 0,8456 | 0,8562 | 0,0071 | +0,0106 | 0,6100 | 0,6362 | PCA (18 comp.), `n_est=282`, `max_depth=9` |
| Logistic Regression | 0,8252 | 0,8358 | 0,0088 | +0,0106 | 0,5703 | 0,7362 | PCA (18 comp.), `C=0,258`, `penalty=l1` |

> **Observações sobre a busca de hiperparâmetros:**
> - **Redução de dimensionalidade:** LightGBM e XGBoost convergiram para RFE com 16 features; Random Forest e Logistic Regression selecionaram PCA com 18 componentes. Isso sugere que os modelos de boosting se beneficiam mais da seleção explícita de features originais (interpretabilidade + remoção de ruído), enquanto os modelos lineares/baseados em árvores rasas aproveitam a compressão do espaço via PCA.
> - **Trade-off recall × ROC-AUC:** XGBoost e Logistic Regression maximizaram recall (>0,73) mas com menor precisão; LightGBM maximizou ROC-AUC (0,8623) com recall mais conservador (0,49 em CV). No holdout, com threshold 0,40, o recall do LightGBM sobe para 0,61.
> - **Tempo de otimização:** LightGBM ~2h, XGBoost ~3h, Random Forest ~1,4h, Logistic Regression ~5min.

#### Avaliação no Holdout — Melhor Modelo (LightGBM)

| Métrica | Valor |
|---------|-------|
| ROC-AUC | **0,8770** |
| F1 (churn) | 0,6417 |
| Recall (churn) | 0,6078 |
| Precision (churn) | 0,6795 |
| Avg Precision | 0,7312 |
| Threshold | 0,40 |

O holdout (ROC-AUC = 0,8770) superou a CV (0,8623 ± 0,0076), indicando boa generalização sem overfitting.

---

### 3.5 Análise Comparativa e Decisão Técnica

**LightGBM foi selecionado como modelo principal** pelos seguintes critérios:

| Critério | LightGBM | XGBoost | Random Forest | Logistic Reg. |
|----------|----------|---------|---------------|---------------|
| ROC-AUC (holdout) | **0,8770** | — | — | — |
| ROC-AUC (CV otimizado) | **0,8623** | 0,8591 | 0,8562 | 0,8358 |
| Desempenho preditivo | ✓ Melhor | ✓ 2º | ✓ 3º | ✗ Mais baixo |
| Custo computacional | ✓ Médio (~2h/30 trials) | ✗ Mais alto (~3h/30 trials) | ✓ Médio (~1,4h) | ✓ Muito baixo (~5min) |
| Tratamento de desbalanceamento | ✓ Nativo (`is_unbalance`) | ✓ (`scale_pos_weight`) | ✓ (`class_weight`) | ✓ (`class_weight`) |
| Interpretabilidade | ✓ `feature_importances_` + SHAP | ✓ idem | ✓ idem | ✓ Alta (coeficientes) |

**Comparação com projeto anterior (E5 — Random Forest sem SMOTE):**
- F1 churn: 0,51 → **0,64** (+25% relativo)
- ROC-AUC: não medido no E5 → **0,8770**

A melhora decorre principalmente de três mudanças: (1) SMOTE corrigindo o desbalanceamento, (2) feature engineering adicional (AgeInactivity, EngagementScore, BalanceSalaryRatio), e (3) otimização bayesiana de hiperparâmetros com Optuna (30 trials/modelo via TPE Sampler).

---

### 3.6 Rastreamento com MLflow

Todos os experimentos foram registrados no MLflow com backend SQLite (`mlflow.db`), sob o experimento `bank-churn-classification`. Para cada modelo foram logados:

- **Parâmetros**: hiperparâmetros padrão e otimizados, método de redução de dimensionalidade, threshold utilizado
- **Métricas por fold**: `fold_roc_auc`, `fold_f1`, `fold_recall`, `fold_precision`, `fold_avg_precision`
- **Métricas agregadas**: média e desvio padrão de cada métrica na CV
- **Métricas do holdout**: avaliação final em dados nunca vistos durante a busca
- **Artefatos**: curva ROC, curva Precision-Recall, matriz de confusão, análise de threshold, importância de features, comparação por fold, distribuição antes/após SMOTE
- **Pipeline serializado**: pipeline completo (imputação + escalonamento + reducer + estimador) salvo como artefato MLflow, pronto para inferência


## Parte 4 — Redução de Dimensionalidade

### 4.1 Análise da Necessidade de Redução

O dataset possui 20 features após o feature engineering da Parte 2. Para avaliar se redução de dimensionalidade é necessária, é preciso considerar três fatores:

**Dimensionalidade:** 20 features é um espaço relativamente compacto. A "maldição da dimensionalidade" se manifesta tipicamente a partir de centenas de features. Com 10.000 amostras e 20 features, a razão amostras/dimensão é ~500:1 — muito acima do limiar problemático.

**Resultados da Parte 3:** o Optuna testou quatro métodos de redução (`none`, `rfe`, `pca`, `kpca`) simultaneamente com os hiperparâmetros do modelo. O LightGBM selecionou RFE com 16 features — eliminando apenas 4 das 20 — o que indica que a maior parte das features é informativa e a redução agressiva não é necessária.

**Conclusão preliminar:** não há necessidade técnica imperativa de redução de dimensionalidade neste dataset. O experimento desta parte serve para **quantificar o trade-off** entre compressão do espaço de features, desempenho preditivo e custo computacional.

---

### 4.2 Técnicas Escolhidas e Justificativas

Foram selecionadas **PCA** e **LDA** para o experimento controlado. t-SNE foi descartado como técnica de pipeline por razão técnica fundamental.

**PCA — Principal Component Analysis (não-supervisionado)**

Escolhida por ser a técnica de referência em redução de dimensionalidade: decomposição espectral da matriz de covariância que projeta os dados na direção de máxima variância. Vantagens para este contexto: implementação estável, produz componentes ordenados por variância explicada, e permite análise de quanto da informação original é preservada. Limitação: ignora completamente os rótulos da variável-alvo — maximiza variância dos dados, não separabilidade entre classes.

Configuração: **18 componentes** (de 20 features), preservando >99% da variância — redução mínima, útil para isolar o efeito da transformação linear sem perda significativa de informação.

**LDA — Linear Discriminant Analysis (supervisionado)**

Escolhida como contraponto supervisionado ao PCA: em vez de maximizar variância, LDA encontra a projeção que **maximiza a separação entre classes**. Para classificação binária, o resultado é sempre **1 único componente** (n_classes − 1 = 1) — compressão máxima: de 20 features para 1 dimensão.

Essa propriedade torna o LDA um experimento extremo: se um único discriminante linear capturar a separabilidade necessária, o modelo é altamente eficiente e interpretável. Se o desempenho cair significativamente, demonstra que a fronteira de decisão tem estrutura não-linear que 1 componente linear não captura.

**Por que t-SNE foi descartado como técnica de pipeline:**

t-SNE é um algoritmo de visualização não-paramétrico — ele não aprende uma transformação generalizável. Não existe `fit()` + `transform()` independentes para novos dados: cada execução otimiza uma embedding específica para o conjunto de entrada, sem capacidade de projetar amostras de teste. Por isso, **não pode ser integrado a um pipeline de inferência**. Seu uso neste projeto se limitaria à visualização exploratória dos dados (EDA), não ao treinamento de classificadores.

---

### 4.3 Integração ao Pipeline e Configuração Experimental

O experimento foi implementado em `notebooks/dimensionality_reduction.py`, reutilizando o componente `src/feature_reducer.py` — que foi estendido com suporte a LDA nesta fase.

**Design do experimento controlado:**

- **Variável independente:** técnica de redução (`none` | `PCA` | `LDA`)
- **Variável controlada:** hiperparâmetros do LightGBM fixados nos valores ótimos da Parte 3 (`n_estimators=528`, `num_leaves=30`, `lr=0.0144`, etc.)
- **Protocolo:** StratifiedKFold (5 folds), threshold=0.40, SMOTE ativado, seed=42

**Pipeline por variante:**

```
none : DataImputer → FeatureReducer(none)   → SMOTE → LGBMClassifier
PCA  : DataImputer → StandardScaler → FeatureReducer(pca,  n=18) → SMOTE → LGBMClassifier
LDA  : DataImputer → StandardScaler → FeatureReducer(lda,  n=1)  → SMOTE → LGBMClassifier
```

StandardScaler foi adicionado antes de PCA e LDA porque ambos são sensíveis à escala absoluta das features (PCA confunde variância com magnitude; LDA maximiza razão de variâncias entre/intra-classes). O LightGBM na variante `none` não recebe scaler pois árvores são invariantes a escala monotônica.

---

### 4.4 Resultados — Comparação com e sem Redução de Dimensionalidade

Experimentos executados em `2026-04-20`, logados no MLflow com tag `stage=dr_comparison`.

| Técnica DR | Features | CV ROC-AUC | ± std | CV F1 | CV Recall | Holdout ROC-AUC | Holdout F1 | Holdout Recall | Holdout Prec. | Tempo CV (s) |
|------------|----------|-----------|-------|-------|-----------|-----------------|------------|----------------|---------------|-------------|
| **Sem DR** | 20 | **0,8598** | 0,0088 | 0,5845 | **0,7791** | **0,8730** | **0,5973** | **0,8162** | 0,4710 | 6,5 |
| PCA (18 comp.) | 18 | 0,8586 | 0,0071 | **0,5941** | 0,7675 | 0,8687 | 0,5875 | 0,7941 | 0,4662 | **5,0** |
| LDA (1 comp.) | 1 | 0,8196 | 0,0107 | 0,5200 | 0,7951 | 0,8435 | 0,5340 | 0,8284 | 0,3939 | **4,8** |

> **Nota sobre os resultados:** os valores acima diferem levemente dos resultados da Parte 3 porque o Optuna usou RFE(16) como redução — não "none". O experimento de DR usa os melhores params do Optuna mas com método de redução controlado, revelando o impacto isolado de cada técnica.

---

### 4.5 Análise dos Trade-offs

#### Impacto no Desempenho de Classificação

**Sem DR vs PCA:** a diferença de ROC-AUC no holdout é mínima (0,8730 → 0,8687, −0,0043). PCA com 18 de 20 componentes preserva virtualmente toda a informação preditiva. A pequena queda decorre das 2 features descartadas e da transformação linear que mistura features interpretáveis em componentes abstratos. F1 do PCA é ligeiramente superior (0,5941 vs 0,5845 em CV) por reduzir marginalmente o ruído.

**Sem DR vs LDA:** queda expressiva em ROC-AUC (0,8730 → 0,8435, −0,0295). Comprimir 20 features em 1 único componente linear é uma hipótese extremamente restritiva: assume que toda a informação discriminante está em uma única direção linear no espaço de features. O resultado mostra que essa hipótese não se sustenta — o problema tem estrutura de separação mais complexa que 1 componente captura. O recall do LDA no holdout (0,8284) supera os demais porque 1 componente tende a criar fronteiras de decisão mais "grossas", deslocando mais predições para a classe positiva.

#### Custo Computacional

| Fase | Sem DR | PCA | LDA |
|------|--------|-----|-----|
| Fit do redutor (CV) | — | ~0,01s/fold | ~0,005s/fold |
| Treino LightGBM (CV) | ~1,3s/fold | ~1,0s/fold | ~0,9s/fold |
| **Total CV (5 folds)** | **6,5s** | **5,0s** | **4,8s** |
| Inferência (por amostra) | baseline | +transformação linear | +transformação linear |

PCA e LDA reduzem o tempo de treino do LightGBM porque o modelo opera em espaço de menor dimensão. A economia é modesta aqui (20 → 18 ou 20 → 1 features) mas seria substancial com centenas de features. O overhead do próprio PCA/LDA é desprezível (<0,1s por fold).

#### Interpretabilidade

| Aspecto | Sem DR | PCA | LDA |
|---------|--------|-----|-----|
| Features originais visíveis no modelo | ✓ Sim | ✗ Não | ✗ Não |
| Importância de features interpretável | ✓ Direto | ✗ Componentes abstratos | ✗ 1 eixo discriminante |
| SHAP values interpretáveis | ✓ Sim | ✗ Requer back-projection | ✗ Limitado |
| Explicabilidade para stakeholders | ✓ Alta | ✗ Baixa | ✗ Muito baixa |

Sem DR, o LightGBM pode indicar "Age tem importância 0,38" de forma direta e acionável. Com PCA, o `pc_0` é uma combinação linear de todas as 20 features — interpretar qual variável de negócio está mais associada ao churn requer projetar os loadings de volta, adicionando complexidade. Com LDA, o único componente tem coeficientes que somam contribuições de todas as features para o eixo de separação de classes, mas o modelo final opera apenas nesse valor escalar — a conexão com variáveis de negócio é indireta.

---

### 4.6 Decisão Técnica — Redução de Dimensionalidade é Adequada?

**Conclusão: redução de dimensionalidade não é recomendada para este problema no estado atual.**

Justificativas:

1. **Sem ganho preditivo relevante:** PCA com 18 componentes produz ROC-AUC 0,43 pontos percentuais menor no holdout — diferença negligenciável, dentro da margem de variação CV (±0,0088). LDA perde 2,95 pontos — queda relevante.

2. **Recall é a métrica crítica de negócio:** o critério definido na Parte 1 é recall ≥ 0,70. Sem DR o holdout atinge recall 0,8162. Com PCA, 0,7941. Com LDA, 0,8284 (por razão espúria — menor precisão, não melhor discriminação). A configuração "Sem DR" entrega o melhor recall com boa precisão.

3. **Custo de interpretabilidade é alto:** o banco precisa explicar decisões de retenção. Sem DR, features como `Age`, `NumOfProducts` e `IsActiveMember` têm importâncias diretas e acionáveis. PCA/LDA destroem essa rastreabilidade.

4. **20 features não é um problema de dimensionalidade:** redução seria valiosa a partir de ~100+ features altamente correlacionadas. Aqui, o feature engineering já produziu um conjunto compacto e informativo.

**Exceção:** se o objetivo fosse visualização exploratória (análise de clusters de clientes, por exemplo), PCA com 2 componentes ou t-SNE seriam ferramentas valiosas — mas para fins de inferência e deploy, a versão sem redução é superior.

---

## Parte 5 — Seleção do Modelo Final

### 5.1 Consolidação dos Experimentos

A tabela abaixo reúne todos os experimentos avaliados no holdout ao longo das Partes 3 e 4. Apenas o LightGBM passou pela avaliação completa de holdout — foi selecionado na Parte 3 como o modelo com maior ROC-AUC em validação cruzada entre os quatro candidatos, e testado em múltiplas configurações de redução de dimensionalidade na Parte 4.

| Experimento | Configuração | CV ROC-AUC | Holdout ROC-AUC | Holdout F1 | Holdout Recall | Holdout Prec. |
|-------------|--------------|-----------|-----------------|------------|----------------|---------------|
| P3 — Baseline | Parâmetros padrão, sem DR | 0,8499 ± 0,0058 | 0,8539 | 0,6103 | 0,5429 | 0,6965 |
| P3 — Optuna | RFE(16), hiperparams otimizados | 0,8623 ± 0,0076 | 0,8770 | 0,6417 | 0,6078 | 0,6795 |
| P4 — DR | Sem DR, hiperparams Optuna | 0,8598 ± 0,0088 | 0,8730 | 0,5973 | **0,8162** | 0,4710 |
| P4 — DR | PCA(18), hiperparams Optuna | 0,8586 ± 0,0071 | 0,8687 | 0,5875 | 0,7941 | 0,4662 |
| P4 — DR | LDA(1), hiperparams Optuna | 0,8196 ± 0,0107 | 0,8435 | 0,5340 | 0,8284 | 0,3939 |

---

### 5.2 Por que o Optuna Escolheu RFE e Não "Sem DR"?

Esse é o ponto central para entender a seleção final.

O Optuna **co-otimizou o método de DR e os hiperparâmetros do modelo simultaneamente** em 30 trials. Cada trial explorava uma combinação diferente dessas duas dimensões:

```
Trial X:  RFE(16) + n_est=528 + num_leaves=30 + lr=0.014  →  CV AUC 0.8623  ← melhor trial
Trial Y:  none    + n_est=300 + num_leaves=50 + lr=0.050  →  CV AUC 0.851
Trial Z:  none    + n_est=150 + num_leaves=20 + lr=0.030  →  CV AUC 0.843
```

O problema é que **nenhum trial testou `none` com os mesmos hiperparâmetros do melhor trial RFE**. O Optuna encontrou um ótimo local: a combinação RFE + aqueles hiperparâmetros específicos. Mas não explorou se esses mesmos hiperparâmetros, sem RFE, seriam equivalentes ou melhores — com apenas 30 trials em um espaço de busca grande, essa combinação simplesmente não foi amostrada.

A Parte 4 fez exatamente esse experimento controlado: pegou os hiperparâmetros do melhor trial Optuna e testou `none`, `PCA` e `LDA`. O resultado revelou que **sem DR, com os mesmos hiperparâmetros, o AUC cai apenas 0,004 e o recall sobe 21 pontos percentuais**.

---

### 5.3 Análise da Diferença de AUC

A diferença de ROC-AUC entre as duas principais configurações:

| Configuração | Holdout ROC-AUC | Diferença | CV std |
|-------------|-----------------|-----------|--------|
| Optuna + RFE(16) | 0,8770 | — | ±0,0076 |
| Optuna + sem DR | 0,8730 | **−0,004** | ±0,0088 |

A diferença de 0,004 é **menor que metade de um desvio padrão** da variação natural entre folds. Não há evidência estatística de que a configuração com RFE discrimine genuinamente melhor — a diferença está dentro do ruído esperado entre execuções.

Em contrapartida, a diferença de recall é **21 pontos percentuais** (0,61 → 0,82) — essa sim está muito além do ruído e representa um efeito real e substancial.

---

### 5.4 Modelo Candidato à Operação

**Modelo selecionado: LightGBM com hiperparâmetros Optuna, sem redução de dimensionalidade.**

#### Justificativa da Seleção

**1. AUC estatisticamente equivalente ao melhor encontrado:** a diferença de 0,004 em relação à configuração Optuna+RFE está dentro da variação natural do CV. As duas configurações têm capacidade discriminativa equivalente.

**2. Recall superior em 21 pontos percentuais:** 0,82 vs 0,61 no holdout. Para o problema de churn bancário — onde o custo de não identificar um cliente que vai sair (falso negativo) é permanente — essa diferença tem impacto direto no resultado de negócio. Em uma base de 10.000 clientes com 20% de churn, isso representa ~429 churners adicionais identificados a cada ciclo.

**3. O recall mais alto não é espúrio:** vem de um modelo com AUC equivalente, não de um modelo que simplesmente "chuta positivo para tudo". A precisão de 0,47 indica que o modelo ainda discrimina — captura 82% dos churners com 47% de precisão, operando em um ponto da curva ROC mais favorável ao recall.

**4. Pipeline mais simples e robusto:** sem RFE, o pipeline elimina um componente stateful que precisaria ser re-calibrado a cada re-treino e é sensível a mudanças no esquema de features. Menos peças que podem falhar em produção.

**5. Interpretabilidade total:** 20 features originais preservadas. Importâncias de features e SHAP values são diretamente rastreáveis a variáveis de negócio — essencial para explicar decisões de retenção para stakeholders.

#### Especificação Técnica

```python
LGBMClassifier(
    n_estimators       = 528,
    num_leaves         = 30,
    learning_rate      = 0.01437,
    subsample          = 0.6872,
    colsample_bytree   = 0.4803,
    min_child_samples  = 69,
    reg_alpha          = 0.00805,
    reg_lambda         = 0.01376,
    is_unbalance       = True,
    random_state       = 42,
)
```

**Pipeline completo:**
```
DataImputer → FeatureReducer(method='none') → SMOTE(k=5) → LGBMClassifier
```

**Threshold de decisão:** 0,40

#### Desempenho Final

| Métrica | Valor | Meta (Parte 1) | Status |
|---------|-------|----------------|--------|
| ROC-AUC (holdout) | 0,8730 | ≥ 0,85 | ✓ Atendida |
| Recall classe churn (holdout) | 0,8162 | ≥ 0,70 | ✓ Atendida |
| F1-score (holdout) | 0,5973 | ≥ 0,60 | ✗ −0,003 |
| Threshold de decisão | 0,40 | — | Ajustado |

> **Nota sobre F1:** o F1 de 0,597 fica 0,003 abaixo da meta de 0,60 — diferença negligenciável. O valor reflete o threshold 0,40, que desloca o modelo para maior recall (0,82) e menor precisão (0,47). Com threshold 0,50 o F1 sobe para ~0,62, mas recall cai para ~0,55. A priorização do recall como critério de negócio principal justifica essa configuração.

#### Comparação com o Projeto Anterior (E5 — Random Forest sem SMOTE)

| Métrica | E5 — Scikit-Learn | Modelo Final — MLOps | Evolução |
|---------|------------------|----------------------|---------|
| Recall churn | 0,36 | **0,82** | **+127%** |
| F1 churn | 0,51 | 0,60 | +17,6% |
| ROC-AUC | não medido | 0,8730 | — |
| Tratamento de desbalanceamento | ✗ | ✓ SMOTE | — |
| Otimização de hiperparâmetros | Grid Search manual | Optuna TPE 30 trials | — |

O salto de recall de 0,36 para 0,82 — mais que o dobro — é o resultado direto de três mudanças em conjunto: SMOTE corrigindo o desbalanceamento, feature engineering adicional (AgeInactivity, EngagementScore, BalanceSalaryRatio) e otimização bayesiana com Optuna.

---

## Parte 6 — Deploy, Monitoramento e Operação do Modelo

### 6.1 Persistência e Versionamento do Modelo

O pipeline completo — DataImputer, FeatureReducer, SMOTE e LGBMClassifier — é serializado localmente via joblib e registrado no MLflow Model Registry sob o nome `bank-churn-lgbm`. Cada novo treino gera uma versão numerada automaticamente.

O fluxo de promoção é controlado por um gate de qualidade: modelos que atingem ROC-AUC ≥ 0,85 e Recall ≥ 0,65 são promovidos automaticamente para **Production**; os demais ficam em **Staging** para inspeção manual. Versões anteriores são arquivadas, mantendo o histórico completo de modelos implantados e permitindo rollback imediato se necessário.

---

### 6.2 Empacotamento como Artefato de Inferência

O modelo é empacotado como artefato autocontido: um único arquivo carregável que inclui todo o pipeline de transformação e predição, sem necessidade de reprocessar os dados de treino. Um arquivo de schema acompanha o artefato, documentando as colunas esperadas, a variável-alvo e o threshold de decisão — formalizando o contrato de inferência entre o modelo e qualquer serviço que o consuma.

---

### 6.3 Serviço de Inferência

O modelo é exposto via API REST (FastAPI) com três endpoints: verificação de saúde, predição individual e predição em lote. Cada resposta inclui a probabilidade de churn, o flag de decisão e um nível de risco categorizado em baixo, médio ou alto — permitindo que a equipe de retenção priorize ações sem precisar interpretar probabilidades brutas.

Uma interface visual (Streamlit) complementa a API, permitindo simulações interativas com formulário de entrada e resultado imediato com indicação visual de risco por cor.

O serviço carrega o modelo diretamente do MLflow Registry (estágio Production) e recorre ao artefato local como fallback automático caso o Registry não esteja disponível.

---

### 6.4 Pipeline CI/CD Simulado

O pipeline de integração e entrega contínua automatiza seis etapas em sequência: validação do ambiente, verificação dos dados de entrada, treino e registro do modelo, gate de qualidade com verificação de métricas mínimas, smoke test da API e ciclo inicial de monitoramento. O pipeline aborta imediatamente em qualquer falha, garantindo que nenhum modelo deficiente seja promovido a produção.

---

### 6.5 Métricas Técnicas e de Impacto de Negócio

#### Métricas Técnicas

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| ROC-AUC | 0,8613 | Discriminação geral entre classes |
| Recall | 0,6740 | Proporção de churners reais identificados |
| F1-score | 0,6104 | Equilíbrio entre precisão e recall |
| Precision | 0,5578 | Confiabilidade das predições positivas |

#### Métricas de Impacto de Negócio

Calculadas para cada batch mensal de produção com base em premissas de negócio: receita anual por cliente retido de €500, custo de €40 por ação de retenção e taxa de sucesso de retenção de 30%.

| Métrica | Referência (batch de 1.000 clientes) |
|---------|--------------------------------------|
| Verdadeiros positivos (churners corretamente identificados) | 130 de 192 |
| Falsos positivos (clientes abordados desnecessariamente) | 104 |
| Receita preservada estimada | €19.500 |
| Custo total de campanhas | €9.360 |
| ROI estimado | 1,1x |

---

### 6.6 Monitoramento Pós-Deploy e Detecção de Drift

O monitoramento é executado a cada batch de dados novos e gera um run no MLflow com tag `stage=monitoring`, permitindo acompanhar a evolução das métricas ao longo do tempo.

**Data drift** é detectado pelo teste de Kolmogorov-Smirnov aplicado feature a feature, comparando a distribuição do conjunto de treino com a do batch de produção. Um p-value abaixo de 0,05 indica que a distribuição de uma feature mudou significativamente. No ciclo inicial simulado: 0 de 20 features com drift detectado.

**Estabilidade das probabilidades** é medida pelo PSI (Population Stability Index) sobre as saídas do modelo. Valores abaixo de 0,10 indicam estabilidade; acima de 0,20, drift severo com re-treino recomendado. Resultado inicial: PSI = 0,028 — estável.

**Model drift** é avaliado pela queda nas métricas entre o batch de referência e o batch de produção. Quedas acima de 3 pontos percentuais em AUC ou 5 pontos em recall acionam alerta. Resultado inicial: sem model drift detectado.

---

### 6.7 Estratégia de Re-treinamento

O re-treino é acionado por quatro gatilhos: três ou mais features com drift detectado pelo KS test, PSI acima de 0,20, queda relevante nas métricas de produção, ou schedule mensal preventivo independente de drift.

A janela de treino utiliza os últimos 12 meses de dados para evitar que padrões obsoletos contaminem o modelo. Antes de qualquer promoção, o novo modelo passa pelo mesmo gate de qualidade do CI/CD. A versão anterior permanece disponível no Registry para rollback imediato caso o novo modelo apresente regressão em produção.