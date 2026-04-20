# Análise Exploratória de Dados — Bank Customer Churn

> **Dataset:** `Customer-Churn-Records.csv` — Kaggle (`radheshyamkollipara/bank-customer-churn`)
> **Análise gerada em:** 11 de abril de 2026
> **Script:** `eda/eda_analysis.py` | **Relatório JSON:** `outputs/eda/eda_report_20260411_114506.json`

---

## Sumário

1. [Contexto de negócio e valor estratégico](#1-contexto-de-negócio-e-valor-estratégico)
2. [Visão geral técnica do dataset](#2-visão-geral-técnica-do-dataset)
3. [Qualidade dos dados](#3-qualidade-dos-dados)
4. [Distribuição da variável alvo — Churn](#4-distribuição-da-variável-alvo--churn)
5. [Análise das variáveis](#5-análise-das-variáveis)
6. [Principais achados e implicações estratégicas](#6-principais-achados-e-implicações-estratégicas)
7. [Alerta crítico: risco de data leakage](#7-alerta-crítico-risco-de-data-leakage)
8. [Recomendações para o modelo de Machine Learning](#8-recomendações-para-o-modelo-de-machine-learning)
9. [Parâmetros derivados para Great Expectations](#9-parâmetros-derivados-para-great-expectations)

---

## 1. Contexto de negócio e valor estratégico

**Churn** — a saída voluntária de clientes — é um dos principais vetores de perda de receita recorrente no setor bancário. Este dataset registra o histórico de **10.000 clientes** de um banco europeu com operações na **França, Alemanha e Espanha**, consolidando variáveis comportamentais, financeiras e de perfil para cada cliente. A variável alvo (`Exited`) indica se o cliente encerrou o relacionamento com o banco.

O objetivo analítico é identificar, de forma proativa e com antecedência, quais clientes apresentam maior probabilidade de saída — viabilizando intervenções de retenção direcionadas antes que a decisão seja tomada.

As variáveis disponíveis cobrem quatro dimensões:

- **Perfil financeiro:** score de crédito, saldo em conta, salário estimado
- **Relacionamento com o banco:** tempo de conta (tenure), número de produtos contratados, atividade, pontos acumulados, possui cartão de crédito, tipo de cartão
- **Dados demográficos:** idade, gênero, país
- **Experiência do cliente:** nota de satisfação, histórico de reclamações

### Impacto financeiro do churn

O custo de aquisição de novos clientes tende a ser significativamente maior do que o custo de retenção dos existentes. Em carteiras com volumes expressivos de clientes, mesmo taxas de churn moderadas resultam em:

- perda relevante de base ativa ao longo do ciclo anual
- redução de receita recorrente proveniente de tarifas, operações de crédito e produtos complementares
- esforço de recomposição via aquisição que, em geral, supera o investimento em ações preventivas de retenção
---

## 2. Visão geral técnica do dataset

| Propriedade | Valor |
|---|---|
| Total de linhas | 10.000 |
| Total de colunas | 18 |
| Memória | 1.68 MB |
| Valores ausentes | **0** (0.0%) |
| Linhas duplicadas | **0** |
| IDs únicos de clientes | 10.000 |

### Schema de colunas

| Coluna | Tipo | Categoria | Descrição |
|---|---|---|---|
| `RowNumber` | int64 | Identificador | Índice sequencial|
| `CustomerId` | int64 | Identificador | ID único do cliente |
| `Surname` | string | Identificador | Sobrenome |
| `CreditScore` | int64 | Numérica contínua | Score de crédito (350–850) |
| `Geography` | string | Categórica nominal | País: France, Germany, Spain |
| `Gender` | string | Categórica nominal | Gênero: Male, Female |
| `Age` | int64 | Numérica contínua | Idade (18–92 anos) |
| `Tenure` | int64 | Numérica discreta | Anos como cliente (0–10) |
| `Balance` | float64 | Numérica contínua | Saldo na conta (0–250.898) |
| `NumOfProducts` | int64 | Numérica discreta | Produtos contratados (1–4) |
| `HasCrCard` | int64 | Binária | Possui cartão de crédito 1=Sim, 0=Não |
| `IsActiveMember` | int64 | Binária | Membro ativo 1=Sim, 0=Não |
| `EstimatedSalary` | float64 | Numérica contínua | Salário estimado (11–199.992) |
| `Exited` | int64 | **Alvo (Target)** | Saiu do banco: 1=Sim, 0=Não |
| `Complain` | int64 | Binária | Registrou reclamação 1=Sim, 0=Não |
| `Satisfaction Score` | int64 | Ordinal | Satisfação: 1 (péssimo) a 5 (ótimo) |
| `Card Type` | string | Categórica nominal | Tipo de cartão: DIAMOND, GOLD, SILVER, PLATINUM |
| `Point Earned` | int64 | Numérica discreta | Pontos de fidelidade acumulados (119-1.000) |

---

## 3. Qualidade dos dados

### Valores ausentes

**Resultado:** Nenhuma das 180.000 células contém valor nulo.

```
Células totais    : 180.000
Células ausentes  : 0  (0.0000%)
Colunas afetadas  : nenhuma
```

**Implicação para MLOps:** Nenhuma estratégia de imputação é necessária para este dataset. No entanto, em produção (dados reais do banco), é provável que surjam nulos — especialmente em `Balance` (clientes novos sem histórico) e `CreditScore` (clientes sem análise de crédito formal). Estratégias preventivas são detalhadas na [seção 8](#8-recomendações-para-o-modelo-de-machine-learning).

### Duplicatas

```
Duplicatas exatas         : 0
Duplicatas por features   : 0  
```

Cada linha representa um cliente único e distinto.

### Distribuições e outliers

| Coluna | Outliers IQR | Outliers Z>3 | Observação |
|---|---|---|---|
| `CreditScore` | 15 (0.15%) | 8 (0.08%) | Negligível |
| `Age` | 359 (3.59%) | 133 (1.33%) | Clientes acima de 62 anos — dados válidos |
| `Balance` | 0 (0.0%) | 0 (0.0%) | Bimodal: 36.17% dos clientes têm saldo zero |
| `NumOfProducts` | 60 (0.6%) | 60 (0.6%) | 4 produtos = sinal de alerta |
| `Tenure` | 0 (0.0%) | 0 (0.0%) | Distribuição uniforme
| `EstimatedSalary` | 0 (0.0%) | 0 (0.0%) | Distribuição uniforme |
| `Satisfaction Score` | 0 (0.0%) | 0 (0.0%) | Distribuição uniforme |
| `Point Earned` | 0 (0.0%) | 0 (0.0%) | Distribuição uniforme |

Os outliers identificados foram mantidos por representarem comportamentos reais de negócio. Os 15 casos extremos de `CreditScore` são estatisticamente insignificantes (0,15%) e não distorcem a distribuição. O `Balance` não possui outliers, mas apresenta distribuição bimodal — 36,17% dos clientes têm saldo zero, refletindo dois perfis distintos de uso bancário que não devem ser tratados como anomalia. Os 359 clientes idosos sinalizados em `Age` são válidos demograficamente; removê-los introduziria viés e eliminaria um segmento relevante para a previsão de churn. Os 60 clientes com 4 produtos em `NumOfProducts`, embora raros (0,6%), carregam potencial preditivo alto e foram mantidos. A performance do modelo sobre esses subgrupos será verificada na etapa de avaliação para detectar eventuais disparidades preditivas.

---

## 4. Distribuição da variável alvo — Churn

```
Total de clientes   : 10.000
Clientes que saíram : 2.038  (20.38%)
Clientes retidos    : 7.962  (79.62%)
Razão de desequilíbrio: 3.91:1  (retidos:saídos)
```

### O problema de classe desbalanceada

A distribuição 80/20 representa um **desequilíbrio moderado** — não é severo, mas é significativo o suficiente para distorcer modelos.

**Sem tratamento do desbalanceamento**, um modelo que classifica todos os clientes como "retidos" atingiria 79.62% de acurácia — resultado numericamente elevado, mas com valor operacional nulo, pois não identifica nenhum caso de churn.

**Por que isso importa para o negócio:** Em retenção de clientes, o custo de um falso negativo (não identificar quem vai sair e perder o cliente) é muito maior do que o custo de um falso positivo. A estratégia do modelo deve refletir esse custo assimétrico.

---

## 5. Análise das variáveis

### Variáveis numéricas — estatísticas completas

| Variável | Mín | Máx | Média | Mediana | Std | Skewness | Outliers IQR |
|---|---|---|---|---|---|---|---|
| `CreditScore` | 350 | 850 | 650.5 | 652.0 | 96.7 | -0.07 (simétrico) | 0.15% |
| `Age` | 18 | 92 | 38.9 | 37.0 | 10.5 | +1.01 (assimétrico direita) | 3.59% |
| `Tenure` | 0 | 10 | 5.01 | 5.0 | 2.89 | ~0 (uniforme) | 0.0% |
| `Balance` | 0 | 250.898 | 76.486 | 97.199 | 62.397 | -0.14 (bimodal) | 0.0% |
| `NumOfProducts` | 1 | 4 | 1.53 | 1.0 | 0.58 | +0.75 (assimétrico) | 0.6% |
| `EstimatedSalary` | 12 | 199.992 | 100.090 | 100.194 | 57.510 | ~0 (uniforme) | 0.0% |
| `Satisfaction Score` | 1 | 5 | 3.01 | 3.0 | 1.41 | ~0 (uniforme) | 0.0% |
| `Point Earned` | 119 | 1.000 | 606.5 | 605.0 | 225.9 | ~0 (uniforme) | 0.0% |

**Observações técnicas:**
- **`CreditScore`:** Distribuição quase simétrica, mas não possui distribuição normal (p < 0.05 nos dois testes).
- **`Age`:** Distribuição fortemente assimétrica à direita. A maioria dos clientes é jovem-adulta (mediana 37 anos), mas há cauda longa de clientes idosos até 92 anos.
- **`Balance`:** Variável bimodal crítica. 36.17% dos clientes têm saldo zero — isso representa clientes que mantêm a conta aberta mas não a usam. Os demais (63.83%) têm saldo médio de ~120 mil.
- **`Tenure`:** Completamente uniforme entre 0 e 10 anos. Nenhum poder preditivo linear com o churn.
- **`EstimatedSalary`:** Uniformemente distribuído entre ~0 e ~200k. Provavelmente gerado sinteticamente neste dataset.

### Variáveis categóricas — distribuição

**Geography:**
| País | Clientes | % do total | Churn Rate |
|---|---|---|---|
| France | 5.014 | 50.14% | 16.2% |
| Germany | 2.509 | 25.09% | **32.4%** |
| Spain | 2.477 | 24.77% | 16.7% |

**Gender:**
| Gênero | Clientes | % do total | Churn Rate |
|---|---|---|---|
| Male | 5.457 | 54.57% | 16.5% |
| Female | 4.543 | 45.43% | **25.1%** |

**Card Type:**
| Tipo | Clientes | % do total | Churn Rate |
|---|---|---|---|
| DIAMOND | 2.507 | 25.07% | 21.8% |
| GOLD | 2.502 | 25.02% | 19.3% |
| PLATINUM | 2.495 | 24.95% | 20.4% |
| SILVER | 2.496 | 24.96% | 20.1% |

> O tipo de cartão está distribuído de forma perfeitamente balanceada (≈25% cada) e **não tem associação significativa com o churn** (χ² p=0.168, Cramér's V=0.02).

---

## 6. Principais achados e implicações estratégicas

Os dados revelam padrões de comportamento com relevância direta para a estratégia de retenção. Cada achado é apresentado com o embasamento estatístico e a implicação operacional correspondente.

---

### Achado 1: Reclamação é o indicador mais fortemente associado à saída

**Correlação de Pearson entre `Complain` e `Exited` = 0.9957**

| Registrou reclamação? | Total | Saíram | Taxa de churn |
|---|---|---|---|
| Não | 7.956 | 4 | **0.05%** |
| Sim | 2.044 | 2.034 | **99.5%** |

A correlação é virtualmente perfeita: clientes que registraram reclamação apresentaram saída em 99.5% dos casos. O lado inverso é igualmente expressivo — entre os clientes sem reclamação, apenas 0.05% encerrou o relacionamento.

**Implicação estratégica:** Cada reclamação não resolvida se converte, na quase totalidade dos casos, em um cliente perdido. O investimento em capacidade de resolução de reclamações representa diretamente uma alavanca de retenção.

**Alerta para o modelo preditivo:** A força desta correlação levanta uma questão de integridade dos dados. Ver [seção 7](#7-alerta-crítico-risco-de-data-leakage).

---

### Achado 2: Clientes mais velhos apresentam maior propensão à saída

**Idade média dos clientes que saíram: 44.8 anos vs. 37.4 anos entre os que permaneceram. Cohen's d = 0.747 (efeito médio-grande).**

A diferença de ~7 anos entre os grupos é estatisticamente robusta (p ≈ 0, Mann-Whitney U). Isso sugere que:

- Clientes mais velhos podem ter relacionamentos mais longos com outros bancos e maior facilidade para migrar
- O banco pode não estar oferecendo produtos adequados para o perfil de cliente maduro (previdência, gestão de patrimônio, etc.)
- Campanhas de retenção devem priorizar a faixa etária 40–60 anos

---

### Achado 3: A operação alemã concentra o dobro do churn das demais praças

**Alemanha: 32.4% de churn vs. 16.2% na França e 16.7% na Espanha. Associação moderada (Cramér's V = 0.17, p < 0.0001).**

Um cliente alemão apresenta aproximadamente o dobro da probabilidade de saída em relação a clientes franceses ou espanhóis. Hipóteses que merecem investigação pelo time de negócio:
- O banco pode ter uma operação menos competitiva na Alemanha
- Diferenças culturais no relacionamento com banco (alemães tendem a ser mais exigentes com qualidade de serviço)
- Concorrência bancária mais intensa no mercado alemão
- Problemas localizados de atendimento ou produto

**O maior efeito estrutural do dataset: clientes alemães têm saldo médio quase o dobro dos demais.**

| País | Saldo médio |
|---|---|
| France | €62.093 |
| Germany | **€119.730** |
| Spain | €61.818 |

A diferença de saldo entre a Alemanha e as demais praças é a **maior associação numérica observada em toda a análise** (ANOVA F=958, p≈0, eta²=0.16 — efeito grande). Clientes alemães são, em média, os mais ricos do portfolio — e, portanto, os alvos mais atrativos para a concorrência bancária. Isso oferece uma explicação estrutural para o churn elevado: não é um problema de qualidade de serviço isolado, é um efeito de mercado.

**O efeito da Alemanha é independente do score de crédito.**

Ao segmentar o churn por quartil de `CreditScore` dentro de cada país, o padrão se mantém estável em todos os níveis:

| Quartil de CreditScore | France | Germany | Spain |
|---|---|---|---|
| Q1 (score mais baixo) | 17.3% | **35.6%** | 17.6% |
| Q2 | 17.5% | **32.0%** | 16.6% |
| Q3 | 15.1% | **28.1%** | 15.4% |
| Q4 (score mais alto) | 14.8% | **33.7%** | 17.1% |

Independentemente da qualidade de crédito do cliente, a probabilidade de churn na Alemanha é sempre o dobro das demais praças. Isso descarta a hipótese de que o problema alemão é causado por inadimplência ou perfil de risco — o fenômeno é sistêmico na operação local.

Adicionalmente, a análise de chi-squared mostra que clientes alemães registram proporcionalmente mais reclamações (Cramér's V = 0.175, p < 0.0001) — o mesmo coeficiente de associação que a própria variável `Geography` tem com `Exited`.

---

### Achado 4: Clientes do sexo feminino apresentam churn significativamente maior

**Mulheres: 25.1% de churn vs. 16.5% entre homens (Cramér's V = 0.11, p < 0.0001).**

Clientes do sexo feminino apresentam 52% mais probabilidade de saída em relação ao sexo masculino. Esse diferencial pode indicar desalinhamento entre o portfólio de produtos e a comunicação do banco com o perfil feminino, ou expectativas específicas de serviço que não estão sendo atendidas adequadamente.

---

### Achado 5: Concentração de produtos acima de 2 é fortemente associada à saída

| Nº de produtos | Total | Saíram | Taxa de churn |
|---|---|---|---|
| 1 | 5.084 | 1.409 | 27.7% |
| 2 | 4.590 | 349 | **7.6%** |
| 3 | 266 | 220 | **82.7%** |
| 4 | 60 | 60 | **100%** |

O padrão é não-linear: o ponto de menor churn é exatamente 2 produtos. A partir de 3 produtos, a taxa de saída eleva-se dramaticamente — atingindo 100% para 4 produtos.

- **1 produto:** Vínculo frágil com o banco — cliente susceptível a propostas da concorrência
- **2 produtos:** Ponto de maior engajamento e menor churn
- **3–4 produtos:** Indicativo de saturação — possível resultado de estratégias agressivas de cross-sell que geraram insatisfação

**Implicação estratégica:** Revisão das metas de cross-sell para clientes que já operam com 2 produtos. O foco de expansão deve recair sobre a transição de 1 para 2 produtos — o intervalo de maior retorno em retenção.

---

### Achado 6: Inatividade como precursor de saída

**Membros inativos: 26.9% de churn vs. 14.3% entre membros ativos. Pearson r = -0.156.**

Clientes sem engajamento recente com os serviços do banco — sem transações, acesso ao aplicativo ou uso de cartão — apresentam quase o dobro da taxa de saída em relação a clientes ativos. A inatividade funciona como um sinal precoce de desengajamento, antecedendo a decisão de encerramento da conta.

**Implicação operacional:** Monitorar quedas de atividade e acionar campanhas de reativação personalizadas antes que o cliente tome a decisão de saída.

---

### Achado 7: Clientes de maior saldo apresentam maior propensão à saída

**Saldo médio dos que saíram: €91.109 vs. €72.743 entre os que permaneceram (+25%). Cohen's d = 0.30 (efeito pequeno).**

O padrão é contra-intuitivo mas coerente: clientes com saldo elevado são exatamente os mais disputados pela concorrência, recebendo propostas ativas de migração. Adicionalmente, 36.17% dos clientes possuem saldo zero — perfil de baixo engajamento financeiro que, paradoxalmente, apresenta menor taxa de saída.

---

### Achado 8: O dataset não possui segmentos naturais nítidos — a divisão é contínua, não discreta

A análise de clustering KMeans identificou **k=2 como partição ótima**, mas com **silhouette score = 0.13** — valor muito próximo de zero, indicando que os grupos têm fronteiras difusas e grande sobreposição. O dataset não apresenta clusters bem definidos; a variação dos clientes é mais contínua do que segmentada.

Os dois segmentos identificados são essencialmente uma divisão pelo saldo em conta:

| Segmento | Clientes | Saldo médio | Produtos (média) | Churn |
|---|---|---|---|---|
| Cluster 0 — saldo baixo | 4.207 | €12.371 | 1,87 | 17,5% |
| Cluster 1 — saldo alto | 5.793 | €123.047 | 1,28 | 22,5% |

O cluster de saldo alto apresenta churn superior (22.5% vs 17.5%), em linha com o Achado 7. Clientes com maior patrimônio no banco e poucos produtos contratados são os que mais saem — reforçando que o engajamento via múltiplos produtos é um fator protetor, mas somente até 2 produtos.

**Implicação para o modelo:** Abordagens baseadas em segmentos predefinidos (ex: modelos separados por cluster) não são recomendadas para este dataset. A baixa separabilidade dos clusters indica que um modelo global, treinado sobre toda a base com features que capturem o contínuo de saldo e engajamento, terá melhor desempenho do que modelos especializados por segmento.

---

### Variáveis sem poder preditivo — e o que isso revela

Os testes estatísticos indicam ausência de associação significativa entre churn e as seguintes variáveis:

| Variável | Pearson r | p-valor | Conclusão |
|---|---|---|---|
| `EstimatedSalary` | +0.012 | 0.211 | Sem associação significativa |
| `Tenure` | -0.014 | 0.172 | Sem associação significativa |
| `Satisfaction Score` | -0.006 | 0.559 | Sem associação significativa |
| `Point Earned` | -0.005 | 0.644 | Sem associação significativa |
| `HasCrCard` | -0.007 | 0.485 | Sem associação significativa |
| `Card Type` | χ² p=0.168 | — | Sem associação significativa |

**O resultado da nota de satisfação merece atenção específica.** A ausência de correlação entre `Satisfaction Score` e `Exited` é um sinal relevante de gestão. Possíveis interpretações:
- Os clientes respondem pesquisas de satisfação sem sinceridade
- A pesquisa de satisfação está sendo aplicada no momento errado da jornada
- Os clientes decidem sair por razões racionais (produtos, preço, concorrência) sem expressar insatisfação explícita

---

## 7. Alerta Crítico: Risco de Data Leakage

### O problema com a variável `Complain`

A correlação entre `Complain` e `Exited` é de **0.9957** — uma magnitude que dificilmente ocorre em dados reais sem que haja **data leakage**: o uso inadvertido de informação futura no treinamento do modelo.

Data leakage ocorre quando uma variável preditora contém informação que, na prática, só estaria disponível **após** o evento que se deseja prever — tornando o modelo artificialmente preciso em treino, mas completamente ineficaz em produção.

**O cenário problemático:**
```
Sem leakage (correto):
  Timeline: [Dados históricos] → [PREDIÇÃO: vai sair?] → [Evento: sai ou fica]

Com leakage (errado):
  Timeline: [Dados históricos + "Reclamou?"] → [PREDIÇÃO] 
  Problema: a reclamação pode ter sido registrada APÓS a decisão de sair
```

Se a reclamação for registrada **após** o processo de churn ter iniciado, o modelo aprenderia a prever um evento já ocorrido, não um evento futuro. Em produção, a variável simplesmente não estaria disponível no momento da predição.

### Cenários possíveis

| Cenário | `Complain` deve ser usada? |
|---|---|
| A reclamação precede a saída em semanas/meses | Sim — é um sinal preditivo legítimo |
| A reclamação é registrada no mesmo momento da saída | Não — é data leakage |
| A reclamação é um campo preenchido pela equipe após confirmação de saída | Não — é definitivamente leakage |

### Recomendação

**Antes de usar `Complain` no modelo**, validar com a equipe de dados do banco:
1. Qual é o timestamp de quando uma reclamação é registrada?
2. Qual é o timestamp de quando o churn é registrado?
3. Há gap temporal entre os dois eventos?

**Para fins de experimentação e benchmark:**
- Treinar **dois modelos**: um com `Complain` (upper bound) e um sem `Complain` (modelo real)
- O modelo sem `Complain` é o que deve ir para produção

---

## 8. Recomendações para o Modelo de Machine Learning

### 8.1 Pré-processamento de features

#### Colunas a descartar
```python
COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
# Identificadores sem valor preditivo
```

#### Encoding de variáveis categóricas

| Variável | Técnica recomendada | Justificativa |
|---|---|---|
| `Geography` | One-Hot Encoding | 3 categorias, sem ordem natural (France, Germany, Spain) |
| `Gender` | Binary Encoding ou OHE | 2 categorias |
| `Card Type` | One-Hot Encoding | 4 categorias sem ordem; efeito não significativo (pode ser descartada) |


#### Escalonamento de variáveis numéricas

**Necessário apenas para modelos baseados em distância/gradiente** (Logistic Regression, SVM, KNN, redes neurais). Árvores de decisão e Random Forest são invariantes a escala.

| Variável | Técnica recomendada | Justificativa |
|---|---|---|
| `CreditScore` | StandardScaler | Distribuição quase normal, sem outliers extremos |
| `Age` | RobustScaler + log transform | Assimétrica (skew=1.01), outliers de idosos válidos |
| `Balance` | RobustScaler | Bimodal com 36% de zeros — evitar StandardScaler |
| `EstimatedSalary` | StandardScaler | Uniforme, sem outliers |
| `Point Earned` | StandardScaler | Uniforme, sem outliers |
| `Tenure` | Nenhum ou MinMaxScaler | Discreta uniforme 0–10 |
| `NumOfProducts` | Nenhum (tratar como ordinal) | Relação não-linear com target |
| `Satisfaction Score` | Nenhum (tratar como ordinal) | Escala 1–5 |

#### Feature Engineering recomendado

Com base nos padrões identificados na EDA, as seguintes features derivadas podem melhorar o modelo:

```python
# 1. Flag de saldo zero — padrão binário forte
df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)

# 2. Produto de age × is_active — combinação de sinais
df["AgeInactivity"] = df["Age"] * (1 - df["IsActiveMember"])

# 3. Flag de alto risco por produtos — relação não-linear
df["HighRiskProducts"] = (df["NumOfProducts"] >= 3).astype(int)

# 4. Score de engajamento composto
df["EngagementScore"] = (
    df["IsActiveMember"] 
    + (df["NumOfProducts"] == 2).astype(int)
    + df["HasCrCard"]
    - (df["Balance"] == 0).astype(int)
)

# 5. Faixa etária (captura relação não-linear com churn)
df["AgeGroup"] = pd.cut(df["Age"], 
    bins=[0, 30, 40, 50, 60, 100], 
    labels=["<30", "30-40", "40-50", "50-60", "60+"]
)

# 6. Razão saldo / salário estimado (indicador de engajamento financeiro)
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
```

### 8.2 Tratamento do desbalanceamento de classes

O dataset tem razão 3.91:1 (retidos:churned). Estratégias recomendadas, em ordem de preferência:

| Estratégia | Quando usar | Como |
|---|---|---|
| **class_weight='balanced'** | Primeira tentativa — simples | Parâmetro nativo no sklearn |
| **SMOTE** (oversampling) | Se class_weight não resolver | `imblearn.over_sampling.SMOTE` |
| **scale_pos_weight** | Para XGBoost/LightGBM | `scale_pos_weight = 7962 / 2038 ≈ 3.91` |
| **Threshold tuning** | Pós-treinamento | Ajustar threshold de 0.5 para ~0.3 |

```python
# XGBoost com class balancing
from xgboost import XGBClassifier
model = XGBClassifier(scale_pos_weight=3.91, ...)

# sklearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', ...)
```

### 8.3 Estratégia para dados ausentes em produção

Embora este dataset não tenha nulos, em produção recomenda-se:

| Variável | Estratégia de imputação | Justificativa |
|---|---|---|
| `CreditScore` | Mediana por Geography | Score varia por região geográfica |
| `Age` | Mediana geral (37 anos) | Distribuição assimétrica — mediana > média |
| `Balance` | Zero | 36% dos clientes têm saldo zero — é um valor legítimo |
| `EstimatedSalary` | Mediana geral (100.194) | Distribuição uniforme |
| `NumOfProducts` | Moda (1) | Valor mais comum |
| `Tenure` | Mediana (5 anos) | Distribuição uniforme |
| Categóricas | Moda | Valor mais frequente |

### 8.4 Modelos recomendados

#### Modelos a experimentar (em ordem de prioridade)

**1. XGBoost / LightGBM (recomendação principal)**
```
Motivo: Melhor para dados tabulares com relações não-lineares
        (a relação NumOfProducts × churn é altamente não-linear)
        Robusto a features sem transformação
        Suporta class_weight nativamente
Configuração inicial:
  - n_estimators: 300–500
  - max_depth: 4–6
  - learning_rate: 0.05–0.1
  - scale_pos_weight: 3.91
  - subsample: 0.8
  - colsample_bytree: 0.8
```

**2. Random Forest**
```
Motivo: Interpretável, robusto a outliers, sem necessidade de escalonamento
        Bom baseline antes de boosting
Configuração inicial:
  - n_estimators: 200–500
  - max_depth: 8–15
  - class_weight: 'balanced'
  - min_samples_leaf: 20
```

**3. Logistic Regression (baseline)**
```
Motivo: Interpretável, rápido, bom para entender importância de features
        Útil para explicabilidade ao time de negócio
Requer: escalonamento + encoding correto
Configuração: class_weight='balanced', C=0.1–1.0
```

**4. CatBoost**
```
Motivo: Lida nativamente com variáveis categóricas (sem OHE manual)
        Excelente para Geography, Gender, Card Type
```

#### Modelos a evitar ou usar com cuidado
- **KNN:** Caro computacionalmente com 10k linhas; sensível a escala e features irrelevantes
- **SVM:** Escalabilidade ruim para produção; difícil de calibrar probabilidades
- **Naive Bayes:** Assume independência entre features — violada (NumOfProducts × Balance correlacionados)

### 8.5 Métricas de avaliação

> **Accuracy não é uma métrica adequada para dados desbalanceados.** Um modelo que classifica todos os clientes como "vai ficar" atingiria 79.62% de acurácia — resultado aparentemente satisfatório, mas com valor preditivo nulo para a classe de interesse.

#### Métricas primárias recomendadas

| Métrica | Por que usar | Meta inicial |
|---|---|---|
| **ROC-AUC** | Mede discriminação geral; independente de threshold | > 0.85 |
| **F1-Score (classe churn)** | Balanceia precisão e recall para a classe minoritária | > 0.60 |
| **Recall (classe churn)** | Minimiza falsos negativos — não perder clientes em risco | > 0.70 |
| **Average Precision (PR-AUC)** | Melhor que ROC-AUC quando há desbalanceamento | > 0.60 |

#### Métricas de negócio

```python
# Custo estimado:
CUSTO_FALSO_NEGATIVO  = 500  # perda de receita anual por cliente perdido não identificado
CUSTO_FALSO_POSITIVO  = 20   # custo de campanha de retenção desnecessária

# Profit curve para threshold otimization
def business_profit(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # perdeu o cliente
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # campanha desnecessária
    return -(fn * CUSTO_FALSO_NEGATIVO + fp * CUSTO_FALSO_POSITIVO)
```

#### Matriz de confusão interpretada

```
                    Previsto: Fica    Previsto: Sai
Real: Fica (7.962)     TN               FP
                    Classificação      Ação de retenção
                    correta            desnecessária
                                       Custo: ~R$20/cliente

Real: Sai (2.038)      FN               TP
                    Cliente perdido    Intervenção bem-
                    sem intervenção    sucedida
                    Custo: ~R$500      Receita preservada
```

### 8.6 Performance baseline esperada

Com base nas características do dataset e na literatura para problemas similares:

| Cenário | ROC-AUC esperado | Notas |
|---|---|---|
| **Com `Complain` (upper bound)** | 0.99–1.00 | Leakage — não usar em produção |
| **Sem `Complain`, XGBoost** | 0.82–0.88 | Target realista para este dataset |
| **Sem `Complain`, Random Forest** | 0.78–0.84 | Bom baseline |
| **Sem `Complain`, Logistic Reg.** | 0.72–0.78 | Baseline mínimo |
| **Modelo trivial (sempre prediz "fica")** | 0.50 | Piso do ROC-AUC |

**Recall para classe churn esperado (sem `Complain`):**
- XGBoost otimizado: 0.65–0.75
- Com SMOTE + XGBoost: 0.70–0.80

### 8.7 Configuração de validação cruzada

```python
from sklearn.model_selection import StratifiedKFold

# Manter proporção de classes em cada fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Split train/test estratificado
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y  # <- essencial para manter 80/20 em cada split
)
```

### 8.8 Features a excluir do modelo final

| Feature | Motivo |
|---|---|
| `Complain` | Leakage potencial — investigar antes de usar |
| `Card Type` | Sem associação significativa com churn (p=0.168) |
| `HasCrCard` | Sem associação significativa (p=0.485) |
| `EstimatedSalary` | Sem associação significativa (p=0.211) |
| `Point Earned` | Sem associação significativa (p=0.644) |
| `Satisfaction Score` | Sem associação significativa (p=0.559) |

> Atenção: "sem associação linear" não significa "sem valor preditivo". Modelos não-lineares (Random Forest, XGBoost) podem extrair valor dessas features via interações. Usar SHAP values para confirmação.

---

## 9. Parâmetros derivados para Great Expectations

Esta seção consolida os parâmetros calculados da EDA para atualização direta do `config/quality.yaml`.

> Todos os valores foram derivados do dataset real usando percentis robustos (p01–p99) + tolerâncias estatísticas.

### 9.1 Expectations de tabela

```yaml
table_expectations:
  - type: expect_table_row_count_to_be_between
    kwargs:
      min_value: 8000    # 10.000 × 0.80 — tolerância de -20%
      max_value: 12000   # 10.000 × 1.20 — tolerância de +20%

  - type: expect_table_columns_to_match_set
    kwargs:
      column_set:
        - CreditScore
        - Geography
        - Gender
        - Age
        - Tenure
        - Balance
        - NumOfProducts
        - HasCrCard
        - IsActiveMember
        - EstimatedSalary
        - Exited
        - Complain
        - Satisfaction Score
        - Card Type
        - Point Earned
      exact_match: true
```

### 9.2 Expectations de coluna — Numéricas contínuas

```yaml
column_expectations:

  CreditScore:
    # Dados observados: min=350, max=850, média=650.5, std=96.7
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 432    # p01 observado
        max_value: 850    # p99 observado (máximo histórico de score)
        mostly: 0.99
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 457.2  # média - 2σ = 650.5 - 2×96.7
        max_value: 843.8  # média + 2σ = 650.5 + 2×96.7
    - type: expect_column_stdev_to_be_between
      kwargs:
        min_value: 48.3   # std × 0.5 = 96.7 × 0.5
        max_value: 145.0  # std × 1.5 = 96.7 × 1.5

  Age:
    # Dados observados: min=18, max=92, média=38.9, std=10.5, assimétrico
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 21     # p01 observado (mínimo legal realista = 18)
        max_value: 72     # p99 observado (alguns clientes chegam a 92, mas são raros)
        mostly: 0.99
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 17.9   # média - 2σ
        max_value: 59.9   # média + 2σ
    - type: expect_column_stdev_to_be_between
      kwargs:
        min_value: 5.2    # std × 0.5
        max_value: 15.7   # std × 1.5

  Balance:
    # ATENÇÃO: 36.17% dos clientes têm saldo = 0 — comportamento normal
    # Dados observados: min=0, max=250.898, mediana=97.199
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 0      # Zero é valor legítimo e frequente
        max_value: 250898 # Máximo histórico observado
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 0      # Média pode variar bastante (bimodal)
        max_value: 201281 # média + 2σ = 76.486 + 2×62.397

  EstimatedSalary:
    # Dados observados: min=11.58, max=199.992, uniforme
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 0        # Não pode ser negativo
        max_value: 200000   # Limite observado (próximo de 200k)
        mostly: 0.99
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 0        # Média - 2σ (pode ser uniforme em outro batch)
        max_value: 215111   # média + 2σ = 100.090 + 2×57.510

  Point Earned:
    # Dados observados: min=119, max=1000, uniforme
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 119    # Mínimo histórico (pode ser 0 em prod com clientes novos)
        max_value: 1000   # Máximo do programa de pontos
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 154.7  # média - 2σ = 606.5 - 2×225.9
        max_value: 1058.4 # média + 2σ (truncar em 1000 na prática)
```

### 9.3 Expectations de coluna — Numéricas discretas

```yaml
  Tenure:
    # Dados observados: min=0, max=10, média=5.01, uniforme
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 0    # Cliente novo
        max_value: 10   # Máximo histórico = 10 anos

  NumOfProducts:
    # ATENÇÃO: relação não-linear com churn. 3-4 produtos = altíssimo churn
    # Dados observados: min=1, max=4, mediana=1
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 1    # Mínimo: pelo menos 1 produto
        max_value: 4    # Máximo histórico observado

  Satisfaction Score:
    # Escala fixa 1–5
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 1    # Pior nota
        max_value: 5    # Melhor nota
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: [1, 2, 3, 4, 5]
```

### 9.4 Expectations de coluna — Binárias

```yaml
  HasCrCard:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: [0, 1]

  IsActiveMember:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: [0, 1]

  Exited:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: [0, 1]

  Complain:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: [0, 1]
```

### 9.5 Expectations de coluna — Categóricas

```yaml
  Geography:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: ["France", "Germany", "Spain"]
    # Distribuição observada: France=50.1%, Germany=25.1%, Spain=24.8%
    # Alerta se Germany > 40% (mudança de distribuição suspeita)

  Gender:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: ["Male", "Female"]

  Card Type:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: ["DIAMOND", "GOLD", "SILVER", "PLATINUM"]
    # Distribuição perfeitamente balanceada (~25% cada)
    # Qualquer desvio grande indicaria problema de encoding
```

---

## Outputs gerados pelo script EDA

```
outputs/eda/
├── eda_report_20260411_114506.json      ← Relatório completo (10 seções)
└── plots/
    ├── 01_target_distribution.png       ← Distribuição do churn (pizza + barras)
    ├── 02_numeric_distributions.png     ← Histogramas de todas as numéricas
    ├── 03_numeric_by_target_boxplot.png ← Boxplots: numéricas × churn
    ├── 04_correlation_heatmap.png       ← Heatmap Pearson (triângulo inferior)
    ├── 05_categorical_by_target.png     ← Geografia, Gênero, Cartão × churn (%)
    ├── 06_age_distribution_by_churn.png ← KDE de idade: retidos vs saídos
    ├── 07_balance_distribution.png      ← Distribuição de saldo (com zeros)
    ├── 08_creditscore_by_geography.png  ← CreditScore por país e churn
    ├── 09_satisfaction_and_points.png   ← Satisfação e pontos × churn
    └── 10_products_and_tenure_churn.png ← Taxa de churn por produtos e tenure
```

---

*Documento gerado a partir da análise exploratória realizada pelo script `eda/eda_analysis.py`.*
*Para regenerar: `python eda/eda_analysis.py` (com venv ativa)*
