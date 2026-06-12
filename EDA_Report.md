# Exploratory Data Analysis — Bank Customer Churn

> **Dataset:** `Customer-Churn-Records.csv` — Kaggle (`radheshyamkollipara/bank-customer-churn`)
> **Analysis generated on:** April 11, 2026
> **Script:** `eda/eda_analysis.py` | **JSON Report:** `outputs/eda/eda_report_20260411_114506.json`

---

## Table of Contents

1. [Business context and strategic value](#1-business-context-and-strategic-value)
2. [Technical overview of the dataset](#2-technical-overview-of-the-dataset)
3. [Data quality](#3-data-quality)
4. [Distribution of the target variable — Churn](#4-distribution-of-the-target-variable--churn)
5. [Analysis of variables](#5-analysis-of-variables)
6. [Key findings and strategic implications](#6-key-findings-and-strategic-implications)
7. [Critical alert: risk of data leakage](#7-critical-alert-risk-of-data-leakage)
8. [Recommendations for the Machine Learning model](#8-recommendations-for-the-machine-learning-model)
9. [Parameters derived for Great Expectations](#9-parameters-derived-for-great-expectations)

---

## 1. Business Context and Strategic Value

**Churn** — the voluntary departure of customers — is one of the primary drivers of recurring revenue loss in the banking sector. This dataset records the history of **10,000 customers** from a European bank with operations in **France, Germany, and Spain**, consolidating behavioral, financial, and profile variables for each customer. The target variable (`Exited`) indicates whether the customer terminated the relationship with the bank.

The analytical objective is to identify proactively and in advance which customers have the highest probability of leaving — enabling targeted retention interventions before the decision is finalized.

The available variables cover four dimensions:

- **Financial profile:** credit score, account balance, estimated salary
- **Relationship with the bank:** account tenure, number of products contracted, activity, accumulated points, credit card ownership, card type
- **Demographic data:** age, gender, country
- **Customer experience:** satisfaction score, complaint history

### Financial Impact of Churn

The cost of acquiring new customers tends to be significantly higher than the cost of retaining existing ones. In portfolios with substantial customer volumes, even moderate churn rates result in:

- relevant loss of active base throughout the annual cycle
- reduction in recurring revenue from fees, credit operations, and complementary products
- effort to rebuild through acquisition that generally exceeds investment in preventive retention actions

---

## 2. Technical Overview of the Dataset

| Property | Value |
|---|---|
| Total rows | 10,000 |
| Total columns | 18 |
| Memory | 1.68 MB |
| Missing values | **0** (0.0%) |
| Duplicate rows | **0** |
| Unique customer IDs | 10,000 |

### Column Schema

| Column | Type | Category | Description |
|---|---|---|---|
| `RowNumber` | int64 | Identifier | Sequential index |
| `CustomerId` | int64 | Identifier | Unique customer ID |
| `Surname` | string | Identifier | Last name |
| `CreditScore` | int64 | Continuous numeric | Credit score (350–850) |
| `Geography` | string | Nominal categorical | Country: France, Germany, Spain |
| `Gender` | string | Nominal categorical | Gender: Male, Female |
| `Age` | int64 | Continuous numeric | Age (18–92 years) |
| `Tenure` | int64 | Discrete numeric | Years as customer (0–10) |
| `Balance` | float64 | Continuous numeric | Account balance (0–250,898) |
| `NumOfProducts` | int64 | Discrete numeric | Products contracted (1–4) |
| `HasCrCard` | int64 | Binary | Has credit card: 1=Yes, 0=No |
| `IsActiveMember` | int64 | Binary | Active member: 1=Yes, 0=No |
| `EstimatedSalary` | float64 | Continuous numeric | Estimated salary (11–199,992) |
| `Exited` | int64 | **Target** | Left the bank: 1=Yes, 0=No |
| `Complain` | int64 | Binary | Registered complaint: 1=Yes, 0=No |
| `Satisfaction Score` | int64 | Ordinal | Satisfaction: 1 (very poor) to 5 (excellent) |
| `Card Type` | string | Nominal categorical | Card type: DIAMOND, GOLD, SILVER, PLATINUM |
| `Point Earned` | int64 | Discrete numeric | Loyalty points accumulated (119-1,000) |

---

## 3. Data Quality

### Missing Values

**Result:** None of the 180,000 cells contain null values.

```
Total cells      : 180,000
Missing cells    : 0  (0.0000%)
Affected columns : none
```

**Implication for MLOps:** No imputation strategy is necessary for this dataset. However, in production (real bank data), nulls are likely to emerge — especially in `Balance` (new customers without history) and `CreditScore` (customers without formal credit analysis). Preventive strategies are detailed in [section 8](#8-recommendations-for-the-machine-learning-model).

### Duplicates

```
Exact duplicates    : 0
Feature duplicates  : 0  
```

Each row represents a unique and distinct customer.

### Distributions and Outliers

| Column | IQR Outliers | Z>3 Outliers | Observation |
|---|---|---|---|
| `CreditScore` | 15 (0.15%) | 8 (0.08%) | Negligible |
| `Age` | 359 (3.59%) | 133 (1.33%) | Customers over 62 years — valid data |
| `Balance` | 0 (0.0%) | 0 (0.0%) | Bimodal: 36.17% of customers have zero balance |
| `NumOfProducts` | 60 (0.6%) | 60 (0.6%) | 4 products = signal of caution |
| `Tenure` | 0 (0.0%) | 0 (0.0%) | Uniform distribution |
| `EstimatedSalary` | 0 (0.0%) | 0 (0.0%) | Uniform distribution |
| `Satisfaction Score` | 0 (0.0%) | 0 (0.0%) | Uniform distribution |
| `Point Earned` | 0 (0.0%) | 0 (0.0%) | Uniform distribution |

Identified outliers were retained as they represent real business behaviors. The 15 extreme cases of `CreditScore` are statistically insignificant (0.15%) and do not distort the distribution. `Balance` has no outliers but presents a bimodal distribution — 36.17% of customers have zero balance, reflecting two distinct profiles of banking usage that should not be treated as anomalies. The 359 elderly customers flagged in `Age` are demographically valid; removing them would introduce bias and eliminate a relevant segment for churn prediction. The 60 customers with 4 products in `NumOfProducts`, though rare (0.6%), carry high predictive potential and were retained. Model performance on these subgroups will be verified in the evaluation stage to detect any predictive disparities.

---

## 4. Distribution of the Target Variable — Churn

```
Total customers      : 10,000
Customers who left   : 2,038  (20.38%)
Customers retained   : 7,962  (79.62%)
Imbalance ratio      : 3.91:1  (retained:exited)
```

### The Class Imbalance Problem

The 80/20 distribution represents a **moderate imbalance** — not severe, but significant enough to distort models.

**Without imbalance treatment**, a model that classifies all customers as "retained" would achieve 79.62% accuracy — a numerically high result, but with zero operational value, as it identifies no churn cases.

**Why this matters to the business:** In customer retention, the cost of a false negative (failing to identify who will leave and losing the customer) is much higher than the cost of a false positive. The model strategy should reflect this asymmetric cost.

---

## 5. Analysis of Variables

### Numeric Variables — Complete Statistics

| Variable | Min | Max | Mean | Median | Std | Skewness | IQR Outliers |
|---|---|---|---|---|---|---|---|
| `CreditScore` | 350 | 850 | 650.5 | 652.0 | 96.7 | -0.07 (symmetric) | 0.15% |
| `Age` | 18 | 92 | 38.9 | 37.0 | 10.5 | +1.01 (right-skewed) | 3.59% |
| `Tenure` | 0 | 10 | 5.01 | 5.0 | 2.89 | ~0 (uniform) | 0.0% |
| `Balance` | 0 | 250,898 | 76,486 | 97,199 | 62,397 | -0.14 (bimodal) | 0.0% |
| `NumOfProducts` | 1 | 4 | 1.53 | 1.0 | 0.58 | +0.75 (skewed) | 0.6% |
| `EstimatedSalary` | 12 | 199,992 | 100,090 | 100,194 | 57,510 | ~0 (uniform) | 0.0% |
| `Satisfaction Score` | 1 | 5 | 3.01 | 3.0 | 1.41 | ~0 (uniform) | 0.0% |
| `Point Earned` | 119 | 1,000 | 606.5 | 605.0 | 225.9 | ~0 (uniform) | 0.0% |

**Technical observations:**
- **`CreditScore`:** Nearly symmetric distribution, but does not follow normal distribution (p < 0.05 in both tests).
- **`Age`:** Distribution strongly right-skewed. Most customers are young-adult (median 37 years), but with a long tail extending to elderly customers up to 92 years.
- **`Balance`:** Critical bimodal variable. 36.17% of customers have zero balance — this represents customers who keep the account open but do not use it. The remaining (63.83%) have average balance of ~120,000.
- **`Tenure`:** Completely uniform between 0 and 10 years. No linear predictive power with churn.
- **`EstimatedSalary`:** Uniformly distributed between ~0 and ~200k. Likely synthetically generated in this dataset.

### Categorical Variables — Distribution

**Geography:**
| Country | Customers | % of Total | Churn Rate |
|---|---|---|---|
| France | 5,014 | 50.14% | 16.2% |
| Germany | 2,509 | 25.09% | **32.4%** |
| Spain | 2,477 | 24.77% | 16.7% |

**Gender:**
| Gender | Customers | % of Total | Churn Rate |
|---|---|---|---|
| Male | 5,457 | 54.57% | 16.5% |
| Female | 4,543 | 45.43% | **25.1%** |

**Card Type:**
| Type | Customers | % of Total | Churn Rate |
|---|---|---|---|
| DIAMOND | 2,507 | 25.07% | 21.8% |
| GOLD | 2,502 | 25.02% | 19.3% |
| PLATINUM | 2,495 | 24.95% | 20.4% |
| SILVER | 2,496 | 24.96% | 20.1% |

> Card type is distributed perfectly balanced (≈25% each) and **has no significant association with churn** (χ² p=0.168, Cramér's V=0.02).

---

## 6. Key Findings and Strategic Implications

The data reveals behavior patterns with direct relevance to the retention strategy. Each finding is presented with statistical backing and corresponding operational implication.

---

### Finding 1: Complaint is the Strongest Indicator of Departure

**Pearson correlation between `Complain` and `Exited` = 0.9957**

| Registered complaint? | Total | Left | Churn Rate |
|---|---|---|---|
| No | 7,956 | 4 | **0.05%** |
| Yes | 2,044 | 2,034 | **99.5%** |

The correlation is virtually perfect: customers who registered a complaint left in 99.5% of cases. The inverse is equally striking — among customers without complaints, only 0.05% terminated the relationship.

**Strategic implication:** Every unresolved complaint converts, in virtually all cases, into a lost customer. Investment in complaint resolution capacity represents a direct lever for retention.

**Alert for the predictive model:** The strength of this correlation raises a data integrity question. See [section 7](#7-critical-alert-risk-of-data-leakage).

---

### Finding 2: Older Customers Show Higher Propensity for Departure

**Average age of customers who left: 44.8 years vs. 37.4 years among those retained. Cohen's d = 0.747 (medium-large effect).**

The difference of ~7 years between groups is statistically robust (p ≈ 0, Mann-Whitney U). This suggests that:

- Older customers may have longer-standing relationships with other banks and greater ease migrating
- The bank may not be offering products suitable for mature customer profiles (retirement planning, wealth management, etc.)
- Retention campaigns should prioritize the 40–60 age range

---

### Finding 3: German Operations Concentrate Double the Churn of Other Markets

**Germany: 32.4% churn vs. 16.2% in France and 16.7% in Spain. Moderate association (Cramér's V = 0.17, p < 0.0001).**

A German customer has approximately double the probability of departure relative to French or Spanish customers. Hypotheses warranting investigation by the business team:
- The bank may have a less competitive operation in Germany
- Cultural differences in banking relationships (Germans tend to demand higher service quality)
- More intense banking competition in the German market
- Localized service or product issues

**The largest structural effect in the dataset: German customers have average balance almost double the others.**

| Country | Average Balance |
|---|---|
| France | €62,093 |
| Germany | **€119,730** |
| Spain | €61,818 |

The balance difference between Germany and other markets is the **largest numerical association observed in the entire analysis** (ANOVA F=958, p≈0, eta²=0.16 — large effect). German customers are on average the wealthiest in the portfolio — and therefore the most attractive targets for competitor banks. This provides a structural explanation for elevated churn: not an isolated service quality issue, but a market effect.

**The Germany effect is independent of credit score.**

When segmenting churn by credit score quartile within each country, the pattern remains stable across all levels:

| Credit Score Quartile | France | Germany | Spain |
|---|---|---|---|
| Q1 (lowest score) | 17.3% | **35.6%** | 17.6% |
| Q2 | 17.5% | **32.0%** | 16.6% |
| Q3 | 15.1% | **28.1%** | 15.4% |
| Q4 (highest score) | 14.8% | **33.7%** | 17.1% |

Regardless of customer credit quality, the probability of churn in Germany is always double the other markets. This rules out the hypothesis that the German problem is caused by defaults or risk profile — the phenomenon is systemic to local operations.

Additionally, chi-squared analysis shows that German customers proportionally file more complaints (Cramér's V = 0.175, p < 0.0001) — the same association coefficient that the `Geography` variable itself has with `Exited`.

---

### Finding 4: Female Customers Show Significantly Higher Churn

**Women: 25.1% churn vs. 16.5% among men (Cramér's V = 0.11, p < 0.0001).**

Female customers show 52% higher probability of departure relative to male customers. This differential may indicate misalignment between the product portfolio and bank communications with the female profile, or unmet specific service expectations.

---

### Finding 5: Product Concentration Above 2 is Strongly Associated with Departure

| No. of Products | Total | Left | Churn Rate |
|---|---|---|---|
| 1 | 5,084 | 1,409 | 27.7% |
| 2 | 4,590 | 349 | **7.6%** |
| 3 | 266 | 220 | **82.7%** |
| 4 | 60 | 60 | **100%** |

The pattern is non-linear: the lowest churn point is exactly 2 products. Beyond 3 products, the exit rate rises dramatically — reaching 100% for 4 products.

- **1 product:** Fragile bond with the bank — customer susceptible to competitor offers
- **2 products:** Peak engagement point and lowest churn
- **3–4 products:** Signal of saturation — possible result of aggressive cross-sell strategies that generated dissatisfaction

**Strategic implication:** Review cross-sell targets for customers already operating with 2 products. Expansion focus should fall on the 1-to-2 product transition — the interval of highest retention return.

---

### Finding 6: Inactivity as a Precursor to Departure

**Inactive members: 26.9% churn vs. 14.3% among active members. Pearson r = -0.156.**

Customers without recent engagement with bank services — no transactions, app access, or card usage — show nearly double the exit rate relative to active customers. Inactivity functions as an early signal of disengagement, preceding the departure decision.

**Operational implication:** Monitor activity declines and trigger personalized reactivation campaigns before the customer decides to leave.

---

### Finding 7: Higher-Balance Customers Show Greater Propensity for Departure

**Average balance of those who left: €91,109 vs. €72,743 among those retained (+25%). Cohen's d = 0.30 (small effect).**

The pattern is counter-intuitive but coherent: customers with high balances are precisely those most targeted by competitors, receiving active migration offers. Additionally, 36.17% of customers have zero balance — a profile of low financial engagement that paradoxically shows lower exit rates.

---

### Finding 8: Dataset Has No Clear Natural Segments — Division is Continuous, Not Discrete

KMeans clustering analysis identified **k=2 as optimal partition**, but with **silhouette score = 0.13** — a value very close to zero, indicating groups have fuzzy boundaries and substantial overlap. The dataset presents no well-defined clusters; customer variation is more continuous than segmented.

The two identified segments are essentially a division by account balance:

| Segment | Customers | Average Balance | Products (avg) | Churn |
|---|---|---|---|---|
| Cluster 0 — low balance | 4,207 | €12,371 | 1.87 | 17.5% |
| Cluster 1 — high balance | 5,793 | €123,047 | 1.28 | 22.5% |

The high-balance cluster shows superior churn (22.5% vs 17.5%), aligned with Finding 7. Customers with greater bank wealth and few contracted products are the ones departing most — reinforcing that product engagement is a protective factor, but only up to 2 products.

**Implication for the model:** Approaches based on predefined segments (e.g., separate models by cluster) are not recommended for this dataset. Low cluster separability indicates that a global model trained on the entire base with features capturing the balance and engagement continuum will outperform specialized segment models.

---

### Variables Without Predictive Power — and What This Reveals

Statistical tests indicate absence of significant association between churn and the following variables:

| Variable | Pearson r | p-value | Conclusion |
|---|---|---|---|
| `EstimatedSalary` | +0.012 | 0.211 | No significant association |
| `Tenure` | -0.014 | 0.172 | No significant association |
| `Satisfaction Score` | -0.006 | 0.559 | No significant association |
| `Point Earned` | -0.005 | 0.644 | No significant association |
| `HasCrCard` | -0.007 | 0.485 | No significant association |
| `Card Type` | χ² p=0.168 | — | No significant association |

**The satisfaction score result merits specific attention.** Absence of correlation between `Satisfaction Score` and `Exited` signals something relevant about management. Possible interpretations:
- Customers respond satisfaction surveys insincerely
- The satisfaction survey is applied at the wrong journey moment
- Customers decide to leave for rational reasons (products, price, competition) without expressing explicit dissatisfaction

---

## 7. Critical Alert: Risk of Data Leakage

### The Problem with the `Complain` Variable

The correlation between `Complain` and `Exited` is **0.9957** — a magnitude that rarely occurs in real data without **data leakage**: the inadvertent use of future information in model training.

Data leakage occurs when a predictor variable contains information that, in practice, would only be available **after** the event being predicted — making the model artificially accurate in training, but completely ineffective in production.

**The problematic scenario:**
```
Without leakage (correct):
  Timeline: [Historical Data] → [PREDICTION: will leave?] → [Event: leaves or stays]

With leakage (wrong):
  Timeline: [Historical Data + "Complained?"] → [PREDICTION] 
  Problem: the complaint may have been registered AFTER departure decision began
```

If the complaint is registered **after** the churn process has commenced, the model would learn to predict an already-occurred event, not a future event. In production, the variable simply would not be available at prediction time.

### Possible Scenarios

| Scenario | Should `Complain` be used? |
|---|---|
| Complaint precedes departure by weeks/months | Yes — is a legitimate predictive signal |
| Complaint registered at same time as departure | No — is data leakage |
| Complaint is a field filled by staff after departure confirmation | No — is definitely leakage |

### Recommendation

**Before using `Complain` in the model**, validate with the bank's data team:
1. What is the timestamp for when a complaint is registered?
2. What is the timestamp for when churn is registered?
3. Is there a time gap between the two events?

**For experimentation and benchmarking purposes:**
- Train **two models**: one with `Complain` (upper bound) and one without `Complain` (real model)
- The model without `Complain` is the one that should go to production

---

## 8. Recommendations for the Machine Learning Model

### 8.1 Feature Pre-processing

#### Columns to Drop
```python
COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
# Identifiers with no predictive value
```

#### Encoding Categorical Variables

| Variable | Recommended Technique | Justification |
|---|---|---|
| `Geography` | One-Hot Encoding | 3 categories, no natural order (France, Germany, Spain) |
| `Gender` | Binary Encoding or OHE | 2 categories |
| `Card Type` | One-Hot Encoding | 4 categories without order; effect not significant (can be dropped) |

#### Scaling Numeric Variables

**Necessary only for distance/gradient-based models** (Logistic Regression, SVM, KNN, neural networks). Decision trees and Random Forest are scale-invariant.

| Variable | Recommended Technique | Justification |
|---|---|---|
| `CreditScore` | StandardScaler | Nearly normal distribution, no extreme outliers |
| `Age` | RobustScaler + log transform | Skewed (skew=1.01), valid elderly outliers |
| `Balance` | RobustScaler | Bimodal with 36% zeros — avoid StandardScaler |
| `EstimatedSalary` | StandardScaler | Uniform, no outliers |
| `Point Earned` | StandardScaler | Uniform, no outliers |
| `Tenure` | None or MinMaxScaler | Discrete uniform 0–10 |
| `NumOfProducts` | None (treat as ordinal) | Non-linear relationship with target |
| `Satisfaction Score` | None (treat as ordinal) | 1–5 scale |

#### Recommended Feature Engineering

Based on patterns identified in the EDA, the following derived features may improve model performance:

```python
# 1. Zero balance flag — strong binary pattern
df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)

# 2. Product of age × is_active — combination of signals
df["AgeInactivity"] = df["Age"] * (1 - df["IsActiveMember"])

# 3. High-risk products flag — non-linear relationship
df["HighRiskProducts"] = (df["NumOfProducts"] >= 3).astype(int)

# 4. Composite engagement score
df["EngagementScore"] = (
    df["IsActiveMember"] 
    + (df["NumOfProducts"] == 2).astype(int)
    + df["HasCrCard"]
    - (df["Balance"] == 0).astype(int)
)

# 5. Age group (captures non-linear relationship with churn)
df["AgeGroup"] = pd.cut(df["Age"], 
    bins=[0, 30, 40, 50, 60, 100], 
    labels=["<30", "30-40", "40-50", "50-60", "60+"]
)

# 6. Balance to estimated salary ratio (financial engagement indicator)
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
```

### 8.2 Class Imbalance Treatment

The dataset has a 3.91:1 ratio (retained:churned). Recommended strategies in order of preference:

| Strategy | When to Use | How |
|---|---|---|
| **class_weight='balanced'** | First attempt — simple | Native parameter in sklearn |
| **SMOTE** (oversampling) | If class_weight insufficient | `imblearn.over_sampling.SMOTE` |
| **scale_pos_weight** | For XGBoost/LightGBM | `scale_pos_weight = 7962 / 2038 ≈ 3.91` |
| **Threshold tuning** | Post-training | Adjust threshold from 0.5 to ~0.3 |

```python
# XGBoost with class balancing
from xgboost import XGBClassifier
model = XGBClassifier(scale_pos_weight=3.91, ...)

# sklearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', ...)
```

### 8.3 Missing Data Strategy for Production

Although this dataset has no nulls, production deployment should account for:

| Variable | Imputation Strategy | Justification |
|---|---|---|
| `CreditScore` | Median by Geography | Score varies by geographic region |
| `Age` | Median overall (37 years) | Skewed distribution — median > mean |
| `Balance` | Zero | 36% of customers have zero balance — is legitimate value |
| `EstimatedSalary` | Median overall (100,194) | Uniform distribution |
| `NumOfProducts` | Mode (1) | Most common value |
| `Tenure` | Median (5 years) | Uniform distribution |
| Categorical | Mode | Most frequent value |

### 8.4 Recommended Models

#### Models to Experiment With (in priority order)

**1. XGBoost / LightGBM (primary recommendation)**
```
Reason: Best for tabular data with non-linear relationships
        (NumOfProducts × churn relationship is highly non-linear)
        Robust to features without transformation
        Supports class_weight natively
Initial configuration:
  - n_estimators: 300–500
  - max_depth: 4–6
  - learning_rate: 0.05–0.1
  - scale_pos_weight: 3.91
  - subsample: 0.8
  - colsample_bytree: 0.8
```

**2. Random Forest**
```
Reason: Interpretable, robust to outliers, no scaling needed
        Good baseline before boosting
Initial configuration:
  - n_estimators: 200–500
  - max_depth: 8–15
  - class_weight: 'balanced'
  - min_samples_leaf: 20
```

**3. Logistic Regression (baseline)**
```
Reason: Interpretable, fast, useful for understanding feature importance
        Useful for business team explainability
Requires: proper scaling + encoding
Configuration: class_weight='balanced', C=0.1–1.0
```

**4. CatBoost**
```
Reason: Handles categorical variables natively (no manual OHE)
        Excellent for Geography, Gender, Card Type
```

#### Models to Avoid or Use with Caution
- **KNN:** Computationally expensive with 10k rows; sensitive to scale and irrelevant features
- **SVM:** Poor scalability for production; difficult to calibrate probabilities
- **Naive Bayes:** Assumes feature independence — violated (NumOfProducts × Balance correlated)

### 8.5 Evaluation Metrics

> **Accuracy is not an appropriate metric for imbalanced data.** A model classifying all customers as "will stay" would achieve 79.62% accuracy — apparently satisfactory, but with zero predictive value for the target class.

#### Primary Recommended Metrics

| Metric | Why Use | Initial Target |
|---|---|---|
| **ROC-AUC** | Measures general discrimination; threshold-independent | > 0.85 |
| **F1-Score (churn class)** | Balances precision and recall for minority class | > 0.60 |
| **Recall (churn class)** | Minimizes false negatives — avoid missing at-risk customers | > 0.70 |
| **Average Precision (PR-AUC)** | Better than ROC-AUC when imbalanced | > 0.60 |

#### Business Metrics

```python
# Estimated costs:
COST_FALSE_NEGATIVE  = 500  # annual revenue loss per unidentified churned customer
COST_FALSE_POSITIVE  = 20   # cost of unnecessary retention campaign

# Profit curve for threshold optimization
def business_profit(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # lost customer
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # unnecessary campaign
    return -(fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE)
```

#### Confusion Matrix Interpretation

```
                    Predicted: Stays    Predicted: Leaves
Real: Stays (7,962)      TN                  FP
                    Correct               Unnecessary retention
                    classification        action
                                         Cost: ~$20/customer

Real: Leaves (2,038)     FN                  TP
                    Lost customer        Successful
                    without intervention intervention
                    Cost: ~$500          Revenue preserved
```

### 8.6 Expected Baseline Performance

Based on dataset characteristics and literature for similar problems:

| Scenario | Expected ROC-AUC | Notes |
|---|---|---|
| **With `Complain` (upper bound)** | 0.99–1.00 | Leakage — do not use in production |
| **Without `Complain`, XGBoost** | 0.82–0.88 | Realistic target for this dataset |
| **Without `Complain`, Random Forest** | 0.78–0.84 | Good baseline |
| **Without `Complain`, Logistic Reg.** | 0.72–0.78 | Minimum baseline |
| **Trivial model (always predicts "stays")** | 0.50 | ROC-AUC floor |

**Expected Recall for churn class (without `Complain`):**
- Optimized XGBoost: 0.65–0.75
- With SMOTE + XGBoost: 0.70–0.80

### 8.7 Cross-Validation Setup

```python
from sklearn.model_selection import StratifiedKFold

# Maintain class proportion in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stratified train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y  # <- essential to maintain 80/20 in each split
)
```

### 8.8 Features to Exclude from Final Model

| Feature | Reason |
|---|---|
| `Complain` | Potential leakage — investigate before using |
| `Card Type` | No significant association with churn (p=0.168) |
| `HasCrCard` | No significant association (p=0.485) |
| `EstimatedSalary` | No significant association (p=0.211) |
| `Point Earned` | No significant association (p=0.644) |
| `Satisfaction Score` | No significant association (p=0.559) |

> Caution: "No linear association" does not mean "no predictive value." Non-linear models (Random Forest, XGBoost) may extract value from these features via interactions. Use SHAP values for confirmation.

---

## 9. Parameters Derived for Great Expectations

This section consolidates parameters calculated from the EDA for direct updating of `config/quality.yaml`.

> All values were derived from the real dataset using robust percentiles (p01–p99) + statistical tolerances.

### 9.1 Table Expectations

```yaml
table_expectations:
  - type: expect_table_row_count_to_be_between
    kwargs:
      min_value: 8000    # 10,000 × 0.80 — tolerance of -20%
      max_value: 12000   # 10,000 × 1.20 — tolerance of +20%

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

### 9.2 Expectations for Numeric Continuous Columns

```yaml
column_expectations:

  CreditScore:
    # Observed data: min=350, max=850, mean=650.5, std=96.7
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 432    # p01 observed
        max_value: 850    # p99 observed (historical maximum score)
        mostly: 0.99
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 457.2  # mean - 2σ = 650.5 - 2×96.7
        max_value: 843.8  # mean + 2σ = 650.5 + 2×96.7
    - type: expect_column_stdev_to_be_between
      kwargs:
        min_value: 48.3   # std × 0.5 = 96.7 × 0.5
        max_value: 145.0  # std × 1.5 = 96.7 × 1.5

  Age:
    # Observed data: min=18, max=92, mean=38.9, std=10.5, skewed
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 21     # p01 observed (legal minimum realistic = 18)
        max_value: 72     # p99 observed (some customers reach 92, but rare)
        mostly: 0.99
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 17.9   # mean - 2σ
        max_value: 59.9   # mean + 2σ
    - type: expect_column_stdev_to_be_between
      kwargs:
        min_value: 5.2    # std × 0.5
        max_value: 15.7   # std × 1.5

  Balance:
    # CAUTION: 36.17% of customers have balance = 0 — normal behavior
    # Observed data: min=0, max=250,898, median=97,199
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 0      # Zero is legitimate and frequent value
        max_value: 250898 # Historical maximum observed
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 0      # Mean can vary significantly (bimodal)
        max_value: 201281 # mean + 2σ = 76,486 + 2×62,397

  EstimatedSalary:
    # Observed data: min=11.58, max=199,992, uniform
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 0        # Cannot be negative
        max_value: 200000   # Observed limit (close to 200k)
        mostly: 0.99
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 0        # Mean - 2σ (may be uniform in other batch)
        max_value: 215111   # mean + 2σ = 100,090 + 2×57,510

  Point Earned:
    # Observed data: min=119, max=1000, uniform
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 119    # Historical minimum (could be 0 in prod with new customers)
        max_value: 1000   # Loyalty program maximum
    - type: expect_column_mean_to_be_between
      kwargs:
        min_value: 154.7  # mean - 2σ = 606.5 - 2×225.9
        max_value: 1058.4 # mean + 2σ (cap at 1000 in practice)
```

### 9.3 Expectations for Discrete Numeric Columns

```yaml
  Tenure:
    # Observed data: min=0, max=10, mean=5.01, uniform
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 0    # New customer
        max_value: 10   # Historical maximum = 10 years

  NumOfProducts:
    # CAUTION: non-linear relationship with churn. 3-4 products = very high churn
    # Observed data: min=1, max=4, median=1
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 1    # Minimum: at least 1 product
        max_value: 4    # Historical maximum observed

  Satisfaction Score:
    # Fixed scale 1–5
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_between
      kwargs:
        min_value: 1    # Worst score
        max_value: 5    # Best score
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: [1, 2, 3, 4, 5]
```

### 9.4 Expectations for Binary Columns

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

### 9.5 Expectations for Categorical Columns

```yaml
  Geography:
    - type: expect_column_values_to_not_be_null
    - type: expect_column_values_to_be_in_set
      kwargs:
        value_set: ["France", "Germany", "Spain"]
    # Observed distribution: France=50.1%, Germany=25.1%, Spain=24.8%
    # Alert if Germany > 40% (suspicious distribution shift)

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
    # Perfectly balanced distribution (~25% each)
    # Any large deviation would signal encoding issue
```

---

## Outputs Generated by EDA Script

```
outputs/eda/
├── eda_report_20260411_114506.json      ← Complete report (10 sections)
└── plots/
    ├── 01_target_distribution.png       ← Churn distribution (pie + bars)
    ├── 02_numeric_distributions.png     ← Histograms of all numeric variables
    ├── 03_numeric_by_target_boxplot.png ← Boxplots: numeric × churn
    ├── 04_correlation_heatmap.png       ← Pearson heatmap (lower triangle)
    ├── 05_categorical_by_target.png     ← Geography, Gender, Card × churn (%)
    ├── 06_age_distribution_by_churn.png ← Age KDE: retained vs. churned
    ├── 07_balance_distribution.png      ← Balance distribution (with zeros)
    ├── 08_creditscore_by_geography.png  ← CreditScore by country and churn
    ├── 09_satisfaction_and_points.png   ← Satisfaction and points × churn
    └── 10_products_and_tenure_churn.png ← Churn rate by products and tenure
```

---

*Document generated from exploratory data analysis performed by script `eda/eda_analysis.py`.*  
*To regenerate: `python eda/eda_analysis.py` (with venv active)*
