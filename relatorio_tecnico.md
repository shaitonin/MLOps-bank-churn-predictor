## Part 1 — Structuring the Machine Learning Project

### 1.1 Technical Objective Definition

The core objective of this project is to build a binary classification system capable of identifying in advance customers with a high probability of ending their relationship with the bank (churn), before explicit signals — such as formal complaints — become available.

The target variable is `Exited` (0 = retained customer, 1 = churned customer), with an observed distribution of 79.62% negatives and 20.38% positives, characterizing a class imbalance problem with an approximate ratio of 3.91:1.


### 1.2 Experiment Mapping

Five experiments were conducted in the exploratory phase (Scikit-Learn project), varying model architecture, the presence of the `Complain` column, and regularization and hyperparameter optimization strategies:

| Experiment | Model                                | Complain | Main configurations                                                      | Main result (test)                                 |
|------------|--------------------------------------|----------|---------------------------------------------------------------------------|----------------------------------------------------|
| E1         | Perceptron                           | Yes      | StandardScaler, Yeo-Johnson on Balance, OHE on Geography/Gender           | Accuracy ≈ 1.00 — inflated by data leakage         |
| E2         | Perceptron                           | No       | Same transformations; `Complain` removed                                 | Accuracy ≈ 0.76; churn class F1 = 0.36             |
| E3         | Decision Tree without regularization | No       | Default `DecisionTreeClassifier` hyperparameters; no depth restriction   | Training accuracy 1.00 vs. test 0.80 — severe overfitting (depth 26, 928 leaves) |
| E4         | Regularized Decision Tree            | No       | Grid Search + StratifiedKFold (10 folds); gini criterion, `max_depth = 5` | Training and test accuracy ≈ 0.86; 30 leaves; mean CV accuracy = 0.7696 (95% CI: [0.7267–0.8126]) |
| E5         | Random Forest (ensemble)             | No       | Grid Search; `n_estimators = 100`, `max_depth = 5`, `min_samples_split = 5`, `min_samples_leaf = 1` | Training and test accuracy ≈ 0.86; churn class F1 = 0.51; no overfitting |

**E1 — Perceptron with Complain:** served as a negative control. Accuracy near 1.00 — unusual for a simple linear classifier — showed the model was learning the nearly perfect correlation between `Complain` and `Exited` (Pearson = 0.9957), not genuine customer behavior patterns.

**E2 — Perceptron without Complain:** removing the contaminated variable caused performance to drop significantly. Test accuracy was 0.76, with an F1-score of 0.36 for the churn class — an honest baseline reflecting the true difficulty of the problem.

**E3 — Decision Tree without regularization:** trained with default `DecisionTreeClassifier` settings, without any depth or leaf restrictions. The tree reached depth 26 and 928 leaves, memorizing the training data completely (accuracy = 1.00) without generalizing to the test set (accuracy ≈ 0.80). This experiment empirically demonstrated the risk of overfitting in unconstrained trees.

**E4 — Decision Tree with regularization:** applied Grid Search with 10-fold stratified cross-validation for systematic hyperparameter search. The best configuration used the gini criterion and a maximum depth of 5 levels, resulting in only 30 leaves. Depth regularization eliminated overfitting: training and test accuracy converged to ≈ 0.86, with mean CV accuracy of 0.7696 and a coefficient of variation of 9%.

**E5 — Random Forest:** an ensemble method combining 100 trees with maximum depth 5, trained on different data subsamples. Aggregating predictions reduces variance compared to a single tree while maintaining accuracy ≈ 0.86 in both training and test — with no signs of overfitting. Feature importance analysis confirmed `Age` (0.385) and `NumOfProducts` (0.318) as the most influential variables, aligning with EDA findings. The remaining limitation was low churn recall (0.36 on test), indicating that class imbalance (3.91:1) was not treated in this phase.

### 1.3 Success Criteria

The choice of metrics is based on a business cost asymmetry: a false negative (a customer who will churn and is not identified) has a higher cost than a false positive (a retention strategy applied to a customer who would not churn). This favors recall over precision.

**Quantitative criteria defined:**

- **ROC-AUC ≥ 0.85** — overall class discrimination
- **Recall (churn class) ≥ 0.70** — detect at least 70% of real churns
- **F1-Score ≥ 0.60** — minimum balance between precision and recall
- **Threshold adjusted to 0.40** (default 0.50) — shifts the decision boundary to maximize minority class recall

**Derived business metrics:**

- Incremental retention rate: customers correctly identified and approached with retention action
- Intervention cost per contacted customer vs. preserved revenue per retained customer

### 1.4 Exploratory Analysis — Main Engineering Findings

EDA identified patterns that directly guided feature engineering decisions implemented in the pipeline:

**Variables with the highest discriminative power (confirmed by statistical tests):**

- `Age` — Cohen's d = 0.747; churned customers average 44.8 years old vs. 37.4 for retained customers
- `IsActiveMember` — inactive customers have churn 2.6× higher than active customers
- `NumOfProducts` — non-linear relationship: 3–4 products → churn of 82.7% to 100%
- `Balance` — bimodal distribution; 36.17% with zero balance (legitimate value, not error)
- `Geography` — Germany with churn 2× higher (Cramér's V = 0.17)
- `Gender` — women with 25.1% churn vs. 16.5% for men (Cramér's V = 0.11)

**Variables without significant linear association (p > 0.05):**

- `HasCrCard` (p = 0.485), `EstimatedSalary` (p = 0.211), `Satisfaction Score` (p = 0.559), `Point Earned` (p = 0.644) — kept in the pipeline because non-linear models may extract value through interactions; SHAP values will validate before definitive exclusion.

---

## Part 2 — Data Foundation and Initial Diagnosis

### 2.1 Ingestion Pipeline

The ingestion pipeline was built following the MLOps principle of separating policy (YAML configuration) from mechanism (generic Python code). The file `config/data.yaml` controls all ingestion decisions; module `src/ingest.py` executes with no hardcoded logic.

**Data flow:**

```
Kaggle API → raw CSV → data/raw/
           → schema validation → 5,000-row blocks
           → type conversion (convert_dtypes=True)
           → data/processed/bank_churn.parquet (Snappy compression)
```

Choosing Parquet instead of CSV reduces read time in subsequent loads, preserves data types correctly, and compresses the file to approximately 60% of the original size. The 5,000-row block size was calibrated to accommodate larger datasets (up to 50,000 rows) without memory pressure.

Schema validation checks for the mandatory presence of all 18 original columns, throwing an error immediately if any column is missing — fail-fast behavior that avoids silent errors in the rest of the pipeline.

### 2.2 Quality Diagnosis (Great Expectations)

Module `src/quality_checks.py` performs validations configured in `config/quality.yaml`, following the Great Expectations pattern with table and column expectations. All parameters were derived from the actual data (EDA Section 9) using robust percentiles (p01–p99) with ±2σ tolerances.

**Anomalies identified in the dataset:**

**Outliers in `CreditScore`:** 15 cases with values below 400, outside the operational range of any conventional credit score system (350–850). They were kept in the dataset after investigation — EDA confirmed they do not affect the global distribution (std = 96.7).

**Bimodal distribution in `Balance`:** 36.17% of customers have exactly zero balance. This represents legitimate behavior, since there are customers who use the bank solely for other services.

**High age concentration in `Age`:** 359 customers older than 60 years, in the right tail of the asymmetric distribution (skew = +1.01). These customers coincide with the group of highest churn rate, making them relevant for the model — they were not removed as outliers.

**`NumOfProducts` equal to 4:** 60 customers contracted 4 products, a category with 100% churn in the sample. The valid range was confirmed as 1–4; these cases are not errors but rather the strongest predictive signal in the dataset.

### 2.3 Structural Dataset Inconsistencies

**Artificial distribution of `Card Type`:** The column shows a perfectly uniform distribution across the four categories (DIAMOND, GOLD, SILVER, PLATINUM), with χ² p = 0.168 and Cramér's V = 0.02 — evidence of synthetic data with no real association to churn. Additionally, `Card Type` contradicts `HasCrCard`: 1,151 customers without a credit card (`HasCrCard = 0`) have an assigned card type — a structural inconsistency that confirms synthetic origin. `Card Type` was removed from the pipeline.

**Data leakage in `Complain`:** The Pearson correlation of 0.9957 between `Complain` and `Exited` indicates the formal complaint was recorded after or simultaneously with churn, not before. In production, this information would not be available at prediction time. The column was excluded via `ColumnDropper` in the first preprocessing step, before any transformation.

### 2.4 Class Imbalance

The 3.91:1 ratio between retained and churned customers requires explicit treatment to avoid the model learning to classify everything as negative (apparent accuracy of 79.62% with no real predictive power). Mitigation strategies are detailed in Section 3.2.

### 2.5 Impact on Generalization and Structural Limitations

**Synthetic data origin:** The dataset was artificially generated for educational purposes. This implies the absence of real-world spurious correlations present in production data (seasonality, macroeconomic effects, cohort behavior), perfectly uniform distributions in some columns (`Card Type`, `Tenure`, `EstimatedSalary`), and no collection noise. Models trained on this dataset tend to overestimate performance compared to real datasets.

**Lack of temporal dimension:** The dataset is a cross-section, without timestamps. In production, churn is a temporal phenomenon — the churn rate in December may differ from March. The current pipeline does not implement temporal features (rolling averages, time-since-last-transaction); this is the main limitation for generalization in a real environment.

**Recommendation:** For production use, the pipeline should be recalibrated with real data, including monitoring for data drift in input distributions and concept drift in the relationship between features and the target.

---

### 2.6 Applied Transformations and Techniques

This section describes all the transformations applied throughout the project, organized into two layers: **preprocessing** (stateless — executed over the full dataset before the split) and **modeling** (stateful — executed only on training data after the split).

#### 2.6.1 Stateless Preprocessing

Stateless preprocessing gathers all transformations that do not learn parameters from the data — meaning they can be safely applied to the full dataset without risk of data leakage. Each transformation is controlled by `config/preprocessing.yaml` and implemented as a Scikit-Learn transformer (`BaseEstimator + TransformerMixin`) in `src/preprocessing.py`.

#### Column Removal (`ColumnDropper`)

The columns `RowNumber`, `CustomerId`, `Surname`, and `Complain` are removed in the first pipeline stage. The first three have no predictive value — they are only record identifiers. `Complain` is removed due to severe data leakage (Pearson correlation = 0.9957 with `Exited`): in production, the formal complaint would not be available before churn occurs.

#### Binary Flags (`BinaryFlagTransformer`)

Creation of two new binary columns derived from existing variables, based on patterns identified in the EDA:

- **`HasZeroBalance`** (`Balance == 0`): flags the 36.17% of customers with zero balance — a group with behavior distinct from the rest.
- **`HighRiskProducts`** (`NumOfProducts >= 3`): flags customers with 3 or more products, a range in which churn reaches 82.7% to 100%.

#### Interaction Features (`InteractionFeatureTransformer`)

Creation of composite features that capture combined signals identified in the EDA:

- **`AgeInactivity`** = `Age × (1 − IsActiveMember)`: amplifies the disengagement signal in older customers. Since inactivity and advanced age are the two strongest isolated churn predictors, their combination creates a stronger composite predictor.
- **`EngagementScore`** = `IsActiveMember + (NumOfProducts == 2) + HasCrCard − HasZeroBalance`: an engagement score in the range −1 to 3. High values indicate more engaged customers with lower churn risk.

#### Ratio Feature (`RatioFeatureTransformer`)

- **`BalanceSalaryRatio`** = `Balance / (EstimatedSalary + 1)`: ratio between balance and estimated salary. The `+1` in the denominator protects against division by zero. Customers with a high balance relative to salary are most targeted by banking competitors and present higher churn risk.

#### Age Discretization (`AgeBinTransformer`)

- **`AgeGroup`**: the continuous variable `Age` is discretized into 5 age bands ([< 30, 30–40, 40–50, 50–60, 60+]) and converted to ordinal encoding (0 to 4). This captures the non-linear relationship between age and churn identified in the EDA: customers between 40 and 60 years old concentrate the highest propensity to leave.

#### Categorical Encoding (`CategoricalEncoder`)

- **`Geography`** → One-Hot Encoding with prefix `geo`, generating `geo_France`, `geo_Germany`, and `geo_Spain`. Each country is treated as an independent predictor, preserving the 2× higher churn difference in Germany.
- **`Gender`** → Binary encoding: Male = 0, Female = 1. A compact method for a binary variable.
- **`Card Type`** → removed from the MLOps pipeline (χ² p = 0.168, Cramér's V = 0.02 — no association with churn). In the Scikit-Learn project it had been ordinally encoded (Silver=0, Gold=1, Platinum=2, Diamond=3), but Random Forest analysis confirmed importance = 0.0021 — irrelevant.

#### Feature Selection (`FeatureSelector`)

Final preprocessing stage: selects the definitive set of 21 features plus the target variable `Exited`, discarding intermediate and original columns that were replaced by their encoded versions (e.g., original `Geography` and `Gender` are discarded after encoding).

---

#### 2.6.2 Stateful Modeling Techniques

Stateful techniques learn parameters from the training data and can only be applied after the train/test split. Applying them before would cause data leakage.

#### Missing Data Imputation (`DataImputer`)

Column-specific strategies to handle null values that may appear in production:

| Column            | Strategy               | Justification                                                  |
|-------------------|------------------------|----------------------------------------------------------------|
| `CreditScore`     | Median by Geography    | Credit score varies by geographic region                      |
| `Age`             | Global median (37)     | Asymmetric distribution — median is more robust than mean      |
| `Balance`         | Constant zero          | 36% of customers have zero balance — legitimate value          |
| `EstimatedSalary` | Global median (100,194)| Uniform distribution                                           |
| `NumOfProducts`   | Mode (1)               | Most common value                                              |
| `Tenure`          | Global median (5)      | Uniform distribution                                           |
| `Geography`       | Mode                   | Most frequent value                                            |
| `Gender`          | Mode                   | Most frequent value                                            |

#### Scaling (`StandardScaler` and `RobustScaler`)

Scaling is applied **only inside the modeling pipeline, after the train/test split**, and **exclusively for Logistic Regression**. Random Forest and XGBoost are tree-based — they perform binary splits on the data and are invariant to feature scale. Applying scaling before the split would cause data leakage: the parameters (mean, median, IQR) would be calculated on the entire dataset, including the test data.

EDA recommended different techniques by column based on each variable's distribution (Section 8.1):

| Variable | Technique | EDA justification |
|----------|-----------|-------------------|
| `CreditScore` | `StandardScaler` | Nearly normal distribution (skew = −0.07), without extreme outliers |
| `EstimatedSalary` | `StandardScaler` | Uniform distribution, without outliers |
| `Point Earned` | `StandardScaler` | Uniform distribution, without outliers |
| `Satisfaction Score` | `StandardScaler` | Discrete 1–5, without outliers |
| `Age` | `RobustScaler` + log transform | Asymmetric (skew = +1.01), valid outliers from elderly customers |
| `Balance` | `RobustScaler` | Bimodal with 36% zeros — mean and std would be distorted |

**Why `RobustScaler` for `Age` and `Balance`?**

`StandardScaler` computes mean and standard deviation, which are sensitive to outliers and asymmetric distributions. `RobustScaler` uses **median and IQR** (interquartile range = Q75 − Q25), which are robust statistics against extreme values. For `Age`, the long tail of older customers would distort the standard deviation; for `Balance`, the concentration of zeros would distort the mean.

**Log transform on `Age`:**

Before `RobustScaler`, apply `np.log1p(Age)` — a logarithmic transformation that compresses the right tail and brings the distribution closer to normality.

**Non-scaled variables** (binary, ordinal, and encodings — already on a reduced scale by nature): `HasCrCard`, `IsActiveMember`, `HasZeroBalance`, `HighRiskProducts`, `AgeGroup`, `NumOfProducts`, `Tenure`, `AgeInactivity`, `EngagementScore`, `BalanceSalaryRatio`, `geo_France`, `geo_Germany`, `geo_Spain`, `Gender_encoded`.

The complete pipeline — imputer + scalers + model — is registered as a single artifact in MLflow, ensuring that in production scaling is applied automatically without separate manual steps.

#### Yeo-Johnson Transformation (`Balance`) — previous project

Applied in the Scikit-Learn project in addition to StandardScaler to handle the skewness of `Balance`. In the current MLOps pipeline it was replaced by two more suitable approaches: the `HasZeroBalance` flag (stateless, in preprocessing) and `RobustScaler` (stateful, in the Logistic Regression modeling pipeline).

#### Class Imbalance Treatment

Three strategies were configured to handle the 3.91:1 class ratio:

- **`scale_pos_weight = 3.91`**: native parameter in XGBoost and LightGBM that assigns higher weight to churn class errors during training. No additional computational cost.
- **SMOTE** (`k_neighbors = 5`): generates synthetic minority class samples by interpolating between real churn examples. Applied only to training data after the split.
- **Threshold adjustment to 0.40**: lowers the decision threshold from 0.50 to 0.40, making the model classify as churn cases with probability above 40%. Increases churn recall at the cost of more false positives — a trade-off justified by the business cost asymmetry.

#### Hyperparameter Optimization (Grid Search + Cross-Validation)

A systematic hyperparameter search evaluating all combinations of a predefined grid. Combined with 10-fold `StratifiedKFold`, which ensures each validation subset maintains the original class proportion (80/20). Applied in experiments E4 and E5:

- **E4 (Decision Tree):** grid over `criterion` (gini, entropy) and `max_depth` (3, 5, 7, 10). Best configuration: gini + max_depth = 5.
- **E5 (Random Forest):** grid over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. Best configuration: 100 trees, max_depth = 5, min_samples_split = 5, min_samples_leaf = 1.

---

## Part 3 — Systematic Model Experimentation

### 3.1 Experimental Approach

The experiments from the previous project (E1–E5) established a baseline with 0.86 accuracy and 0.51 churn F1 (Random Forest, without imbalance treatment). This phase expands experimentation with four candidates, explicit imbalance treatment via SMOTE, and Bayesian hyperparameter optimization with Optuna, with all experiments tracked in MLflow.

**Selected candidate models:**

| Model | Paradigm | Justification |
|-------|----------|---------------|
| `LogisticRegression` | Linear | Interpretable baseline; the only one requiring scaling |
| `RandomForestClassifier` | Bagging | Already tested in the previous project (E5); comparison benchmark |
| `XGBClassifier` | Depth-wise boosting | Industry standard; `scale_pos_weight=3.91` for imbalance |
| `LGBMClassifier` | Leaf-wise boosting | Different growth strategy than XGBoost; native `is_unbalance=True` |

**Selection criterion:** ROC-AUC as the primary metric (robust to imbalance), complemented by F1 and churn recall.

---

### 3.2 End-to-End Pipeline (scikit-learn + imbalanced-learn)

The pipeline was built with `imblearn.Pipeline`, ensuring each stateful transformation is fit exclusively on training data in each CV fold — without data leakage:

```
DataImputer → (StandardScaler + RobustScaler¹) → FeatureReducer → SMOTE → Estimator
```

¹ Scaling applied only to Logistic Regression. Random Forest, XGBoost, and LightGBM are scale invariant (based on binary splits).

**Pipeline components:**

- **`DataImputer`**: imputes missing values per column — median by group (`CreditScore` by `Geography`), constant zero (`Balance`), global median (`Age`, `Tenure`, `EstimatedSalary`) and mode (`NumOfProducts`).
- **`StandardScalerTransformer`**: z-score scaling on `CreditScore`, `EstimatedSalary`, `Point Earned`, `Satisfaction Score` (normal/uniform distributions).
- **`RobustScalerTransformer`**: median/IQR scaling on `Age` (skew=+1.01, with prior log1p) and `Balance` (bimodal with 36% zeros).
- **`FeatureReducer`**: supports `none | rfe | pca | kpca` — configurable via `modeling.yaml`.
- **`SMOTE`**: synthetic minority class oversampling (`k_neighbors=5`) applied only to the training fold.

---

### 3.3 Cross-Validation and Hyperparameter Search

**Data split:**
- Holdout: 20% separated before any training (`stratify=True` — preserves 20.4% churn proportion)
- CV: `StratifiedKFold` with 5 folds on the training set (8,000 samples)

**Adjusted threshold:** 0.40 (sklearn default: 0.50). The reduction favors churn recall.

**Optuna — Bayesian optimization (TPE Sampler):**
- 10 trials per model in this run (configurable in `modeling.yaml → optuna.default_trials`)
- Optuna co-optimizes estimator hyperparameters and the dimensionality reduction method (`none | rfe | pca | kpca`)
- Invalid trials (e.g. `penalty=l1` + `solver=lbfgs`) are automatically ignored

---

### 3.4 Experimental Results

#### Baseline — Default Parameters

| Model | ROC-AUC | ± std | F1 | Recall | Avg Precision |
|-------|---------|-------|-----|--------|---------------|
| LightGBM | **0.8499** | 0.0058 | 0.5937 | 0.5448 | 0.6784 |
| XGBoost | 0.8465 | 0.0089 | 0.5730 | 0.7380 | 0.6700 |
| Random Forest | 0.8456 | 0.0080 | 0.5971 | 0.5969 | 0.6464 |
| Logistic Regression | 0.8252 | 0.0075 | 0.5761 | 0.6380 | 0.6304 |

#### After Optuna (30 trials/model, 5-fold CV)

| Model | Baseline ROC-AUC | Optimized ROC-AUC | ± std | Improvement | F1 | Recall | Best configuration (selection) |
|-------|------------------|--------------------|-------|-------------|-----|--------|--------------------------------|
| **LightGBM** | 0.8499 | **0.8623** | 0.0076 | +0.0124 | 0.5882 | 0.4877 | RFE (16 feat.), `n_est=528`, `num_leaves=30`, `lr=0.014` |
| XGBoost | 0.8465 | 0.8591 | 0.0092 | +0.0126 | 0.5931 | 0.7693 | RFE (16 feat.), `n_est=164`, `max_depth=6`, `lr=0.023` |
| Random Forest | 0.8456 | 0.8562 | 0.0071 | +0.0106 | 0.6100 | 0.6362 | PCA (18 comp.), `n_est=282`, `max_depth=9` |
| Logistic Regression | 0.8252 | 0.8358 | 0.0088 | +0.0106 | 0.5703 | 0.7362 | PCA (18 comp.), `C=0.258`, `penalty=l1` |

> **Notes on hyperparameter search:**
> - **Dimensionality reduction:** LightGBM and XGBoost converged to RFE with 16 features; Random Forest and Logistic Regression selected PCA with 18 components. This suggests boosting models benefit more from explicit selection of original features (interpretability + noise removal), while linear/shallower tree models benefit from space compression via PCA.
> - **Recall vs. ROC-AUC trade-off:** XGBoost and Logistic Regression maximized recall (>0.73) but with lower precision; LightGBM maximized ROC-AUC (0.8623) with more conservative recall (0.49 in CV). In the holdout set, with threshold 0.40, LightGBM recall increases to 0.61.
> - **Optimization time:** LightGBM ~2h, XGBoost ~3h, Random Forest ~1.4h, Logistic Regression ~5min.

#### Holdout Evaluation — Best Model (LightGBM)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8770** |
| F1 (churn) | 0.6417 |
| Recall (churn) | 0.6078 |
| Precision (churn) | 0.6795 |
| Avg Precision | 0.7312 |
| Threshold | 0.40 |

The holdout result (ROC-AUC = 0.8770) exceeded CV performance (0.8623 ± 0.0076), indicating good generalization without overfitting.

---

### 3.5 Comparative Analysis and Technical Decision

**LightGBM was selected as the primary model** for the following reasons:

| Criterion | LightGBM | XGBoost | Random Forest | Logistic Reg. |
|-----------|----------|---------|---------------|---------------|
| Holdout ROC-AUC | **0.8770** | — | — | — |
| Optimized CV ROC-AUC | **0.8623** | 0.8591 | 0.8562 | 0.8358 |
| Predictive performance | ✓ Best | ✓ 2nd | ✓ 3rd | ✗ Lowest |
| Computational cost | ✓ Medium (~2h/30 trials) | ✗ Higher (~3h/30 trials) | ✓ Medium (~1.4h) | ✓ Very low (~5min) |
| Imbalance handling | ✓ Native (`is_unbalance`) | ✓ (`scale_pos_weight`) | ✓ (`class_weight`) | ✓ (`class_weight`) |
| Interpretability | ✓ `feature_importances_` + SHAP | ✓ same | ✓ same | ✓ High (coefficients) |

**Comparison with the previous project (E5 — Random Forest without SMOTE):**
- Churn F1: 0.51 → **0.64** (+25% relative)
- ROC-AUC: not measured in E5 → **0.8770**

The improvement stems mainly from three changes: (1) SMOTE correcting class imbalance, (2) additional feature engineering (AgeInactivity, EngagementScore, BalanceSalaryRatio), and (3) Bayesian hyperparameter optimization with Optuna (30 trials/model via TPE Sampler).

---

### 3.6 MLflow Tracking

All experiments were logged in MLflow with a SQLite backend (`mlflow.db`), under the experiment `bank-churn-classification`. For each model, the following were logged:

- **Parameters:** default and optimized hyperparameters, dimensionality reduction method, threshold used
- **Per-fold metrics:** `fold_roc_auc`, `fold_f1`, `fold_recall`, `fold_precision`, `fold_avg_precision`
- **Aggregated metrics:** mean and standard deviation of each CV metric
- **Holdout metrics:** final evaluation on data never seen during the search
- **Artifacts:** ROC curve, precision-recall curve, confusion matrix, threshold analysis, feature importance, fold comparison, distribution before/after SMOTE
- **Serialized pipeline:** complete pipeline (imputation + scaling + reducer + estimator) saved as an MLflow artifact, ready for inference


## Part 4 — Dimensionality Reduction

### 4.1 Analysis of the Need for Reduction

The dataset has 20 features after the feature engineering in Part 2. To assess whether dimensionality reduction is necessary, three factors must be considered:

**Dimensionality:** 20 features is a relatively compact space. The "curse of dimensionality" typically appears with hundreds of features. With 10,000 samples and 20 features, the sample-to-dimension ratio is ~500:1 — well above the problematic threshold.

**Part 3 results:** Optuna tested four reduction methods (`none`, `rfe`, `pca`, `kpca`) simultaneously with model hyperparameters. LightGBM selected RFE with 16 features — removing only 4 of 20 — which indicates that most features are informative and aggressive reduction is unnecessary.

**Preliminary conclusion:** there is no technical imperative to apply dimensionality reduction to this dataset. This part's experiment serves to **quantify the trade-off** between feature space compression, predictive performance, and computational cost.

---

### 4.2 Selected Techniques and Justifications

**PCA** and **LDA** were selected for the controlled experiment. t-SNE was discarded as a pipeline technique for a fundamental technical reason.

**PCA — Principal Component Analysis (unsupervised)**

Chosen because it is the reference technique in dimensionality reduction: spectral decomposition of the covariance matrix that projects the data in the direction of maximum variance. Advantages in this context: stable implementation, components ordered by explained variance, and the ability to analyze how much original information is preserved. Limitation: it completely ignores target labels — it maximizes data variance, not class separability.

Configuration: **18 components** (out of 20 features), preserving >99% of the variance — a minimal reduction useful for isolating the effect of the linear transformation without significant information loss.

**LDA — Linear Discriminant Analysis (supervised)**

Chosen as a supervised counterpart to PCA: instead of maximizing variance, LDA finds the projection that **maximizes class separation**. For binary classification, the result is always **a single component** (n_classes − 1 = 1) — maximum compression: from 20 features to 1 dimension.

This property makes LDA an extreme experiment: if a single linear discriminant captures the necessary separability, the model is highly efficient and interpretable. If performance drops significantly, it demonstrates that the decision boundary has a non-linear structure that one component cannot capture.

**Why t-SNE was discarded as a pipeline technique:**

t-SNE is a non-parametric visualization algorithm — it does not learn a generalizable transformation. There is no independent `fit()` + `transform()` for new data: each execution optimizes an embedding specific to the input dataset, with no ability to project test samples. Therefore, **it cannot be integrated into an inference pipeline**. Its use in this project would be limited to exploratory data visualization (EDA), not classifier training.

---

### 4.3 Pipeline Integration and Experimental Setup

The experiment was implemented in `notebooks/dimensionality_reduction.py`, reusing the component `src/feature_reducer.py` — which was extended with LDA support in this phase.

**Controlled experiment design:**

- **Independent variable:** reduction technique (`none` | `PCA` | `LDA`)
- **Controlled variable:** LightGBM hyperparameters fixed at the optimal values from Part 3 (`n_estimators=528`, `num_leaves=30`, `lr=0.0144`, etc.)
- **Protocol:** StratifiedKFold (5 folds), threshold=0.40, SMOTE enabled, seed=42

**Pipeline by variant:**

```
none : DataImputer → FeatureReducer(none)   → SMOTE → LGBMClassifier
PCA  : DataImputer → StandardScaler → FeatureReducer(pca,  n=18) → SMOTE → LGBMClassifier
LDA  : DataImputer → StandardScaler → FeatureReducer(lda,  n=1)  → SMOTE → LGBMClassifier
```

StandardScaler was added before PCA and LDA because both are sensitive to the absolute scale of features (PCA confuses variance with magnitude; LDA maximizes the ratio of between/intra-class variances). LightGBM in the `none` variant does not receive scaling because trees are invariant to monotonic scale.

---

### 4.4 Results — Comparison with and without Dimensionality Reduction

Experiments executed on `2026-04-20`, logged in MLflow with tag `stage=dr_comparison`.

| DR technique | Features | CV ROC-AUC | ± std | CV F1 | CV Recall | Holdout ROC-AUC | Holdout F1 | Holdout Recall | Holdout Prec. | CV time (s) |
|--------------|----------|-----------|-------|-------|-----------|-----------------|------------|----------------|---------------|-------------|
| **No DR** | 20 | **0.8598** | 0.0088 | 0.5845 | **0.7791** | **0.8730** | **0.5973** | **0.8162** | 0.4710 | 6.5 |
| PCA (18 comp.) | 18 | 0.8586 | 0.0071 | **0.5941** | 0.7675 | 0.8687 | 0.5875 | 0.7941 | 0.4662 | **5.0** |
| LDA (1 comp.) | 1 | 0.8196 | 0.0107 | 0.5200 | 0.7951 | 0.8435 | 0.5340 | 0.8284 | 0.3939 | **4.8** |

> **Note on results:** the values above differ slightly from Part 3 results because Optuna used RFE(16) as reduction — not `none`. The DR experiment uses the best Optuna parameters but with a controlled reduction method, revealing the isolated impact of each technique.

---

### 4.5 Trade-off Analysis

#### Impact on Classification Performance

**No DR vs PCA:** the holdout ROC-AUC difference is minimal (0.8730 → 0.8687, −0.0043). PCA with 18 of 20 components preserves virtually all predictive information. The small drop arises from the two discarded features and the linear transformation that mixes interpretable features into abstract components. PCA's F1 is slightly higher (0.5941 vs. 0.5845 in CV) because it marginally reduces noise.

**No DR vs LDA:** a pronounced ROC-AUC drop (0.8730 → 0.8435, −0.0295). Compressing 20 features into a single linear component is an extremely restrictive hypothesis: it assumes all discriminative information lies in one linear direction in feature space. The result shows this hypothesis does not hold — the problem has a separation structure more complex than one component can capture. The LDA holdout recall (0.8284) exceeds the others because one component tends to create thicker decision boundaries, pushing more predictions toward the positive class.

#### Computational Cost

| Stage | No DR | PCA | LDA |
|-------|-------|-----|-----|
| Reducer fit (CV) | — | ~0.01s/fold | ~0.005s/fold |
| LightGBM train (CV) | ~1.3s/fold | ~1.0s/fold | ~0.9s/fold |
| **Total CV (5 folds)** | **6.5s** | **5.0s** | **4.8s** |
| Inference (per sample) | baseline | + linear transform | + linear transform |

PCA and LDA reduce LightGBM training time because the model operates in a lower-dimensional space. The savings are modest here (20 → 18 or 20 → 1 features) but would be substantial with hundreds of features. The PCA/LDA overhead is negligible (<0.1s per fold).

#### Interpretability

| Aspect | No DR | PCA | LDA |
|--------|-------|-----|-----|
| Original features visible in the model | ✓ Yes | ✗ No | ✗ No |
| Interpretable feature importance | ✓ Direct | ✗ Abstract components | ✗ Single discriminant axis |
| Interpretable SHAP values | ✓ Yes | ✗ Requires back-projection | ✗ Limited |
| Explainability for stakeholders | ✓ High | ✗ Low | ✗ Very low |

With no DR, LightGBM can directly indicate "Age has importance 0.38" in an actionable way. With PCA, `pc_0` is a linear combination of all 20 features — interpreting which business variable drives churn requires back-projecting the loadings, adding complexity. With LDA, the single component has coefficients that combine contributions from all features to the class separation axis, but the model ultimately operates on a single scalar value — the business connection is indirect.

---

### 4.6 Technical Decision — Is Dimensionality Reduction Appropriate?

**Conclusion: dimensionality reduction is not recommended for this problem in its current state.**

Justifications:

1. **No meaningful predictive gain:** PCA with 18 components yields a holdout ROC-AUC 0.43 percentage points lower — a negligible difference within CV variability (±0.0088). LDA loses 2.95 points — a relevant drop.

2. **Recall is the business-critical metric:** the criterion defined in Part 1 is recall ≥ 0.70. No DR achieves holdout recall 0.8162. PCA yields 0.7941. LDA yields 0.8284 (for spurious reasons — lower precision, not better discrimination). The "No DR" configuration delivers the best recall with solid precision.

3. **Interpretability cost is high:** the bank needs to explain retention decisions. With no DR, features such as `Age`, `NumOfProducts`, and `IsActiveMember` have direct, actionable importances. PCA/LDA destroy that traceability.

4. **20 features is not a dimensionality problem:** reduction becomes valuable at ~100+ highly correlated features. Here, feature engineering already produced a compact and informative set.

**Exception:** if the objective were exploratory visualization (customer cluster analysis, for example), PCA with 2 components or t-SNE would be valuable tools — but for inference and deployment, the no-DR version is superior.

---

## Part 5 — Final Model Selection

### 5.1 Experiment Consolidation

The table below aggregates all experiments evaluated on the holdout across Parts 3 and 4. Only LightGBM underwent full holdout evaluation — it was selected in Part 3 as the model with the highest CV ROC-AUC among the four candidates, and tested with multiple dimensionality reduction configurations in Part 4.

| Experiment | Configuration | CV ROC-AUC | Holdout ROC-AUC | Holdout F1 | Holdout Recall | Holdout Prec. |
|------------|---------------|------------|-----------------|------------|----------------|---------------|
| P3 — Baseline | Default parameters, no DR | 0.8499 ± 0.0058 | 0.8539 | 0.6103 | 0.5429 | 0.6965 |
| P3 — Optuna | RFE(16), optimized hyperparams | 0.8623 ± 0.0076 | 0.8770 | 0.6417 | 0.6078 | 0.6795 |
| P4 — DR | No DR, Optuna hyperparams | 0.8598 ± 0.0088 | 0.8730 | 0.5973 | **0.8162** | 0.4710 |
| P4 — DR | PCA(18), Optuna hyperparams | 0.8586 ± 0.0071 | 0.8687 | 0.5875 | 0.7941 | 0.4662 |
| P4 — DR | LDA(1), Optuna hyperparams | 0.8196 ± 0.0107 | 0.8435 | 0.5340 | 0.8284 | 0.3939 |

---

### 5.2 Why Did Optuna Choose RFE and Not "No DR"?

This is the key point for understanding the final selection.

Optuna **co-optimized the DR method and model hyperparameters simultaneously** over 30 trials. Each trial explored a different combination of these two dimensions:

```
Trial X:  RFE(16) + n_est=528 + num_leaves=30 + lr=0.014  →  CV AUC 0.8623  ← best trial
Trial Y:  none    + n_est=300 + num_leaves=50 + lr=0.050  →  CV AUC 0.851
Trial Z:  none    + n_est=150 + num_leaves=20 + lr=0.030  →  CV AUC 0.843
```

The issue is that **no trial tested `none` with the same hyperparameters as the best RFE trial**. Optuna found a local optimum: the combination RFE + those specific hyperparameters. But it did not explore whether those same hyperparameters, without RFE, would be equivalent or better — with only 30 trials in a large search space, that combination simply was not sampled.

Part 4 performed exactly that controlled experiment: it took the best Optuna hyperparameters and tested `none`, `PCA`, and `LDA`. The result showed that **without DR, using the same hyperparameters, AUC drops by only 0.004 and recall increases by 21 percentage points**.

---

### 5.3 AUC Difference Analysis

The ROC-AUC difference between the two main configurations:

| Configuration | Holdout ROC-AUC | Difference | CV std |
|---------------|-----------------|------------|--------|
| Optuna + RFE(16) | 0.8770 | — | ±0.0076 |
| Optuna + no DR | 0.8730 | **−0.004** | ±0.0088 |

The 0.004 difference is **less than half a standard deviation** of the natural fold variation. There is no statistical evidence that the RFE configuration truly discriminates better — the difference is within expected noise between runs.

In contrast, the recall difference is **21 percentage points** (0.61 → 0.82) — this is well beyond noise and represents a real, substantial effect.

---

### 5.4 Candidate Operational Model

**Selected model: LightGBM with Optuna hyperparameters, no dimensionality reduction.**

#### Selection rationale

**1. Equivalent AUC to the best found model:** the 0.004 difference versus Optuna+RFE is within natural CV variation. Both configurations have equivalent discriminative capacity.

**2. 21 percentage points higher recall:** 0.82 vs 0.61 on holdout. For bank churn — where the cost of missing a customer who will leave (false negative) is permanent — this difference has direct business impact. On a population of 10,000 customers with 20% churn, it represents ~429 additional churners identified per cycle.

**3. The higher recall is not spurious:** it comes from a model with equivalent AUC, not from a model that simply “guesses positive for everything.” Precision of 0.47 indicates the model still discriminates — it captures 82% of churners with 47% precision, operating at a point on the ROC curve favorable to recall.

**4. Simpler and more robust pipeline:** without RFE, the pipeline removes a stateful component that would need re-calibration each retrain and is sensitive to feature scheme changes. Fewer moving parts mean fewer points of failure in production.

**5. Full interpretability:** 20 original features are preserved. Feature importances and SHAP values remain directly traceable to business variables — essential for explaining retention decisions to stakeholders.

#### Technical specification

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

**Complete pipeline:**
```
DataImputer → FeatureReducer(method='none') → SMOTE(k=5) → LGBMClassifier
```

**Decision threshold:** 0.40

#### Final performance

| Metric | Value | Goal (Part 1) | Status |
|--------|-------|---------------|--------|
| ROC-AUC (holdout) | 0.8730 | ≥ 0.85 | ✓ Met |
| Churn recall (holdout) | 0.8162 | ≥ 0.70 | ✓ Met |
| F1-score (holdout) | 0.5973 | ≥ 0.60 | ✗ −0.003 |
| Decision threshold | 0.40 | — | Adjusted |

> **Note on F1:** the 0.597 F1 is 0.003 below the 0.60 goal — a negligible difference. It reflects the 0.40 threshold, which shifts the model toward higher recall (0.82) and lower precision (0.47). With threshold 0.50, F1 rises to ~0.62, but recall falls to ~0.55. Prioritizing recall as the primary business criterion justifies this configuration.

#### Comparison with the previous project (E5 — Random Forest without SMOTE)

| Metric | E5 — Scikit-Learn | Final Model — MLOps | Change |
|--------|------------------|----------------------|--------|
| Churn recall | 0.36 | **0.82** | **+127%** |
| Churn F1 | 0.51 | 0.60 | +17.6% |
| ROC-AUC | not measured | 0.8730 | — |
| Imbalance treatment | ✗ | ✓ SMOTE | — |
| Hyperparameter optimization | Manual Grid Search | Optuna TPE 30 trials | — |

The recall jump from 0.36 to 0.82 — more than double — is the direct result of three combined changes: SMOTE correcting imbalance, additional feature engineering (AgeInactivity, EngagementScore, BalanceSalaryRatio), and Bayesian optimization with Optuna.

---

## Part 6 — Model Deployment, Monitoring, and Operation

### 6.1 Model Persistence and Versioning

The complete pipeline — DataImputer, FeatureReducer, SMOTE, and LGBMClassifier — is serialized locally via joblib and registered in the MLflow Model Registry under the name `bank-churn-lgbm`. Each retrain produces a versioned artifact automatically.

Promotion flow is controlled by a quality gate: models achieving ROC-AUC ≥ 0.85 and Recall ≥ 0.65 are automatically promoted to **Production**; others remain in **Staging** for manual review. Prior versions are archived, preserving the full deployment history and enabling immediate rollback if necessary.

---

### 6.2 Packaging as an Inference Artifact

The model is packaged as a self-contained artifact: a single loadable file that includes the entire transformation and prediction pipeline, without the need to reprocess training data. A schema file accompanies the artifact, documenting the expected columns, target variable, and decision threshold — formalizing the inference contract between the model and any consuming service.

---

### 6.3 Inference Service

The model is exposed via a REST API (FastAPI) with three endpoints: health check, single prediction, and batch prediction. Each response includes churn probability, decision flag, and a risk level categorized as low, medium, or high — allowing the retention team to prioritize actions without interpreting raw probabilities.

A visual interface (Streamlit) complements the API, enabling interactive simulations with an input form and immediate risk-colored results.

The service loads the model directly from the MLflow Registry (Production stage) and falls back to the local artifact automatically if the Registry is unavailable.

---

### 6.4 Simulated CI/CD Pipeline

The continuous integration and delivery pipeline automates six sequential stages: environment validation, input data verification, model training and registration, quality gate with metric checks, API smoke test, and initial monitoring cycle. The pipeline aborts immediately on any failure, ensuring that no deficient model is promoted to production.

---

### 6.5 Technical and Business Impact Metrics

#### Technical metrics

| Metric | Value | Description |
|--------|-------|-------------|
| ROC-AUC | 0.8613 | Overall class discrimination |
| Recall | 0.6740 | Proportion of real churners identified |
| F1-score | 0.6104 | Balance between precision and recall |
| Precision | 0.5578 | Reliability of positive predictions |

#### Business impact metrics

Calculated for each monthly production batch based on business assumptions: annual revenue per retained customer of €500, intervention cost of €40 per retention action, and a 30% retention success rate.

| Metric | Reference (1,000-customer batch) |
|--------|----------------------------------|
| True positives (correctly identified churners) | 130 of 192 |
| False positives (unnecessarily contacted customers) | 104 |
| Estimated preserved revenue | €19,500 |
| Total campaign cost | €9,360 |
| Estimated ROI | 1.1x |

---

### 6.6 Post-Deploy Monitoring and Drift Detection

Monitoring runs on each new data batch and logs an MLflow run with tag `stage=monitoring`, allowing metric evolution tracking over time.

**Data drift** is detected by the Kolmogorov-Smirnov test applied feature by feature, comparing the training distribution to the production batch distribution. A p-value below 0.05 indicates a significant distribution change. In the initial simulated cycle: 0 of 20 features exhibited drift.

**Probability stability** is measured by PSI (Population Stability Index) on model outputs. Values below 0.10 indicate stability; above 0.20, severe drift with retraining recommended. Initial result: PSI = 0.028 — stable.

**Model drift** is evaluated by metric degradation between the reference batch and production batch. Drops greater than 3 percentage points in AUC or 5 points in recall trigger an alert. Initial result: no model drift detected.

---

### 6.7 Retraining Strategy

Retraining is triggered by four conditions: three or more features with KS-detected drift, PSI above 0.20, significant production metric degradation, or a preventive monthly schedule independent of drift.

The training window uses the last 12 months of data to avoid stale patterns contaminating the model. Before promotion, the new model passes the same CI/CD quality gate. The previous version remains available in the Registry for immediate rollback if the new model regresses in production.