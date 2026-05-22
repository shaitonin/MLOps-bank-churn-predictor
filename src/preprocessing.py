"""
preprocessing.py — Preprocessing and Feature Engineering Transformers.

scikit-learn interface (BaseEstimator + TransformerMixin):
  Each class inherits from sklearn.base.BaseEstimator and TransformerMixin.

  BaseEstimator  → provides get_params() and set_params() automatically,
                   required for GridSearchCV and Pipeline cloning.
                   RULE: all __init__ parameters must have the same name as
                   the instance attribute (e.g. self.group_col).

  TransformerMixin → provides fit_transform(X) automatically as
                     self.fit(X).transform(X), compatible with sklearn.Pipeline.

  This allows composing transformers in a Pipeline:
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('dropper',   ColumnDropper(...)),
        ('flags',     BinaryFlagTransformer(...)),
        ('interact',  InteractionFeatureTransformer(...)),
        ('ratios',    RatioFeatureTransformer(...)),
        ('age_bins',  AgeBinTransformer(...)),
        ('encoder',   CategoricalEncoder(...)),
        ('selector',  FeatureSelector(...)),
    ])
    pipe.fit_transform(df)

Stateless transformers (fit is a no-op — returns self without learning):
  ColumnDropper, BinaryFlagTransformer, InteractionFeatureTransformer,
  RatioFeatureTransformer, AgeBinTransformer, CategoricalEncoder, FeatureSelector

Stateful transformers (fit learns train data parameters):
  GroupMedianImputer        → learns group medians
  DataImputer               → learns medians, modes and constants per column
  StandardScalerTransformer → learns mean and stddev (CreditScore, EstimatedSalary, Point Earned)
  RobustScalerTransformer   → learns median and IQR (Age with log1p, Balance)
"""
import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configurable Column Imputation
# ─────────────────────────────────────────────────────────────────────────────

class DataImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using column-specific configured strategies.

    Supported strategies (EDA Section 8.3):
    - "median"          : global median — for skewed distributions (Age, Tenure)
    - "constant"        : fixed value (fill_value) — Balance → 0 (common legit value)
    - "mode"            : mode (most frequent value) — for categoricals and NumOfProducts
    - "median_by_group" : group-stratified median — CreditScore by Geography

    Why differentiated strategies?
    - Balance: 36.17% zeros are legitimate → fill_value=0, not median
    - CreditScore: varies by geographic region → median by Geography, not global
    - Age: skewed distribution (skew=1.01) → median (37) is better than mean (38.9)

    ⚠ MLOps WARNING — Data Leakage:
    Stateful transformer: learns medians and modes in fit() using only training data.
    It must be used in the modeling Pipeline AFTER the train/holdout split — never before.

    Attributes learned in fit:
        fill_values_ (dict): {column → learned fill value}
          For "median_by_group": {"by_group": {group: median}, "global": global_median,
                                    "group_col": group_column_name}

    Config (preprocessing.yaml → imputation):
        - column: "CreditScore"
          strategy: "median_by_group"
          group_column: "Geography"
        - column: "Balance"
          strategy: "constant"
          fill_value: 0
        - column: "Age"
          strategy: "median"
          fallback_value: 37

    Example (modeling pipeline):
        imp_cfg = config['imputation']
        imputer = DataImputer(imputation_config=imp_cfg, logger=logger)
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test  = imputer.transform(X_test)
    """

    def __init__(self, imputation_config: list[dict], logger: Any = None) -> None:
        self.imputation_config = imputation_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "DataImputer":
        """
        Learns the fill value for each configured column.

        MLOps WARNING: call fit() only on the training set.
        """
        self.fill_values_: dict = {}

        for spec in self.imputation_config:
            col = spec["column"]
            strategy = spec["strategy"]

            if col not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "DataImputer.fit: column '%s' not found — skipped.", col
                    )
                continue

            if strategy == "median":
                self.fill_values_[col] = float(X[col].median())

            elif strategy == "constant":
                self.fill_values_[col] = spec["fill_value"]

            elif strategy == "mode":
                mode_series = X[col].mode()
                self.fill_values_[col] = (
                    mode_series.iloc[0] if not mode_series.empty
                    else spec.get("fallback_value")
                )

            elif strategy == "median_by_group":
                group_col = spec["group_column"]
                if group_col not in X.columns:
                    if self.logger:
                        self.logger.warning(
                            "DataImputer.fit: group column '%s' not found "
                            "for '%s' — using global median.", group_col, col
                        )
                    self.fill_values_[col] = float(X[col].median())
                else:
                    self.fill_values_[col] = {
                        "by_group": X.groupby(group_col)[col].median().to_dict(),
                        "global":   float(X[col].median()),
                        "group_col": group_col,
                    }

            else:
                if self.logger:
                    self.logger.warning(
                        "DataImputer.fit: strategy '%s' not supported for '%s' — skipped.",
                        strategy, col,
                    )
                continue

            self._log(
                "DataImputer.fit: '%s' → strategy=%s | learned value: %s",
                col, strategy,
                self.fill_values_[col] if strategy != "median_by_group"
                else self.fill_values_[col]["by_group"],
            )

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fills NaN in each column with the value learned in fit.

        Columns without NaN are kept intact (safe operation).
        """
        if not hasattr(self, "fill_values_"):
            raise RuntimeError(
                "DataImputer has not been fitted. Call fit() before transform()."
            )

        X = X.copy()

        for spec in self.imputation_config:
            col = spec["column"]
            strategy = spec["strategy"]

            if col not in X.columns or col not in self.fill_values_:
                continue

            n_before = int(X[col].isna().sum())
            if n_before == 0:
                continue

            if strategy == "median_by_group":
                fill_info = self.fill_values_[col]
                group_col = fill_info["group_col"]

                def _fill_by_group(row: pd.Series) -> Any:
                    if pd.isna(row[col]):
                        return fill_info["by_group"].get(row[group_col], fill_info["global"])
                    return row[col]

                X[col] = X.apply(_fill_by_group, axis=1)
            else:
                X[col] = X[col].fillna(self.fill_values_[col])

            n_after = int(X[col].isna().sum())
            self._log(
                "DataImputer.transform: '%s' (%s) — NaN: %d → %d",
                col, strategy, n_before, n_after,
            )

        return X


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Group Median Imputation (kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using group-based (stratified) median.

    Why group median?
    - CreditScore varies systematically by Geography (France, Germany, Spain).
    - Imputing with global median ignores this regional heterogeneity.
    - EDA Section 8.3: "CreditScore → median by Geography (score varies by region)"

    ⚠ MLOps WARNING — Data Leakage:
    This transformer is STATEFUL: it learns medians in fit() using only training data.
    It must be applied INSIDE the modeling Pipeline AFTER the train/holdout split — never before.

    sklearn.Pipeline compatibility:
        BaseEstimator provides get_params()/set_params() by introspecting
        __init__ parameters (group_col, target_col, logger).
        TransformerMixin provides fit_transform().

    Attributes learned in fit:
        medians_       (dict): {group_value → median}
        global_median_ (float): fallback for groups not seen in fit
    """

    def __init__(
        self,
        group_col: str,
        target_col: str,
        logger: Any = None,
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        """
        Learns the median of target_col for each group_col value.

        The y=None parameter exists by scikit-learn API convention —
        it is not used (unsupervised transformer).

        Raises:
            KeyError: If group_col or target_col are missing from the DataFrame.
        """
        missing_cols = [c for c in [self.group_col, self.target_col] if c not in X.columns]
        if missing_cols:
            raise KeyError(
                f"GroupMedianImputer.fit: missing columns in DataFrame: {missing_cols}"
            )

        self.medians_ = (
            X.groupby(self.group_col)[self.target_col]
            .median()
            .to_dict()
        )
        self.global_median_ = float(X[self.target_col].median())

        self._log(
            "GroupMedianImputer.fit: medians learned by '%s' for '%s': %s",
            self.group_col, self.target_col,
            {k: round(v, 1) for k, v in self.medians_.items()},
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fills NaN in target_col with the corresponding group median.

        Rows whose group was not seen in fit receive the global median.
        """
        if not hasattr(self, "medians_"):
            raise RuntimeError(
                "GroupMedianImputer has not been fitted. Call fit() before transform()."
            )

        X = X.copy()
        n_before = int(X[self.target_col].isna().sum())

        def _fill(row: pd.Series) -> float:
            if pd.isna(row[self.target_col]):
                return self.medians_.get(row[self.group_col], self.global_median_)
            return row[self.target_col]

        X[self.target_col] = X.apply(_fill, axis=1)
        n_after = int(X[self.target_col].isna().sum())

        self._log(
            "GroupMedianImputer.transform: '%s' — NaN before=%d, after=%d",
            self.target_col, n_before, n_after,
        )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 2. Identifier Removal
# ─────────────────────────────────────────────────────────────────────────────

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Removes columns with no predictive value (identifiers and personal data).

    Why drop identifiers?
    - RowNumber: sequential index with no causal relation to churn.
    - CustomerId: unique ID — if seen during training, the model may memorize
      specific customers instead of learning generalizable patterns.
    - Surname: personal identification text with no predictive power.
    - EDA Section 8.1: COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

    Config (preprocessing.yaml → drop_columns):
        - "RowNumber"
        - "CustomerId"
        - "Surname"

    Example:
        dropper = ColumnDropper(columns=config['drop_columns'], logger=logger)
        df = dropper.fit_transform(df)
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        to_drop = [c for c in self.columns if c in X.columns]
        skipped = [c for c in self.columns if c not in X.columns]

        if skipped and self.logger:
            self.logger.warning(
                "ColumnDropper: columns not found (ignored): %s", skipped
            )

        X.drop(columns=to_drop, inplace=True)
        self._log(
            "ColumnDropper: %d columns dropped: %s", len(to_drop), to_drop
        )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 3. Binary Flags
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Adds binary (0/1) indicator columns for relevant business patterns.

    Why binary flags?
    - Balance == 0: 36.17% of customers — two distinct banking usage profiles.
      Models need to explicitly recognize zero balance as a state,
      not an outlier to impute or ignore.
    - NumOfProducts >= 3: abrupt saturation threshold (churn 82.7%–100%).
      Without the flag, a model would need to discover this threshold on its own.

    Supported operators:
      "==" (default): X[column] == value
      ">="         : X[column] >= value
      "<="         : X[column] <= value

    Config (preprocessing.yaml → binary_flags):
        - column: "Balance"
          value: 0
          operator: "=="
          new_column: "HasZeroBalance"
          inference_safe: true
        - column: "NumOfProducts"
          value: 3
          operator: ">="
          new_column: "HighRiskProducts"
          inference_safe: true

    Example:
        flags_cfg = config['binary_flags']
        transformer = BinaryFlagTransformer(flags=flags_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    _DEFAULTS = [
        {"column": "Balance", "value": 0, "operator": "==", "new_column": "HasZeroBalance"},
        {"column": "NumOfProducts", "value": 3, "operator": ">=", "new_column": "HighRiskProducts"},
    ]

    def __init__(self, flags: list[dict] = None, logger: Any = None) -> None:
        self.flags = flags if flags is not None else self._DEFAULTS
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "BinaryFlagTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.flags:
            col = spec["column"]
            val = spec["value"]
            new_col = spec["new_column"]
            operator = spec.get("operator", "==")

            if col not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "BinaryFlagTransformer: column '%s' not found — flag '%s' skipped.",
                        col, new_col,
                    )
                continue

            if operator == "==":
                mask = X[col] == val
            elif operator == ">=":
                mask = X[col] >= val
            elif operator == "<=":
                mask = X[col] <= val
            else:
                if self.logger:
                    self.logger.warning(
                        "BinaryFlagTransformer: operator '%s' not supported for '%s' — skipped.",
                        operator, new_col,
                    )
                continue

            X[new_col] = mask.astype(int)
            n_flagged = int(X[new_col].sum())
            self._log(
                "BinaryFlagTransformer: '%s' %s %s → '%s': %d rows flagged (%.2f%%)",
                col, operator, val, new_col, n_flagged, 100 * n_flagged / len(X),
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interaction Features
# ─────────────────────────────────────────────────────────────────────────────

class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates interaction and composite features to capture non-linear patterns.

    Supported types:

    product_complement: col_a × (1 − col_b)
      AgeInactivity = Age × (1 − IsActiveMember)
      Amplifies the inactivity signal for older customers — the group
      with higher churn propensity. An inactive 50-year-old will have
      AgeInactivity=50; an active customer will have 0.

    engagement_composite: customer engagement composite score
      EngagementScore = IsActiveMember + (NumOfProducts == 2) + HasCrCard − HasZeroBalance
      Positive components: active member (+1), exactly 2 products (+1, the
      lowest churn point in EDA), has credit card (+1).
      Negative component: zero balance (−1, financial disengagement).
      Resulting range: -1 (fully disengaged) to 3 (highly engaged).
      ⚠ Requires HasZeroBalance created by BinaryFlagTransformer.
      Required column order: [IsActiveMember, NumOfProducts, HasCrCard, HasZeroBalance]

    Config (preprocessing.yaml → interaction_features):
        - name: "AgeInactivity"
          type: "product_complement"
          col_a: "Age"
          col_b: "IsActiveMember"
        - name: "EngagementScore"
          type: "engagement_composite"
          columns: ["IsActiveMember", "NumOfProducts", "HasCrCard", "HasZeroBalance"]

    Example:
        interact_cfg = config['interaction_features']
        transformer = InteractionFeatureTransformer(features_config=interact_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, features_config: list[dict], logger: Any = None) -> None:
        self.features_config = features_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "InteractionFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []

        for spec in self.features_config:
            name = spec["name"]
            ftype = spec["type"]

            if ftype == "product_complement":
                col_a = spec["col_a"]
                col_b = spec["col_b"]
                missing = [c for c in [col_a, col_b] if c not in X.columns]
                if missing:
                    if self.logger:
                        self.logger.warning(
                            "InteractionFeatureTransformer: missing columns %s — '%s' skipped.",
                            missing, name,
                        )
                    continue
                X[name] = X[col_a] * (1 - X[col_b])
                created.append(name)

            elif ftype == "engagement_composite":
                cols = spec["columns"]  # [IsActiveMember, NumOfProducts, HasCrCard, HasZeroBalance]
                missing = [c for c in cols if c not in X.columns]
                if missing:
                    if self.logger:
                        self.logger.warning(
                            "InteractionFeatureTransformer: missing columns %s — '%s' skipped.",
                            missing, name,
                        )
                    continue
                X[name] = (
                    X[cols[0]]                           # IsActiveMember
                    + (X[cols[1]] == 2).astype(int)      # NumOfProducts == 2
                    + X[cols[2]]                         # HasCrCard
                    - X[cols[3]]                         # HasZeroBalance
                )
                created.append(name)

            else:
                if self.logger:
                    self.logger.warning(
                        "InteractionFeatureTransformer: type '%s' not supported for '%s' — skipped.",
                        ftype, name,
                    )

        self._log("InteractionFeatureTransformer: features created: %s", created)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ratio Features
# ─────────────────────────────────────────────────────────────────────────────

class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates ratio features: numerator / (denominator + offset).

    Why BalanceSalaryRatio?
    - Customers with high balance relative to salary are the most targeted
      by competitors — exactly the profile with the highest churn in the dataset.
    - The offset (denominator_offset=1) protects against division by zero
      without distorting values where EstimatedSalary > 0.

    Safe division:
    - (denominator + offset) == 0 → NaN (avoids division by zero if offset=0)
    - Inf replaced by NaN

    Config (preprocessing.yaml → ratio_features):
        - name: "BalanceSalaryRatio"
          numerator: "Balance"
          denominator: "EstimatedSalary"
          denominator_offset: 1

    Example:
        ratios_cfg = config['ratio_features']
        transformer = RatioFeatureTransformer(ratios=ratios_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, ratios: list[dict], logger: Any = None) -> None:
        self.ratios = ratios
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "RatioFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []

        for spec in self.ratios:
            name = spec["name"]
            num = spec["numerator"]
            den = spec["denominator"]
            offset = spec.get("denominator_offset", 0)

            if num not in X.columns or den not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "RatioFeatureTransformer: columns '%s' or '%s' missing — '%s' skipped.",
                        num, den, name,
                    )
                continue

            denominator = X[den] + offset
            X[name] = (X[num] / denominator.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
            created.append(name)

        self._log("RatioFeatureTransformer: features created: %s", created)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 6. Age Bins
# ─────────────────────────────────────────────────────────────────────────────

class AgeBinTransformer(BaseEstimator, TransformerMixin):
    """
    Creates age bins (AgeGroup) from the age column.

    Why discretize age?
    - The relationship between age and churn is non-linear: peak between 40–60.
    - EDA: average age of churned = 44.8 years vs 37.4 of retained
      (Cohen's d = 0.747 — medium-large effect).
    - Bins allow linear models to capture this curved effect
      without requiring polynomial transformation.

    Ordinal encoding (ordinal_encoding: true):
    - Converts categorical labels to integers (0, 1, 2, 3, 4).
    - Preserves the natural order of age (important for GBM/XGBoost).
    - For linear regression: apply OHE on AgeGroup in the modeling pipeline.

    Config (preprocessing.yaml → age_group):
        column: "Age"
        new_column: "AgeGroup"
        bins: [0, 30, 40, 50, 60, 100]
        labels: ["<30", "30-40", "40-50", "50-60", "60+"]
        ordinal_encoding: true
        ordinal_map:
          "<30": 0
          "30-40": 1
          ...

    Example:
        age_cfg = config['age_group']
        transformer = AgeBinTransformer(age_config=age_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, age_config: dict = None, logger: Any = None) -> None:
        self.age_config = age_config if age_config is not None else {
            "ordinal_encoding": True,
            "ordinal_map": {"<30": 0, "30-40": 1, "40-50": 2, "50-60": 3, "60+": 4},
        }
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "AgeBinTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        col = self.age_config.get("column", "Age")
        new_col = self.age_config.get("new_column", "AgeGroup")
        bins = self.age_config.get("bins", [0, 30, 40, 50, 60, 100])
        labels = self.age_config.get("labels", ["<30", "30-40", "40-50", "50-60", "60+"])
        ordinal_encoding = self.age_config.get("ordinal_encoding", True)
        ordinal_map: dict = self.age_config.get("ordinal_map", {})

        if col not in X.columns:
            if self.logger:
                self.logger.warning(
                    "AgeBinTransformer: column '%s' not found — '%s' skipped.",
                    col, new_col,
                )
            return X

        X = X.copy()
        X[new_col] = pd.cut(X[col], bins=bins, labels=labels)

        if ordinal_encoding and ordinal_map:
            X[new_col] = X[new_col].map(ordinal_map)
            n_unknown = int(X[new_col].isna().sum())
            if n_unknown > 0 and self.logger:
                self.logger.warning(
                    "AgeBinTransformer: %d values of '%s' were not mapped → NaN",
                    n_unknown, new_col,
                )

        self._log(
            "AgeBinTransformer: '%s' → '%s' | bins=%s | ordinal=%s",
            col, new_col, bins, ordinal_encoding,
        )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 7. Categorical Variable Encoding
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Applies encoding to multiple categorical variables.

    Supported methods:

    one_hot: pd.get_dummies with configurable prefix
      Geography → geo_France, geo_Germany, geo_Spain
      Card Type → card_DIAMOND, card_GOLD, card_PLATINUM, card_SILVER
      drop_first=False keeps all categories for maximum interpretability.

    binary: manual mapping to 0/1
      Gender → Gender_encoded (Male=0, Female=1)
      More compact than OHE for variables with exactly 2 categories.

    Why OHE for Geography and not ordinal?
    - There is no natural order among France, Germany, Spain.
    - OHE ensures the model treats each country as an independent predictor.
    - EDA: Germany has churn 2× higher (Cramér's V=0.17) — a non-linear effect
      that would be distorted by arbitrary ordinal encoding.

    Original categorical columns are kept in the DataFrame for
    inspection. FeatureSelector later drops those that are not used in the model.

    Config (preprocessing.yaml → categorical_encoding):
        - column: "Geography"
          method: "one_hot"
          prefix: "geo"
          drop_first: false
        - column: "Gender"
          method: "binary"
          mapping: {"Male": 0, "Female": 1}
          new_column: "Gender_encoded"

    Example:
        enc_cfg = config['categorical_encoding']
        encoder = CategoricalEncoder(encoding_config=enc_cfg, logger=logger)
        df = encoder.fit_transform(df)
    """

    _DEFAULTS = [
        {"column": "Geography", "method": "one_hot", "prefix": "Geography", "drop_first": False},
        {"column": "Gender", "method": "one_hot", "prefix": "Gender", "drop_first": False},
    ]

    def __init__(self, encoding_config: list[dict] = None, logger: Any = None) -> None:
        self.encoding_config = encoding_config if encoding_config is not None else self._DEFAULTS
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        for spec in self.encoding_config:
            col = spec["column"]
            method = spec.get("method", "one_hot")

            if col not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "CategoricalEncoder: column '%s' not found — encoding skipped.",
                        col,
                    )
                continue

            if method == "one_hot":
                prefix = spec.get("prefix", col.lower().replace(" ", "_"))
                drop_first: bool = spec.get("drop_first", False)
                dummies = pd.get_dummies(
                    X[col],
                    prefix=prefix,
                    drop_first=drop_first,
                ).astype(int)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                self._log(
                    "CategoricalEncoder: OHE '%s' (prefix='%s') → %s",
                    col, prefix, list(dummies.columns),
                )

            elif method == "binary":
                mapping: dict = spec.get("mapping", {})
                new_col: str = spec.get("new_column", f"{col}_encoded")
                X[new_col] = X[col].map(mapping)
                n_unknown = int(X[new_col].isna().sum())
                if n_unknown > 0 and self.logger:
                    self.logger.warning(
                        "CategoricalEncoder: %d values of '%s' were not mapped → NaN",
                        n_unknown, col,
                    )
                self._log(
                    "CategoricalEncoder: binary '%s' → '%s' (map=%s)",
                    col, new_col, mapping,
                )

            else:
                if self.logger:
                    self.logger.warning(
                        "CategoricalEncoder: method '%s' not supported for '%s' — skipped.",
                        method, col,
                    )

        return X


# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects the final set of features for modeling.

    Why explicit selection?
    - After transformations, the DataFrame has 30+ columns (original + engineered).
    - Original categorical columns (Geography, Gender, Card Type) are replaced
      by encoding — keeping them would cause redundancy.
    - Keeping only what goes into the model prevents accidental data leakage.

    Tolerant behavior:
    - Missing columns generate a WARNING (not an exception) — allowing the pipeline
      to continue even if a prior transformation was skipped.
    - Only available columns are selected.

    Config (preprocessing.yaml → feature_selection.features_to_keep):
        - "CreditScore"
        - "Age"
        - "geo_France"
        - ...

    Example:
        sel_cfg = config['feature_selection']
        selector = FeatureSelector(features_to_keep=sel_cfg['features_to_keep'], logger=logger)
        df = selector.fit_transform(df)
    """

    def __init__(self, features_to_keep: list[str], logger: Any = None) -> None:
        self.features_to_keep = features_to_keep
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        """Validates that all configured features exist in the DataFrame."""
        missing = [c for c in self.features_to_keep if c not in X.columns]
        if missing and self.logger:
            self.logger.warning(
                "FeatureSelector.fit: %d config columns missing from DataFrame "
                "(will be ignored): %s",
                len(missing), missing,
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        available = [c for c in self.features_to_keep if c in X.columns]
        dropped = len(self.features_to_keep) - len(available)

        if dropped > 0 and self.logger:
            self.logger.warning(
                "FeatureSelector.transform: %d/%d requested columns missing — ignored.",
                dropped, len(self.features_to_keep),
            )

        self._log(
            "FeatureSelector.transform: %d/%d features selected | shape: %s → %s",
            len(available), len(self.features_to_keep),
            X.shape, (len(X), len(available)),
        )
        return X[available].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Scaling (StandardScaler)
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Z-score normalization: z = (x − μ) / σ.

    Why StandardScaler?
    - Logistic regression, SVM and KNN are sensitive to feature scale.
    - Gradient boosting and Random Forest do NOT require scaling —
      but keeping the dataset scaled eases model comparison.
    - Binary (0/1) and ordinal features have interpretable scale → do not scale.

    EDA Section 8.1 — techniques by variable:
      CreditScore      → StandardScaler (near-normal distribution, skew=-0.07)
      EstimatedSalary  → StandardScaler (uniform, no outliers)
      Point Earned     → StandardScaler (uniform, no outliers)
      Age              → RobustScaler   (skewed, skew=+1.01) — apply in modeling
      Balance          → RobustScaler   (bimodal, 36% zeros)      — apply in modeling
      Tenure           → MinMaxScaler or none (uniform discrete 0–10)
      NumOfProducts    → none (treat as ordinal)

    ⚠ MLOps WARNING — Data Leakage:
    This transformer is STATEFUL: it learns mean and standard deviation in fit()
    using only training data. It must be applied INSIDE the modeling Pipeline,
    AFTER the train/holdout split — never before.

    Parameters learned in fit (only on the training dataset!):
        mean_  (dict): {column: mean}
        std_   (dict): {column: standard deviation}

    Columns with std=0 are ignored (constant — no information).

    Example (modeling pipeline):
        scaler = StandardScalerTransformer(columns=scale_cols, logger=logger)
        scaler.fit(X_train)            # learns only on training data
        X_train_sc = scaler.transform(X_train)
        X_test_sc  = scaler.transform(X_test)
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScalerTransformer":
        """
        Learns mean and standard deviation of the specified columns.

        MLOps WARNING: always call fit() only on the training set.
        Use transform() on validation/test sets to avoid data leakage.
        """
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        skipped: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                skipped.append(col)
                continue

            mu = float(X[col].mean())
            sigma = float(X[col].std())

            if sigma == 0:
                if self.logger:
                    self.logger.warning(
                        "StandardScalerTransformer.fit: '%s' has std=0 (constant) — skipped.",
                        col,
                    )
                continue

            self.mean_[col] = mu
            self.std_[col] = sigma

        if skipped and self.logger:
            self.logger.warning(
                "StandardScalerTransformer.fit: missing columns ignored: %s", skipped
            )

        self._log(
            "StandardScalerTransformer.fit: parameters learned for %d columns.",
            len(self.mean_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Applies Z-score scaling to columns fitted in fit.

        Columns not seen in fit are left unchanged.
        """
        if not hasattr(self, "mean_"):
            raise RuntimeError(
                "StandardScalerTransformer has not been fitted. Call fit() before transform()."
            )

        X = X.copy()
        scaled: list[str] = []

        for col, mu in self.mean_.items():
            if col not in X.columns:
                continue
            X[col] = (X[col] - mu) / self.std_[col]
            scaled.append(col)

        self._log(
            "StandardScalerTransformer.transform: %d columns scaled (z-score).",
            len(scaled),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Returns a DataFrame with learned mean and standard deviation (useful for auditing)."""
        return pd.DataFrame(
            {"mean": self.mean_, "std": self.std_}
        ).rename_axis("feature")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Robust Scaling (RobustScaler)
# ─────────────────────────────────────────────────────────────────────────────

class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies robust scaling: z = (x − median) / IQR.

    Why RobustScaler for Age and Balance?
    - Age: skewed distribution (skew=+1.01) with valid outliers of older customers
      up to age 92. StandardScaler would be distorted by the long tail —
      mean and standard deviation are influenced by extremes.
    - Balance: bimodal distribution with 36% zeros. Median and IQR better capture
      true dispersion than mean and standard deviation.

    Optional log transform (log_columns):
    - Age receives np.log1p (log(1+x)) before RobustScaler to compress
      the right tail and approximate normality.
    - np.log1p is preferable to np.log because it supports zero values.
    - The log transform is learned implicitly: median and IQR are calculated
      on the already transformed series (log1p(Age)), ensuring consistency between
      fit and transform.

    ⚠ MLOps WARNING — Data Leakage:
    Stateful: learns median and IQR in fit() using only training data.
    Must be applied INSIDE the modeling Pipeline, AFTER the train/holdout split.

    Parameters learned in fit:
        median_ (dict): {column: median (of post-log1p series if applicable)}
        iqr_    (dict): {column: IQR Q75 − Q25 (of post-log1p series if applicable)}

    Config (preprocessing.yaml → scaling.robust_columns / log_transform_before_robust):
        robust_columns: ["Age", "Balance"]
        log_transform_before_robust: ["Age"]

    Example (modeling pipeline — Logistic Regression):
        robust_cfg = config['scaling']
        scaler = RobustScalerTransformer(
            columns=robust_cfg['robust_columns'],
            log_columns=robust_cfg.get('log_transform_before_robust', []),
            logger=logger
        )
        scaler.fit(X_train)
        X_train_sc = scaler.transform(X_train)
        X_test_sc  = scaler.transform(X_test)
    """

    def __init__(
        self,
        columns: list[str],
        log_columns: list[str] | None = None,
        logger: Any = None,
    ) -> None:
        self.columns = columns
        self.log_columns = log_columns or []
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "RobustScalerTransformer":
        """
        Learns median and IQR of the specified columns.

        For columns in log_columns, computes median and IQR on log1p(series).
        MLOps WARNING: always call fit() only on the training set.
        """
        self.median_: dict[str, float] = {}
        self.iqr_: dict[str, float] = {}
        skipped: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                skipped.append(col)
                continue

            series = X[col].copy().astype(float)
            if col in self.log_columns:
                series = np.log1p(series)

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25

            if iqr == 0:
                if self.logger:
                    self.logger.warning(
                        "RobustScalerTransformer.fit: '%s' has IQR=0 (constant) — skipped.",
                        col,
                    )
                continue

            self.median_[col] = float(series.median())
            self.iqr_[col] = iqr

            self._log(
                "RobustScalerTransformer.fit: '%s'%s → median=%.4f | IQR=%.4f",
                col,
                " (log1p)" if col in self.log_columns else "",
                self.median_[col],
                self.iqr_[col],
            )

        if skipped and self.logger:
            self.logger.warning(
                "RobustScalerTransformer.fit: missing columns ignored: %s", skipped
            )

        self._log(
            "RobustScalerTransformer.fit: parameters learned for %d columns.",
            len(self.median_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Applies robust scaling to columns fitted in fit.

        For columns in log_columns, applies log1p before scaling.
        Columns not seen in fit are left unchanged.
        """
        if not hasattr(self, "median_"):
            raise RuntimeError(
                "RobustScalerTransformer has not been fitted. Call fit() before transform()."
            )

        X = X.copy()
        scaled: list[str] = []

        for col, median in self.median_.items():
            if col not in X.columns:
                continue

            series = X[col].copy().astype(float)
            if col in self.log_columns:
                series = np.log1p(series)

            X[col] = (series - median) / self.iqr_[col]
            scaled.append(col)

        self._log(
            "RobustScalerTransformer.transform: %d columns scaled (robust).",
            len(scaled),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Returns a DataFrame with learned median and IQR (useful for auditing)."""
        return pd.DataFrame(
            {"median": self.median_, "iqr": self.iqr_}
        ).rename_axis("feature")
