"""
eda/eda_analysis.py — Exploratory Data Analysis

Main objective:
    Derive real statistical parameters from the dataset to calibrate
    Great Expectations expectations (quality.yaml) and inform feature engineering.


Analysis sections:
    1.  Dataset overview
    2.  Missing values and duplicates
    3.  Univariate statistics (numeric + categorical)
    4.  Target variable analysis — churn rate
    5.  Correlations (Pearson, Spearman, Cramér's V)
    6.  Univariate tests (Mann-Whitney, Welch t-test, Cohen's d)
    7.  ANOVA + Kruskal-Wallis + Tukey HSD (group differences)
    8.  Levene test (variance homogeneity)
    9.  Eta-squared (effect size for categorical variables)
    10. Chi-squared between all categoricals
    11. 2-way interactions between key features
    12. Simple clustering (KMeans) for segmentation
    13. Parameter suggestions for Great Expectations

Outputs:
    outputs/eda/eda_report_<timestamp>.json  — complete JSON report
    outputs/eda/plots/*.png
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"
PLOTS_DIR  = OUTPUT_DIR / "plots"

# ── logger ─────────────────────────────────────────────────────────────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml

_pipeline_cfg = load_yaml(ROOT / "config" / "pipeline.yaml")
logger = get_logger("eda", _pipeline_cfg.get("logging", {}))

# ── schema metadata ────────────────────────────────────────────────────────────
TARGET_COL       = "Exited"
ID_COLS          = ["RowNumber", "CustomerId", "Surname"]

NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary", "Satisfaction Score", "Point Earned",
]
BINARY_COLS      = ["HasCrCard", "IsActiveMember", "Exited", "Complain"]
CATEGORICAL_COLS = ["Geography", "Gender", "Card Type"]

# relevant percentiles
PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# interaction pairs: priority features for classification
INTERACTION_PAIRS = [
    ("Age", "Tenure"),
    ("Balance", "NumOfProducts"),
    ("CreditScore", "Geography"),
    ("Satisfaction Score", "IsActiveMember"),
]


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python types (JSON-serializable)."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _f(val: Any) -> Any:
    """Convert value to native Python type."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(val, np.bool_):
        return bool(val)
    return val


# ── main class ─────────────────────────────────────────────────────────────────
class EDAAnalyzer:

    def __init__(self, df: pd.DataFrame, target_col: str = TARGET_COL) -> None:
        self.df     = df.copy()
        self.target = target_col
        self._plots: list[tuple[str, plt.Figure]] = []
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ── section 1: overview ───────────────────────────────────────────────────
    def _overview(self) -> dict:
        df = self.df
        return {
            "n_rows":           int(df.shape[0]),
            "n_cols":           int(df.shape[1]),
            "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "columns":          list(df.columns),
            "dtypes":           {col: str(dtype) for col, dtype in df.dtypes.items()},
            "id_columns":       ID_COLS,
            "numeric_columns":  NUMERIC_COLS,
            "binary_columns":   BINARY_COLS,
            "categorical_columns": CATEGORICAL_COLS,
        }

    # ── section 2: missing values ──────────────────────────────────────────────
    def _missing_values(self) -> dict:
        df    = self.df
        total = len(df)
        missing = df.isnull().sum()
        per_col = {
            col: {
                "missing_count": int(missing[col]),
                "missing_pct":   round(int(missing[col]) / total * 100, 4),
                "has_missing":   bool(missing[col] > 0),
            }
            for col in df.columns
        }
        return {
            "total_missing_cells":  int(missing.sum()),
            "total_cells":          int(total * len(df.columns)),
            "overall_missing_pct":  round(missing.sum() / (total * len(df.columns)) * 100, 4),
            "columns_with_missing": [c for c in df.columns if missing[c] > 0],
            "per_column":           per_col,
        }

    # ── section 3: duplicates ──────────────────────────────────────────────────
    def _duplicates(self) -> dict:
        df           = self.df
        feature_cols = [c for c in df.columns if c not in ID_COLS]
        return {
            "exact_duplicates":                int(df.duplicated().sum()),
            "feature_duplicates_excl_ids":     int(df[feature_cols].duplicated().sum()),
            "unique_customer_ids":             int(df["CustomerId"].nunique())
                                               if "CustomerId" in df.columns else None,
        }

    # ── section 4: numeric statistics ─────────────────────────────────────────
    def _numeric_stats(self) -> dict:
        df     = self.df
        result = {}

        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            s = df[col].dropna()

            pct_vals = {f"p{int(p * 100):02d}": _f(s.quantile(p)) for p in PERCENTILES}

            # Normality tests
            normality: dict = {}
            stat_dp, p_dp = stats.normaltest(s)
            normality["dagostino_pearson"] = {
                "statistic":       _f(stat_dp),
                "p_value":         _f(p_dp),
                "is_normal_a05":   bool(p_dp > 0.05),
            }

            stat_ks, p_ks = stats.kstest((s - s.mean()) / s.std() if s.std() > 0 else s, "norm")
            normality["kolmogorov_smirnov"] = {
                "statistic":     _f(stat_ks),
                "p_value":       _f(p_ks),
                "is_normal_a05": bool(p_ks > 0.05),
            }

            # Outliers IQR
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            fence_lo = q1 - 1.5 * iqr
            fence_hi = q3 + 1.5 * iqr
            outlier_mask_iqr = (s < fence_lo) | (s > fence_hi)
            z_scores = np.abs(stats.zscore(s))

            result[col] = {
                "count":                   int(s.count()),
                "mean":                    _f(s.mean()),
                "std":                     _f(s.std()),
                "min":                     _f(s.min()),
                "max":                     _f(s.max()),
                "median":                  _f(s.median()),
                "skewness":                _f(s.skew()),
                "kurtosis":                _f(s.kurtosis()),
                "percentiles":             pct_vals,
                "iqr":                     _f(iqr),
                "outliers_iqr_count":      int(outlier_mask_iqr.sum()),
                "outliers_iqr_pct":        round(float(outlier_mask_iqr.mean()) * 100, 3),
                "outliers_zscore_count":   int((z_scores > 3).sum()),
                "normality_tests":         normality,
            }

        return result

    # ── section 5: categorical statistics ─────────────────────────────────────
    def _categorical_stats(self) -> dict:
        df     = self.df
        result = {}

        for col in CATEGORICAL_COLS + BINARY_COLS:
            if col not in df.columns:
                continue
            s  = df[col].dropna()
            vc = s.value_counts()
            n  = len(s)

            result[col] = {
                "dtype":                str(df[col].dtype),
                "unique_count":         int(s.nunique()),
                "unique_values":        sorted([str(v) for v in s.unique().tolist()]),
                "value_counts":         {str(k): int(v) for k, v in vc.items()},
                "value_frequencies_pct":{str(k): round(float(v / n * 100), 3) for k, v in vc.items()},
                "entropy_bits":         _f(stats.entropy(vc.values, base=2)),
            }

        return result

    # ── section 6: target variable analysis ───────────────────────────────────
    def _target_analysis(self) -> dict:
        df = self.df
        if self.target not in df.columns:
            return {}

        vc      = df[self.target].value_counts().sort_index()
        n_total = len(df)
        n_churn = int(vc.get(1, 0))
        n_keep  = int(vc.get(0, 0))

        def _churn_by_group(col: str) -> dict:
            if col not in df.columns:
                return {}
            g = df.groupby(col)[self.target].agg(["sum", "count", "mean"]).reset_index()
            return {
                str(row[col]): {
                    "churn_count":    int(row["sum"]),
                    "total_count":    int(row["count"]),
                    "churn_rate_pct": round(float(row["mean"]) * 100, 3),
                }
                for _, row in g.iterrows()
            }

        churn_rate = round(n_churn / n_total * 100, 4)
        return {
            "target_column":          self.target,
            "n_total":                n_total,
            "n_churned":              n_churn,
            "n_retained":             n_keep,
            "churn_rate_pct":         churn_rate,
            "retention_rate_pct":     round(100 - churn_rate, 4),
            "class_imbalance_ratio":  round(n_keep / n_churn, 4) if n_churn > 0 else None,
            "churn_by_geography":     _churn_by_group("Geography"),
            "churn_by_gender":        _churn_by_group("Gender"),
            "churn_by_num_products":  _churn_by_group("NumOfProducts"),
            "churn_by_card_type":     _churn_by_group("Card Type"),
            "churn_by_tenure":        _churn_by_group("Tenure"),
            "churn_by_satisfaction":  _churn_by_group("Satisfaction Score"),
        }

    # ── section 7: correlations ───────────────────────────────────────────────
    def _correlations(self) -> dict:
        df       = self.df
        num_cols = [c for c in NUMERIC_COLS + BINARY_COLS if c in df.columns]

        pearson_df  = df[num_cols].corr(method="pearson")
        spearman_df = df[num_cols].corr(method="spearman")

        target_corr: dict = {}
        for col in num_cols:
            if col == self.target:
                continue
            valid = df[[col, self.target]].dropna()
            r_p, p_p = stats.pearsonr(valid[col], valid[self.target])
            r_s, p_s = stats.spearmanr(valid[col], valid[self.target])
            target_corr[col] = {
                "pearson_r":        _f(r_p),
                "pearson_p":        _f(p_p),
                "spearman_r":       _f(r_s),
                "spearman_p":       _f(p_s),
                "significant_a05":  bool(p_p < 0.05),
            }

        # Chi-square + Cramér's V for categoricals vs target
        chi2_results: dict = {}
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            ct           = pd.crosstab(df[col], df[self.target])
            chi2, p_val, dof, _ = stats.chi2_contingency(ct)
            n            = int(ct.sum().sum())
            k            = min(ct.shape) - 1
            cramers_v    = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0

            chi2_results[col] = {
                "chi2_statistic":    _f(chi2),
                "p_value":           _f(p_val),
                "cramers_v":         round(cramers_v, 4),
                "significant_a05":   bool(p_val < 0.05),
            }

        sorted_target = sorted(
            target_corr.items(),
            key=lambda x: abs(x[1]["pearson_r"] or 0),
            reverse=True,
        )

        return {
            "pearson_matrix":  {
                col: {c2: _f(v) for c2, v in row.items()}
                for col, row in pearson_df.to_dict().items()
            },
            "spearman_matrix": {
                col: {c2: _f(v) for c2, v in row.items()}
                for col, row in spearman_df.to_dict().items()
            },
            "target_correlations":          target_corr,
            "target_correlations_ranked":   [
                {"feature": k, "pearson_r": v["pearson_r"], "spearman_r": v["spearman_r"]}
                for k, v in sorted_target
            ],
            "chi2_tests_vs_target":         chi2_results,
        }

    # ── section 8: univariate tests (Mann-Whitney, Welch, Cohen's d) ─────────
    def _statistical_tests(self) -> dict:
        df      = self.df
        churned = df[df[self.target] == 1]
        kept    = df[df[self.target] == 0]
        result: dict = {}

        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            g1 = churned[col].dropna()
            g0 = kept[col].dropna()

            u_stat, p_mw = stats.mannwhitneyu(g1, g0, alternative="two-sided")
            t_stat, p_tt = stats.ttest_ind(g1, g0, equal_var=False)

            pooled_std = float(np.sqrt((g1.std() ** 2 + g0.std() ** 2) / 2))
            cohens_d   = float((g1.mean() - g0.mean()) / pooled_std) if pooled_std > 0 else 0.0

            result[col] = {
                "group_churned":  {
                    "mean":   _f(g1.mean()),
                    "median": _f(g1.median()),
                    "std":    _f(g1.std()),
                    "n":      int(len(g1)),
                },
                "group_retained": {
                    "mean":   _f(g0.mean()),
                    "median": _f(g0.median()),
                    "std":    _f(g0.std()),
                    "n":      int(len(g0)),
                },
                "mean_difference":     _f(g1.mean() - g0.mean()),
                "mann_whitney_u":      {
                    "statistic":      _f(u_stat),
                    "p_value":        _f(p_mw),
                    "significant_a05":bool(p_mw < 0.05),
                },
                "welch_t_test":        {
                    "statistic":      _f(t_stat),
                    "p_value":        _f(p_tt),
                    "significant_a05":bool(p_tt < 0.05),
                },
                "cohens_d":            _f(cohens_d),
            }

        return result

    # ── section 9: ANOVA 1-way + Kruskal-Wallis + Tukey HSD ───────────────────
    def _anova_and_kruskal(self) -> dict:
        """Group differences (e.g.: churn × Geography, churn × NumOfProducts)."""
        df = self.df
        result: dict = {}

        grouping_cols = ["Geography", "Gender", "Card Type", "NumOfProducts", "Tenure"]

        for numeric_col in NUMERIC_COLS:
            if numeric_col not in df.columns:
                continue
            result[numeric_col] = {}

            for group_col in grouping_cols:
                if group_col not in df.columns:
                    continue

                try:
                    groups = [group[numeric_col].dropna().values
                              for name, group in df.groupby(group_col)]
                    if len(groups) < 2:
                        continue

                    f_stat, p_anova = stats.f_oneway(*groups)
                    h_stat, p_kw = stats.kruskal(*groups)

                    tukey_data = []
                    if p_anova < 0.05:
                        try:
                            res = pairwise_tukeyhsd(
                                df[numeric_col].dropna(),
                                df.loc[df[numeric_col].notna(), group_col],
                                alpha=0.05
                            )
                            tukey_summary = str(res).split('\n')[1:]
                            tukey_data = [line.strip() for line in tukey_summary if line.strip()]
                        except:
                            tukey_data = []

                    result[numeric_col][group_col] = {
                        "anova_f_statistic":        _f(f_stat),
                        "anova_p_value":            _f(p_anova),
                        "anova_significant_a05":    bool(p_anova < 0.05),
                        "kruskal_wallis_h":         _f(h_stat),
                        "kruskal_wallis_p":         _f(p_kw),
                        "kw_significant_a05":       bool(p_kw < 0.05),
                        "tukey_hsd_significant":    len(tukey_data) > 0,
                        "n_groups":                 len(groups),
                    }
                except Exception as e:
                    result[numeric_col][group_col] = {"error": str(e)}

        return result

    # ── section 10: Levene test (variance homogeneity) ────────────────────────
    def _levene_test(self) -> dict:
        """Test whether variances are equal across groups (churned vs retained)."""
        df = self.df
        result: dict = {}

        churned = df[df[self.target] == 1]
        retained = df[df[self.target] == 0]

        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            g1 = churned[col].dropna()
            g0 = retained[col].dropna()

            try:
                stat, p_val = stats.levene(g1, g0)
                result[col] = {
                    "statistic":           _f(stat),
                    "p_value":             _f(p_val),
                    "equal_variance_a05":  bool(p_val > 0.05),
                    "variance_ratio":      _f(g1.var() / g0.var()) if g0.var() > 0 else None,
                }
            except:
                result[col] = {"error": "Could not compute"}

        return result

    # ── section 11: Eta-squared (categorical effect size) ─────────────────────
    def _eta_squared(self) -> dict:
        """Eta-squared: proportion of variance explained by the categorical variable."""
        df = self.df
        result: dict = {}

        for numeric_col in NUMERIC_COLS:
            if numeric_col not in df.columns:
                continue
            result[numeric_col] = {}

            for cat_col in ["Geography", "Gender", "Card Type", "NumOfProducts"]:
                if cat_col not in df.columns:
                    continue

                try:
                    data = df[[numeric_col, cat_col]].dropna()
                    grand_mean = data[numeric_col].mean()

                    ss_between = sum(
                        len(group) * (group[numeric_col].mean() - grand_mean) ** 2
                        for _, group in data.groupby(cat_col)
                    )

                    ss_total = sum((data[numeric_col] - grand_mean) ** 2)

                    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

                    result[numeric_col][cat_col] = {
                        "eta_squared": round(eta_sq, 4),
                        "effect_interpretation": (
                            "large" if eta_sq >= 0.14
                            else "medium" if eta_sq >= 0.06
                            else "small" if eta_sq >= 0.01
                            else "negligible"
                        ),
                    }
                except:
                    result[numeric_col][cat_col] = {"error": "Could not compute"}

        return result

    # ── section 12: Chi-squared between ALL categoricals ──────────────────────
    def _chi2_between_categoricals(self) -> dict:
        """Chi-squared tests between all pairs of categorical variables."""
        df = self.df
        all_cat = CATEGORICAL_COLS + BINARY_COLS
        result: dict = {}

        for i, col1 in enumerate(all_cat):
            if col1 not in df.columns:
                continue
            for col2 in all_cat[i+1:]:
                if col2 not in df.columns:
                    continue

                ct = pd.crosstab(df[col1], df[col2])
                chi2, p_val, dof, _ = stats.chi2_contingency(ct)
                n = int(ct.sum().sum())
                k = min(ct.shape) - 1
                cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0

                pair_key = f"{col1} × {col2}"
                result[pair_key] = {
                    "chi2_statistic":    _f(chi2),
                    "p_value":           _f(p_val),
                    "cramers_v":         round(cramers_v, 4),
                    "significant_a05":   bool(p_val < 0.05),
                }

        return result

    # ── section 13: 2-way interactions ────────────────────────────────────────
    def _interactions_2way(self) -> dict:
        """Analyze interactions between key features and their relation to churn."""
        df = self.df
        result: dict = {}

        for feat1, feat2 in INTERACTION_PAIRS:
            if feat1 not in df.columns or feat2 not in df.columns:
                continue

            key = f"{feat1} × {feat2}"
            data = df[[feat1, feat2, self.target]].dropna()

            if feat1 in NUMERIC_COLS and feat2 in CATEGORICAL_COLS:
                data[f"{feat1}_bin"] = pd.qcut(data[feat1], q=4, labels=False, duplicates="drop")
                pivot = pd.crosstab(
                    data[f"{feat1}_bin"],
                    data[feat2],
                    values=data[self.target],
                    aggfunc="mean"
                ) * 100

                result[key] = {
                    "interaction_type": "numeric_vs_categorical",
                    "churn_rate_by_bins": pivot.to_dict(orient="index"),
                }
            elif feat1 in CATEGORICAL_COLS and feat2 in NUMERIC_COLS:
                data[f"{feat2}_bin"] = pd.qcut(data[feat2], q=4, labels=False, duplicates="drop")
                pivot = pd.crosstab(
                    data[feat1],
                    data[f"{feat2}_bin"],
                    values=data[self.target],
                    aggfunc="mean"
                ) * 100
                result[key] = {
                    "interaction_type": "categorical_vs_numeric",
                    "churn_rate_by_bins": pivot.to_dict(orient="index"),
                }
            elif feat1 in CATEGORICAL_COLS and feat2 in CATEGORICAL_COLS:
                pivot = pd.crosstab(
                    data[feat1],
                    data[feat2],
                    values=data[self.target],
                    aggfunc="mean"
                ) * 100
                result[key] = {
                    "interaction_type": "categorical_vs_categorical",
                    "churn_rate_by_groups": pivot.to_dict(orient="index"),
                }

        return result

    # ── section 14: Clustering (KMeans) for segmentation ──────────────────────
    def _clustering(self) -> dict:
        """KMeans clustering to discover natural customer segments."""
        df = self.df
        numeric_subdf = df[NUMERIC_COLS].dropna()

        if len(numeric_subdf) == 0:
            return {"error": "No numeric data available"}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_subdf)

        result: dict = {}
        inertias = []
        silhouettes = []

        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(float(kmeans.inertia_))

            from sklearn.metrics import silhouette_score
            sil = silhouette_score(X_scaled, kmeans.labels_)
            silhouettes.append(float(sil))

        result["elbow_inertias"] = inertias
        result["silhouette_scores"] = silhouettes

        optimal_k = silhouettes.index(max(silhouettes)) + 2
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans_final.fit_predict(X_scaled)

        cluster_profiles = {}
        for c in range(optimal_k):
            mask = labels == c
            cluster_df = numeric_subdf[mask]
            cluster_profiles[f"cluster_{c}"] = {
                "size": int(mask.sum()),
                "churn_rate_pct": round(df.loc[cluster_df.index, self.target].mean() * 100, 2),
                "feature_means": {
                    col: _f(cluster_df[col].mean())
                    for col in NUMERIC_COLS
                },
            }

        result["optimal_k"] = int(optimal_k)
        result["cluster_profiles"] = cluster_profiles

        return result

    # ── section 15: Great Expectations suggestions ────────────────────────────
    def _ge_suggestions(
        self,
        numeric_stats:     dict,
        categorical_stats: dict,
    ) -> dict:
        """Generate expectation parameters based on real data."""
        df = self.df
        suggestions: dict = {
            "table_expectations":  [],
            "column_expectations": {},
        }

        n = len(df)

        suggestions["table_expectations"].extend([
            {
                "type":      "expect_table_row_count_to_be_between",
                "rationale": f"Current dataset has {n} rows. Tolerance ±20%.",
                "kwargs":    {
                    "min_value": int(n * 0.8),
                    "max_value": int(n * 1.2),
                },
            },
            {
                "type":      "expect_table_columns_to_match_set",
                "rationale": "Stable schema (no extra or missing columns).",
                "kwargs":    {
                    "column_set":  [c for c in df.columns if c not in ID_COLS],
                    "exact_match": True,
                },
            },
        ])

        for col in NUMERIC_COLS:
            if col not in df.columns or col not in numeric_stats:
                continue
            st  = numeric_stats[col]
            pct = st["percentiles"]
            exps = []

            exps.append({
                "type":      "expect_column_values_to_not_be_null",
                "rationale": "Required numeric column.",
                "kwargs":    {},
            })
            exps.append({
                "type":      "expect_column_values_to_be_between",
                "rationale": f"1st–99th percentile: {pct.get('p01')} — {pct.get('p99')}",
                "kwargs":    {
                    "min_value": pct.get('p01'),
                    "max_value": pct.get('p99'),
                },
            })

            suggestions["column_expectations"][col] = exps

        for col in CATEGORICAL_COLS + BINARY_COLS:
            if col not in df.columns or col not in categorical_stats:
                continue
            st   = categorical_stats[col]
            vals = st.get("unique_values", [])

            suggestions["column_expectations"][col] = [
                {
                    "type":      "expect_column_values_to_be_in_set",
                    "rationale": f"Observed values: {vals}",
                    "kwargs":    {"value_set": vals},
                },
            ]

        return suggestions

    # ── section 16: plot generation ────────────────────────────────────────────
    def _generate_plots(self) -> None:
        """Generate 15+ visualizations focused on classification."""
        df = self.df

        # 1. Target distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        vc = df[self.target].value_counts().sort_index()
        labels = ["Retained", "Churned"]
        colors = ["steelblue", "tomato"]
        axes[0].bar(labels, vc.values, color=colors, edgecolor="white", alpha=0.85)
        axes[0].set_title("Target Variable Distribution", fontweight="bold")
        axes[0].set_ylabel("Count")
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center")

        axes[1].pie(vc.values, labels=labels, autopct="%1.1f%%", colors=colors,
                    startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        axes[1].set_title("Churn Rate", fontweight="bold")
        plt.suptitle("Class Imbalance: Churn vs Retained", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("01_target_distribution", fig))
        plt.close(fig)

        # 2. Numeric — histograms
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i, col in enumerate(NUMERIC_COLS):
            if col not in df.columns:
                continue
            ax = axes[i // 4][i % 4]
            df[col].hist(ax=ax, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(df[col].mean(),   color="red",    linestyle="--", linewidth=1.5, label="mean")
            ax.axvline(df[col].median(), color="orange", linestyle=":",  linewidth=1.5, label="median")
            ax.set_title(col, fontweight="bold", fontsize=10)
            ax.set_xlabel("")
        axes[0][0].legend(fontsize=8)
        plt.suptitle("Numeric Variables Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("02_numeric_distributions", fig))
        plt.close(fig)

        # 3. Boxplots: numeric × churn
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i, col in enumerate(NUMERIC_COLS):
            if col not in df.columns:
                continue
            ax = axes[i // 4][i % 4]
            df.boxplot(column=col, by=self.target, ax=ax, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.6),
                       medianprops=dict(color="red", linewidth=2),
                       flierprops=dict(marker="o", markersize=3, alpha=0.4, color="grey"))
            ax.set_title(col, fontweight="bold", fontsize=10)
            ax.set_xlabel("Exited (0=Retained, 1=Churned)", fontsize=9)
        plt.suptitle("Numeric Comparison: Churned vs Retained (Boxplots)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("03_numeric_by_target_boxplot", fig))
        plt.close(fig)

        # 4. Correlation heatmap
        num_bin_cols = [c for c in NUMERIC_COLS + BINARY_COLS if c in df.columns]
        corr = df[num_bin_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    mask=mask, ax=ax, linewidths=0.4, vmin=-1, vmax=1, annot_kws={"size": 8})
        ax.set_title("Correlation Matrix (Pearson)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("04_correlation_heatmap", fig))
        plt.close(fig)

        # 5. Categoricals × churn (stacked bars)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(CATEGORICAL_COLS):
            if col not in df.columns:
                continue
            ct = pd.crosstab(df[col], df[self.target], normalize="index") * 100
            ct.columns = ["Retained (%)", "Churned (%)"]
            ct.plot(kind="bar", ax=axes[i], color=["steelblue", "tomato"],
                    edgecolor="white", alpha=0.85)
            axes[i].set_title(f"{col} × Churn", fontweight="bold", fontsize=11)
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=15)
            axes[i].legend(loc="upper right", fontsize=9)
            axes[i].set_ylim(0, 110)
        plt.suptitle("Categoricals × Churn (normalized %)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("05_categorical_by_target", fig))
        plt.close(fig)

        # 6. Age × Churn
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for target_val, label, color, ax in [
            (0, "Retained", "steelblue", axes[0]),
            (1, "Churned",  "tomato",    axes[1]),
        ]:
            sub = df[df[self.target] == target_val]["Age"]
            sub.hist(bins=35, ax=ax, color=color, edgecolor="white", alpha=0.85, density=True)
            sub.plot(kind="kde", ax=ax, color="black", linewidth=1.5)
            ax.set_title(f"Age — {label}", fontweight="bold")
            ax.set_xlabel("Age")
            ax.set_ylabel("Density")
        plt.suptitle("Age Distribution (Separated by Churn)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("06_age_distribution_by_churn", fig))
        plt.close(fig)

        # 7. Balance zero analysis
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        zero_pct = round((df["Balance"] == 0).mean() * 100, 1)
        df["Balance"].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white", alpha=0.85)
        axes[0].set_title(f"Balance Distribution\n(zeros = {zero_pct}%)", fontweight="bold")
        axes[0].set_xlabel("Balance")

        nz = df[df["Balance"] > 0]
        for val, label, color in [(0, "Retained", "steelblue"), (1, "Churned", "tomato")]:
            nz[nz[self.target] == val]["Balance"].hist(
                bins=40, ax=axes[1], alpha=0.6, label=label,
                color=color, edgecolor="white")
        axes[1].set_title("Balance (non-zero) × Churn", fontweight="bold")
        axes[1].set_xlabel("Balance")
        axes[1].legend()
        plt.suptitle("Balance Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("07_balance_distribution", fig))
        plt.close(fig)

        # 8. CreditScore × Geography
        geos = sorted(df["Geography"].dropna().unique())
        fig, axes = plt.subplots(1, len(geos), figsize=(6 * len(geos), 5))
        if len(geos) == 1:
            axes = [axes]
        for ax, geo in zip(axes, geos):
            sub = df[df["Geography"] == geo]
            for val, label, color in [(0, "Retained", "steelblue"), (1, "Churned", "tomato")]:
                sub[sub[self.target] == val]["CreditScore"].hist(
                    ax=ax, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
            ax.set_title(f"CreditScore — {geo}", fontweight="bold")
            ax.set_xlabel("CreditScore")
            ax.legend(fontsize=8)
        plt.suptitle("CreditScore by Geography and Churn", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("08_creditscore_by_geography", fig))
        plt.close(fig)

        # 9. Satisfaction × Points
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        ct_sat = pd.crosstab(df["Satisfaction Score"], df[self.target])
        ct_sat.columns = ["Retained", "Churned"]
        ct_sat.plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"],
                    edgecolor="white", alpha=0.85)
        axes[0].set_title("Satisfaction Score × Churn", fontweight="bold")
        axes[0].set_xlabel("Satisfaction Score")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend()

        for val, label, color in [(0, "Retained", "steelblue"), (1, "Churned", "tomato")]:
            df[df[self.target] == val]["Point Earned"].hist(
                bins=40, ax=axes[1], alpha=0.6, label=label, color=color, edgecolor="white")
        axes[1].set_title("Point Earned × Churn", fontweight="bold")
        axes[1].set_xlabel("Point Earned")
        axes[1].legend()
        plt.suptitle("Satisfaction and Points", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("09_satisfaction_and_points", fig))
        plt.close(fig)

        # 10. NumOfProducts × Tenure churn rate
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        pivot_prod = df.groupby("NumOfProducts")[self.target].mean().reset_index()
        axes[0].bar(pivot_prod["NumOfProducts"].astype(str),
                    pivot_prod[self.target] * 100,
                    color="tomato", edgecolor="white", alpha=0.85)
        axes[0].set_title("Churn Rate (%) by NumOfProducts", fontweight="bold")
        axes[0].set_xlabel("NumOfProducts")
        axes[0].set_ylabel("Churn Rate (%)")
        for _, row in pivot_prod.iterrows():
            axes[0].text(str(row["NumOfProducts"]), row[self.target] * 100 + 0.5,
                        f"{row[self.target]*100:.1f}%", ha="center", fontsize=10)

        pivot_ten = df.groupby("Tenure")[self.target].mean().reset_index()
        axes[1].plot(pivot_ten["Tenure"], pivot_ten[self.target] * 100,
                    marker="o", color="tomato", linewidth=2, markersize=6)
        axes[1].set_title("Churn Rate (%) by Tenure", fontweight="bold")
        axes[1].set_xlabel("Tenure (years)")
        axes[1].set_ylabel("Churn Rate (%)")
        axes[1].grid(True, alpha=0.3)
        plt.suptitle("Churn Rate by NumOfProducts and Tenure",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("10_products_and_tenure_churn", fig))
        plt.close(fig)

        # 11. Churn rate by Geography
        fig, ax = plt.subplots(figsize=(10, 6))
        geo_churn = df.groupby("Geography")[self.target].agg(["sum", "count", "mean"]).reset_index()
        geo_churn.columns = ["Geography", "Churned", "Total", "ChurnRate"]
        geo_churn["ChurnRate"] *= 100
        bars = ax.bar(geo_churn["Geography"], geo_churn["ChurnRate"],
                     color="tomato", edgecolor="white", alpha=0.85)
        ax.set_title("Churn Rate by Geography", fontweight="bold", fontsize=12)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Geography")
        for bar, rate in zip(bars, geo_churn["ChurnRate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("11_churn_by_geography", fig))
        plt.close(fig)

        # 12. Gender × Churn
        fig, ax = plt.subplots(figsize=(10, 6))
        gen_churn = df.groupby("Gender")[self.target].agg(["sum", "count", "mean"]).reset_index()
        gen_churn.columns = ["Gender", "Churned", "Total", "ChurnRate"]
        gen_churn["ChurnRate"] *= 100
        bars = ax.bar(gen_churn["Gender"], gen_churn["ChurnRate"],
                     color=["steelblue", "tomato"], edgecolor="white", alpha=0.85)
        ax.set_title("Churn Rate by Gender", fontweight="bold", fontsize=12)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Gender")
        for bar, rate in zip(bars, gen_churn["ChurnRate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("12_churn_by_gender", fig))
        plt.close(fig)

        # 13. Age bins × Churn (segmentation strategy)
        fig, ax = plt.subplots(figsize=(12, 6))
        df_copy = df.copy()
        df_copy["Age_bin"] = pd.cut(df_copy["Age"], bins=[0, 30, 40, 50, 60, 100],
                               labels=["<30", "30-40", "40-50", "50-60", ">60"])
        age_churn = df_copy.groupby("Age_bin")[self.target].agg(["sum", "count", "mean"]).reset_index()
        age_churn.columns = ["AgeGroup", "Churned", "Total", "ChurnRate"]
        age_churn["ChurnRate"] *= 100
        bars = ax.bar(age_churn["AgeGroup"], age_churn["ChurnRate"],
                     color="tomato", edgecolor="white", alpha=0.85)
        ax.set_title("Churn Rate by Age Group", fontweight="bold", fontsize=12)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Age Group")
        for bar, rate, count in zip(bars, age_churn["ChurnRate"], age_churn["Total"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{rate:.1f}%\n(n={count})", ha="center", fontsize=9)
        plt.tight_layout()
        self._plots.append(("13_churn_by_age_bins", fig))
        plt.close(fig)

        # 14. Cumulative churn by Tenure
        fig, ax = plt.subplots(figsize=(12, 6))
        tenure_churn = df.groupby("Tenure")[self.target].agg(["sum", "count", "mean"]).reset_index()
        tenure_churn = tenure_churn.sort_values("Tenure")
        tenure_churn["cum_churned"] = tenure_churn["sum"].cumsum() / df[df[self.target]==1].shape[0] * 100
        ax.plot(tenure_churn["Tenure"], tenure_churn["cum_churned"],
               marker="o", color="tomato", linewidth=2, markersize=4)
        ax.set_title("Cumulative Churned Proportion by Tenure", fontweight="bold", fontsize=12)
        ax.set_ylabel("Cumulative % Churned")
        ax.set_xlabel("Tenure (years)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._plots.append(("14_cumulative_churn_by_tenure", fig))
        plt.close(fig)

        # 15. IsActiveMember × Complain × Churn (3D effect)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for i, active in enumerate([0, 1]):
            sub = df[df["IsActiveMember"] == active]
            ct = pd.crosstab(sub["Complain"], sub[self.target], normalize="index") * 100
            ct.columns = ["Retained", "Churned"]
            ct.plot(kind="bar", ax=axes[i], color=["steelblue", "tomato"],
                   edgecolor="white", alpha=0.85)
            label = "Active" if active == 1 else "Inactive"
            axes[i].set_title(f"IsActiveMember={label}: Complain × Churn", fontweight="bold")
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_xlabel("Complain (0=No, 1=Yes)")
            axes[i].tick_params(axis="x", rotation=0)
            axes[i].legend()
            axes[i].set_ylim(0, 110)
        plt.suptitle("Effect of Member Activity and Complaint on Churn",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._plots.append(("15_active_member_complain_churn", fig))
        plt.close(fig)

    # ── main orchestrator ──────────────────────────────────────────────────────
    def run(self) -> dict:
        """Execute all analysis sections."""
        steps = [
            ("Overview",                   self._overview),
            ("Missing values",             self._missing_values),
            ("Duplicates",                 self._duplicates),
            ("Numeric statistics",         self._numeric_stats),
            ("Categorical statistics",     self._categorical_stats),
            ("Target analysis",            self._target_analysis),
            ("Correlations",               self._correlations),
            ("Univariate tests",           self._statistical_tests),
            ("ANOVA + Kruskal-Wallis",     self._anova_and_kruskal),
            ("Levene test",                self._levene_test),
            ("Eta-squared",                self._eta_squared),
            ("Chi-squared categoricals",   self._chi2_between_categoricals),
            ("2-way interactions",         self._interactions_2way),
            ("Clustering (KMeans)",        self._clustering),
        ]

        results: dict = {}
        keys = [
            "overview", "missing_values", "duplicates",
            "numeric_statistics", "categorical_statistics",
            "target_analysis", "correlations", "statistical_tests",
            "anova_kruskal_wallis", "levene_test", "eta_squared",
            "chi2_categoricals", "interactions_2way", "clustering",
        ]

        logger.info("Running exploratory analysis...")
        for (label, fn), key in zip(steps, keys):
            idx = keys.index(key) + 1
            try:
                results[key] = fn()
                logger.info("  [%2d/%d] %s ✓", idx, len(steps), label)
            except Exception as e:
                logger.error("  [%2d/%d] %s ✗ (error: %s)", idx, len(steps), label, str(e)[:50])
                results[key] = {"error": str(e)}

        try:
            results["great_expectations_suggestions"] = self._ge_suggestions(
                results["numeric_statistics"],
                results["categorical_statistics"],
            )
            logger.info("  [%2d/%d] Great Expectations Suggestions ✓", len(steps) + 1, len(steps) + 1)
        except Exception as e:
            logger.error("  [%2d/%d] Great Expectations Suggestions ✗ (%s)", len(steps) + 1, len(steps) + 1, e)

        return {
            "metadata": {
                "generated_at":    datetime.utcnow().isoformat() + "Z",
                "dataset_path":    str(DATA_PATH),
                "analysis_version": "2.0.0",
                "context":         "Classification (Bank Churn)",
            },
            **results,
        }

    def save_report(self, report: dict, output_dir: Path) -> Path:
        """Save report as JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"eda_report_{ts}.json"

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        logger.info("JSON report: %s", report_path.name)
        return report_path

    def save_plots(self, output_dir: Path) -> list[Path]:
        """Generate and save all plots."""
        logger.info("Generating visualizations...")
        self._generate_plots()

        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving %d visualizations...", len(self._plots))
        saved: list[Path] = []
        for name, fig in self._plots:
            path = plots_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(path)
            logger.info("  ✓ %s.png", name)
        plt.close("all")
        return saved


# ── entrypoint ─────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 70)
    logger.info("  BANK CHURN — Exploratory Data Analysis (EDA v2.0 — Classification)")
    logger.info("=" * 70)
    logger.info("  Dataset    : %s", DATA_PATH)
    logger.info("  Output     : %s", OUTPUT_DIR)
    logger.info("=" * 70)

    if not DATA_PATH.exists():
        logger.error("File not found: %s", DATA_PATH)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset loaded: %d rows x %d columns", df.shape[0], df.shape[1])

    analyzer = EDAAnalyzer(df)
    report = analyzer.run()

    logger.info("Saving outputs...")
    report_path = analyzer.save_report(report, OUTPUT_DIR)
    plot_paths  = analyzer.save_plots(OUTPUT_DIR)

    # ── final summary ──────────────────────────────────────────────────────────
    overview = report["overview"]
    target   = report["target_analysis"]
    missing  = report["missing_values"]

    logger.info("=" * 70)
    logger.info("  ANALYSIS SUMMARY")
    logger.info("=" * 70)
    logger.info("  Rows x Columns           : %d x %d", overview['n_rows'], overview['n_cols'])
    logger.info("  Memory                   : %s MB", overview['memory_mb'])
    logger.info("  Churn Rate               : %.2f%% (%d/%d)", target['churn_rate_pct'], target['n_churned'], target['n_total'])
    logger.info("  Class Imbalance (1:N)    : 1:%.2f", target['class_imbalance_ratio'])
    logger.info("  Missing cells            : %d (%.4f%%)", missing['total_missing_cells'], missing['overall_missing_pct'])
    logger.info("  Visualizations generated : %d", len(plot_paths))
    logger.info("  JSON saved               : %s", report_path.name)
    logger.info("=" * 70)

    # Top correlations
    ranked = report["correlations"].get("target_correlations_ranked", [])
    if ranked:
        logger.info("Top Features (Pearson |r| with Exited):")
        for entry in ranked[:5]:
            r = entry.get("pearson_r")
            if r:
                logger.info("  %-22s r = %+.4f", entry['feature'], r)

    # Significant differences
    stat_tests = report.get("statistical_tests", {})
    sig_cols = [col for col, res in stat_tests.items() if res.get("mann_whitney_u", {}).get("significant_a05")]
    if sig_cols:
        logger.info("Features with significant difference (Mann-Whitney p<0.05):")
        for col in sig_cols[:5]:
            d = stat_tests[col].get("cohens_d", 0)
            logger.info("  %-22s Cohen's d = %+.3f", col, d)

    # Clustering insights
    clustering_info = report.get("clustering", {})
    if "optimal_k" in clustering_info:
        logger.info("Clustering: %d customer segments found", clustering_info['optimal_k'])

    logger.info("See the JSON for detailed analyses (ANOVA, Tukey, Interactions)")
    logger.info("Use the plots for classification-focused visualizations")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
