"""
feature_reducer.py — Dimensionality reduction.

Supports: none | rfe | pca | kpca | lda
Configured via modeling.yaml → feature_reduction.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.ensemble import RandomForestClassifier as _RF


class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for dimensionality reduction methods.

    Parameters
    ----------
    method               : 'none' | 'rfe' | 'pca' | 'kpca' | 'lda'
    n_features_to_select : number of features (RFE)
    rfe_estimator        : 'random_forest' | 'logistic_regression' (RFE)
    n_components         : number of components (PCA / KPCA / LDA)
                           LDA: limited to n_classes-1 (1 for binary classification)
    kernel               : KPCA kernel ('rbf' | 'poly' | 'cosine')
    gamma                : kernel parameter (None = 1/n_features)
    degree               : degree of the poly kernel
    coef0                : independent coefficient (poly / sigmoid)
    random_state         : random seed

    Learned attributes in fit
    -------------------------
    selected_features    : list of feature names after reduction
                           RFE  → original selected names
                           PCA/KPCA → ['pc_0', 'pc_1', ...] / ['kpc_0', ...]
                           LDA  → ['lda_0'] (binary) or ['lda_0', 'lda_1', ...]
    """

    def __init__(
        self,
        method: str = 'none',
        n_features_to_select: int = 15,
        rfe_estimator: str = 'random_forest',
        n_components: int = 15,
        kernel: str = 'rbf',
        gamma=None,
        degree: int = 3,
        coef0: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.rfe_estimator = rfe_estimator
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, y=None) -> "FeatureReducer":
        cols = list(X.columns) if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
        X_arr = X.values if hasattr(X, 'values') else np.array(X)

        if self.method == 'none':
            self.selected_features = cols

        elif self.method == 'rfe':
            if self.rfe_estimator == 'random_forest':
                base = _RF(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                base = _LR(random_state=self.random_state, max_iter=1000, n_jobs=-1)
            n = min(self.n_features_to_select, len(cols))
            self.selector_ = RFE(estimator=base, n_features_to_select=n)
            self.selector_.fit(X_arr, y)
            self.selected_features = [c for c, s in zip(cols, self.selector_.support_) if s]

        elif self.method == 'pca':
            n = min(self.n_components, len(cols))
            self.reducer_ = PCA(n_components=n, random_state=self.random_state)
            self.reducer_.fit(X_arr)
            self.selected_features = [f'pc_{i}' for i in range(n)]

        elif self.method == 'kpca':
            n = min(self.n_components, len(cols))
            self.reducer_ = KernelPCA(
                n_components=n,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                random_state=self.random_state,
            )
            self.reducer_.fit(X_arr)
            self.selected_features = [f'kpc_{i}' for i in range(n)]

        elif self.method == 'lda':
            n_classes = len(np.unique(y)) if y is not None else 2
            # LDA maximum: n_classes - 1 components
            n = min(self.n_components, n_classes - 1)
            self.reducer_ = _LDA(n_components=n)
            self.reducer_.fit(X_arr, y)
            self.selected_features = [f'lda_{i}' for i in range(n)]

        else:
            raise ValueError(f"method={self.method!r} is not supported. Use: none | rfe | pca | kpca | lda")

        return self

    def transform(self, X):
        idx = X.index if hasattr(X, 'index') else None
        X_arr = X.values if hasattr(X, 'values') else np.array(X)

        if self.method == 'none':
            return X
        elif self.method == 'rfe':
            result = self.selector_.transform(X_arr)
            return pd.DataFrame(result, index=idx, columns=self.selected_features)
        elif self.method in ('pca', 'kpca', 'lda'):
            result = self.reducer_.transform(X_arr)
            return pd.DataFrame(result, index=idx, columns=self.selected_features)

        return X
