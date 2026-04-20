"""
preprocessing.py — Transformadores de Pré-processamento e Feature Engineering.

Interface scikit-learn (BaseEstimator + TransformerMixin):
  Cada classe herda de sklearn.base.BaseEstimator e TransformerMixin.

  BaseEstimator  → fornece get_params() e set_params() automaticamente,
                   necessários para GridSearchCV e clone() de Pipeline.
                   REGRA: todos os parâmetros do __init__ devem ter o
                   mesmo nome que o atributo de instância (ex: self.group_col).

  TransformerMixin → fornece fit_transform(X) automaticamente como
                     self.fit(X).transform(X), compatível com sklearn.Pipeline.

  Isso permite compor os transformadores em um Pipeline:
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

Transformadores stateless (fit é no-op — retorna self sem aprender):
  ColumnDropper, BinaryFlagTransformer, InteractionFeatureTransformer,
  RatioFeatureTransformer, AgeBinTransformer, CategoricalEncoder, FeatureSelector

Transformadores stateful (fit aprende parâmetros dos dados de treino):
  GroupMedianImputer        → aprende medianas por grupo
  DataImputer               → aprende medianas, modas e constantes por coluna
  StandardScalerTransformer → aprende média e desvio padrão (CreditScore, EstimatedSalary, Point Earned)
  RobustScalerTransformer   → aprende mediana e IQR (Age com log1p, Balance)
"""
import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imputação Configurável por Coluna
# ─────────────────────────────────────────────────────────────────────────────

class DataImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores ausentes usando estratégias configuradas por coluna.

    Estratégias suportadas (EDA Seção 8.3):
    - "median"          : mediana global — para distribuições assimétricas (Age, Tenure)
    - "constant"        : valor fixo (fill_value) — Balance → 0 (valor legítimo frequente)
    - "mode"            : moda (valor mais frequente) — para categóricas e NumOfProducts
    - "median_by_group" : mediana estratificada por grupo — CreditScore por Geography

    Por que estratégias diferenciadas?
    - Balance: 36.17% de zeros são legítimos → fill_value=0, não mediana
    - CreditScore: varia por região geográfica → mediana por Geography, não global
    - Age: distribuição assimétrica (skew=1.01) → mediana (37) supera a média (38.9)

    ⚠ AVISO MLOps — Data Leakage:
    Transformer STATEFUL: aprende medianas e modas no fit() usando apenas
    os dados de treino. Deve ser usado no Pipeline de modelagem, APÓS o
    split treino/holdout — nunca antes.

    Atributos aprendidos no fit:
        fill_values_ (dict): {coluna → valor de preenchimento aprendido}
          Para "median_by_group": {"by_group": {grupo: mediana}, "global": mediana_global,
                                    "group_col": nome_da_coluna_de_grupo}

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

    Exemplo (pipeline de modelagem):
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
        Aprende o valor de preenchimento para cada coluna configurada.

        ATENÇÃO MLOps: chamar fit() apenas no conjunto de treino.
        """
        self.fill_values_: dict = {}

        for spec in self.imputation_config:
            col = spec["column"]
            strategy = spec["strategy"]

            if col not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "DataImputer.fit: coluna '%s' não encontrada — ignorada.", col
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
                            "DataImputer.fit: coluna de grupo '%s' não encontrada "
                            "para '%s' — usando mediana global.", group_col, col
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
                        "DataImputer.fit: estratégia '%s' não suportada para '%s' — ignorada.",
                        strategy, col,
                    )
                continue

            self._log(
                "DataImputer.fit: '%s' → strategy=%s | valor aprendido: %s",
                col, strategy,
                self.fill_values_[col] if strategy != "median_by_group"
                else self.fill_values_[col]["by_group"],
            )

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Preenche NaN em cada coluna com o valor aprendido no fit.

        Colunas sem NaN são mantidas intactas (operação segura).
        """
        if not hasattr(self, "fill_values_"):
            raise RuntimeError(
                "DataImputer não foi ajustado. Chame fit() antes de transform()."
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
# 1b. Imputação por Mediana de Grupo (mantida para compatibilidade)
# ─────────────────────────────────────────────────────────────────────────────

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores ausentes usando a mediana do grupo (estratificada).

    Por que mediana por grupo?
    - CreditScore varia sistematicamente por Geography (France, Germany, Spain).
    - Imputar com a mediana global ignora essa heterogeneidade regional.
    - EDA Seção 8.3: "CreditScore → mediana por Geography (score varia por região)"

    ⚠ AVISO MLOps — Data Leakage:
    Este transformador é STATEFUL: aprende as medianas no fit() usando apenas
    os dados de treino. Deve ser aplicado DENTRO do Pipeline de modelagem,
    APÓS o split treino/holdout — nunca antes.

    Compatibilidade com sklearn.Pipeline:
        BaseEstimator fornece get_params()/set_params() via introspecção dos
        parâmetros do __init__ (group_col, target_col, logger).
        TransformerMixin fornece fit_transform().

    Atributos aprendidos no fit:
        medians_       (dict): {valor_do_grupo → mediana}
        global_median_ (float): fallback para grupos não vistos no fit
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
        Aprende a mediana de target_col para cada valor de group_col.

        O parâmetro y=None existe por convenção da API scikit-learn —
        não é utilizado (transformador não supervisionado).

        Raises:
            KeyError: Se group_col ou target_col não existirem no DataFrame.
        """
        missing_cols = [c for c in [self.group_col, self.target_col] if c not in X.columns]
        if missing_cols:
            raise KeyError(
                f"GroupMedianImputer.fit: colunas ausentes no DataFrame: {missing_cols}"
            )

        self.medians_ = (
            X.groupby(self.group_col)[self.target_col]
            .median()
            .to_dict()
        )
        self.global_median_ = float(X[self.target_col].median())

        self._log(
            "GroupMedianImputer.fit: medianas aprendidas por '%s' para '%s': %s",
            self.group_col, self.target_col,
            {k: round(v, 1) for k, v in self.medians_.items()},
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Preenche NaN em target_col com a mediana do grupo correspondente.

        Linhas cujo grupo não foi visto no fit recebem a mediana global.
        """
        if not hasattr(self, "medians_"):
            raise RuntimeError(
                "GroupMedianImputer não foi ajustado. Chame fit() antes de transform()."
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
            "GroupMedianImputer.transform: '%s' — NaN antes=%d, depois=%d",
            self.target_col, n_before, n_after,
        )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 2. Remoção de Identificadores
# ─────────────────────────────────────────────────────────────────────────────

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Remove colunas sem valor preditivo (identificadores e dados pessoais).

    Por que remover identificadores?
    - RowNumber: índice sequencial sem relação causal com churn.
    - CustomerId: ID único — se visto em treino, o modelo pode memorizar
      clientes específicos em vez de aprender padrões generalizáveis.
    - Surname: texto de identificação pessoal sem poder preditivo.
    - EDA Seção 8.1: COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

    Config (preprocessing.yaml → drop_columns):
        - "RowNumber"
        - "CustomerId"
        - "Surname"

    Exemplo:
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
                "ColumnDropper: colunas não encontradas (ignoradas): %s", skipped
            )

        X.drop(columns=to_drop, inplace=True)
        self._log(
            "ColumnDropper: %d colunas removidas: %s", len(to_drop), to_drop
        )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 3. Flags Binárias
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Adiciona colunas binárias (0/1) indicando padrões de negócio relevantes.

    Por que flags binárias?
    - Balance == 0: 36.17% dos clientes — dois perfis distintos de uso bancário.
      Modelos precisam reconhecer explicitamente que saldo zero é um estado,
      não um outlier a ser imputado ou ignorado.
    - NumOfProducts >= 3: limiar de saturação abrupto (churn de 82.7%–100%).
      Sem a flag, um modelo precisaria descobrir esse limiar sozinho.

    Operadores suportados:
      "==" (padrão): X[column] == value
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

    Exemplo:
        flags_cfg = config['binary_flags']
        transformer = BinaryFlagTransformer(flags=flags_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, flags: list[dict], logger: Any = None) -> None:
        self.flags = flags
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
                        "BinaryFlagTransformer: coluna '%s' não encontrada — flag '%s' ignorada.",
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
                        "BinaryFlagTransformer: operador '%s' não suportado para '%s' — ignorada.",
                        operator, new_col,
                    )
                continue

            X[new_col] = mask.astype(int)
            n_flagged = int(X[new_col].sum())
            self._log(
                "BinaryFlagTransformer: '%s' %s %s → '%s': %d linhas flagadas (%.2f%%)",
                col, operator, val, new_col, n_flagged, 100 * n_flagged / len(X),
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. Features de Interação
# ─────────────────────────────────────────────────────────────────────────────

class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Cria features de interação e compostas para capturar padrões não-lineares.

    Tipos suportados:

    product_complement: col_a × (1 − col_b)
      AgeInactivity = Age × (1 − IsActiveMember)
      Amplifica o sinal de inatividade em clientes mais velhos — o grupo
      com maior propensão ao churn. Um cliente de 50 anos inativo terá
      AgeInactivity=50; um ativo terá 0.

    engagement_composite: score composto de engajamento do cliente
      EngagementScore = IsActiveMember + (NumOfProducts == 2) + HasCrCard − HasZeroBalance
      Componentes positivos: membro ativo (+1), exatamente 2 produtos (+1, ponto
      de menor churn na EDA), possui cartão de crédito (+1).
      Componente negativo: saldo zero (−1, desengajamento financeiro).
      Range resultante: -1 (totalmente desengajado) a 3 (muito engajado).
      ⚠ Requer HasZeroBalance criado pelo BinaryFlagTransformer.
      Ordem obrigatória das colunas: [IsActiveMember, NumOfProducts, HasCrCard, HasZeroBalance]

    Config (preprocessing.yaml → interaction_features):
        - name: "AgeInactivity"
          type: "product_complement"
          col_a: "Age"
          col_b: "IsActiveMember"
        - name: "EngagementScore"
          type: "engagement_composite"
          columns: ["IsActiveMember", "NumOfProducts", "HasCrCard", "HasZeroBalance"]

    Exemplo:
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
                            "InteractionFeatureTransformer: colunas ausentes %s — '%s' ignorada.",
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
                            "InteractionFeatureTransformer: colunas ausentes %s — '%s' ignorada.",
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
                        "InteractionFeatureTransformer: tipo '%s' não suportado para '%s' — ignorada.",
                        ftype, name,
                    )

        self._log("InteractionFeatureTransformer: features criadas: %s", created)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 5. Features de Razão
# ─────────────────────────────────────────────────────────────────────────────

class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Cria features de razão: numerador / (denominador + offset).

    Por que BalanceSalaryRatio?
    - Clientes com saldo alto relativo ao salário são os mais disputados
      pela concorrência — exatamente o perfil com maior churn no dataset.
    - O offset (denominator_offset=1) no denominador protege contra divisão
      por zero sem distorcer valores onde EstimatedSalary > 0.

    Divisão segura:
    - (denominador + offset) == 0 → NaN (evita divisão por zero se offset=0)
    - Inf substituído por NaN

    Config (preprocessing.yaml → ratio_features):
        - name: "BalanceSalaryRatio"
          numerator: "Balance"
          denominator: "EstimatedSalary"
          denominator_offset: 1

    Exemplo:
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
                        "RatioFeatureTransformer: colunas '%s' ou '%s' ausentes — '%s' ignorada.",
                        num, den, name,
                    )
                continue

            denominator = X[den] + offset
            X[name] = (X[num] / denominator.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
            created.append(name)

        self._log("RatioFeatureTransformer: features criadas: %s", created)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 6. Faixas Etárias
# ─────────────────────────────────────────────────────────────────────────────

class AgeBinTransformer(BaseEstimator, TransformerMixin):
    """
    Cria faixa etária (AgeGroup) a partir da coluna de idade.

    Por que discretizar a idade?
    - A relação entre idade e churn é não-linear: pico entre 40–60 anos.
    - EDA: idade média dos que saíram = 44.8 anos vs 37.4 dos que ficaram
      (Cohen's d = 0.747 — efeito médio-grande).
    - Bins permitem que modelos lineares capturem esse efeito curvilíneo
      sem exigir transformação polinomial.

    Encoding ordinal (ordinal_encoding: true):
    - Converte labels categóricos para inteiros (0, 1, 2, 3, 4).
    - Preserva a ordem natural da idade (importante para GBM/XGBoost).
    - Para regressão linear: aplicar OHE sobre AgeGroup no pipeline de modelagem.

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

    Exemplo:
        age_cfg = config['age_group']
        transformer = AgeBinTransformer(age_config=age_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, age_config: dict, logger: Any = None) -> None:
        self.age_config = age_config
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
                    "AgeBinTransformer: coluna '%s' não encontrada — '%s' ignorada.",
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
                    "AgeBinTransformer: %d valores de '%s' não mapeados → NaN",
                    n_unknown, new_col,
                )

        self._log(
            "AgeBinTransformer: '%s' → '%s' | bins=%s | ordinal=%s",
            col, new_col, bins, ordinal_encoding,
        )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 7. Encoding de Variáveis Categóricas
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica encoding em múltiplas variáveis categóricas.

    Métodos suportados:

    one_hot: pd.get_dummies com prefixo configurável
      Geography → geo_France, geo_Germany, geo_Spain
      Card Type → card_DIAMOND, card_GOLD, card_PLATINUM, card_SILVER
      drop_first=False mantém todas as categorias para máxima interpretabilidade.

    binary: mapeamento manual para 0/1
      Gender → Gender_encoded (Male=0, Female=1)
      Mais compacto que OHE para variáveis com exatamente 2 categorias.

    Por que OHE para Geography e não ordinal?
    - Não há ordem natural entre France, Germany, Spain.
    - OHE garante que o modelo trate cada país como preditor independente.
    - EDA: Germany tem churn 2× maior (Cramér's V=0.17) — efeito não-linear
      que seria distorcido por um encoding ordinal arbitrário.

    As colunas categóricas originais são mantidas no DataFrame para
    inspeção. O FeatureSelector ao final exclui o que não entra no modelo.

    Config (preprocessing.yaml → categorical_encoding):
        - column: "Geography"
          method: "one_hot"
          prefix: "geo"
          drop_first: false
        - column: "Gender"
          method: "binary"
          mapping: {"Male": 0, "Female": 1}
          new_column: "Gender_encoded"

    Exemplo:
        enc_cfg = config['categorical_encoding']
        encoder = CategoricalEncoder(encoding_config=enc_cfg, logger=logger)
        df = encoder.fit_transform(df)
    """

    def __init__(self, encoding_config: list[dict], logger: Any = None) -> None:
        self.encoding_config = encoding_config
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
                        "CategoricalEncoder: coluna '%s' não encontrada — encoding ignorado.",
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
                X = pd.concat([X, dummies], axis=1)
                self._log(
                    "CategoricalEncoder: OHE '%s' (prefixo='%s') → %s",
                    col, prefix, list(dummies.columns),
                )

            elif method == "binary":
                mapping: dict = spec.get("mapping", {})
                new_col: str = spec.get("new_column", f"{col}_encoded")
                X[new_col] = X[col].map(mapping)
                n_unknown = int(X[new_col].isna().sum())
                if n_unknown > 0 and self.logger:
                    self.logger.warning(
                        "CategoricalEncoder: %d valores de '%s' não mapeados → NaN",
                        n_unknown, col,
                    )
                self._log(
                    "CategoricalEncoder: binary '%s' → '%s' (mapa=%s)",
                    col, new_col, mapping,
                )

            else:
                if self.logger:
                    self.logger.warning(
                        "CategoricalEncoder: método '%s' não suportado para '%s' — ignorado.",
                        method, col,
                    )

        return X


# ─────────────────────────────────────────────────────────────────────────────
# 8. Seleção de Features
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Seleciona o conjunto final de features para modelagem.

    Por que seleção explícita?
    - Após as transformações, o DataFrame tem 30+ colunas (originais + engineered).
    - Colunas categóricas originais (Geography, Gender, Card Type) são substituídas
      pelo encoding — mantê-las causaria redundância.
    - Manter apenas o que vai para o modelo previne vazamento acidental de dados.

    Comportamento tolerante:
    - Colunas ausentes geram WARNING (não exceção) — permite que o pipeline
      continue mesmo que uma transformação anterior tenha sido pulada.
    - Apenas as colunas disponíveis são selecionadas.

    Config (preprocessing.yaml → feature_selection.features_to_keep):
        - "CreditScore"
        - "Age"
        - "geo_France"
        - ...

    Exemplo:
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
        """Valida que todas as features configuradas existem no DataFrame."""
        missing = [c for c in self.features_to_keep if c not in X.columns]
        if missing and self.logger:
            self.logger.warning(
                "FeatureSelector.fit: %d colunas da config ausentes no DataFrame "
                "(serão ignoradas): %s",
                len(missing), missing,
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        available = [c for c in self.features_to_keep if c in X.columns]
        dropped = len(self.features_to_keep) - len(available)

        if dropped > 0 and self.logger:
            self.logger.warning(
                "FeatureSelector.transform: %d/%d colunas solicitadas ausentes — ignoradas.",
                dropped, len(self.features_to_keep),
            )

        self._log(
            "FeatureSelector.transform: %d/%d features selecionadas | shape: %s → %s",
            len(available), len(self.features_to_keep),
            X.shape, (len(X), len(available)),
        )
        return X[available].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Escalonamento (StandardScaler)
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica Z-score normalization: z = (x − μ) / σ.

    Por que StandardScaler?
    - Regressão logística, SVM e KNN são sensíveis à escala das features.
    - Gradient boosting e Random Forest NÃO precisam de escalonamento —
      mas manter o dataset escalado facilita a comparação entre modelos.
    - Binárias (0/1) e ordinais têm escala interpretável → não escalonar.

    EDA Seção 8.1 — Técnicas por variável:
      CreditScore      → StandardScaler (distribuição quase normal, skew=-0.07)
      EstimatedSalary  → StandardScaler (uniforme, sem outliers)
      Point Earned     → StandardScaler (uniforme, sem outliers)
      Age              → RobustScaler   (assimétrica, skew=+1.01) — aplicar no modelo
      Balance          → RobustScaler   (bimodal, 36% zeros)      — aplicar no modelo
      Tenure           → MinMaxScaler ou nenhum (uniforme discreta 0–10)
      NumOfProducts    → nenhum (tratar como ordinal)

    ⚠ AVISO MLOps — Data Leakage:
    Este transformador é STATEFUL: aprende média e desvio padrão no fit()
    usando apenas os dados de treino. Deve ser aplicado DENTRO do Pipeline
    de modelagem, APÓS o split treino/holdout — nunca antes.

    Parâmetros aprendidos no fit (só no dataset de treino!):
        mean_  (dict): {coluna: média}
        std_   (dict): {coluna: desvio padrão}

    Colunas com std=0 são ignoradas (constantes — sem informação).

    Exemplo (pipeline de modelagem):
        scaler = StandardScalerTransformer(columns=scale_cols, logger=logger)
        scaler.fit(X_train)            # aprende apenas no treino
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
        Aprende média e desvio padrão das colunas especificadas.

        ATENÇÃO MLOps: sempre chamar fit() apenas no conjunto de treino.
        Usar transform() no conjunto de validação/teste para evitar data leakage.
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
                        "StandardScalerTransformer.fit: '%s' tem std=0 (constante) — ignorada.",
                        col,
                    )
                continue

            self.mean_[col] = mu
            self.std_[col] = sigma

        if skipped and self.logger:
            self.logger.warning(
                "StandardScalerTransformer.fit: colunas ausentes ignoradas: %s", skipped
            )

        self._log(
            "StandardScalerTransformer.fit: parâmetros aprendidos para %d colunas.",
            len(self.mean_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Aplica Z-score nas colunas ajustadas no fit.

        Colunas não vistas no fit são mantidas sem alteração.
        """
        if not hasattr(self, "mean_"):
            raise RuntimeError(
                "StandardScalerTransformer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        scaled: list[str] = []

        for col, mu in self.mean_.items():
            if col not in X.columns:
                continue
            X[col] = (X[col] - mu) / self.std_[col]
            scaled.append(col)

        self._log(
            "StandardScalerTransformer.transform: %d colunas escalonadas (z-score).",
            len(scaled),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Retorna DataFrame com média e desvio padrão aprendidos (útil para auditoria)."""
        return pd.DataFrame(
            {"mean": self.mean_, "std": self.std_}
        ).rename_axis("feature")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Escalonamento Robusto (RobustScaler)
# ─────────────────────────────────────────────────────────────────────────────

class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica escalonamento robusto: z = (x − mediana) / IQR.

    Por que RobustScaler para Age e Balance?
    - Age: distribuição assimétrica (skew=+1.01) com outliers válidos de clientes
      idosos até 92 anos. StandardScaler seria distorcido pela cauda longa —
      média e desvio padrão são influenciados pelos extremos.
    - Balance: distribuição bimodal com 36% de zeros. Mediana e IQR capturam
      melhor a dispersão real do que média e desvio padrão.

    Log transform opcional (log_columns):
    - Age recebe np.log1p (log(1+x)) antes do RobustScaler para comprimir
      a cauda direita e aproximar a distribuição da normalidade.
    - np.log1p é preferível a np.log pois é compatível com valor zero.
    - O log transform é aprendido implicitamente: mediana e IQR são calculados
      sobre a série já transformada (log1p(Age)), garantindo consistência entre
      fit e transform.

    ⚠ AVISO MLOps — Data Leakage:
    STATEFUL: aprende mediana e IQR no fit() usando apenas os dados de treino.
    Deve ser aplicado DENTRO do Pipeline de modelagem, APÓS o split treino/holdout.

    Parâmetros aprendidos no fit:
        median_ (dict): {coluna: mediana (da série pós log1p se aplicável)}
        iqr_    (dict): {coluna: IQR Q75 − Q25 (da série pós log1p se aplicável)}

    Config (preprocessing.yaml → scaling.robust_columns / log_transform_before_robust):
        robust_columns: ["Age", "Balance"]
        log_transform_before_robust: ["Age"]

    Exemplo (pipeline de modelagem — Regressão Logística):
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
        Aprende mediana e IQR das colunas especificadas.

        Para colunas em log_columns, calcula mediana e IQR sobre log1p(série).
        ATENÇÃO MLOps: sempre chamar fit() apenas no conjunto de treino.
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
                        "RobustScalerTransformer.fit: '%s' tem IQR=0 (constante) — ignorada.",
                        col,
                    )
                continue

            self.median_[col] = float(series.median())
            self.iqr_[col] = iqr

            self._log(
                "RobustScalerTransformer.fit: '%s'%s → mediana=%.4f | IQR=%.4f",
                col,
                " (log1p)" if col in self.log_columns else "",
                self.median_[col],
                self.iqr_[col],
            )

        if skipped and self.logger:
            self.logger.warning(
                "RobustScalerTransformer.fit: colunas ausentes ignoradas: %s", skipped
            )

        self._log(
            "RobustScalerTransformer.fit: parâmetros aprendidos para %d colunas.",
            len(self.median_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Aplica escalonamento robusto nas colunas ajustadas no fit.

        Para colunas em log_columns, aplica log1p antes de escalonar.
        Colunas não vistas no fit são mantidas sem alteração.
        """
        if not hasattr(self, "median_"):
            raise RuntimeError(
                "RobustScalerTransformer não foi ajustado. Chame fit() antes de transform()."
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
            "RobustScalerTransformer.transform: %d colunas escalonadas (robust).",
            len(scaled),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Retorna DataFrame com mediana e IQR aprendidos (útil para auditoria)."""
        return pd.DataFrame(
            {"median": self.median_, "iqr": self.iqr_}
        ).rename_axis("feature")
