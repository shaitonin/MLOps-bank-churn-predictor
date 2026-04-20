# %%
#
# Transformações cobertas (baseadas no EDA Report — Seções 8.1, 8.2, 8.3):
#   1. Remoção de identificadores — RowNumber, CustomerId, Surname (8.1)
#   2. Imputação       — delegada ao Pipeline de modelagem (stateful — ver nota) (8.3)
#   3. Flags binárias  — HasZeroBalance, HighRiskProducts (8.1)
#   4. Interações      — AgeInactivity, EngagementScore (8.1)
#   5. Razões          — BalanceSalaryRatio (8.1)
#   6. Faixas etárias  — AgeGroup (encoding ordinal) (8.1)
#   7. Encoding        — Geography (OHE), Gender (binary), Card Type (OHE) (8.1)
#   8. Seleção         — features_to_keep do EDA Report (8.1)
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Configuração do Ambiente
import sys
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

# Definições de caminhos
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]
prep_yaml_path = CONFIG_DIR / 'preprocessing.yaml'

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml

# ── Importa todos os transformadores do módulo src/preprocessing.py ──────────
from src.preprocessing import (
    DataImputer,
    ColumnDropper,
    BinaryFlagTransformer,
    InteractionFeatureTransformer,
    RatioFeatureTransformer,
    AgeBinTransformer,
    CategoricalEncoder,
    FeatureSelector,
)

# %%
config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
preprocessing = load_yaml(prep_yaml_path)
config.update(preprocessing)  # Mescla as configs (pipeline + preprocessing)

# Configura o logger
log_cfg = config.get('logging')
logger = get_logger(
    name='preprocessamento',
    logging_config=log_cfg
)

logger.info('=== Pré-processamento e Feature Engineering — Bank Churn ===')
logger.info('Config carregada: pipeline.yaml + preprocessing.yaml')

# ─────────────────────────────────────────────────────────────────────────────
# Inspeciona a configuração carregada
#
# Ponto de ensino: o aluno vê exatamente o que foi lido do YAML.
# Qualquer mudança no preprocessing.yaml aparece aqui sem alterar o código.
# ─────────────────────────────────────────────────────────────────────────────

# %%
prep_cfg    = config.get('preprocessing', {})
output_dir  = ROOT_DIR / prep_cfg.get('output_dir', 'data/features')
output_path = output_dir / prep_cfg.get('output_filename', 'bank_churn_features.parquet')
compression = prep_cfg.get('compression', 'snappy')

logger.info('Saída      : %s', output_path)
logger.info('Compressão : %s', compression)

# %%
# Lista as transformações configuradas no YAML
logger.info('Colunas a remover       : %d', len(config.get('drop_columns', [])))
logger.info('Flags binárias          : %d', len(config.get('binary_flags', [])))
logger.info('Features de interação   : %d', len(config.get('interaction_features', [])))
logger.info('Features de razão       : %d', len(config.get('ratio_features', [])))
logger.info('Encoding categórico     : %d', len(config.get('categorical_encoding', [])))
logger.info('Features para seleção   : %d',
            len(config.get('feature_selection', {}).get('features_to_keep', [])))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Carregamento do Dataset
# %%
# Caminho do Parquet gerado pela etapa de ingestão + qualidade
processed_dir = ROOT_DIR / config['paths']['processed_data_dir']
parquet_path  = processed_dir / config['paths']['output_filename']

logger.info('Lendo: %s', parquet_path)

if not parquet_path.exists():
    raise FileNotFoundError(
        f"Arquivo Parquet não encontrado: {parquet_path}\n"
        "Execute ingestao.py e qualidade.py antes deste script."
    )

# %%
# Inspeciona o schema sem carregar os dados (leitura de metadados é barata)
schema = pq.read_schema(str(parquet_path))
logger.info('Schema original (%d colunas):', len(schema))
for field in schema:
    logger.info('  %-30s %s', field.name, field.type)

# %%
# Carrega o DataFrame completo
df = pq.read_table(str(parquet_path)).to_pandas()
logger.info('Shape original: %s', df.shape)

# %%
# Visão rápida antes das transformações
logger.info(df.head())

# Estatísticas descritivas e nulos — baseline antes do pré-processamento
logger.info('Valores ausentes por coluna:')
null_counts = df.isna().sum()
if null_counts.sum() == 0:
    logger.info('  Nenhum valor ausente encontrado — dataset limpo.')
else:
    for col, n_null in null_counts[null_counts > 0].items():
        logger.info('  %-25s %d (%.2f%%)', col, n_null, 100 * n_null / len(df))

# %%
logger.info(df.describe())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Remoção de Identificadores
#
# RowNumber, CustomerId, Surname não têm relação causal com churn.
# Removê-los garante que o modelo aprenda padrões de comportamento,
# não correlações espúrias com IDs específicos de clientes.
# EDA Seção 8.1: COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
# ─────────────────────────────────────────────────────────────────────────────

# %%
drop_cols = config.get('drop_columns', [])
logger.info('─' * 60)
logger.info('SEÇÃO 2: Remoção de Identificadores')
logger.info('Colunas a remover: %s', drop_cols)

# %%
# Instancia e aplica o dropper (stateless — fit é no-op)
dropper = ColumnDropper(columns=drop_cols, logger=logger)
df = dropper.fit_transform(df)

# %%
logger.info('Shape após remoção: %s', df.shape)
logger.info('Colunas restantes: %s', list(df.columns))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — Imputação de Valores Ausentes  [MOVIDA PARA O PIPELINE DE MODELAGEM]
#
# ⚠ AVISO MLOps — Data Leakage
# Imputação é STATEFUL: GroupMedianImputer aprende medianas/modas no fit()
# e aplica no transform(). Ajustar ANTES do split treino/holdout usa
# informação do conjunto de teste para preencher NaN no conjunto de treino
# → data leakage.
#
# Este dataset não possui valores ausentes (EDA Seção 3: 0 nulos em 180.000
# células). No entanto, em produção (dados reais do banco) esperam-se nulos
# — especialmente em Balance (clientes novos) e CreditScore (sem análise formal).
#
# SOLUÇÃO: GroupMedianImputer é aplicado DENTRO do Pipeline de modelagem
# (modelagem.py), APÓS o split treino/holdout.
# As estratégias estão configuradas em preprocessing.yaml → imputation.
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SEÇÃO 3: Imputação — delegada ao Pipeline de modelagem (ver modelagem.py)')
imp_specs = config.get('imputation', [])
logger.info('Especificações configuradas (referência): %d colunas', len(imp_specs))
for spec in imp_specs:
    logger.info('  %-20s → strategy=%s', spec['column'], spec['strategy'])

n_null_total = int(df.isna().sum().sum())
logger.info(
    'NaN total no dataset atual: %d — %s',
    n_null_total,
    'dataset limpo, imputação não necessária neste batch'
    if n_null_total == 0
    else 'NaN encontrados — serão tratados no pipeline de modelagem',
)

# %%
# Referência de uso — DataImputer será instanciado no pipeline de modelagem:
#
#   imp_cfg = config['imputation']
#   imputer = DataImputer(imputation_config=imp_cfg, logger=logger)
#   imputer.fit(X_train)           # aprende medianas/modas APENAS no treino
#   X_train = imputer.transform(X_train)
#   X_test  = imputer.transform(X_test)
#
# Estratégias configuradas (preprocessing.yaml → imputation):
imp_cfg = config.get('imputation', [])
for spec in imp_cfg:
    logger.info(
        '  %-20s → strategy=%-20s %s',
        spec['column'],
        spec['strategy'],
        f"(group={spec['group_column']})" if spec.get('group_column') else
        f"(fill={spec['fill_value']})"    if spec.get('fill_value') is not None else '',
    )

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Flags Binárias
#
# Dois padrões têm significado especial neste dataset:
#
# 1. Balance == 0 → HasZeroBalance
#    - 36.17% dos clientes mantêm conta com saldo zero — não é erro de dados,
#      é um comportamento distinto de uso bancário.
#    - Paradoxalmente, apresentam MENOR churn (clientes "ativos" sem saldo
#      mas que mantêm a conta ativa por outros motivos).
#    - A flag permite que o modelo aprenda esse padrão sem confundir
#      saldo zero com "dado ausente".
#
# 2. NumOfProducts >= 3 → HighRiskProducts
#    - 3 produtos: 82.7% de churn | 4 produtos: 100% de churn
#    - O ponto de inflexão é abrupto e altamente não-linear.
#    - Sem a flag, um modelo linear não conseguiria capturar esse limiar.
# ─────────────────────────────────────────────────────────────────────────────

# %%
flags_cfg = config.get('binary_flags', [])
logger.info('─' * 60)
logger.info('SEÇÃO 4: Flags Binárias')

# %%
# Instancia e aplica o transformador de flags (stateless — fit é no-op)
flag_transformer = BinaryFlagTransformer(flags=flags_cfg, logger=logger)
df = flag_transformer.transform(df)

# %%
# Inspeciona os resultados
for spec in flags_cfg:
    new_col = spec['new_column']
    logger.info(
        "'%s': %d linhas = 1 (%.2f%%)",
        new_col,
        int(df[new_col].sum()),
        100 * df[new_col].mean(),
    )

# %%
# Validação cruzada com o target — confirma poder preditivo das flags
logger.info('Taxa de churn por HasZeroBalance:')
logger.info(df.groupby('HasZeroBalance')['Exited'].mean().round(4))

logger.info('Taxa de churn por HighRiskProducts:')
logger.info(df.groupby('HighRiskProducts')['Exited'].mean().round(4))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — Features de Interação
#
# AgeInactivity: amplifica o sinal de desengajamento em clientes mais velhos.
#   Fórmula: Age × (1 − IsActiveMember)
#   Clientes inativos e mais velhos apresentam churn muito mais alto.
#   Exemplo: cliente de 50 anos inativo → AgeInactivity = 50
#            cliente de 50 anos ativo   → AgeInactivity = 0
#   A feature captura a combinação dos dois maiores preditores de churn.
#
# EngagementScore: score composto de engajamento do cliente (range: -1 a 3).
#   Fórmula: IsActiveMember + (NumOfProducts == 2) + HasCrCard − HasZeroBalance
#   Score 3: cliente ativo, com 2 produtos e cartão, com saldo → menor risco
#   Score -1: cliente inativo, sem 2 produtos, sem cartão, com saldo zero → maior risco
#
# ⚠ EngagementScore usa HasZeroBalance criado na Seção 4 — ordem importa.
# ─────────────────────────────────────────────────────────────────────────────

# %%
interact_cfg = config.get('interaction_features', [])
logger.info('─' * 60)
logger.info('SEÇÃO 5: Features de Interação')

# %%
interact_transformer = InteractionFeatureTransformer(
    features_config=interact_cfg, logger=logger
)
df = interact_transformer.transform(df)

# %%
# Estatísticas das novas features
new_interact_cols = [spec['name'] for spec in interact_cfg]
logger.info(df[new_interact_cols].describe())

# %%
# Correlação com o target — valida poder preditivo
logger.info('Correlação das features de interação com Exited:')
for col in new_interact_cols:
    if col in df.columns:
        corr = df[col].corr(df['Exited'])
        logger.info('  %-20s r = %.4f', col, corr)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 6 — Features de Razão
#
# BalanceSalaryRatio: razão entre saldo em conta e salário estimado.
#   Fórmula: Balance / (EstimatedSalary + 1)
#   Captura clientes que concentram patrimônio no banco (saldo alto
#   relativo ao salário) — exatamente os mais disputados pela concorrência.
#   EDA (Achado 7): clientes de maior saldo apresentam 25% mais churn.
#   +1 no denominador: proteção contra divisão por zero.
# ─────────────────────────────────────────────────────────────────────────────

# %%
ratio_cfg = config.get('ratio_features', [])
logger.info('─' * 60)
logger.info('SEÇÃO 6: Features de Razão')

# %%
ratio_transformer = RatioFeatureTransformer(ratios=ratio_cfg, logger=logger)
df = ratio_transformer.transform(df)

# %%
# Estatísticas da nova feature
new_ratio_cols = [spec['name'] for spec in ratio_cfg]
logger.info(df[new_ratio_cols].describe())

# %%
# Correlação com o target
logger.info('Correlação das features de razão com Exited:')
for col in new_ratio_cols:
    if col in df.columns:
        corr = df[col].corr(df['Exited'])
        logger.info('  %-20s r = %.4f', col, corr)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 7 — Faixas Etárias (AgeGroup)
#
# A relação entre idade e churn é fortemente não-linear:
#   < 30 anos  : churn moderado
#   30–40 anos : menor churn (clientes estabilizados, engajados)
#   40–50 anos : churn crescente (maior poder de decisão e mobilidade)
#   50–60 anos : pico de churn (maior patrimônio, alvo principal da concorrência)
#   60+ anos   : churn cai (menor mobilidade, relações mais consolidadas)
#
# EDA: idade média dos que saíram = 44.8 anos vs 37.4 dos que ficaram
#      Cohen's d = 0.747 — efeito médio-grande
#
# Encoding ordinal (0–4): preserva a ordem natural e é compacto.
# Modelos baseados em árvore (XGBoost, RF) trabalham bem com ordinais.
# ─────────────────────────────────────────────────────────────────────────────

# %%
age_cfg = config.get('age_group', {})
logger.info('─' * 60)
logger.info('SEÇÃO 7: Faixas Etárias (AgeGroup)')
logger.info('Bins  : %s', age_cfg.get('bins'))
logger.info('Labels: %s', age_cfg.get('labels'))

# %%
age_transformer = AgeBinTransformer(age_config=age_cfg, logger=logger)
df = age_transformer.transform(df)

# %%
# Distribuição por faixa etária
age_col = age_cfg.get('new_column', 'AgeGroup')
logger.info('Distribuição de AgeGroup (ordinal):')
logger.info(df[age_col].value_counts().sort_index())

# %%
# Taxa de churn por faixa etária — valida o padrão identificado na EDA
logger.info('Taxa de churn por AgeGroup:')
logger.info(df.groupby(age_col)['Exited'].agg(['mean', 'count']).round(4))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 8 — Encoding de Variáveis Categóricas
#
# Geography (3 categorias sem ordem) → One-Hot: geo_France, geo_Germany, geo_Spain
#   EDA: Germany tem churn 2× maior que France/Spain (Cramér's V=0.17, p<0.0001)
#   OHE mantém cada país como preditor independente — sem assumir ordem.
#
# Gender (2 categorias) → Binary: Gender_encoded (Male=0, Female=1)
#   EDA: mulheres com 25.1% de churn vs 16.5% dos homens (Cramér's V=0.11)
#   Encoding binário é compacto para variáveis com 2 categorias.
#
# Card Type (4 categorias) → One-Hot: card_DIAMOND, card_GOLD, card_PLATINUM, card_SILVER
#   EDA: χ² p=0.168, Cramér's V=0.02 — sem associação linear significativa.
#   Mantido: XGBoost/RF podem extrair interações via combinação de features.
# ─────────────────────────────────────────────────────────────────────────────

# %%
enc_cfg = config.get('categorical_encoding', [])
logger.info('─' * 60)
logger.info('SEÇÃO 8: Encoding de Variáveis Categóricas')
for spec in enc_cfg:
    logger.info('  %-12s → método=%s', spec['column'], spec['method'])

# %%
encoder = CategoricalEncoder(encoding_config=enc_cfg, logger=logger)
df = encoder.transform(df)

# %%
# Verificação das colunas OHE criadas para Geography
logger.info('Colunas OHE de Geography:')
geo_cols = [c for c in df.columns if c.startswith('geo_')]
logger.info('  %s', geo_cols)
logger.info(df[geo_cols].sum())

# %%
# Verificação do encoding binário de Gender
logger.info('Encoding binário de Gender:')
if 'Gender_encoded' in df.columns:
    logger.info(df['Gender_encoded'].value_counts())
    # Valida taxa de churn por gênero (deve replicar EDA: Female > Male)
    logger.info('Taxa de churn por Gender_encoded (0=Male, 1=Female):')
    logger.info(df.groupby('Gender_encoded')['Exited'].mean().round(4))

# %%
# Verificação das colunas OHE criadas para Card Type
logger.info('Colunas OHE de Card Type:')
card_cols = [c for c in df.columns if c.startswith('card_')]
logger.info('  %s', card_cols)
logger.info(df[card_cols].sum())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 9 — Seleção de Features
#
# Após as transformações, o DataFrame tem 30+ colunas.
# Precisamos selecionar o subconjunto recomendado pelo EDA Report:
#
#   ✓ Numéricas originais (sem identificadores)
#   ✓ Binárias originais (HasCrCard, IsActiveMember, Complain*)
#   ✓ Features derivadas (flags, interações, razão, faixa etária)
#   ✓ Encoding de Geography, Gender, Card Type
#   ✗ Identificadores (já removidos na Seção 2)
#   ✗ Categóricas originais (Geography, Gender, Card Type — substituídas)
#
# * Complain incluída com alerta de leakage — verificar antes de usar em prod.
#
# FeatureSelector emite WARNING (não exceção) para colunas ausentes,
# garantindo que o pipeline continue mesmo que alguma transformação
# anterior tenha sido pulada.
# ─────────────────────────────────────────────────────────────────────────────

# %%
sel_cfg = config.get('feature_selection', {})
features_to_keep = sel_cfg.get('features_to_keep', [])
logger.info('─' * 60)
logger.info('SEÇÃO 9: Seleção de Features')
logger.info('Shape antes da seleção: %s', df.shape)
logger.info('Features solicitadas: %d', len(features_to_keep))

# %%
selector = FeatureSelector(features_to_keep=features_to_keep, logger=logger)
df = selector.fit_transform(df)

# %%
logger.info('Shape após seleção: %s', df.shape)
logger.info('Colunas selecionadas:')
for i, col in enumerate(df.columns, 1):
    logger.info('  %2d. %s', i, col)

# %%
logger.info(df.head())

# %%
# Estatísticas finais do dataset de features
logger.info('Estatísticas finais — dataset de features:')
logger.info(df.describe())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 10 — Persistência do Resultado
#
# Salva o dataset de features em Parquet (formato colunar, comprimido).
# Este arquivo é a ENTRADA da etapa de modelagem (modelagem.py).
#
# O arquivo pode ser:
#   • Versionado no repositório (rastreabilidade de dados entre execuções)
#   • Carregado diretamente pelo script de treinamento
#   • Comparado entre execuções para detectar drift de features ao longo do tempo
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Cria o diretório de saída se não existir
output_dir.mkdir(parents=True, exist_ok=True)
logger.info('─' * 60)
logger.info('SEÇÃO 10: Persistência')
logger.info('Diretório de saída: %s', output_dir)

# %%
# Salva em Parquet
df.to_parquet(str(output_path), compression=compression, index=False)
size_mb = output_path.stat().st_size / (1024 ** 2)
logger.info('Arquivo salvo: %s (%.2f MB)', output_path, size_mb)

# %%
# Validação pós-escrita: lê o schema sem carregar os dados
schema_out = pq.read_schema(str(output_path))
logger.info('Schema de saída (%d colunas):', len(schema_out))
for field in schema_out:
    logger.info('  %-35s %s', field.name, field.type)

# %%
# Lê uma amostra para confirmação visual
df_check = pd.read_parquet(str(output_path))
logger.info('Verificação pós-leitura — shape: %s', df_check.shape)
logger.info(df_check.head())
