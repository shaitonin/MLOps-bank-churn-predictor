import sys
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# %%
# definitions
ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
QUALITY_CONFIG = CONFIG_DIR / 'quality.yaml'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

# add root and config to system path
for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.quality_checks import run_quality_checks, save_quality_report

# Load configuration files
pipeline_config = load_yaml(PIPELINE_CONFIG)
data_config = load_yaml(DATA_CONFIG)
quality_config = load_yaml(QUALITY_CONFIG)

# Merge into a single dictionary
config = {
    **pipeline_config,
    **data_config,
    **quality_config
}

# Create logger
log_cfg = config.get('logging')
logger = get_logger(
    name='quality',
    logging_config=log_cfg
)

#%%
# get input data paths
processed_dir = ROOT_DIR / config.get('paths').get('processed_data_dir')
parquet_path = processed_dir / config.get('paths').get('output_filename')

logger.info('Parquet path: %s', parquet_path)

if not parquet_path.exists():
    logger.error('Parquet file not found: %s', parquet_path)
    raise FileNotFoundError(f'Parquet file not found: {parquet_path}')

# schema inspection
schema = pq.read_schema(parquet_path)
logger.info('Schema (%d columns):', len(schema))
for i, field in enumerate(schema):
    logger.info('  %d: %s (%s)', i + 1, field.name, field.type)

# load data into a Pandas DataFrame
df = pd.read_parquet(parquet_path)
logger.info('Data loaded: %d rows, %d columns', df.shape[0], df.shape[1])


#%%
# run quality validation — using YAML as single source of truth
summary = run_quality_checks(
    df=df,
    config=config,
    logging_config=log_cfg
)
logger.info(summary)
# %%
# persist quality report
output_dir = ROOT_DIR / config.get('output_dir', 'outputs/quality')

# save report as JSON
report_path = save_quality_report(
    summary=summary,
    output_dir=output_dir,
    logging_config=log_cfg
)

fail_on_err = config.get('fail_pipeline_on_error', True)

if summary['success']:
    logger.info('=== Quality PASSED — data ready for the next step ===')
else:
    logger.warning(
        '=== Quality with FAILURES: %d/%d checks failed ===',
        summary['failed'],
        summary['total'],
    )
    logger.warning(
        'fail_pipeline_on_error=%s — %s',
        fail_on_err,
        'RuntimeError will be raised' if fail_on_err else 'continuing anyway (exploration mode)',
    )

# %%
# Summary
logger.info('─' * 60)
logger.info('Input    : %s', parquet_path)
logger.info('Report   : %s', report_path)
logger.info('Checks   : %d total | %d passed | %d failed',
            summary['total'], summary['passed'], summary['failed'])
logger.info('Status   : %s', 'PASSED' if summary['success'] else 'FAILED')
