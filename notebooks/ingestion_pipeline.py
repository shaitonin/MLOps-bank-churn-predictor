# %%
# Environment setup
import sys
from pathlib import Path
# %%
# definitions
ROOT_DIR = Path(__file__).resolve().parent.parent # creates a path to the main project folder (MLOps project); if I only want this file path, remove .parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

# add root and config to system path
for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config_loader import load_yaml  # importing the function from config_loader.py to read the configuration files
from src.utils.logger import get_logger
from src.downloader import check_kaggle_credentials, list_remote_files, download_dataset
from src.ingestion import ingest_csv_to_parquet
# %%
# read configuration files
data_cfg = load_yaml(DATA_CONFIG)
pipeline_cfg = load_yaml(PIPELINE_CONFIG)

# %%
# get logging configuration
log_cfg = pipeline_cfg.get('logging')

# create logger
logger = get_logger(
    name='ingestion',
    logging_config=log_cfg
)

# %%
# checking environment variable
if check_kaggle_credentials(secrets_path=SECRETS_PATH):
    logger.info('Kaggle credentials are set!')
else:
    logger.error('Kaggle credentials not set')
# %%
### Data discovery
# pull data settings from data.yaml
dataset = data_cfg.get('kaggle').get('dataset')
file_pattern = data_cfg.get('kaggle').get('file_pattern', "*.csv")
expected_files = data_cfg.get('kaggle').get('expected_files')

# log the discovered information
logger.info('Dataset  : %s', dataset)
logger.info('Pattern  : %s', file_pattern)
logger.info('Files    : %s', expected_files or '(auto-discovery)')

if not expected_files:
    expected_files = list_remote_files(
        dataset=dataset,
        file_pattern=file_pattern,
        logging_config=log_cfg
    )
    logger.info('Files found: %s', expected_files)
    #%%
# define raw data destination directory, download data, and create folder if it does not exist
raw_dir = ROOT_DIR / pipeline_cfg.get('paths').get('raw_data_dir')
raw_dir.mkdir(parents=True, exist_ok=True)

skip_download = pipeline_cfg.get('execution').get('skip_download_if_exists')
force_download = pipeline_cfg.get('execution').get('force_redownload')

# download data - Extract phase of an ETL / ELT
downloaded = download_dataset(
    dataset=dataset,
    expected_files=expected_files, # expected files list is now filled after the discovery step
    destination_dir=raw_dir,
    skip_if_exists=skip_download,
    force=force_download,
    logging_config=log_cfg
)
logger.info('Files ready: %d', len(downloaded))

# check raw directory contents
for f in sorted(raw_dir.glob('*.csv')):
    logger.info('  %s (%.1f KB)', f.name, f.stat().st_size / 1024)

# %%
# define Parquet output path
processed_dir = ROOT_DIR / pipeline_cfg['paths']['processed_data_dir']
output_path = processed_dir / pipeline_cfg['paths']['output_filename']

logger.info('Output: %s', output_path)

# get processing configurations
compression = data_cfg.get('ingest').get('compression', 'snappy')
chunk_size = data_cfg.get('ingest').get('chunk_size_rows', 5_000)
validate = data_cfg.get('ingest').get('validate_schema')
required_cols = data_cfg.get('schema').get('required_columns')
skip_ingest = pipeline_cfg.get('execution').get('skip_ingest_if_exists')
force_ingest = pipeline_cfg.get('execution').get('force_ingest')

result_path = ingest_csv_to_parquet(
    raw_dir=raw_dir,
    output_path=output_path,
    compression=compression,
    chunk_size_rows=chunk_size,
    validate_schema=validate,
    skip_if_exists=skip_ingest,
    force=force_ingest,
    logging_config=log_cfg
)