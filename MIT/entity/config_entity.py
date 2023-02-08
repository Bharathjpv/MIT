import os, sys
from MIT.constants import *

from dataclasses import dataclass

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = SOURCE_DIR_NAME
    artifact_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_artifact_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACTS_DIR)
    download_dir: str = os.path.join(ROOT, DATA_DIR_NAME)
    raw_data_dir: str = os.path.join(download_dir, RAW_DATA_DIR_NAME)
    unzip_data_dir: str = os.path.join(download_dir, UNZIPPED_FOLDER_NAME)