import os, sys
from MIT.constants import *
from from_root import from_root

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

@dataclass
class ModelTrainerConfig:
    model_training_artifact_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINIG_ARTIFACT_DIR)
    model_path: str = os.path.join(model_training_artifact_dir, MODEL_NAME)
    transformer_object_path: str = os.path.join(model_training_artifact_dir, TRAINER_OBJECT_NAME)
    lable_file_path: str = os.path.join(model_training_artifact_dir, LABLE_FILE_NAME)

@dataclass
class ModelEvaluationConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    model_evaluation_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR)
    best_model_dir: str = os.path.join(model_evaluation_artifacts_dir, S3_MODEL_DIR_NAME)
    best_model: str = os.path.join(best_model_dir, S3_MODEL_NAME)

@dataclass
class PredictionPipelineConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    prediction_artifact_dir = os.path.join(from_root(), PREDICTION_PIPELINE_DIR_NAME)
    model_download_path = os.path.join(prediction_artifact_dir, MODEL_NAME)
    transforms_path = os.path.join(prediction_artifact_dir, TRAINER_OBJECT_NAME)
    classes_file_path = os.path.join(prediction_artifact_dir, LABLE_FILE_NAME)