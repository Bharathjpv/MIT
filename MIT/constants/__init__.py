import os
import torch
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ROOT = os.getcwd()

ARTIFACTS_DIR: str = "artifacts"
SOURCE_DIR_NAME: str = 'MIT'

S3_BUCKET_DATA_URI = "s3://mit-indoor-classification/data/"

# constants related to data ingestion
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion_artifacts"
RAW_DATA_DIR_NAME: str = "raw_data"
ZIP_FILE_NAME: str = "MITData.zip"
UNZIPPED_FOLDER_NAME: str = "data"
DATA_DIR_NAME: str = "data"

FOLDER_NAME: str = "indoorCVPR_09"
IMAGE_FOLDER_NAME: str = "Images"
TRAIN_FILE_NAME: str = "TrainImages.txt"
TEST_FILE_NAME: str = "TestImages.txt"

# constants realted to model training
MODEL_TRAINIG_ARTIFACT_DIR: str = "model_training_artifacts"
MODEL_NAME: str = "model.pt"
TRAINER_OBJECT_NAME: str = "transform.pkl"
LABLE_FILE_NAME: str = 'lable.txt'

SPLIT: int = 4500
BATCH_SIZE: int = 8
EPOCHS: int = 1
LEARNING_RATE: int = 6e-5
OPTIMIZER = torch.optim.RMSprop

# constants realted to modle evaluation
S3_BUCKET_MODEL_URI: str = "s3://mit-indoor-classification/model/"
MODEL_EVALUATION_DIR: str = "model_evaluation_artifacts"
S3_MODEL_DIR_NAME: str = "s3_model"
S3_MODEL_NAME: str = "model.pt"

BASE_LOSS: int = 4.00

PREDICTION_PIPELINE_DIR_NAME: str = "prediction_artifacts"