import os
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