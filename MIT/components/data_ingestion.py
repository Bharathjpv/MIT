import os
import sys
import zipfile
import shutil

from MIT.logger import logging
from MIT.exceptions import CustomException
from MIT.constants import *

from MIT.entity.config_entity import DataIngestionConfig
from MIT.entity.artifact_entity import DataIngestionArtifacts
from MIT.cloud_storage.s3_operations import S3Sync

class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig)-> None:

        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
            self.data_ingestion_artifact = self.data_ingestion_config.data_ingestion_artifact_dir
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_from_cloud(self) -> None:
        try:
            logging.info("Initiating data download from s3 bucket...")
            self.raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.isdir(self.raw_data_dir):
                logging.info(f"Data is already available in {self.raw_data_dir}. Hence skipping this step")

            else:
                os.makedirs(self.raw_data_dir, exist_ok=True)
                self.s3_sync.sync_folder_from_s3(
                        folder=self.raw_data_dir, aws_bucket_url=S3_BUCKET_DATA_URI)
                logging.info(
                        f"Data is downloaded from s3 bucket to Download directory: {self.raw_data_dir}.")

        except Exception as e:
            raise CustomException(e, sys)

    def unzip_data(self) -> None:

        try:
            logging.info("Unzipping the downloaded zip file from download directory...")
            raw_zip_path = os.path.join(self.raw_data_dir, ZIP_FILE_NAME)

            if os.path.isdir(self.data_ingestion_config.unzip_data_dir):
                logging.info(f'Unzipped folder already exist in {self.data_ingestion_config.unzip_data_dir}, Hence skipping unzipping')
            else:
                os.makedirs(self.data_ingestion_config.unzip_data_dir)

                with zipfile.ZipFile(raw_zip_path, "r") as f:
                    f.extractall(self.data_ingestion_config.unzip_data_dir)

                logging.info(f"Unzipping of data completd and extracted at {self.data_ingestion_config.unzip_data_dir}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("Initiating the data ingestion component...")
            self.get_data_from_cloud()
            self.unzip_data()

            logging.info("Createing dataingestion artifact folder")

            os.makedirs(self.data_ingestion_artifact)

            ext_train_file_path = os.path.join(self.data_ingestion_config.unzip_data_dir, TRAIN_FILE_NAME)
            ext_test_file_path = os.path.join(self.data_ingestion_config.unzip_data_dir, TEST_FILE_NAME)

            train_file_path = os.path.join(self.data_ingestion_artifact , TRAIN_FILE_NAME)
            test_file_path = os.path.join(self.data_ingestion_artifact, TEST_FILE_NAME)

            logging.info(f"copying the train annotatin file to {train_file_path}")
            shutil.copy(ext_train_file_path, train_file_path)
            logging.info(f"copying the test annotatin file to {test_file_path}")
            shutil.copy(ext_test_file_path, test_file_path)

            image_folder = os.path.join(self.data_ingestion_config.unzip_data_dir, FOLDER_NAME, IMAGE_FOLDER_NAME)

            data_ingestion_artifact = DataIngestionArtifacts(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                image_folder_name=image_folder
            )
            logging.info(f"Data Ingestion Artifact {data_ingestion_artifact}")
            logging.info('Data ingestion is completed Successfully.')

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)