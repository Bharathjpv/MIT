import os
import sys
import joblib
import numpy as np

from MIT.logger import logging
from MIT.exceptions import CustomException
from MIT.constants import *

from MIT.cloud_storage.s3_operations import S3Sync
from MIT.entity.config_entity import ModelEvaluationConfig
from MIT.entity.artifact_entity import ModelTrainerArtifacts, DataIngestionArtifacts, ModelEvaluationArtifacts
from MIT.entity.dataset import MitDataset
from MIT.entity.custom_model import ResNet_152
from MIT.utils import DeviceDataLoader
from MIT.utils import to_device, get_default_device, evaluate

from torch.utils.data import DataLoader

DEVICE = get_default_device()


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifacts=DataIngestionArtifacts, model_trainer_artifacts = ModelTrainerArtifacts, ) -> None:
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts

    def get_best_model_path(self):
        try:
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            os.makedirs(os.path.dirname(best_model_dir), exist_ok=True)
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(
                folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pt"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info(
                        "Model is not available in best_model_directory")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys)

    def get_test_data_loader(self):
        try:
            logging.info("Enter the load_dataset method of model evaluation")
            self.num_classes = os.listdir(self.data_ingestion_artifacts.image_folder_name)
            transform = joblib.load(self.model_trainer_artifacts.transformer_path)

            logging.info("creating class to index dictionary")
            class_to_index = {i:j for j, i in enumerate(self.num_classes)}
            test_dataset = MitDataset(annotation_file_path=self.data_ingestion_artifacts.test_file_path, image_dir=self.data_ingestion_artifacts.image_folder_name, class_to_index=class_to_index, transformation=transform)

            test_dl = DataLoader(test_dataset, BATCH_SIZE, num_workers=2)
            return test_dl

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self):
        try:
            test_dl = self.get_test_data_loader()
            best_model_path = self.get_best_model_path()
            if best_model_path is not None:
                # load back the model
                model = ResNet_152(self.num_classes)
                logging.info("loand production model to gpu")
                model = to_device(model, DEVICE)
                model.load_state_dict(torch.load(self.model_evaluation_config.best_model))
                model.eval()
                logging.info(f"load the data to {DEVICE}")
                test_dl = DeviceDataLoader(test_dl, DEVICE)

                logging.info("evaluate production model on test data")
                result = evaluate(model=model, val_loader=test_dl)

                s3_model_loss = result["validation_step_loss"]
                
            else:
                logging.info(
                    "Model is not found on production server, So couldn't evaluate")
                s3_model_loss = None
            return s3_model_loss
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        try:
            s3_model_loss = self.evaluate_model()
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss
            trained_model_loss = self.model_trainer_artifacts.result["validation_step_loss"]
            evaluation_response = tmp_best_model_loss > trained_model_loss and trained_model_loss < BASE_LOSS
            model_evaluation_artifacts = ModelEvaluationArtifacts(
                s3_model_loss=tmp_best_model_loss,
                is_model_accepted=evaluation_response,
                trained_model_path=os.path.dirname(
                self.model_trainer_artifacts.model_path),
                s3_model_path=self.model_evaluation_config.s3_model_path
            )
            logging.info(f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)