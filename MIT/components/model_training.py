import os
import sys
import joblib

from MIT.logger import logging
from MIT.exceptions import CustomException
from MIT.constants import *

from MIT.entity.config_entity import ModelTrainerConfig
from MIT.entity.artifact_entity import DataIngestionArtifacts, ModelTrainerArtifacts
from MIT.entity.dataset import MitDataset
from MIT.entity.custom_model import ResNet_152
from MIT.utils import DeviceDataLoader
from MIT.utils import get_default_device, to_device, fit

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

DEVICE = get_default_device()

class ModelTrainer:
    """
    Model Trainer
    """
    def __init__(
        self, model_trainer_config: ModelTrainerConfig, data_ingestion_artifact:DataIngestionArtifacts
        ) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def get_tranformer_object(self):
        try:
            logging.info("Enter the get_transformer_object")
            transformed = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                    ])
            os.makedirs(self.model_trainer_config.model_training_artifact_dir)

            logging.info("Saving transformer oblect for prediction")
            joblib.dump(transformed, self.model_trainer_config.transformer_object_path)

            logging.info("Exit the get_transformer_object")

            return transformed
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_dataset(self):
        try:
            logging.info("Enter the load_dataset method of model trainer")
            transform = self.get_tranformer_object()
            self.num_classes = os.listdir(self.data_ingestion_artifact.image_folder_name)

            with open(self.model_trainer_config.lable_file_path, 'w') as f:
                f.write(' '.join(self.num_classes))

            logging.info("creating class to index dictionary")
            class_to_index = {i:j for j, i in enumerate(self.num_classes)}
            train_dataset = MitDataset(annotation_file_path=self.data_ingestion_artifact.train_file_path, image_dir=self.data_ingestion_artifact.image_folder_name, class_to_index=class_to_index, transformation=transform)

            logging.info("Exiting the load_dataset method of model trainer")

            return train_dataset
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_loader(self):
        try:

            logging.info("Enter get_data_loader method of model trainer")
            dataset = self.load_dataset()

            logging.info("Shuffle and split the data")
            training, valid = random_split(dataset, [SPLIT, len(dataset) - SPLIT])

            training_dl = DataLoader(training, BATCH_SIZE, shuffle=True, num_workers=2)
            valid_dl = DataLoader(valid, BATCH_SIZE, shuffle=True, num_workers=2)

            logging.info("Exit get_data_loader method of model trainer")
            return training_dl, valid_dl

        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self):
        try:
            logging.info("getting the pre-trained resnet model")
            model = ResNet_152(self.num_classes)

            logging.info("Exceting the get_model method of model training")
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_to_GPU(self, training_dl, valid_dl, model):
        try:
            logging.info('loading model to GPU')
            model = to_device(model, DEVICE)

            logging.info('loading data to GPU')
            training_dl = DeviceDataLoader(training_dl, DEVICE)
            valid_dl = DeviceDataLoader(valid_dl, DEVICE)

            logging.info("loading data and model to GPU is done")
            return training_dl, valid_dl, model
        except Exception as e:
            raise CustomException(e,sys)

    def train(self, model, train_dl, valid_dl):
        try:
            logging.info("Model training started")
            fitted_model , result = fit(epochs=EPOCHS, lr=LEARNING_RATE, model=model, train_loader=train_dl, val_loader=valid_dl, opt_func=OPTIMIZER)
            logging.info("Model training done")
            return fitted_model, result
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self):
        try:
            logging.info("initiating model training")
            logging.info("loading train and valid data loader")
            train_dl, valid_dl = self.get_data_loader()

            logging.info("load the model")
            model = self.get_model()
            torch.cuda.empty_cache()

            logging.info("loading requirements to GPU")
            training_dl, valid_dl, model = self.load_to_GPU(train_dl, valid_dl, model)

            logging.info("Train the model")
            fitted_model, result = self.train(model=model, train_dl=training_dl, valid_dl=valid_dl)

            logging.info(f"savind the model at {self.model_trainer_config.model_path}")
            torch.save(model.state_dict(), self.model_trainer_config.model_path)

            model_trainer_artifact = ModelTrainerArtifacts(
                model_path=self.model_trainer_config.model_path,
                result=result,
                transformer_path=self.model_trainer_config.transformer_object_path
            )
            logging.info(f"modler trainer artifact {model_trainer_artifact}")
            logging.info("model training completed")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)