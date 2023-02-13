import os
import sys
import torch
import joblib
from PIL import Image

from MIT.logger import logging
from MIT.exceptions import CustomException

from MIT.entity.config_entity import PredictionPipelineConfig
from MIT.cloud_storage.s3_operations import S3Sync
from MIT.entity.custom_model import ResNet_152
from MIT.utils import to_device, predict_image, get_default_device

DEVICE = get_default_device()

class SinglePrediction:
    def __init__(self):
        try: 
            self.s3_sync = S3Sync()
            self.prediction_config = PredictionPipelineConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def _get_model_in_production(self):
        try:
            s3_model_path = self.prediction_config.s3_model_path
            model_download_path = self.prediction_config.prediction_artifact_dir
            os.makedirs(model_download_path, exist_ok=True)
            if len(os.listdir(model_download_path)) == 0:
                self.s3_sync.sync_folder_from_s3(folder=model_download_path, aws_bucket_url=s3_model_path)
        except Exception as e:
            raise CustomException(e, sys)

    
    def get_model(self):
        try:
            self._get_model_in_production()

            prediction_model_path = self.prediction_config.model_download_path

            with open(self.prediction_config.classes_file_path, 'r') as cls:
                lines = cls.readlines()
                self.num_classes = lines[0].split()
            
            prediction_model = to_device(ResNet_152(self.num_classes), DEVICE)

            prediction_model.load_state_dict(torch.load(prediction_model_path, map_location=torch.device('cpu')))

            # for gpu devices
            # prediction_model.load_state_dict(torch.load(prediction_model_path))

            prediction_model.eval()

            return prediction_model
        except Exception as e:
            raise CustomException(e, sys)

    def _get_image_tensor(self, image_path):
        try:
            img = Image.open(image_path)
            transforms = joblib.load(self.prediction_config.transforms_path)
            img_tensor = transforms(img)

            return img_tensor
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, image_path):
        try:
            model = self.get_model()

            image = self._get_image_tensor(image_path)
            

            result = predict_image(image, model, DEVICE, self.num_classes)

            return result
        except Exception as e:
            raise CustomException(e, sys)