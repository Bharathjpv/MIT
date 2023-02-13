import os
import sys

from MIT.logger import logging
from MIT.exceptions import CustomException

from MIT.entity.config_entity import DataIngestionConfig, ModelTrainerConfig, ModelEvaluationConfig
from MIT.entity.artifact_entity import DataIngestionArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts, ModelPusherArtifacts

from MIT.components.data_ingestion import DataIngestion
from MIT.components.model_training import ModelTrainer
from MIT.components.model_evaluation import ModelEvaluation
from MIT.components.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try: 
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion step completed successfully in train pipeline")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_training(self, data_ingestion_artifacts: DataIngestionArtifacts) -> ModelTrainerArtifacts:
        logging.info("Starting model training in training pipeline")
        try: 
            logging.info("Instantiating train and validation dataset from custom dataset class...")
            model_trainer = ModelTrainer(self.model_trainer_config, data_ingestion_artifacts)
                                        
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Model trainer step completed successfully in train pipeline")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_evaluation(self,data_ingestion_artifacts: DataIngestionArtifacts,model_trainer_artifacts: ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Starting model evaluation in training pipeline")
        try: 
            model_evaluation = ModelEvaluation(self.model_evaluation_config, data_ingestion_artifacts, model_trainer_artifacts)
            logging.info("Evaluating current trained model")
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation step completed successfully in train pipeline")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        logging.info("Starting model pusher in training pipeline")
        try: 
            model_pusher = ModelPusher(model_evaluation_artifacts=model_evaluation_artifacts)
            logging.info("If model is accepted in model evaluation. Pushing the model into production storage")
            model_pusher_artifacts = model_pusher.initiate_model_pusher()
            logging.info("Model pusher step completed successfully in train pipeline")
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self) -> None:
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            model_trainer_artifacts = self.start_model_training(data_ingestion_artifacts=data_ingestion_artifacts)

            model_evaluation_artifacts = self.start_model_evaluation(data_ingestion_artifacts, model_trainer_artifacts)
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifacts=model_evaluation_artifacts)

            logging.info("<<<< Training pipeline completed >>>>")
        except Exception as e:
            raise CustomException(e, sys)