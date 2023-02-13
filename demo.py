
from MIT.entity.config_entity import DataIngestionConfig, ModelTrainerConfig

from MIT.components.data_ingestion import DataIngestion
from MIT.components.model_training import ModelTrainer


di_ins = DataIngestion(DataIngestionConfig)

di_art = di_ins.initiate_data_ingestion()

mt_ins = ModelTrainer(ModelTrainerConfig, di_art)

mt_ins.initiate_model_trainer()