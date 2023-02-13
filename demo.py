
# from MIT.entity.config_entity import DataIngestionConfig, ModelTrainerConfig, ModelEvaluationConfig
# from MIT.components.data_ingestion import DataIngestion
# from MIT.components.model_training import ModelTrainer
# from MIT.components.model_evaluation import ModelEvaluation
# from MIT.components.model_pusher import ModelPusher

# di_ins = DataIngestion(DataIngestionConfig)

# di_art = di_ins.initiate_data_ingestion()

# mt_ins = ModelTrainer(ModelTrainerConfig, di_art)

# mt_art = mt_ins.initiate_model_trainer()

# me_ins = ModelEvaluation(ModelEvaluationConfig, di_art, mt_art)

# me_art = me_ins.initiate_model_evaluation()

# mp_ins = ModelPusher(me_art)

# mp_art = mp_ins.initiate_model_pusher()

# from MIT.pipeline.training_pipeline import TrainingPipeline

# tr_pipeline = TrainingPipeline()

# tr_pipeline.run_pipeline()

from MIT.pipeline.prediction_pipeline import SinglePrediction

pred = SinglePrediction()

result = pred.predict("data/data/indoorCVPR_09/Images/artstudio/art_painting_studio_01_13_altavista.jpg")

print(result)