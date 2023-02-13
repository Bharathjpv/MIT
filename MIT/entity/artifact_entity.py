from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str
    image_folder_name: str

@dataclass
class ModelTrainerArtifacts:
    model_path: str
    transformer_path: str
    result: dict


@dataclass
class ModelEvaluationArtifacts:
    s3_model_loss: float
    is_model_accepted: bool
    trained_model_path: str
    s3_model_path: str

@dataclass
class ModelPusherArtifacts:
    response: dict