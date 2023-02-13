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