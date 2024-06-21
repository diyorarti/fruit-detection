ARTIFACTS_DIR: str = "artifacts"

"""
Data ingestion related constants start with DATA_INGESTION var name
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1rA0zZokY4sO8_DzVUC9N4rCOu-2SMePO/view?usp=sharing"

"""
Validation related constants start with DATA_VALIDATION var name
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_STATUS_FILE: str = 'status.txt'
DATA_VALIDATION_ALL_REQUIRED_FILES = ["Train File", "Valid File"]


""""
Model Trainer related constants start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov5s.pt"
MODEL_TRAINER_NO_EPOCHS: int = 1
MODEL_TRAINER_BATCH_SIZE: int = 16