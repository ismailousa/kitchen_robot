from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareDatasetConfig:
    root_dir: Path
    environment_data: Path
    historical_data: Path
    feedback_data: Path
    train_eval_data: Path
    test_data: Path
    train_eval_dataset: Path
    test_dataset: Path


@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    feature_size: int
    target_size: int
    base_model_path: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    feature_size: int
    target_size: int
    batch_size: int
    epochs: int
    train_eval_dataset: Path
    base_model_path: Path
    updated_model_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    feature_size: int
    target_size: int
    batch_size: int
    updated_model_path: Path
    test_dataset: Path
    model_performance_report: Path


@dataclass(frozen=True)
class ModelInferenceConfig:
    feature_size: int
    target_size: int
    model_path: Path
    features: dict
