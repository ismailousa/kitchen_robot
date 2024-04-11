from kitchen_robot.constants import *
from kitchen_robot.entity.config_entity import (
    DataIngestionConfig,
    ModelEvaluationConfig,
    ModelInferenceConfig,
    ModelTrainingConfig,
    PrepareDatasetConfig,
    PrepareModelConfig,
)
from kitchen_robot.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_dataset_config(self) -> PrepareDatasetConfig:
        config = self.config.prepare_dataset
        create_directories([config.root_dir])

        prepare_dataset_config = PrepareDatasetConfig(
            root_dir=config.root_dir,
            environment_data=config.environment_data,
            historical_data=config.historical_data,
            feedback_data=config.feedback_data,
            train_eval_data=config.train_eval_data,
            test_data=config.test_data,
            train_eval_dataset=config.train_eval_dataset,
            test_dataset=config.test_dataset,
        )

        return prepare_dataset_config

    def get_prepare_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        create_directories([config.root_dir])

        prepare_model_config = PrepareModelConfig(
            root_dir=config.root_dir,
            feature_size=config.feature_size,
            target_size=config.target_size,
            base_model_path=config.base_model_path,
        )

        return prepare_model_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            feature_size=self.config.prepare_model.feature_size,
            target_size=self.config.prepare_model.target_size,
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
            train_eval_dataset=config.train_eval_dataset,
            base_model_path=config.base_model_path,
            updated_model_path=config.updated_model_path,
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            feature_size=self.config.prepare_model.feature_size,
            target_size=self.config.prepare_model.target_size,
            batch_size=self.params.batch_size,
            updated_model_path=config.updated_model_path,
            test_dataset=config.test_dataset,
            model_performance_report=config.model_performance_report,
        )

        return model_evaluation_config

    def get_model_inference_config(self) -> ModelInferenceConfig:
        config = self.config.prepare_model
        model_inference_config = ModelInferenceConfig(
            feature_size=config.feature_size,
            target_size=config.target_size,
            model_path=self.config.model_evaluation.updated_model_path,
            features=self.params.features,
        )

        return model_inference_config
