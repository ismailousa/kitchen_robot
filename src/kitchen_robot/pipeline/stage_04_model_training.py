from kitchen_robot.config.configuration import ConfigurationManager
from kitchen_robot.components.model_training import ModelTraining
from kitchen_robot import logger

STAGE_NAME = "Model training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_model_training_config()
        training = ModelTraining(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == "__main__":
    try:
        logger.info("*********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
