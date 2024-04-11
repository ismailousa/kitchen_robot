from kitchen_robot.config.configuration import ConfigurationManager
from kitchen_robot.components.prepare_model import PrepareModel
from kitchen_robot import logger

STAGE_NAME = "Prepare base model"


class PrepareModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.get_prepare_model_config()
        prepare_model = PrepareModel(prepare_model_config)
        prepare_model.generate_base_model()


if __name__ == "__main__":
    try:
        logger.info("*********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
