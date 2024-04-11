from kitchen_robot.config.configuration import ConfigurationManager
from kitchen_robot.components.model_evaluation import ModelEvaluation
from kitchen_robot import logger

STAGE_NAME = "Model evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(config=evaluation_config)
        evaluation.load_model()
        evaluation.evaluate()


if __name__ == "__main__":
    try:
        logger.info("*********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
