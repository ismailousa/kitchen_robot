from kitchen_robot.config.configuration import ConfigurationManager
from kitchen_robot.components.model_inference import ModelInference
from kitchen_robot import logger

STAGE_NAME = "Model inference"


class ModelInferencePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        inference_config = config.get_model_inference_config()
        inference = ModelInference(config=inference_config)
        inference.load_model()
        return inference.infer()


if __name__ == "__main__":
    try:
        logger.info("*********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        logger.info("Inference not running")
        # obj = ModelInferencePipeline()
        # prediction = obj.main()
        # logger.info(f">>>>>> prediction : {prediction} <<<<<<")
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
