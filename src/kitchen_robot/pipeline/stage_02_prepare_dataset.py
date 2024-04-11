from kitchen_robot.config.configuration import ConfigurationManager
from kitchen_robot.components.prepare_dataset import PrepareDataset
from kitchen_robot import logger

STAGE_NAME = "Prepare dataset"


class PrepareDatasetPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_dataset_config = config.get_prepare_dataset_config()
        prepare_dataset = PrepareDataset(prepare_dataset_config)
        prepare_dataset.get_features()
        prepare_dataset.generate_dataset()


if __name__ == "__main__":
    try:
        logger.info("*********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareDatasetPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
