from kitchen_robot import logger
from kitchen_robot.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from kitchen_robot.pipeline.stage_02_prepare_dataset import PrepareDatasetPipeline
from kitchen_robot.pipeline.stage_03_prepare_model import PrepareModelPipeline
from kitchen_robot.pipeline.stage_04_model_training import ModelTrainingPipeline
from kitchen_robot.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info("*********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare dataset"

try:
    logger.info("*********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareDatasetPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"
try:
    logger.info("*********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareModelPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME =   "Model Training"
try:
    logger.info("*********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME =   "Model Evaluation"
try:
    logger.info("*********************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
