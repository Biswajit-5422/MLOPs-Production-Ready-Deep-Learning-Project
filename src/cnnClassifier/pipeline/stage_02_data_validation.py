from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_validation import DataValidation
from cnnClassifier import logger


STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        report = data_validation.validate()

        if not report["validation_status"]:
            raise ValueError(
                "Data validation failed - see artifacts/data_validation/validation_report.json"
            )


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
