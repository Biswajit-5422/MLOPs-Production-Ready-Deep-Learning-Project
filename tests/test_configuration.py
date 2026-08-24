from pathlib import Path

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)


def test_get_data_ingestion_config_matches_config_yaml():
    config = ConfigurationManager().get_data_ingestion_config()

    assert isinstance(config, DataIngestionConfig)
    assert config.root_dir == "artifacts/data_ingestion"
    assert config.local_data_file == "artifacts/data_ingestion/data.zip"


def test_get_prepare_base_model_config_matches_params_yaml():
    config = ConfigurationManager().get_prepare_base_model_config()

    assert isinstance(config, PrepareBaseModelConfig)
    assert config.params_image_size == [224, 224, 3]
    assert config.params_classes == 2
    assert config.params_weights == "imagenet"
    assert config.params_include_top is False


def test_get_training_config_derives_training_data_path():
    config = ConfigurationManager().get_training_config()

    assert isinstance(config, TrainingConfig)
    assert config.training_data == Path("artifacts/data_ingestion/Chest-CT-Scan-data")
    assert config.params_epochs == 10
    assert config.params_batch_size == 16


def test_get_evaluation_config_points_at_trained_model():
    config = ConfigurationManager().get_evaluation_config()

    assert isinstance(config, EvaluationConfig)
    assert config.path_of_model == "artifacts/training/model.h5"
    assert config.params_batch_size == 16
