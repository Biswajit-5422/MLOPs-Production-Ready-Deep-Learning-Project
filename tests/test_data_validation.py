import json

from PIL import Image

from cnnClassifier.components.data_validation import DataValidation
from cnnClassifier.entity.config_entity import DataValidationConfig


def _make_config(tmp_path, data_dir, max_imbalance_ratio=3.0):
    return DataValidationConfig(
        root_dir=tmp_path,
        data_dir=data_dir,
        status_file=tmp_path / "status.txt",
        report_file=tmp_path / "validation_report.json",
        allowed_extensions=(".jpg", ".jpeg", ".png"),
        max_imbalance_ratio=max_imbalance_ratio,
    )


def _write_image(path, size=(8, 8), color=(255, 0, 0)):
    Image.new("RGB", size, color).save(path)


def test_validate_passes_on_well_formed_balanced_dataset(tmp_path):
    data_dir = tmp_path / "data"
    for cls, count in [("Normal", 5), ("Cancer", 5)]:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(count):
            _write_image(cls_dir / f"img_{i}.jpg")

    report = DataValidation(_make_config(tmp_path, data_dir)).validate()

    assert report["validation_status"] is True
    assert report["num_classes"] == 2
    assert report["total_images"] == 10
    assert report["corrupt_files"] == []
    assert report["imbalance_ratio"] == 1.0


def test_validate_flags_corrupt_files_and_fails(tmp_path):
    data_dir = tmp_path / "data"
    cls_dir = data_dir / "Normal"
    cls_dir.mkdir(parents=True)
    _write_image(cls_dir / "good.jpg")
    (cls_dir / "bad.jpg").write_bytes(b"not a real image")

    report = DataValidation(_make_config(tmp_path, data_dir)).validate()

    assert report["validation_status"] is False
    assert str(cls_dir / "bad.jpg") in report["corrupt_files"]


def test_validate_computes_imbalance_ratio(tmp_path):
    data_dir = tmp_path / "data"
    for cls, count in [("Normal", 2), ("Cancer", 8)]:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(count):
            _write_image(cls_dir / f"img_{i}.jpg")

    report = DataValidation(_make_config(tmp_path, data_dir)).validate()

    assert report["imbalance_ratio"] == 4.0
    assert report["validation_status"] is True  # no corrupt files, still "valid" just imbalanced


def test_validate_fails_when_data_dir_missing(tmp_path):
    missing_dir = tmp_path / "does-not-exist"

    report = DataValidation(_make_config(tmp_path, missing_dir)).validate()

    assert report["validation_status"] is False
    assert report["total_images"] == 0


def test_validate_writes_status_and_report_files(tmp_path):
    data_dir = tmp_path / "data"
    cls_dir = data_dir / "Normal"
    cls_dir.mkdir(parents=True)
    _write_image(cls_dir / "img_0.jpg")
    # A second class dir so validation_status can pass.
    (data_dir / "Cancer").mkdir()
    _write_image(data_dir / "Cancer" / "img_0.jpg")

    config = _make_config(tmp_path, data_dir)
    DataValidation(config).validate()

    assert config.status_file.exists()
    assert "Validation status: True" in config.status_file.read_text()
    assert json.loads(config.report_file.read_text())["validation_status"] is True
