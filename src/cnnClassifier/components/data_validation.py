from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataValidationConfig
from cnnClassifier.utils.common import save_json


class DataValidation:
    """Checks the ingested dataset is well-formed before it reaches training:
    every class folder is present, every file actually decodes as an image,
    and the classes aren't so imbalanced that accuracy alone would be misleading.
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self) -> dict:
        data_dir = Path(self.config.data_dir)
        class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir()) if data_dir.exists() else []

        class_counts = {}
        corrupt_files = []

        for class_dir in class_dirs:
            valid_count = 0
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() not in self.config.allowed_extensions:
                    continue
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    valid_count += 1
                except (UnidentifiedImageError, OSError):
                    corrupt_files.append(str(file_path))
            class_counts[class_dir.name] = valid_count

        total_images = sum(class_counts.values())
        counts = list(class_counts.values())
        imbalance_ratio = round(max(counts) / min(counts), 2) if counts and min(counts) > 0 else None

        status = (
            data_dir.exists()
            and len(class_dirs) >= 2
            and total_images > 0
            and len(corrupt_files) == 0
        )

        report = {
            "validation_status": status,
            "num_classes": len(class_dirs),
            "class_counts": class_counts,
            "total_images": total_images,
            "imbalance_ratio": imbalance_ratio,
            "corrupt_files": corrupt_files,
        }

        save_json(path=self.config.report_file, data=report)
        with open(self.config.status_file, "w") as f:
            f.write(f"Validation status: {status}")

        if not status:
            logger.warning(f"Data validation failed: {report}")
        elif imbalance_ratio and imbalance_ratio > self.config.max_imbalance_ratio:
            logger.warning(
                f"Class imbalance ratio {imbalance_ratio} exceeds threshold "
                f"{self.config.max_imbalance_ratio} ({class_counts}) - "
                "consider class weighting or resampling before training."
            )
        else:
            logger.info(f"Data validation passed: {report}")

        return report
