import subprocess
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.keras
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

# Backend must be set before pyplot is imported - there's no display in CI/Docker.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dagshub  # noqa: E402
dagshub.init(repo_owner='biswajitdas542002', repo_name='MLOPs-Production-Ready-Deep-Learning-Project', mlflow=True)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self._compute_classification_metrics()

    def _compute_classification_metrics(self):
        """Runs a full prediction pass over the validation set (shuffle=False, so
        prediction order matches self.valid_generator.classes) to get per-class
        precision/recall/f1 and a confusion matrix - accuracy alone hides whether
        the model is actually catching the cancer class.
        """
        self.valid_generator.reset()
        probabilities = self.model.predict(self.valid_generator, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        y_true = self.valid_generator.classes
        self.class_labels = sorted(
            self.valid_generator.class_indices, key=self.valid_generator.class_indices.get
        )

        self.confusion = confusion_matrix(y_true, y_pred)
        self.report = classification_report(
            y_true, y_pred, target_names=self.class_labels, output_dict=True, zero_division=0
        )

    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "precision_macro": self.report["macro avg"]["precision"],
            "recall_macro": self.report["macro avg"]["recall"],
            "f1_macro": self.report["macro avg"]["f1-score"],
        }
        save_json(path=Path("scores.json"), data=scores)

    def save_confusion_matrix(self):
        out_dir = Path(self.config.eval_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            self.confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_labels, yticklabels=self.class_labels, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Validation Set")
        fig.tight_layout()

        self.confusion_matrix_path = out_dir / "confusion_matrix.png"
        fig.savefig(self.confusion_matrix_path)
        plt.close(fig)

        self.classification_report_path = out_dir / "classification_report.json"
        save_json(path=self.classification_report_path, data=self.report)

    @staticmethod
    def _current_git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    def log_into_mlflow(self):
        """Ties this MLflow run back to the exact pipeline state that produced it:
        the git commit (code) and dvc.lock (data/model artifact hashes), so a run's
        metrics can always be reproduced with `git checkout <commit> && dvc repro`.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            for class_name, metrics in self.report.items():
                if not isinstance(metrics, dict):
                    continue
                safe_name = class_name.replace(" ", "_")
                for metric_name in ("precision", "recall", "f1-score"):
                    if metric_name in metrics:
                        mlflow.log_metric(f"{safe_name}_{metric_name}", metrics[metric_name])

            mlflow.set_tag("git_commit", self._current_git_commit())
            if Path("dvc.lock").exists():
                mlflow.log_artifact("dvc.lock")
            if hasattr(self, "confusion_matrix_path"):
                mlflow.log_artifact(str(self.confusion_matrix_path))
            if hasattr(self, "classification_report_path"):
                mlflow.log_artifact(str(self.classification_report_path))

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
