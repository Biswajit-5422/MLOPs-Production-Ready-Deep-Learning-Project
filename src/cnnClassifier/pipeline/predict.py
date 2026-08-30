import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_LABELS = {0: "Adenocarcinoma Cancer", 1: "Normal"}
DEFAULT_MODEL_PATH = os.path.join("model", "best_model.h5")


class PredictionPipeline:
    """Loads the trained model once; `predict()` takes the image path per call
    instead of storing it on the instance, so one PredictionPipeline can safely
    serve concurrent requests without them clobbering a shared filename.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model = load_model(model_path)

    def predict(self, image_path: str) -> dict:
        test_image = image.load_img(image_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        probabilities = self.model.predict(test_image, verbose=0)
        predicted_index = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities))

        return {
            "label": CLASS_LABELS[predicted_index],
            "confidence": round(confidence * 100, 2),
        }
