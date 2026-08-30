import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module


class FakeClassifier:
    """Stands in for the real (TensorFlow-backed) PredictionPipeline. Still
    opens the file with PIL so the "not a valid image" error path is
    genuinely exercised, without ever importing TensorFlow in this test file.
    """

    def __init__(self, label="Normal", confidence=91.23):
        self.label = label
        self.confidence = confidence

    def predict(self, image_path):
        with Image.open(image_path) as img:
            img.verify()
        return {"label": self.label, "confidence": self.confidence}


def _b64_jpeg(size=(16, 16), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "_load_classifier", lambda: FakeClassifier())
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def client_model_unavailable(monkeypatch):
    def _raise():
        raise RuntimeError("model file missing")

    monkeypatch.setattr(app_module, "_load_classifier", _raise)
    with TestClient(app_module.app) as c:
        yield c


def test_health_is_always_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}


def test_ready_returns_503_when_model_failed_to_load(client_model_unavailable):
    res = client_model_unavailable.get("/ready")
    assert res.status_code == 503


def test_home_page_serves_frontend(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Chest CT-Scan" in res.text


def test_predict_returns_label_and_confidence(client):
    res = client.post("/predict", json={"image": _b64_jpeg()})
    assert res.status_code == 200
    body = res.json()
    assert body == {"label": "Normal", "confidence": 91.23}


def test_predict_rejects_invalid_base64(client):
    res = client.post("/predict", json={"image": "not-valid-base64!!"})
    assert res.status_code == 400


def test_predict_rejects_non_image_bytes(client):
    garbage = base64.b64encode(b"definitely not an image").decode()
    res = client.post("/predict", json={"image": garbage})
    assert res.status_code == 400


def test_predict_missing_field_is_422(client):
    res = client.post("/predict", json={})
    assert res.status_code == 422


def test_predict_returns_503_when_model_not_loaded(client_model_unavailable):
    res = client_model_unavailable.post("/predict", json={"image": _b64_jpeg()})
    assert res.status_code == 503


def test_train_starts_in_background_and_returns_immediately(client, monkeypatch):
    called = {"ran": False}
    monkeypatch.setattr(app_module, "_run_dvc_repro", lambda: called.__setitem__("ran", True))

    res = client.post("/train")

    assert res.status_code == 200
    assert res.json()["status"] == "accepted"
    assert called["ran"] is True  # TestClient runs BackgroundTasks synchronously


def test_metrics_endpoint_exposes_prediction_counter(client):
    client.post("/predict", json={"image": _b64_jpeg()})

    res = client.get("/metrics")

    assert res.status_code == 200
    assert "predictions_total" in res.text
    assert "http_requests_total" in res.text
