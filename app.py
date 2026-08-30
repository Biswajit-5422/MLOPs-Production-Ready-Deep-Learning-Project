import binascii
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import UnidentifiedImageError

from cnnClassifier import logger
from cnnClassifier.api.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_DURATION_SECONDS,
    PrometheusMiddleware,
    metrics_response,
)
from cnnClassifier.api.schemas import HealthResponse, PredictRequest, PredictResponse, TrainResponse
from cnnClassifier.utils.common import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

templates = Jinja2Templates(directory="templates")


def _load_classifier():
    # Imported lazily (not at module level) so importing this module - which
    # tests do via TestClient - never pulls in TensorFlow. Tests monkeypatch
    # this whole function instead of mocking deep inside the real pipeline.
    from cnnClassifier.pipeline.predict import PredictionPipeline
    return PredictionPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    try:
        app.state.classifier = _load_classifier()
        logger.info("Model loaded successfully.")
    except Exception:
        logger.exception("Failed to load model at startup.")
        app.state.classifier = None
    yield
    app.state.classifier = None


app = FastAPI(
    title="Chest CT-Scan Cancer Classifier",
    description="Serves predictions from the MLOps pipeline's trained VGG16 classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness probe: the process is up. Doesn't say the model loaded - see /ready."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse, tags=["ops"])
def ready(request: Request):
    """Readiness probe: only reports ready once the model is actually loaded."""
    if getattr(request.app.state, "classifier", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ready")


@app.get("/metrics", tags=["ops"])
def metrics():
    return metrics_response()


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    responses={400: {"description": "Malformed image data"}, 503: {"description": "Model not loaded"}},
)
def predict(payload: PredictRequest, request: Request):
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    fd, image_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        try:
            decodeImage(payload.image, image_path)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

        try:
            start = time.perf_counter()
            result = classifier.predict(image_path)
            PREDICTION_DURATION_SECONDS.observe(time.perf_counter() - start)
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Uploaded data is not a valid image") from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    PREDICTIONS_TOTAL.labels(label=result["label"]).inc()
    return PredictResponse(**result)


def _run_dvc_repro():
    try:
        subprocess.run(["dvc", "repro"], check=True)
        logger.info("dvc repro completed successfully.")
    except subprocess.CalledProcessError:
        logger.exception("dvc repro failed.")
    except FileNotFoundError:
        logger.exception("dvc executable not found.")


@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(background_tasks: BackgroundTasks):
    """Kicks off `dvc repro` in the background and returns immediately, rather
    than blocking the request for the full training run like the old Flask
    route did.
    """
    background_tasks.add_task(_run_dvc_repro)
    return TrainResponse(status="accepted", detail="Training started in the background (dvc repro).")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
