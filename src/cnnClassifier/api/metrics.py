import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Generic HTTP traffic metrics - the kind any service should expose.
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds", "HTTP request latency in seconds", ["method", "path"]
)

# Model-specific business metrics: the predicted-class counter is what lets you
# notice prediction/data drift in production (e.g. the cancer-class rate
# suddenly jumping) without needing ground-truth labels.
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total predictions served, by predicted label", ["label"]
)
PREDICTION_DURATION_SECONDS = Histogram(
    "prediction_duration_seconds", "Model inference latency in seconds (excludes HTTP overhead)"
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Records request count and latency for every request. Uses the matched
    route template (e.g. /predict) rather than the raw path as the label, so a
    path with an id in it can't blow up metric cardinality.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        route = request.scope.get("route")
        path_template = route.path if route is not None else request.url.path

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method, path=path_template, status_code=response.status_code
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method, path=path_template
        ).observe(duration)

        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
