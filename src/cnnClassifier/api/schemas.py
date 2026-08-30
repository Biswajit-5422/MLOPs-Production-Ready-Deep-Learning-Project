from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded JPEG/PNG image data (no data: URI prefix).",
    )


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted class: 'Normal' or 'Adenocarcinoma Cancer'.")
    confidence: float = Field(..., ge=0, le=100, description="Model confidence as a percentage.")


class HealthResponse(BaseModel):
    status: str


class TrainResponse(BaseModel):
    status: str
    detail: str


class ErrorResponse(BaseModel):
    detail: str
