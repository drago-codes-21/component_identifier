from pathlib import Path
import os
import sys
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import load_assets, predict  # type: ignore  # pylint: disable=wrong-import-position


class PredictRequest(BaseModel):
    text: str = Field(..., description="User story or requirement description.")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for label selection.")


class ComponentScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    components: List[ComponentScore]
    threshold: float


app = FastAPI(
    title="Component Identifier",
    version="1.0.0",
    summary="Predict impacted components from requirement statements.",
)

# CORS for React UI
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",") if os.getenv("ALLOW_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_component_classifier"
MODEL = None
TOKENIZER = None
LABELS = None


@app.on_event("startup")
async def _load_model() -> None:
    global MODEL, TOKENIZER, LABELS  # pylint: disable=global-statement
    if MODEL_DIR.exists():
        MODEL, TOKENIZER, LABELS = load_assets(str(MODEL_DIR))
    else:
        # Defer loading; endpoint guard will respond 503 until trained
        MODEL, TOKENIZER, LABELS = None, None, None


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/readyz")
async def readyz() -> dict:
    return {"ready": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict_components(payload: PredictRequest) -> PredictResponse:
    if MODEL is None or TOKENIZER is None or LABELS is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    predictions = predict(
        model_dir=str(MODEL_DIR),
        texts=[payload.text],
        threshold=payload.threshold,
        model=MODEL,
        tokenizer=TOKENIZER,
        label_names=LABELS,
    )[0]

    predictions.sort(key=lambda item: item["score"], reverse=True)
    scores = [ComponentScore(**item) for item in predictions]
    return PredictResponse(components=scores, threshold=payload.threshold)


# Optional Prometheus metrics if installed and enabled
if os.getenv("ENABLE_METRICS", "0") == "1":
    try:
        from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore

        Instrumentator().instrument(app).expose(app)
    except Exception:  # pragma: no cover - optional dependency
        pass
