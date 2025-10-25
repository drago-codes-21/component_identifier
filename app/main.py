from pathlib import Path
import sys
from typing import List

from fastapi import FastAPI, HTTPException
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

MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_component_classifier"
MODEL = None
TOKENIZER = None
LABELS = None


@app.on_event("startup")
async def _load_model() -> None:
    global MODEL, TOKENIZER, LABELS  # pylint: disable=global-statement
    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory '{MODEL_DIR}' not found. Train the model before starting the API."
        )
    MODEL, TOKENIZER, LABELS = load_assets(str(MODEL_DIR))


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
