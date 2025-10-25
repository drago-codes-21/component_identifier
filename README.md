# Component Identification with DistilBERT

Predict the software components impacted by a user story or requirement description using a multi-label DistilBERT classifier. The project covers data preparation, fine-tuning, batch inference, and serving a FastAPI endpoint.

## Project Layout

```
component_identifier/
├── app/
│   └── main.py                          # FastAPI service
├── data/
│   ├── generate_synthetic_baseline.py   # Script to rebuild the demo dataset
│   └── train.csv                        # Synthetic baseline training set (3,000 stories)
├── models/
│   └── distilbert_component_classifier/ # Saved weights/tokenizer after training
├── src/
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Dataset

- Input CSV columns:
  - `text`: user story or requirement.
  - `labels`: impacted components separated by `|` (multi-label, 20 components in the baseline taxonomy).
- `data/train.csv` contains 3,000 synthetic user stories derived from 25 domain scenarios. Each scenario maps deterministically to the impacted components to provide a balanced starting point for fine-tuning.
- Regenerate the demo data anytime:

  ```bash
  python data/generate_synthetic_baseline.py
  ```

- Swap in your labeled requirements before training for production scenarios, but keep the same schema and ensure every component has dozens of examples for stable learning.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

```bash
python src/train.py \
  --train_path data/train.csv \
  --output_dir models/distilbert_component_classifier \
  --batch_size 4 \
  --num_epochs 4 \
  --max_length 256
```

Key features:

- Uses `DistilBertTokenizerFast` and `DistilBertForSequenceClassification`.
- Multi-label setup with sigmoid activation and `BCEWithLogitsLoss`.
- Train/validation split (default 90/10), micro precision/recall/F1 metrics, and Hugging Face `Trainer`.
- Model + tokenizer + label map saved via `save_pretrained()`.
- Reference run (distilbert-base-uncased, synthetic baseline dataset, batch size 4, 4 epochs) previously achieved micro Precision 1.00, Recall 0.96, F1 0.98 on the held-out validation split. Expect metrics to differ once you introduce real data.

## Local Inference

```bash
python src/predict.py \
  --model_dir models/distilbert_component_classifier \
  --text "As an admin I need to monitor feature rollout to stay compliant."
```

The script prints component predictions with confidence scores. Adjust `--threshold` (default 0.5) for stricter or more permissive outputs.

## FastAPI Service

1. Ensure the fine-tuned artifacts exist in `models/distilbert_component_classifier/`.
2. Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

3. Send a request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "text": "As a customer I need to reset my password to reduce support calls." }'
```

Response example:

```json
{
  "components": [
    {"label": "auth", "score": 0.91},
    {"label": "support", "score": 0.54}
  ],
  "threshold": 0.5
}
```

## Optional: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t component-identifier .
docker run -p 8000:8000 component-identifier
```

## Notes

- Training defaults target CPU-friendly settings (batch size 4, max length 256, 3–5 epochs). Adjust `--num_epochs`, `--learning_rate`, and other CLI flags as needed.
- The provided synthetic dataset is only for demonstration. Replace it with real, labeled production data for meaningful predictions.
