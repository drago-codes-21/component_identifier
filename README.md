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

## MLOps Additions (Open Source + Free)

- Reproducible training: `src/train.py` now accepts `--seed` and fixes RNGs. After training it writes `metrics.json` and `run_info.json` into the output model folder with args, data hash, and git commit (if available) for traceability.
- Health/CORS/metrics: FastAPI exposes `/healthz` and `/readyz`. CORS is enabled for React by default. Optional Prometheus metrics (set `ENABLE_METRICS=1`) if `prometheus-fastapi-instrumentator` is installed.
- Tests: Added unit and API tests under `tests/`. Dev deps in `dev-requirements.txt`. CI via GitHub Actions runs tests on pushes/PRs in public repos for free.

### Configure CORS for React

- By default all origins are allowed. To restrict:

```bash
set ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000  # Windows PowerShell: $env:ALLOW_ORIGINS="http://..."
uvicorn app.main:app --reload --port 8000
```

### Enable Prometheus metrics (optional)

```bash
export ENABLE_METRICS=1   # Windows PowerShell: $env:ENABLE_METRICS=1
uvicorn app.main:app --reload --port 8000
```

### Run Tests Locally

```bash
pip install -r requirements.txt -r dev-requirements.txt
pytest -q
```

### Demo-Ready Metrics Report

Share visuals and plain-English talking points with stakeholders in one command:

```bash
pip install -r requirements.txt -r dev-requirements.txt
python reports/generate_report.py \
  --model_dir models/distilbert_component_classifier \
  --train_path data/train.csv \
  --report_dir reports/latest
```

You will get:

- `reports/latest/report_summary.md`: non-technical explanation of precision, recall, F1, exact-match accuracy, and loss trends.
- `reports/latest/report_data.json`: structured metrics and per-label stats ready for slide tables or dashboards.
- `reports/latest/figures/*.png`: validation F1/loss curves plus a top-component coverage bar chart for quick storytelling.

Run `python reports/generate_report.py -h` to tweak thresholds, validation split, or the destination folder.

### Manual Scenario Audit (Edge Cases)

When you need to show strengths *and* improvement areas, run the curated scenario harness:

```bash
python reports/run_manual_eval.py \
  --cases_path reports/manual_eval_cases_100.jsonl \
  --model_dir models/distilbert_component_classifier \
  --output_dir reports/manual_eval \
  --top_k_fallback 3
```

Artifacts:

- `reports/manual_eval/manual_eval_results.{json,csv}`: per-scenario expectations vs. predictions, true/false positives, misses.
- `reports/manual_eval/manual_eval_summary.md`: bullet-point narrative calling out gaps for non-technical leads.
- `reports/manual_eval/figures/*.png`: outcome distribution, per-case recall, and top missed components for slide-ready visuals.

Use `--top_k_fallback` (default 0) to add the best-scoring labels even when the sigmoid score is below the threshold—handy for exploratory edge-case analysis. Edit `reports/manual_eval_cases_100.jsonl` directly or regenerate it with:

```bash
python reports/build_manual_cases.py \
  --limit 100 \
  --output_path reports/manual_eval_cases_100.jsonl
```

The generator spans authentication, lending, collections, KYC, payments, reporting, disputes, core integration, omni-channel comms, and regulatory scenarios so each component in the taxonomy appears multiple times.

### Lightweight Run Tracking Artifacts

- After training, check your output dir (e.g., `models/distilbert_component_classifier/`) for:
  - `metrics.json`: evaluation metrics from the Trainer
  - `run_info.json`: training args, data SHA256, git commit
  - `label2id.json`: label mapping used at inference

These files are simple, portable, and versionable in git or any storage.

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

## Frontend Integration (React)

- Point your React app to the FastAPI endpoint:

```ts
// Example using fetch
async function predictComponents(text: string, threshold = 0.5) {
  const resp = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, threshold }),
  });
  if (!resp.ok) throw new Error("Prediction failed");
  return await resp.json();
}
```

- Ensure the backend has CORS configured (default is permissive) or set `ALLOW_ORIGINS` accordingly in production.
