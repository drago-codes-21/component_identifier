import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_label_names(model_dir: Path, config: Optional[object] = None) -> List[str]:
    label_file = model_dir / "label2id.json"
    if label_file.exists():
        with label_file.open("r", encoding="utf-8") as fp:
            label2id = json.load(fp)
        return [label for label, _ in sorted(label2id.items(), key=lambda item: item[1])]

    if config and getattr(config, "id2label", None):
        return [config.id2label[idx] for idx in sorted(config.id2label.keys())]

    raise ValueError("No label metadata detected. Train the model first.")


def load_assets(model_dir: str) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast, List[str]]:
    model_path = Path(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
    labels = load_label_names(model_path, model.config)
    return model, tokenizer, labels


def predict(
    model_dir: str,
    texts: Sequence[str],
    max_length: int = 256,
    threshold: float = 0.5,
    model: Optional[DistilBertForSequenceClassification] = None,
    tokenizer: Optional[DistilBertTokenizerFast] = None,
    label_names: Optional[List[str]] = None,
):
    if model is None or tokenizer is None or label_names is None:
        model, tokenizer, label_names = load_assets(model_dir)

    inputs = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)

    outputs = []
    for row in probs:
        values = row.tolist()
        predictions = []
        for idx, score in enumerate(values):
            if score >= threshold:
                predictions.append({"label": label_names[idx], "score": round(score, 4)})
        if not predictions:
            # Return the best candidate even if it is below threshold to keep output informative.
            best_idx = max(range(len(values)), key=lambda i: values[i])
            predictions.append({"label": label_names[best_idx], "score": round(values[best_idx], 4)})
        outputs.append(predictions)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned DistilBERT model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "distilbert_component_classifier"),
    )
    parser.add_argument("--text", type=str, required=True, help="User story or requirement description.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = predict(
        model_dir=args.model_dir,
        texts=[args.text],
        max_length=args.max_length,
        threshold=args.threshold,
    )
    print(json.dumps(predictions[0], indent=2))


if __name__ == "__main__":
    main()
