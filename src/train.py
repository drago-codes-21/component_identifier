import argparse
import json
import os
import hashlib
import random
import subprocess
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import (
    ComponentDataset,
    build_label_mappings,
    encode_labels,
    load_multilabel_data,
    save_label_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for component identification.")
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "train.csv"),
        help="Path to the labeled CSV file with columns 'text' and 'labels'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base Hugging Face checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "distilbert_component_classifier"),
        help="Directory where the trained model and tokenizer will be stored.",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for validation metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def build_datasets(
    tokenizer: DistilBertTokenizerFast,
    csv_path: str,
    max_length: int,
    val_split: float,
) -> Tuple[ComponentDataset, ComponentDataset, Dict[str, int], Dict[int, str]]:
    df, label_space = load_multilabel_data(csv_path)
    label2id, id2label = build_label_mappings(label_space)
    labels_tensor = encode_labels(df["label_list"], label2id)

    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        shuffle=True,
        random_state=42,
    )

    def subset(batch: Dict[str, torch.Tensor], idx: np.ndarray) -> Dict[str, torch.Tensor]:
        idx_tensor = torch.tensor(idx, dtype=torch.long)
        return {key: tensor.index_select(0, idx_tensor) for key, tensor in batch.items()}

    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long)
    train_dataset = ComponentDataset(subset(encodings, train_idx), labels_tensor.index_select(0, train_idx_tensor))
    val_dataset = ComponentDataset(subset(encodings, val_idx), labels_tensor.index_select(0, val_idx_tensor))
    return train_dataset, val_dataset, label2id, id2label


def create_trainer(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    train_dataset: ComponentDataset,
    val_dataset: ComponentDataset,
    args: argparse.Namespace,
):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= args.threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        return {
            "precision_micro": precision,
            "recall_micro": recall,
            "f1_micro": f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


def main():
    args = parse_args()
    # Reproducibility
    try:
        from transformers import set_seed  # type: ignore
        set_seed(args.seed)
    except Exception:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    train_dataset, val_dataset, label2id, id2label = build_datasets(
        tokenizer=tokenizer,
        csv_path=args.train_path,
        max_length=args.max_length,
        val_split=args.val_split,
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_label_index(Path(args.output_dir), label2id)

    # Evaluate and persist run metadata for lightweight tracking
    metrics = trainer.evaluate()
    # Also include our micro metrics computed during eval
    metrics_path = Path(args.output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Save training args and environment info
    run_info = {
        "args": vars(args),
        "model_name": args.model_name,
        "num_labels": len(label2id),
    }

    # Attach git commit if available
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT))
            .decode("utf-8")
            .strip()
        )
        run_info["git_commit"] = commit
    except Exception:
        run_info["git_commit"] = None

    # Attach data hash for traceability
    try:
        hasher = hashlib.sha256()
        with open(args.train_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        run_info["data_sha256"] = hasher.hexdigest()
    except Exception:
        run_info["data_sha256"] = None

    run_info_path = Path(args.output_dir) / "run_info.json"
    with run_info_path.open("w", encoding="utf-8") as fp:
        json.dump(run_info, fp, indent=2)


if __name__ == "__main__":
    main()
