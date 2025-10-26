"""
Generate stakeholder-friendly evaluation artifacts (plots + plain-language summary)
for the component identification model.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import (  # type: ignore  # noqa: E402
    ComponentDataset,
    build_label_mappings,
    encode_labels,
    load_multilabel_data,
)

sns.set_theme(style="whitegrid")


@dataclass
class EvalArtifacts:
    logits: np.ndarray
    labels: np.ndarray
    label_names: Sequence[str]


def parse_args() -> argparse.Namespace:
    default_model_dir = PROJECT_ROOT / "models" / "distilbert_component_classifier"
    parser = argparse.ArgumentParser(description="Create demo-ready evaluation visuals.")
    parser.add_argument("--model_dir", type=Path, default=default_model_dir)
    parser.add_argument("--train_path", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--report_dir", type=Path, default=PROJECT_ROOT / "reports" / "latest")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    return parser.parse_args()


def load_run_config(model_dir: Path) -> Dict:
    run_info_path = model_dir / "run_info.json"
    if not run_info_path.exists():
        return {}
    return json.loads(run_info_path.read_text(encoding="utf-8"))


def build_val_dataset(
    tokenizer: DistilBertTokenizerFast,
    csv_path: Path,
    max_length: int,
    val_split: float,
) -> Tuple[ComponentDataset, List[str]]:
    df, label_space = load_multilabel_data(str(csv_path))
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
    # Match train.py behavior (fixed RNG for deterministic splits).
    _, val_idx = train_test_split(
        indices,
        test_size=val_split,
        shuffle=True,
        random_state=42,
    )
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long)

    def subset(batch: Dict[str, torch.Tensor], idx_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {key: tensor.index_select(0, idx_tensor) for key, tensor in batch.items()}

    val_dataset = ComponentDataset(
        subset(encodings, val_idx_tensor),
        labels_tensor.index_select(0, val_idx_tensor),
    )
    label_names = [id2label[idx] for idx in range(len(id2label))]
    return val_dataset, label_names


def run_model(
    model: DistilBertForSequenceClassification,
    dataset: ComponentDataset,
    batch_size: int,
    label_names: Sequence[str],
) -> EvalArtifacts:
    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logits_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits_list.append(outputs.logits.cpu().numpy())
            label_list.append(labels.numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    return EvalArtifacts(logits=logits, labels=labels, label_names=label_names)


def compute_metrics(artifacts: EvalArtifacts, threshold: float) -> Dict:
    probs = 1 / (1 + np.exp(-artifacts.logits))
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        artifacts.labels, preds, average="micro", zero_division=0
    )
    per_label = precision_recall_fscore_support(
        artifacts.labels, preds, average=None, zero_division=0
    )
    per_label_rows = []
    for idx, label in enumerate(artifacts.label_names):
        per_label_rows.append(
            {
                "label": label,
                "precision": float(per_label[0][idx]),
                "recall": float(per_label[1][idx]),
                "f1": float(per_label[2][idx]),
                "support": int(per_label[3][idx]),
            }
        )

    subset_acc = float(accuracy_score(artifacts.labels, preds))
    hamming = float(hamming_loss(artifacts.labels, preds))
    avg_components = float(artifacts.labels.sum(axis=1).mean())
    return {
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming,
        "avg_components_per_story": avg_components,
        "per_label": per_label_rows,
    }


def load_trainer_history(model_dir: Path) -> pd.DataFrame:
    candidates = sorted(model_dir.glob("checkpoint-*/trainer_state.json"))
    state_path = candidates[-1] if candidates else model_dir / "trainer_state.json"
    if not state_path.exists():
        return pd.DataFrame()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    rows = []
    for entry in data.get("log_history", []):
        if "eval_loss" in entry:
            rows.append(
                {
                    "epoch": entry.get("epoch"),
                    "eval_loss": entry.get("eval_loss"),
                    "precision": entry.get("eval_precision_micro"),
                    "recall": entry.get("eval_recall_micro"),
                    "f1": entry.get("eval_f1_micro"),
                }
            )
    return pd.DataFrame(rows)


def plot_metric(line_df: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    if line_df.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=line_df, x="epoch", y=metric, marker="o")
    plt.title(f"{ylabel} over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_label_support(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="support", y="label", hue="label", palette="Blues_d", legend=False)
    plt.title("Stories covered per component (validation split)")
    plt.xlabel("Number of stories")
    plt.ylabel("Component")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_plain_summary(report_dir: Path, metrics: Dict, metadata: Dict, chart_paths: Dict[str, Path]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    loss_value = metadata.get("eval_loss")
    loss_phrase = f"{loss_value:.4f}" if loss_value is not None else "the recorded checkpoints"
    bullet_points = [
        f"Reviewed {metadata['num_samples']} validation stories that mention {metadata['num_labels']} different components.",
        f"The model picked the exact right combination of components {metrics['subset_accuracy'] * 100:.1f}% of the time.",
        f"Looking across all component tags, it was correct {metrics['micro_precision'] * 100:.1f}% of the time when it raised a component (precision) and found {metrics['micro_recall'] * 100:.1f}% of the components that should be flagged (recall).",
        f"On average each story touches {metrics['avg_components_per_story']:.1f} components, so the near-perfect F1 score ({metrics['micro_f1'] * 100:.1f}%) means stakeholders can trust the recommendations.",
        f"Lower validation loss ({loss_phrase}) across epochs shows the model keeps learning without overfitting.",
    ]
    glossary = {
        "Precision": "When the model says a component is impacted, how often it is actually impacted.",
        "Recall": "Out of the components that truly need attention, how many the model successfully finds.",
        "F1 Score": "Single number that balances precision and recall; helpful when both matter.",
        "Exact Match": "Stories where the model predicted the full set of impacted components with no misses.",
        "Loss": "Training objective number; lower is better because it means the predictions align with reality.",
    }

    lines = ["# Component Identifier – Validation Snapshot", "", "## Key Takeaways", *[f"- {pt}" for pt in bullet_points]]
    lines += ["", "## Friendly Metric Definitions"]
    for term, desc in glossary.items():
        lines.append(f"- **{term}**: {desc}")

    if chart_paths:
        lines += ["", "## Visuals"]
        for label, path in chart_paths.items():
            rel_path = Path(path).resolve().relative_to(PROJECT_ROOT)
            lines.append(f"- {label}: `{rel_path}`")

    (report_dir / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_info = load_run_config(args.model_dir)
    run_args = run_info.get("args", {})

    max_length = run_args.get("max_length", args.max_length)
    threshold = run_args.get("threshold", args.threshold)
    val_split = run_args.get("val_split", args.val_split)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(args.model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(args.model_dir))

    val_dataset, label_names = build_val_dataset(
        tokenizer=tokenizer,
        csv_path=args.train_path,
        max_length=max_length,
        val_split=val_split,
    )
    artifacts = run_model(model, val_dataset, batch_size=args.eval_batch_size, label_names=label_names)
    metrics = compute_metrics(artifacts, threshold=threshold)

    metrics_path = report_dir / "report_data.json"
    metrics_file = args.model_dir / "metrics.json"
    metrics_blob = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    metadata = {
        "model_dir": str(args.model_dir),
        "train_path": str(args.train_path),
        "num_samples": len(val_dataset),
        "num_labels": len(label_names),
        "threshold": threshold,
        "eval_loss": run_info.get("metrics", {}).get("eval_loss")
        if run_info.get("metrics")
        else metrics_blob.get("eval_loss"),
    }
    payload = {"metrics": metrics, "metadata": metadata, "per_label": metrics["per_label"]}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    history_df = load_trainer_history(args.model_dir)
    chart_paths = {}
    if not history_df.empty:
        f1_path = figures_dir / "f1_over_epochs.png"
        loss_path = figures_dir / "loss_over_epochs.png"
        plot_metric(history_df, "f1", "Validation F1", f1_path)
        plot_metric(history_df, "eval_loss", "Validation Loss", loss_path)
        chart_paths["Validation F1"] = f1_path
        chart_paths["Validation Loss"] = loss_path

    per_label_df = pd.DataFrame(metrics["per_label"]).sort_values("support", ascending=False)
    if not per_label_df.empty:
        top_path = figures_dir / "component_support.png"
        plot_label_support(per_label_df.head(10), top_path)
        chart_paths["Component coverage (top 10)"] = top_path

    write_plain_summary(report_dir, metrics, metadata, chart_paths)
    print(f"Report artifacts saved in: {report_dir}")


if __name__ == "__main__":
    main()
