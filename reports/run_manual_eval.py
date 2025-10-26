"""
Run curated user stories through the trained classifier, compare against the
expected components, and emit stakeholder-ready artifacts (JSON + charts).
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(PROJECT_ROOT / "src"))
from predict import load_assets  # type: ignore  # noqa: E402

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    default_cases = PROJECT_ROOT / "reports" / "manual_eval_cases_100.jsonl"
    parser = argparse.ArgumentParser(description="Manual scenario evaluation runner.")
    parser.add_argument("--cases_path", type=Path, default=default_cases, help="JSONL file with fields id,text,expected.")
    parser.add_argument("--model_dir", type=Path, default=PROJECT_ROOT / "models" / "distilbert_component_classifier")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "reports" / "manual_eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--top_k_fallback",
        type=int,
        default=0,
        help="Adds up to K highest-confidence labels even if they fall below the threshold.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> List[Dict]:
    cases: List[Dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def classify_cases(
    cases: List[Dict],
    model_dir: Path,
    threshold: float,
    max_length: int,
    batch_size: int,
    top_k_fallback: int,
):
    model, tokenizer, label_names = load_assets(str(model_dir))
    texts = [case["text"] for case in cases]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_probs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}
        with torch.no_grad():
            logits = model(**encodings).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    probs_matrix = np.concatenate(all_probs, axis=0)

    results = []
    for case, prob_row in zip(cases, probs_matrix):
        sorted_idx = np.argsort(prob_row)[::-1]
        predictions_detail = []
        added_labels = set()
        for idx in sorted_idx:
            score = float(prob_row[idx])
            if score < threshold:
                break
            label = label_names[idx]
            predictions_detail.append({"label": label, "score": round(score, 4), "source": "threshold"})
            added_labels.add(label)

        fallback_added = 0
        if top_k_fallback > 0:
            for idx in sorted_idx:
                if fallback_added >= top_k_fallback:
                    break
                label = label_names[idx]
                if label in added_labels:
                    continue
                score = float(prob_row[idx])
                predictions_detail.append({"label": label, "score": round(score, 4), "source": "fallback"})
                added_labels.add(label)
                fallback_added += 1

        if not predictions_detail:
            top_idx = sorted_idx[0]
            predictions_detail.append(
                {"label": label_names[top_idx], "score": round(float(prob_row[top_idx]), 4), "source": "best_guess"}
            )

        predicted_labels = [item["label"] for item in predictions_detail]
        expected_labels = case.get("expected", [])
        predicted_set = set(predicted_labels)
        expected_set = set(expected_labels)
        true_pos = sorted(predicted_set & expected_set)
        false_pos = sorted(predicted_set - expected_set)
        false_neg = sorted(expected_set - predicted_set)
        label = "exact" if not false_pos and not false_neg else ("partial" if true_pos else "miss")
        results.append(
            {
                "id": case["id"],
                "text": case["text"],
                "expected": expected_labels,
                "predicted": predicted_labels,
                "prediction_details": predictions_detail,
                "true_positives": true_pos,
                "false_positives": false_pos,
                "false_negatives": false_neg,
                "match_type": label,
                "rationale": case.get("rationale"),
            }
        )
    return results


def summarize(results: List[Dict]) -> Dict:
    total_cases = len(results)
    counts = Counter(item["match_type"] for item in results)
    tp = sum(len(item["true_positives"]) for item in results)
    fp = sum(len(item["false_positives"]) for item in results)
    fn = sum(len(item["false_negatives"]) for item in results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    subset_accuracy = counts.get("exact", 0) / total_cases if total_cases else 0.0
    return {
        "total_cases": total_cases,
        "match_counts": dict(counts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "subset_accuracy": subset_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def save_outputs(output_dir: Path, results: List[Dict], summary: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": results}
    (output_dir / "manual_eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(output_dir / "manual_eval_results.csv", index=False)


def create_visuals(
    output_dir: Path, results: List[Dict], summary: Dict, miss_counter: Counter
) -> Dict[str, Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: Dict[str, Path] = {}

    # Match type distribution
    match_df = pd.DataFrame(
        [
            {"match_type": mt.capitalize(), "count": summary["match_counts"].get(mt, 0)}
            for mt in ["exact", "partial", "miss"]
        ]
    )
    plt.figure(figsize=(5, 4))
    sns.barplot(data=match_df, x="match_type", y="count", hue="match_type", palette="crest", dodge=False, legend=False)
    plt.title("Manual Scenario Outcomes")
    plt.xlabel("Outcome Type")
    plt.ylabel("# Cases")
    plt.tight_layout()
    match_path = figures_dir / "match_type_counts.png"
    plt.savefig(match_path, dpi=300)
    plt.close()
    chart_paths["Outcome distribution"] = match_path

    # Case-level recall chart
    per_case = []
    for item in results:
        expected = len(item["expected"])
        recall = len(item["true_positives"]) / expected if expected else 0.0
        per_case.append({"id": item["id"], "recall": recall, "match_type": item["match_type"].capitalize()})
    per_case_df = pd.DataFrame(per_case)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=per_case_df, x="id", y="recall", hue="match_type", palette="Set2")
    plt.title("Per-case recall (fraction of expected components found)")
    plt.xlabel("Scenario ID")
    plt.ylabel("Recall")
    plt.ylim(0, 1.05)
    plt.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    recall_path = figures_dir / "per_case_recall.png"
    plt.savefig(recall_path, dpi=300)
    plt.close()
    chart_paths["Per-case recall"] = recall_path

    # False negative hot spots
    if miss_counter:
        miss_df = (
            pd.DataFrame(miss_counter.items(), columns=["label", "misses"])
            .sort_values("misses", ascending=False)
            .head(10)
        )
        plt.figure(figsize=(7, 4))
        sns.barplot(data=miss_df, x="misses", y="label", hue="label", palette="rocket", legend=False)
        plt.title("Top missing components (edge cases)")
        plt.xlabel("# Misses across manual scenarios")
        plt.ylabel("Component")
        plt.tight_layout()
        miss_path = figures_dir / "top_missed_components.png"
        plt.savefig(miss_path, dpi=300)
        plt.close()
        chart_paths["Top misses"] = miss_path

    return chart_paths


def write_text_summary(
    output_dir: Path,
    summary: Dict,
    results: List[Dict],
    miss_counter: Counter,
    cases_path: Path,
    threshold: float,
    top_k_fallback: int,
) -> None:
    lines = ["# Manual Scenario Evaluation", ""]
    lines.append(f"- Source cases: `{cases_path}`")
    lines.append(f"- Total scenarios reviewed: {summary['total_cases']}")
    lines.append(f"- Threshold: {threshold} | Top-k fallback: {top_k_fallback}.")
    lines.append(
        f"- Precision {summary['precision']*100:.1f}% ({summary['tp']} correct vs {summary['fp']} incorrect component calls)."
    )
    denom = summary["tp"] + summary["fn"]
    lines.append(
        f"- Recall {summary['recall']*100:.1f}% (found {summary['tp']} of {denom} expected components)."
    )
    lines.append(f"- Exact matches: {int(summary['subset_accuracy'] * summary['total_cases'])}/{summary['total_cases']}.")

    if miss_counter:
        top_misses = ", ".join(f"{label} ({count})" for label, count in miss_counter.most_common(5))
        lines.append(f"- Most frequently missed components: {top_misses}.")

    lines += ["", "## Scenario Breakdown"]
    for item in results:
        tp = ", ".join(item["true_positives"]) or "none"
        fp = ", ".join(item["false_positives"]) or "none"
        fn = ", ".join(item["false_negatives"]) or "none"
        lines.append(
            f"- {item['id']} [{item['match_type'].upper()}] expected {', '.join(item['expected'])}; "
            f"model predicted {', '.join(item['predicted']) or 'none'}. "
            f"Hits: {tp}. Extra: {fp}. Missed: {fn}."
        )
        if item.get("rationale"):
            lines.append(f"  - Why it matters: {item['rationale']}")

    (output_dir / "manual_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases_path)
    results = classify_cases(
        cases,
        args.model_dir,
        args.threshold,
        args.max_length,
        batch_size=args.batch_size,
        top_k_fallback=args.top_k_fallback,
    )
    summary = summarize(results)
    miss_counter = Counter()
    for item in results:
        miss_counter.update(item["false_negatives"])
    save_outputs(args.output_dir, results, summary)
    charts = create_visuals(args.output_dir, results, summary, miss_counter)
    write_text_summary(
        args.output_dir,
        summary,
        results,
        miss_counter,
        args.cases_path,
        args.threshold,
        args.top_k_fallback,
    )

    print("Manual evaluation complete:")
    print(json.dumps(summary, indent=2))
    for label, path in charts.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
