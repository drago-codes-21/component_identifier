import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def load_multilabel_data(csv_path: str, label_delimiter: str = "|") -> Tuple[pd.DataFrame, List[str]]:
    """
    Read the training CSV file and return a dataframe with a normalized list of labels per row
    along with a sorted list of all labels present in the dataset.
    """
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError("Expected columns 'text' and 'labels' in the dataset.")

    def _split(labels: str) -> List[str]:
        if pd.isna(labels) or not str(labels).strip():
            return []
        return [label.strip() for label in str(labels).split(label_delimiter) if label.strip()]

    df["label_list"] = df["labels"].apply(_split)
    label_space = sorted({label for labels in df["label_list"] for label in labels})
    return df, label_space


def build_label_mappings(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(label_lists: Iterable[Sequence[str]], label2id: Dict[str, int]) -> torch.Tensor:
    """
    Convert a collection of string label lists into a multi-hot tensor.
    """
    encoded = []
    num_labels = len(label2id)
    for labels in label_lists:
        row = [0.0] * num_labels
        for label in labels:
            if label not in label2id:
                continue
            row[label2id[label]] = 1.0
        encoded.append(row)
    return torch.tensor(encoded, dtype=torch.float)


class ComponentDataset(Dataset):
    """
    Simple torch Dataset that wraps tokenized encodings and dense label vectors.
    """

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def save_label_index(model_dir: Path, label2id: Dict[str, int]) -> None:
    """
    Persist label mappings next to the trained model for use at inference time.
    """
    target = Path(model_dir) / "label2id.json"
    with target.open("w", encoding="utf-8") as fp:
        json.dump(label2id, fp, indent=2)
