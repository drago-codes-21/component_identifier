from pathlib import Path

import pandas as pd

from src.utils import load_multilabel_data, build_label_mappings, encode_labels


def test_load_multilabel_data(tmp_path: Path):
    data = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "labels": ["x|y", "y", ""],
        }
    )
    fp = tmp_path / "train.csv"
    data.to_csv(fp, index=False)

    df, label_space = load_multilabel_data(str(fp))

    assert set(label_space) == {"x", "y"}
    assert isinstance(df.loc[0, "label_list"], list)


def test_label_encoding_roundtrip():
    labels = ["a", "b", "c"]
    l2i, i2l = build_label_mappings(labels)
    assert l2i["a"] == 0 and i2l[0] == "a"

    enc = encode_labels([["a", "c"], ["b"]], l2i)
    assert enc.shape == (2, 3)
    assert enc[0].tolist() == [1.0, 0.0, 1.0]
    assert enc[1].tolist() == [0.0, 1.0, 0.0]

