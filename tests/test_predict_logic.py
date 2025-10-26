import types
import torch

from src.predict import predict


class DummyModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, **kwargs):  # type: ignore[override]
        return types.SimpleNamespace(logits=self._logits)


class DummyTokenizer:
    def __call__(self, texts, truncation=True, padding=True, max_length=256, return_tensors="pt"):
        # Return minimal dict with input_ids/attention_mask of the right shape
        batch = len(texts)
        return {
            "input_ids": torch.zeros((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


def test_predict_threshold_and_fallback(tmp_path):
    # Two labels; logits produce probs below threshold 0.9
    model = DummyModel(logits=[-2.0, -1.0])
    labels = ["L0", "L1"]
    outputs = predict(
        model_dir=str(tmp_path),
        texts=["hello"],
        max_length=8,
        threshold=0.9,
        model=model,
        tokenizer=DummyTokenizer(),
        label_names=labels,
    )
    # Should return the best candidate even below threshold
    assert outputs[0][0]["label"] == "L1"

