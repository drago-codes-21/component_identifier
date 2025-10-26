from starlette.testclient import TestClient

import app.main as app_main


def test_api_predict_with_stub(monkeypatch):
    # Stub model assets so startup doesn't fail
    app_main.MODEL = object()
    app_main.TOKENIZER = object()
    app_main.LABELS = ["x", "y"]

    def fake_predict(**kwargs):
        return [[{"label": "y", "score": 0.7}, {"label": "x", "score": 0.3}]]

    monkeypatch.setattr(app_main, "predict", lambda **kwargs: fake_predict(**kwargs))

    client = TestClient(app_main.app)
    resp = client.post("/predict", json={"text": "hello", "threshold": 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["components"][0]["label"] == "y"
    assert body["threshold"] == 0.5

    # Health and readiness
    assert client.get("/healthz").status_code == 200
    rdy = client.get("/readyz").json()
    assert rdy["ready"] is True

