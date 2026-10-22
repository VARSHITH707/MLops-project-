from fastapi.testclient import TestClient

from tests.conftest import API_KEY, SAMPLE


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]


def test_predict_success_and_logged(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["failure_probability"] <= 1
    assert body["risk_category"] in {"low", "medium", "high"}
    assert body["failure_predicted"] == (body["failure_probability"] >= body["threshold"])
    recent = client.get("/predictions/recent").json()
    assert recent[0]["id"] == body["prediction_id"]


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"readings": [SAMPLE, {**SAMPLE, "tool_wear": 10}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_invalid_requests_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "machine_type": "Z"}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "torque": -1}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in SAMPLE.items() if k != "torque"}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "extra": 1}).status_code == 422
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422


def test_model_info_and_metrics(client):
    info = client.get("/model-info").json()
    assert info["metadata"]["model_version"] == info["pointer"]["current"]
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "pm_predictions_total 1" in text and "pm_model_loaded 1" in text


def test_record_outcome(client):
    pid = client.post("/predict", json=SAMPLE).json()["prediction_id"]
    assert client.post("/outcomes", json={"prediction_id": pid, "actual_failure": 1}).status_code == 200
    assert client.post("/outcomes", json={"prediction_id": "nope", "actual_failure": 1}).status_code == 404


def test_admin_requires_key(client):
    assert client.post("/admin/retrain", json={}).status_code == 401
    assert client.post("/admin/rollback", headers={"X-API-Key": "wrong"}).status_code == 401


def test_admin_disabled_without_configured_key(client, monkeypatch):
    monkeypatch.delenv("PM_ADMIN_API_KEY")
    assert client.post("/admin/retrain", json={}, headers={"X-API-Key": API_KEY}).status_code == 503


def test_missing_model_returns_503(settings):
    from app.main import create_app

    with TestClient(create_app()) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=SAMPLE).status_code == 503
        assert c.get("/model-info").status_code == 503
