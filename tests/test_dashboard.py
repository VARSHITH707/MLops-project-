"""Dashboard tests with Streamlit's AppTest. HTTP calls are routed to the real FastAPI app."""

import requests
from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT

DASHBOARD = str(PROJECT_ROOT / "app" / "dashboard.py")


def test_dashboard_shows_clear_error_when_api_down(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", refuse)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    assert any("Cannot reach the API" in e.value for e in at.error)


def test_dashboard_renders_with_live_api(client, monkeypatch):
    def via_testclient(method, url, timeout=None, **kwargs):
        return client.request(method, url.replace("http://localhost:8000", ""), **kwargs)

    monkeypatch.setattr(requests, "request", via_testclient)
    client.post("/admin/simulate", json={"n": 60, "seed": 1}, headers={"X-API-Key": "test-admin-key"})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert any(m.label == "Recall" for m in at.metric)
    assert any(m.label == "Predictions logged" and m.value == "60" for m in at.metric)

    at.button[0].click().run()  # "Predict failure risk" form button
    assert not at.exception
    assert any(m.label.startswith("Failure probability") for m in at.metric)
