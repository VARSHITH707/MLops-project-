"""Streamlit dashboard. It only talks to the FastAPI service (no separate model).

Run:  python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120

# Deterministic demo inputs (same result every time for the same model).
SAMPLES = {
    "Healthy machine": {
        "machine_id": "M-001", "machine_type": "H", "air_temperature": 299.8,
        "process_temperature": 310.1, "rotational_speed": 1520, "torque": 38.5,
        "tool_wear": 40, "operating_hours": 2500, "maintenance_history_count": 0,
    },
    "Worn tool, high torque": {
        "machine_id": "M-007", "machine_type": "L", "air_temperature": 302.5,
        "process_temperature": 310.9, "rotational_speed": 1380, "torque": 58.4,
        "tool_wear": 215, "operating_hours": 15200, "maintenance_history_count": 5,
    },
    "Overheating, low speed": {
        "machine_id": "M-023", "machine_type": "M", "air_temperature": 304.0,
        "process_temperature": 311.5, "rotational_speed": 1250, "torque": 62.0,
        "tool_wear": 120, "operating_hours": 9000, "maintenance_history_count": 2,
    },
}


def api(method: str, path: str, **kwargs):
    """Call the API; returns (json, error_message)."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL} ({type(exc).__name__}). Is it running?"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        return None, f"API error {resp.status_code}: {detail}"
    return resp.json(), None


st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ Predictive Maintenance MLOps Platform")
st.caption("Educational simulation on synthetic sensor data - not a certified industrial safety system.")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_URL)
    health, err = api("GET", "/health")
    if err:
        st.error(err)
    elif health["model_loaded"]:
        st.success(f"API healthy - model {health['model_version']}")
    else:
        st.warning(health["detail"])
    admin_key = st.text_input("Admin API key (for retraining/simulation)", type="password")
    admin_headers = {"X-API-Key": admin_key} if admin_key else {}

tab_overview, tab_predict, tab_monitor, tab_retrain = st.tabs(
    ["Overview & model", "Predict", "Monitoring", "Retraining"]
)

with tab_overview:
    st.markdown(
        "Industrial machines report sensor readings. The model estimates the probability "
        "that a machine **fails within the next 24 operating hours**, so maintenance can be "
        "planned before a breakdown. Around the model sits the MLOps lifecycle: validation, "
        "tracking (MLflow), versioning (DVC + model registry), serving (FastAPI), monitoring "
        "(SQLite + PSI drift) and guarded retraining with rollback."
    )
    info, err = api("GET", "/model-info")
    if err:
        st.error(err)
    else:
        meta = info["metadata"]
        m = meta["metrics"]
        c = st.columns(4)
        c[0].metric("Recall", f"{m['recall']:.3f}")
        c[1].metric("Precision", f"{m['precision']:.3f}")
        c[2].metric("F1", f"{m['f1']:.3f}")
        c[3].metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a")
        st.write(f"**Model** {meta['model_type'].replace('_', ' ')} · "
                 f"**Version** `{meta['model_version']}` · **trained** {meta['trained_at']} · "
                 f"**threshold** {meta['threshold']} · **dataset fingerprint** `{meta['dataset_fingerprint']}`")
        cm = m["confusion_matrix"]
        st.write("Confusion matrix on held-out test data")
        st.table(pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["actual: no failure", "actual: failure"],
            columns=["predicted: no failure", "predicted: failure"],
        ))
        st.write("All model versions")
        st.dataframe(pd.DataFrame(info["versions"]), width="stretch")

with tab_predict:
    sample_name = st.selectbox("Load a demo sample", list(SAMPLES))
    s = SAMPLES[sample_name]
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        reading = {
            "machine_id": c1.text_input("Machine ID", s["machine_id"]),
            "machine_type": c1.selectbox("Machine type", ["L", "M", "H"], ["L", "M", "H"].index(s["machine_type"])),
            "air_temperature": c1.number_input("Air temperature (K)", 250.0, 350.0, s["air_temperature"]),
            "process_temperature": c2.number_input("Process temperature (K)", 250.0, 400.0, s["process_temperature"]),
            "rotational_speed": c2.number_input("Rotational speed (rpm)", 0.0, 5000.0, float(s["rotational_speed"])),
            "torque": c2.number_input("Torque (Nm)", 0.0, 150.0, s["torque"]),
            "tool_wear": c3.number_input("Tool wear (min)", 0.0, 400.0, float(s["tool_wear"])),
            "operating_hours": c3.number_input("Operating hours", 0.0, 200000.0, float(s["operating_hours"])),
            "maintenance_history_count": c3.number_input(
                "Past maintenance events", 0, 1000, s["maintenance_history_count"]
            ),
        }
        submitted = st.form_submit_button("Predict failure risk")
    if submitted:
        result, err = api("POST", "/predict", json=reading)
        if err:
            st.error(err)
        else:
            colour = {"low": "green", "medium": "orange", "high": "red"}[result["risk_category"]]
            st.markdown(f"### Risk: :{colour}[{result['risk_category'].upper()}]")
            st.metric("Failure probability (next 24h)", f"{result['failure_probability']:.1%}")
            st.progress(min(1.0, result["failure_probability"]))
            st.json(result)

with tab_monitor:
    summary, err = api("GET", "/monitoring/summary")
    if err:
        st.error(err)
    else:
        c = st.columns(4)
        c[0].metric("Predictions logged", summary["total_predictions"])
        c[1].metric("Predicted failures", summary["predicted_failures"])
        c[2].metric("Labelled outcomes", summary["labelled_outcomes"])
        c[3].metric("API errors", summary["api_errors"])

    st.subheader("Data drift (PSI vs training data)")
    drift, err = api("GET", "/monitoring/drift")
    if err:
        st.error(err)
    elif drift["status"] == "insufficient_data":
        st.info(f"Need at least {drift['required']} logged predictions for drift (have {drift['samples']}).")
    else:
        badge = {"stable": "✅ stable", "moderate": "⚠️ moderate drift", "significant": "🚨 significant drift"}
        st.write(f"Overall: **{badge[drift['status']]}** over {drift['samples']} recent predictions")
        st.dataframe(pd.DataFrame(drift["features"]).T, width="stretch")
        st.caption(drift["note"])

    st.subheader("Live performance (needs recorded outcomes)")
    perf, err = api("GET", "/monitoring/performance")
    if err:
        st.error(err)
    elif perf["status"] != "ok":
        st.info(f"Not enough labelled outcomes yet ({perf['labelled']}/{perf['required']}).")
    else:
        pm = perf["metrics"]
        st.write(f"Based on {perf['labelled']} outcomes (sources: {', '.join(perf['outcome_sources'])})")
        c = st.columns(3)
        c[0].metric("Live recall", f"{pm['recall']:.3f}")
        c[1].metric("Live precision", f"{pm['precision']:.3f}")
        c[2].metric("Live F1", f"{pm['f1']:.3f}")

    st.subheader("Recent predictions")
    recent, err = api("GET", "/predictions/recent", params={"limit": 200})
    if err:
        st.error(err)
    elif not recent:
        st.info("No predictions logged yet.")
    else:
        df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "features"}, **r["features"]} for r in recent])
        st.dataframe(df[["ts", "machine_id", "risk", "probability", "prediction", "actual",
                         "outcome_source", "model_version"]].head(50), width="stretch")
        feature = st.selectbox("Sensor distribution of logged readings", [
            "air_temperature", "process_temperature", "rotational_speed", "torque",
            "tool_wear", "operating_hours", "maintenance_history_count"])
        counts, edges = np.histogram(df[feature].astype(float), bins=20)
        st.bar_chart(pd.DataFrame({"count": counts}, index=[f"{e:.0f}" for e in edges[:-1]]))

with tab_retrain:
    st.subheader("Admin actions (need the API key in the sidebar)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**1. Simulate new readings + SIMULATED outcomes**")
        n = st.number_input("Readings", 10, 5000, 300)
        drift_level = st.slider("Sensor drift", 0.0, 3.0, 1.0, 0.5)
        if st.button("Simulate"):
            res, err = api("POST", "/admin/simulate", json={"n": int(n), "seed": 7, "drift": drift_level},
                           headers=admin_headers)
            if err:
                st.error(err)
            else:
                st.success(
                    f"Logged {res['simulated_readings']} SIMULATED readings with SIMULATED outcomes "
                    f"(drift {res['drift']}, {res['predicted_failures']} predicted failures)."
                )
    with c2:
        st.write("**2. Retrain (candidate is promoted only if it passes the checks)**")
        if st.button("Retrain now"):
            with st.spinner("Training and evaluating candidate..."):
                res, err = api("POST", "/admin/retrain", json={"reason": "dashboard", "wait": True},
                               headers=admin_headers)
            if err:
                st.error(err)
            else:
                message = f"{res['status'].upper()}: " + "; ".join(res.get("decision_reasons", []))
                if res["status"] == "promoted":
                    st.success(message)
                else:
                    st.warning(message)
                st.json(res)
        if st.button("Roll back to previous model"):
            res, err = api("POST", "/admin/rollback", headers=admin_headers)
            if err:
                st.error(err)
            else:
                st.success(f"Now serving {res['current']}")

    st.subheader("Retraining status")
    status, err = api("GET", "/retrain/status")
    if err:
        st.error(err)
    else:
        t = status["triggers"]
        st.write(f"**Retraining running:** {status['in_progress']} · "
                 f"**new labelled outcomes since last run:** {t['new_labels']} · "
                 f"**scheduler enabled:** {status['scheduler_enabled']}")
        if t["should_retrain"]:
            st.warning("Trigger conditions met: " + "; ".join(t["reasons"]))
        else:
            st.info("No retraining trigger condition is currently met.")
        if status["history"]:
            st.write("Retraining history")
            st.dataframe(pd.DataFrame(status["history"]), width="stretch")
