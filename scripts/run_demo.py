"""Scripted end-to-end demo against a RUNNING API (all new data is SIMULATED).

    $env:PM_ADMIN_API_KEY = "demo-key"   # same key the API was started with
    python scripts/run_demo.py

Steps: health -> predictions -> simulate drifted readings + outcomes -> drift ->
live performance -> retrain (candidate vs current) -> show the decision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

SAMPLES = {
    "healthy": {"machine_id": "M-001", "machine_type": "H", "air_temperature": 299.8,
                "process_temperature": 310.1, "rotational_speed": 1520, "torque": 38.5,
                "tool_wear": 40, "operating_hours": 2500, "maintenance_history_count": 0},
    "worn tool": {"machine_id": "M-007", "machine_type": "L", "air_temperature": 302.5,
                  "process_temperature": 310.9, "rotational_speed": 1380, "torque": 58.4,
                  "tool_wear": 215, "operating_hours": 15200, "maintenance_history_count": 5},
}


def call(method: str, url: str, **kwargs) -> dict:
    resp = requests.request(method, url, timeout=300, **kwargs)
    if resp.status_code >= 400:
        print(f"  ! {method} {url} -> {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("PM_ADMIN_API_KEY"))
    parser.add_argument("--drift", type=float, default=1.5)
    args = parser.parse_args()
    api = args.api.rstrip("/")
    if not args.api_key:
        print("Set PM_ADMIN_API_KEY (or pass --api-key) to the key the API was started with.")
        return 2
    admin = {"X-API-Key": args.api_key}

    try:
        health = call("GET", f"{api}/health")
    except requests.RequestException:
        print(f"API not reachable at {api}. Start it first: python -m uvicorn app.main:app --port 8000")
        return 2
    print(f"1. Health: {health['status']} - serving {health['model_version']}")

    print("2. Predictions:")
    for name, reading in SAMPLES.items():
        r = call("POST", f"{api}/predict", json=reading)
        print(f"   {name:10s} -> probability {r['failure_probability']:.3f}, risk {r['risk_category'].upper()}")

    sim = call("POST", f"{api}/admin/simulate", json={"n": 300, "seed": 11, "drift": args.drift}, headers=admin)
    print(f"3. Simulated {sim['simulated_readings']} drifted readings (drift={args.drift}) with SIMULATED outcomes")

    drift = call("GET", f"{api}/monitoring/drift")
    print(f"4. Drift status: {drift['status']} - drifted features: {drift.get('drifted_features')}")

    perf = call("GET", f"{api}/monitoring/performance")
    if perf["status"] == "ok":
        m = perf["metrics"]
        print(f"5. Live performance on {perf['labelled']} outcomes: recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
    else:
        print(f"5. Live performance: {perf['status']}")

    print("6. Retraining (candidate is compared with the current model on the same data)...")
    rep = call("POST", f"{api}/admin/retrain", json={"reason": "run_demo", "wait": True}, headers=admin)
    cur, cand = rep.get("current_metrics") or {}, rep.get("candidate_metrics") or {}
    print(f"   current   {rep.get('current_version')}: F1 {cur.get('f1', float('nan')):.3f} "
          f"recall {cur.get('recall', float('nan')):.3f}")
    print(f"   candidate {rep.get('candidate_version')}: F1 {cand.get('f1', float('nan')):.3f} "
          f"recall {cand.get('recall', float('nan')):.3f}")
    print(f"   decision: {rep['status'].upper()} - {'; '.join(rep['decision_reasons'])}")
    print(f"7. Now serving: {call('GET', f'{api}/health')['model_version']}")
    print("\nFull report:", json.dumps({k: rep[k] for k in ('run_id', 'report_path')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
