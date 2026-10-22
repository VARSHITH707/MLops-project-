import json

import numpy as np
import pandas as pd

from src.models import registry
from src.models.evaluate import compute_metrics


def test_compute_metrics_known_values():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == 0.75


def test_roc_auc_none_with_single_class():
    assert compute_metrics([0, 0], [0.2, 0.3], 0.5)["roc_auc"] is None


def test_training_produces_artifacts_and_metadata(trained):
    s = trained
    version = registry.current_version(s.artifacts_dir)
    assert version is not None
    folder = s.artifacts_dir / "models" / version
    assert (folder / "model.joblib").exists()
    meta = json.loads((folder / "metadata.json").read_text())
    for key in ("model_version", "trained_at", "dataset_fingerprint", "feature_names",
                "random_seed", "metrics", "threshold", "model_type"):
        assert key in meta
    assert meta["metrics"]["f1"] > 0.5
    assert (s.reports_dir / "metrics.json").exists()
    assert (s.reports_dir / "training_report.json").exists()


def test_loaded_model_gives_valid_probabilities(trained):
    model, _ = registry.load_current(trained.artifacts_dir)
    proba = model.predict_proba(pd.read_csv(trained.test_csv))[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_script_missing_data_exit_code(settings):
    from scripts import train_model

    assert train_model.main([]) == 2
