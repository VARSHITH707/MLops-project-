# Viva Questions and Answers

## A. Problem and machine learning basics

**1. What is predictive maintenance?**
Using equipment data to predict failures so repairs happen *before* a breakdown. It sits between reactive maintenance (fix after it breaks) and preventive maintenance (fix on a fixed schedule whether needed or not).

**2. What exactly does your model predict?**
The probability that a machine fails within the next 24 operating hours (`failure_within_horizon`), given one sensor reading. It's binary classification with a probability output.

**3. Why classification and not regression?**
The maintenance decision is yes/no (inspect or not). Predicting "remaining useful life" (regression) needs run-to-failure histories, which this setup doesn't simulate. The probability still gives a graded risk.

**4. How is the synthetic label generated?**
A hidden risk score adds penalties for tool wear above 170 minutes, power above 8.5 kW or below 3.8 kW, a process–air temperature gap below 9 K, age, maintenance history and low machine quality, plus random noise. A sigmoid turns it into a probability, and a Bernoulli draw produces the 0/1 label. The randomness means the classes overlap, as in reality.

**5. What are the limits of synthetic data?**
The model can only learn the relationships I wrote into the generator, so high metrics prove the pipeline works, not that it would predict real machines. Real data adds sensor faults, label delays, machine-specific behaviour and time dependence.

**6. Why two models?**
Logistic Regression is a simple, explainable baseline. Random Forest captures non-linear effects such as "risk only rises *after* wear exceeds a level". The baseline proves the complex model earns its complexity: F1 0.690 vs 0.493.

**7. How do you prevent target leakage?**
`machine_id`, `timestamp` and the target are excluded from features. Scaling and encoding are fitted only on the training split, inside a Pipeline. There is no post-failure information in the inputs.

## B. Metrics

**8. Define precision, recall and F1.**
Precision = TP / (TP + FP): of the machines flagged, how many really fail (0.580). Recall = TP / (TP + FN): of real failures, how many we catch (0.852). F1 = 2·P·R / (P + R), the harmonic mean (0.690).

**9. Read your confusion matrix.**
On 1,600 test rows: 1,196 true negatives, 154 false positives (unneeded inspections), 37 false negatives (missed breakdowns), 213 true positives.

**10. Why not just use accuracy?**
Only about 16% of rows are failures. Predicting "never fails" gives about 84% accuracy and zero value. Accuracy hides the rare class that matters.

**11. Why does a false negative matter more here?**
A missed failure means an unplanned stoppage, expensive emergency repair and possible injury. A false alarm costs one inspection. So we tune for high recall.

**12. Why is the threshold 0.35?**
Lowering it below 0.5 flags more machines and raises recall at the cost of precision, which suits the cost asymmetry. It's configurable in `params.yaml` and stored in each model's metadata, so every prediction uses the threshold of the model that made it.

**13. What is ROC-AUC and why report it?**
The probability that a random failure gets a higher score than a random non-failure (0.935 here). It is threshold-independent, so it measures ranking quality separately from the threshold choice.

## C. Data validation and features

**14. What does your validation check?**
Required columns, numeric types, missing values, infinite values, physical ranges, unknown machine types, parseable timestamps, duplicate (machine, timestamp) readings, a binary target, and empty datasets. API requests are checked separately by Pydantic (types, ranges, allowed categories, no extra fields, no NaN or infinity).

**15. Why reject instead of filling bad values?**
Silently imputing a torque of −5 hides a sensor or software fault and produces a confident but wrong prediction. Rejecting makes the problem visible.

**16. Which features did you engineer and why?**
Temperature difference (cooling), mechanical power in kW (overload or stall), and wear × torque (a worn tool under load). They encode physical intuition, so the model needs fewer examples to learn.

**17. Why is a random split a limitation?**
Sensor data is a time series. A random split can put later readings of a machine in training and earlier ones in testing, which slightly leaks the future. Production would use a time-based split.

## D. MLflow, DVC and versioning

**18. What does MLflow store in your project?**
For each run: model type, hyperparameters, threshold, training-set size, metrics (precision, recall, F1, ROC-AUC, accuracy, training seconds), tags (dataset fingerprint, model version, selected) and, for the chosen model, the model file and metadata. The backend is SQLite at `mlruns/mlflow.db`.

**19. What does DVC do that Git doesn't?**
It versions large data files by hash and defines the pipeline as stages with dependencies, parameters and outputs. `dvc repro` reruns only what changed, and `dvc.lock` pins the exact hashes. There is no remote configured yet, so the cache is local.

**20. How do you know which data a model was trained on?**
Each model's `metadata.json` and MLflow run carry a dataset fingerprint (a SHA-256 hash of the training data's content), and `dvc.lock` records the data file hashes.

**21. How are models versioned?**
Each version is a folder `artifacts/models/v<timestamp>-<id>/` with `model.joblib` and `metadata.json`. `artifacts/current.json` points to the served version and the previous one. It is replaced atomically with `os.replace`.

## E. Serving: FastAPI, Streamlit, Docker

**22. Why FastAPI?**
Type-checked request validation through Pydantic, automatic Swagger documentation, high performance and simple code.

**23. What happens if no model exists?**
`/health` reports `degraded`, and `/predict` and `/model-info` return HTTP 503 with an instruction to run `scripts/setup.py`. The service never invents a prediction.

**24. How are admin actions protected?**
They require the `X-API-Key` header to equal the `PM_ADMIN_API_KEY` environment variable (constant-time comparison). If the variable is unset, admin routes are disabled (503). No secret is in the code.

**25. Why does the dashboard call the API instead of loading the model?**
Single source of truth: the dashboard shows exactly what the production service returns, and there is no second model that could drift out of sync.

**26. What does Docker Compose start?**
`init` (one-shot: data plus first model if missing), `api`, `dashboard` and `mlflow`, with named volumes for data, models, logs, reports and MLflow, health checks, a non-root user, and the dashboard reaching the API at `http://api:8000`.

## F. Monitoring, drift and retraining

**27. What is data drift and how do you measure it?**
A change in the input distribution compared with training. I use the Population Stability Index per feature: bins from training quantiles, then Σ (cur% − ref%) · ln(cur% / ref%). Below 0.1 is stable, 0.1–0.25 moderate, 0.25 or more significant.

**28. Does drift mean the model is wrong?**
No. Inputs can shift in ways the model still handles. Drift is a warning that triggers investigation. Proof of degradation requires labelled outcomes, which is why performance monitoring is separate and drift alone never promotes a model.

**29. When does retraining trigger?**
Manually (button or API), or, with the scheduler on, when 200 or more new labelled outcomes exist since the last run, or when live F1 is 0.10 or more below the training F1.

**30. How do you decide whether to promote the new model?**
Both models are evaluated on the same evaluation set (original test set plus 30% of the new labelled data). The candidate must reach F1 ≥ 0.50, F1 ≥ the current F1, and recall no more than 0.02 below the current recall. Otherwise it is rejected.

**31. Why not always deploy the newest model?**
Newer isn't better: new data can be noisy, mislabelled or unrepresentative. In my demo run a candidate was correctly **rejected** (F1 0.727 vs current 0.732).

**32. How does rollback work?**
`POST /admin/rollback` promotes the previous version recorded in `current.json`. Old versions are never deleted, so it is instant, and the API reloads automatically on the next request.

**33. What stops two retrainings running at once?**
A lock file created with `O_CREAT | O_EXCL`, an atomic "create only if absent". A second attempt gets HTTP 409. The lock is released in a `finally` block, and a lock older than 30 minutes is considered left over from a crash and reclaimed.

**34. What if the API runs with several workers and the scheduler on?**
Only the worker that obtains `.scheduler.lock` starts the scheduler. It refreshes the lock's timestamp every cycle, and other workers back off. A test checks the second start returns False.

## G. Failure scenarios

**35. A sensor sends `torque = -40`. What happens?**
The API returns 422 naming `torque`; the error is counted in `api_errors`; no prediction is logged.

**36. Retraining crashes halfway (e.g. disk full).**
The exception is caught, the report says `failed` with the reason, the lock is released, and `current.json` was never touched, so the old model keeps serving. A test covers this.

**37. New labelled data contains corrupt rows.**
Validation fails, the run ends `failed_validation`, nothing is trained or promoted, and the lock is released.

**38. Someone deploys a bad model by mistake.**
Roll back with one call, then use `/model-info` to see the promotion history with reasons and timestamps.

**39. MLflow's database is locked or missing.**
Training logs a warning and continues. Tracking is best-effort and never blocks the pipeline.

**40. CI is red after a push.**
Open the Actions log. Each step (lint, tests, pipeline, Docker) is separate, so the failing stage is obvious. Reproduce locally with the same command (`ruff check .`, `pytest -q`).
