# Cloud Deployment Guide

> Nothing here has been deployed. Following these steps creates real cloud resources that can cost money. Check the provider's current pricing page first. Free tiers and credits change and are never guaranteed to last.

## 1. Recommended architecture for a student

**One small Linux virtual machine running the existing `docker-compose.yml`.**

```
Internet ──► VM (Ubuntu, Docker)
               ├─ api        :8000  (FastAPI)
               ├─ dashboard  :8501  (Streamlit)
               ├─ mlflow     :5000  (keep private)
               └─ Docker volumes: data, artifacts, logs, reports, mlruns  (on the VM disk)
```

Why a VM rather than serverless or container-app platforms:
- The app keeps state on disk (SQLite logs, model versions, MLflow DB). A VM disk persists. Many serverless platforms wipe the filesystem on restart.
- It reuses the exact Compose setup you tested locally, with no new code.
- One machine is easy to understand and cheap to stop.

Any provider works: AWS EC2 or Lightsail, Azure Virtual Machines, Google Compute Engine, DigitalOcean, Hetzner and others. Pick a size with at least **2 vCPU and 4 GB RAM** (MLflow plus scikit-learn training need the memory) and about 20 GB of disk.

## 2. Required environment variables

Create `.env` on the VM (never commit it):

| Variable | Value |
|---|---|
| `PM_ADMIN_API_KEY` | a long random string, e.g. from `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `PM_SCHEDULER_ENABLED` | `true` to let condition-based retraining run, `false` otherwise |
| `API_URL` | leave it unset; Compose already sets `http://api:8000` for the dashboard |
| `PM_DISABLE_MLFLOW` | `0` |

## 3. Step by step (Ubuntu VM)

```bash
# 1. Create the VM in your provider's console (Ubuntu 22.04/24.04 LTS, 2 vCPU / 4 GB).
#    Allow inbound SSH (22) from YOUR IP only, plus 8501 (and 8000 if the API must be public).
ssh ubuntu@<vm-public-ip>

# 2. Install Docker Engine + Compose plugin (official convenience script)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 3. Get the code and configure secrets
git clone https://github.com/VARSHITH707/MLops-project-.git
cd MLops-project-
cp .env.example .env
nano .env          # set PM_ADMIN_API_KEY

# 4. Build and start (init trains the first model automatically)
docker compose up --build -d
docker compose ps                      # all services "healthy" after ~1-2 min
curl http://localhost:8000/health
```

Then browse to `http://<vm-public-ip>:8501` (dashboard) and `http://<vm-public-ip>:8000/docs` (API).

## 4. Health checks
- Built into Compose: the API checks `/health`, the dashboard `/_stcore/health`, and MLflow `/health`. Unhealthy containers are restarted (`restart: unless-stopped`).
- External: add an uptime monitor (most providers offer one) that calls `http://<ip>:8000/health` and alerts when `model_loaded` is not `true`.
- Prometheus-style counters are at `/metrics` if you add Prometheus or Grafana later.

## 5. Persistent model and data storage
- Docker named volumes (`pm-data`, `pm-artifacts`, `pm-logs`, `pm-reports`, `pm-mlruns`) live on the VM disk and survive `docker compose down` and restarts. **`docker compose down -v` deletes them.**
- Back up regularly, e.g. take a disk snapshot in the provider console, or:
  ```bash
  docker run --rm -v mlops-project-_pm-artifacts:/src -v $PWD:/backup alpine tar czf /backup/artifacts.tgz -C /src .
  ```
  (Check exact volume names with `docker volume ls`.)
- For larger setups: move datasets to object storage with a DVC remote (`dvc remote add -d storage s3://...`).

## 6. MLflow persistence
MLflow uses SQLite in the `pm-mlruns` volume, so runs persist with the VM. **Do not expose port 5000 publicly.** The MLflow UI has no authentication. View it through an SSH tunnel:
```bash
ssh -L 5000:localhost:5000 ubuntu@<vm-public-ip>     # then open http://localhost:5000 locally
```
Upgrade path: a managed database (Postgres) as `--backend-store-uri` and object storage for artifacts, with `MLFLOW_TRACKING_URI` pointing to that server.

## 7. Secret configuration
- Secrets live only in `.env` on the VM (`chmod 600 .env`), or in the provider's secret manager injected as environment variables.
- `.env` is excluded by `.gitignore` and `.dockerignore`, so it never enters Git or the image.
- Rotate `PM_ADMIN_API_KEY` by editing `.env` and running `docker compose up -d` (this recreates the containers).

## 8. Securing the public endpoints (recommended)
- Firewall: open only what you need (8501 for the dashboard). Keep 5000 closed and SSH limited to your IP.
- Put a reverse proxy with HTTPS in front (e.g. Caddy, which obtains certificates automatically if you own a domain), and close 8000/8501 to the outside.
- The admin API key travels in a header: only send it over HTTPS on a public network.

## 9. Deployment and redeployment
```bash
cd MLops-project-
git pull
docker compose up --build -d          # rebuilds the image; volumes (models, logs) are kept
docker compose logs -f api            # watch startup
```
Model updates do **not** need redeployment. They happen through retraining or rollback inside the running system. The CI workflow already proves the image builds and passes its health check before you pull.

## 10. Limitations and cost notes
- A single VM is a single point of failure, and SQLite suits one host only.
- The VM costs money while it runs, even when idle. **Stop or delete it after the demo.** Disks, snapshots and static IPs can be billed even when the VM is stopped.
- Training runs on the VM's CPU. Large datasets would need a bigger machine or a separate training job.
- Opt-in only: there is no automatic cloud deployment in CI. Adding one would require cloud credentials stored as GitHub secrets and explicit approval.
