"""First-run helper: generate data -> validate -> split -> train + promote.

    python scripts/setup.py              # skip if a model is already served
    python scripts/setup.py --force      # rebuild data and train again
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from scripts import train_model  # noqa: E402
from src.data import generate_data, validate_data  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import registry  # noqa: E402

log = logging.getLogger("setup")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()

    if registry.current_version(settings.artifacts_dir) and not args.force:
        log.info("Model %s already served - nothing to do (use --force to rebuild)",
                 registry.current_version(settings.artifacts_dir))
        return 0

    for step, func in (
        ("generate data", lambda: generate_data.main([])),
        ("validate data", lambda: validate_data.main([])),
        ("split data", lambda: build_features.main([])),
        ("train model", lambda: train_model.main(["--promote"] if args.force else [])),
    ):
        log.info("== %s ==", step)
        code = func()
        if code != 0:
            log.error("Step '%s' failed with exit code %s", step, code)
            return code
    log.info("Setup complete. Start the API with: python -m uvicorn app.main:app --port 8000")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
