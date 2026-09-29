"""Run the full project pipeline: training, evaluation, and artifact verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_models import main as evaluate_main
from scripts.train_models import main as train_main


def main() -> None:
    """Train the models then evaluate the saved artifacts."""
    train_main()
    evaluate_main()


if __name__ == "__main__":
    main()
