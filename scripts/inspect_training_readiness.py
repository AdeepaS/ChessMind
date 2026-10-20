"""Print a read-only readiness report; exit 0 valid, 2 invalid input, 1 unexpected failure."""

import argparse
import json
import sys
from pathlib import Path

from src.chessmind.ml.data.dataset_readiness import inspect_training_readiness, report_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "candidate_path": "data/processed/ml/candidate_moves.jsonl",
        "feature_path": "data/processed/ml/candidate_features.jsonl",
        "evidence_path": "data/labeled/ml/mistake_label_proposals.jsonl",
        "source_dataset_path": "data/raw/ml/games/lichess_game_sample.pgn",
        "source_manifest_path": "data/raw/ml/games/source_manifest.json",
        "candidate_manifest_path": None, "feature_manifest_path": None,
        "labeling_manifest_path": None, "adjudications_path": None, "approvals_path": None,
    }
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=default)
    args = parser.parse_args(argv)
    try:
        report = inspect_training_readiness(**vars(args))
        print(report_json(report), end="")
        return 0 if report["integrity_valid"] else 2
    except Exception as exc:
        print(json.dumps({"error": "unexpected_failure", "message": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
