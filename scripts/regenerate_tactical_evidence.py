"""Regenerate the historical 24-position cohort into a new, validated v2 bundle."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chessmind.ml.labeling.regeneration import regenerate_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(candidate_path="data/processed/ml/candidate_moves.jsonl",
                    feature_path="data/processed/ml/candidate_features.jsonl",
                    evidence_path="data/labeled/ml/mistake_label_proposals.jsonl",
                    source_dataset_path="data/raw/ml/games/lichess_game_sample.pgn",
                    source_manifest_path="data/raw/ml/games/source_manifest.json",
                    output_dir="data/labeled/c3_v2", engine_path="tools/stockfish/stockfish.exe")
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=Path(default))
    args = parser.parse_args(argv)
    try:
        audit = regenerate_evidence(**vars(args), progress=lambda message: print(message, file=sys.stderr))
    except KeyboardInterrupt:
        print("Regeneration interrupted; no accepted C3 bundle published.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Regeneration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(audit, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
