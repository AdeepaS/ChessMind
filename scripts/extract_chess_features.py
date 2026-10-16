"""Generate lightweight chess features from candidate JSONL; no Stockfish or labels."""

import argparse
import json
from pathlib import Path

from src.chessmind.ml.features.pipeline import generate_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path,
                        default=Path("data/processed/ml/candidate_moves.jsonl"))
    parser.add_argument("--output", dest="output_path", type=Path,
                        default=Path("data/processed/ml/candidate_features.jsonl"))
    parser.add_argument("--source-manifest", dest="source_manifest_path", type=Path)
    parser.add_argument("--permissive", action="store_true", help="count and skip invalid records; default is strict")
    parser.add_argument("--max-line-chars", type=int, default=2_000_000)
    options = vars(parser.parse_args())
    options["strict"] = not options.pop("permissive")
    try:
        manifest = generate_features(**options)
    except KeyboardInterrupt:
        parser.exit(130, "Feature extraction interrupted; no completed output pair published.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Feature extraction failed: {exc}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
