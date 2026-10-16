"""Propose narrow, provisional chess concepts; default pilot is 24 candidates at depth 12."""

import argparse
import json
import sys
from pathlib import Path

import chess.engine

from src.chessmind.ml.labeling.pipeline import run_labeling_pipeline
from src.chessmind.ml.labeling.schema import LabelingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Output directory must be new. All proposals require expert review.")
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/ml/candidate_moves.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/ml/candidate_features.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/labeled/ml"))
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--engine-path", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--permissive", action="store_true")
    mode.add_argument("--strict", action="store_true", help="default: abort on input validation errors")
    parser.add_argument("--skip-verification", action="store_true", help="explicit abstention-only run")
    args = parser.parse_args()
    try:
        config = LabelingConfig(args.max_candidates, args.depth, not args.skip_verification,
                                args.threads, args.hash_mb, not args.permissive)
        manifest = run_labeling_pipeline(args.candidates, args.features, args.output_dir, config=config,
            engine_path=args.engine_path, progress=lambda message: print(message, file=sys.stderr))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; no output bundle published.\n")
    except (ValueError, TypeError, OSError, chess.engine.EngineError) as exc:
        parser.exit(1, f"Labeling failed; no output bundle published: {exc}\n")
    print(json.dumps(manifest, indent=2))
    if manifest["verification_failures"] or manifest["input_diagnostics"]:
        parser.exit(2, "Published explicit failures/abstentions; inspect the manifest.\n")


if __name__ == "__main__":
    main()
