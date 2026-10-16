"""Extract provisional candidates from a small human-game batch; no tactical labels."""

import argparse
import json
import sys
from pathlib import Path

import chess.engine
import zstandard

from src.chessmind.ml.data.candidate_pipeline import run_candidate_pipeline
from src.chessmind.ml.data.candidate_schema import ExtractionConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "The output parent must be a new directory so JSONL and manifest publish together. "
        "Defaults: 10 games, depth 8. These are pilot candidates, not training labels."))
    parser.add_argument("--input", dest="input_path", type=Path,
                        default=Path("data/raw/ml/games/lichess_game_sample.pgn"))
    parser.add_argument("--output", dest="output_path", type=Path,
                        default=Path("data/processed/ml/candidate_moves.jsonl"))
    parser.add_argument("--source-manifest", dest="source_manifest_path", type=Path)
    parser.add_argument("--engine-path", type=Path)
    parser.add_argument("--max-games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--cp-threshold", type=float, default=100)
    parser.add_argument("--control-threshold", type=float, default=20)
    parser.add_argument("--max-controls-per-game", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-mate-candidates", action="store_true")
    args = vars(parser.parse_args())
    try:
        args["config"] = ExtractionConfig(args.pop("cp_threshold"), args.pop("control_threshold"),
            args.pop("max_controls_per_game"), args.pop("seed"), not args.pop("exclude_mate_candidates"))
        manifest = run_candidate_pipeline(**args, progress=lambda message: print(message, file=sys.stderr))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; no output bundle published.\n")
    except (OSError, ValueError, TypeError, chess.engine.EngineError, zstandard.ZstdError) as exc:
        parser.exit(1, f"Extraction failed; no output bundle published: {exc}\n")
    print(json.dumps({k: manifest[k] for k in (
        "games_analyzed", "games_failed", "total_moves_analyzed", "selection_counts",
        "mate_transition_counts", "elapsed_seconds", "average_seconds_per_successful_game",
        "cache", "output_sha256")}, indent=2))
    if not manifest["complete_requested_batch"]:
        parser.exit(2, "Partial batch published; inspect candidate_manifest.json for failures/shortage.\n")


if __name__ == "__main__":
    main()
