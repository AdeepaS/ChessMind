"""Validate a local puzzle sample and print streaming summary statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import zstandard

from src.chessmind.ml.data.puzzle_loader import open_puzzles


def inspect_puzzles(path: str | Path, *, strict: bool = False,
                    max_records: int | None = None) -> dict:
    themes = Counter()
    sample_ids = []
    rating_sum = 0
    rating_min = rating_max = None
    with open_puzzles(path, strict=strict, max_records=max_records) as puzzles:
        for puzzle in puzzles:
            if len(sample_ids) < 5:
                sample_ids.append(puzzle.puzzle_id)
            themes.update(dict.fromkeys(puzzle.themes, 1))
            rating_sum += puzzle.rating
            rating_min = puzzle.rating if rating_min is None else min(rating_min, puzzle.rating)
            rating_max = puzzle.rating if rating_max is None else max(rating_max, puzzle.rating)
        stats = puzzles.stats
    return {
        "processed_rows": stats.processed,
        "accepted_rows": stats.accepted,
        "rejected_rows": stats.rejected,
        "unique_puzzles": stats.unique_puzzles,
        "sample_ids": sample_ids,
        "common_themes": dict(themes.most_common(10)),
        "ratings": {"min": rating_min, "max": rating_max,
                    "mean": rating_sum / stats.accepted if stats.accepted else None},
        "rejection_counts": dict(stats.rejection_counts),
        "error_examples": stats.error_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    try:
        result = inspect_puzzles(**vars(args))
    except (OSError, ValueError, zstandard.ZstdError) as exc:
        parser.exit(1, f"Inspection failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
