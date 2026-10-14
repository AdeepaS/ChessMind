"""Descriptive statistics for source puzzle annotations, never mistake labels."""

import csv
import hashlib
import json
from bisect import bisect_right
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

from src.chessmind.ml.data.puzzle_loader import PuzzleRecord, open_puzzles


DEFAULT_RATING_BUCKETS = (0, 500, 1000, 1500, 2000, 2500, 3000)


def numeric_summary(values: Iterable[int]) -> dict:
    """Exact median and population standard deviation; null summaries if empty."""
    values = list(values)
    if any(type(value) is not int for value in values):
        raise ValueError("Metadata values must be integers")
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "population_stddev": pstdev(values) if values else None,
    }


def _bucket_edges(edges: Iterable[int]) -> tuple[int, ...]:
    edges = tuple(edges)
    if (not edges or any(type(edge) is not int for edge in edges)
            or edges[0] != 0 or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError("Rating bucket lower bounds must start at 0 and strictly increase")
    return edges


def _percentage(count: int, total: int) -> float:
    return round(100 * count / total, 6) if total else 0.0


def rating_distribution(ratings: Iterable[int], bucket_edges=DEFAULT_RATING_BUCKETS) -> list[dict]:
    """Left-inclusive, right-exclusive bins; the last bin has no upper bound."""
    edges = _bucket_edges(bucket_edges)
    counts = [0] * len(edges)
    for rating in ratings:
        if type(rating) is not int or rating < 0:
            raise ValueError("Ratings must be nonnegative integers")
        counts[bisect_right(edges, rating) - 1] += 1
    total = sum(counts)
    return [
        {"lower_inclusive": lower,
         "upper_exclusive": edges[index + 1] if index + 1 < len(edges) else None,
         "count": counts[index], "percentage": _percentage(counts[index], total)}
        for index, lower in enumerate(edges)
    ]


def _labels(labels, name):
    if not isinstance(labels, (tuple, list)) or any(
            not isinstance(label, str) or not label or any(c.isspace() for c in label)
            for label in labels):
        raise ValueError(f"{name} must be a sequence of nonempty theme/tag identifiers")
    return set(labels)


def explore_puzzles(puzzles: Iterable[PuzzleRecord], *, bucket_edges=DEFAULT_RATING_BUCKETS,
                    rare_threshold: int = 5) -> dict:
    """Consume records once, retaining numeric columns and IDs, not full records.

    Every percentage uses all accepted records as its denominator. Each theme
    and unordered distinct pair counts at most once per puzzle. All ties sort
    lexicographically. Rare means count <= rare_threshold. Direct callers may
    supply duplicate IDs: counts describe all supplied rows and unique count
    explicitly distinguishes them. File loading rejects duplicates upstream.
    Board/move validation belongs to puzzle_loader; this validates consumed
    metadata. Exact medians require O(N) numeric storage.
    """
    edges = _bucket_edges(bucket_edges)
    if type(rare_threshold) is not int or rare_threshold < 1:
        raise ValueError("rare_threshold must be a positive integer")
    ratings, popularity, plays, theme_sizes = [], [], [], []
    ids, themes, pairs = set(), Counter(), Counter()
    missing = dict(themes_empty=0, opening_tags_empty=0,
                   daily_date_empty=0, daily_date_column_absent=0)
    duplicate_theme_tokens = 0
    for puzzle in puzzles:
        if not isinstance(puzzle, PuzzleRecord):
            raise TypeError("Expected PuzzleRecord instances from puzzle_loader")
        for name in ("rating", "popularity", "number_of_plays"):
            value = getattr(puzzle, name)
            if type(value) is not int or (not -100 <= value <= 100 if name == "popularity" else value < 0):
                raise ValueError(f"Invalid {name}: {value!r}")
        if not isinstance(puzzle.puzzle_id, str) or not puzzle.puzzle_id.strip():
            raise ValueError("Puzzle ID must be a nonempty string")
        labels = _labels(puzzle.themes, "themes")
        openings = _labels(puzzle.opening_tags, "opening_tags")
        ids.add(puzzle.puzzle_id)
        ratings.append(puzzle.rating)
        popularity.append(puzzle.popularity)
        plays.append(puzzle.number_of_plays)
        theme_sizes.append(len(labels))
        duplicate_theme_tokens += len(puzzle.themes) - len(labels)
        themes.update(labels)
        pairs.update(combinations(sorted(labels), 2))
        missing["themes_empty"] += not labels
        missing["opening_tags_empty"] += not openings
        raw = dict(puzzle.source_fields)
        missing["daily_date_column_absent"] += "DailyDate" not in raw
        missing["daily_date_empty"] += "DailyDate" in raw and not raw["DailyDate"].strip()
    total = len(ratings)
    theme_rows = [
        {"theme": theme, "count": count, "percentage": _percentage(count, total)}
        for theme, count in sorted(themes.items(), key=lambda item: (-item[1], item[0]))
    ]
    pair_rows = [
        {"theme_a": a, "theme_b": b, "count": count, "percentage": _percentage(count, total)}
        for (a, b), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "general": {"total_puzzles": total, "unique_puzzles": len(ids),
                    "duplicate_record_ids": total - len(ids), "ratings": numeric_summary(ratings)},
        "rating_distribution": rating_distribution(ratings, edges),
        "themes": theme_rows,
        "top_themes": theme_rows[:15],
        "rare_threshold": rare_threshold,
        "rare_themes": [row for row in theme_rows if row["count"] <= rare_threshold],
        "distinct_themes": len(themes),
        "total_theme_assignments": sum(themes.values()),
        "duplicate_theme_tokens": duplicate_theme_tokens,
        "themes_per_puzzle": numeric_summary(theme_sizes),
        "theme_count_distribution": [
            {"theme_count": size, "count": count, "percentage": _percentage(count, total)}
            for size, count in sorted(Counter(theme_sizes).items())
        ],
        "theme_cooccurrence": pair_rows,
        "metadata": {"popularity": numeric_summary(popularity),
                     "number_of_plays": numeric_summary(plays), "missing_optional": missing},
    }


def analyze_dataset(path: str | Path, *, manifest_path: str | Path | None = None,
                    strict: bool = False, bucket_edges=DEFAULT_RATING_BUCKETS,
                    rare_threshold: int = 5) -> dict:
    """Reuse loader validation; optionally verify source provenance before analysis."""
    path = Path(path)
    with path.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = None
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("sha256") != sha256:
            raise ValueError("Source manifest SHA-256 does not match the input dataset")
    with open_puzzles(path, strict=strict) as puzzles:
        result = explore_puzzles(puzzles, bucket_edges=bucket_edges, rare_threshold=rare_threshold)
        stats = puzzles.stats
        result["validation"] = {
            "processed": stats.processed, "accepted": stats.accepted,
            "rejected": stats.rejected, "rejection_counts": dict(stats.rejection_counts),
            "error_examples": list(stats.error_examples),
            "mode": "strict" if strict else "permissive",
        }
    result["source"] = {
        "input_file": path.as_posix(), "sha256": sha256, "manifest": manifest,
        "manifest_hash_verified": manifest is not None,
        "manifest_count_matches": (
            manifest.get("sample_count") == result["general"]["total_puzzles"] if manifest is not None else None
        ),
    }
    return result


def write_reports(analysis: dict, output_dir: str | Path) -> None:
    """Write stable CSV/JSON bytes with no generation timestamp or platform newline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = (
        ("theme_counts.csv", "themes", ("theme", "count", "percentage")),
        ("rating_distribution.csv", "rating_distribution",
         ("lower_inclusive", "upper_exclusive", "count", "percentage")),
        ("theme_cooccurrence.csv", "theme_cooccurrence", ("theme_a", "theme_b", "count", "percentage")),
    )
    for filename, key, columns in tables:
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(analysis[key])
    with (output_dir / "dataset_quality.json").open("w", encoding="utf-8", newline="\n") as target:
        json.dump(analysis, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
