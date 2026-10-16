"""Inspect human games without engine analysis; python -m scripts.inspect_lichess_games."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.chessmind.ml.data.game_dataset_loader import open_games


def inspect_games(path, *, strict=False, filters=None, manifest_path=None):
    players, ratings, controls, speeds, results, missing, groups = (Counter() for _ in range(7))
    lengths, averages = [], []
    annotated = clocks = 0
    with open_games(path, strict=strict, filters=filters) as reader:
        for game in reader:
            players.update(game.player_keys)
            ratings.update((game.white_rating, game.black_rating))
            controls[game.time_control] += 1
            speeds[game.time_class] += 1
            results[game.result] += 1
            lengths.append(game.move_count)
            averages.append(game.average_rating)
            group = next((f"{lo}-{hi - 1}" for lo, hi in
                          ((800, 1200), (1200, 1800), (1800, 2400))
                          if lo <= game.average_rating < hi), "outside_target_ranges")
            groups[group] += 1
            for name in ("utc_date", "utc_time", "opening", "eco"):
                missing[name] += getattr(game, name) is None
            annotated += "[%eval " in game.original_pgn
            clocks += "[%clk " in game.original_pgn
        stats = reader.stats
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "accepted_games": stats.accepted, "unique_game_ids": stats.accepted,
        "unique_players": len(players), "games_per_player": dict(sorted(players.items())),
        "individual_rating_histogram": dict(sorted(ratings.items())),
        "average_rating_groups": dict(groups),
        "average_ratings": {"min": min(averages, default=None),
                            "max": max(averages, default=None),
                            "mean": mean(averages) if averages else None},
        "time_controls": dict(controls), "time_classes": dict(speeds),
        "mean_plies": mean(lengths) if lengths else None,
        "median_plies": median(lengths) if lengths else None,
        "results": dict(results), "missing_metadata": dict(missing),
        "games_with_eval_annotations": annotated, "games_with_clock_annotations": clocks,
        "processed_games": stats.processed, "rejected_games": stats.rejected,
        "rejection_counts": dict(stats.rejection_counts), "sha256": digest,
        "model_limitation": "Few repeat players cannot establish recurring weaknesses. "
        "Acquire chronological multi-game histories and separate players across train/test sets."
    }
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        report["manifest_verified"] = (manifest["sha256"] == digest and
            manifest["accepted_count"] == stats.accepted and not stats.rejected and
            all(sum(1 for r in averages if g["low"] <= r < g["high"]) == g["accepted"]
                for g in manifest["rating_groups"]))
        if not report["manifest_verified"]:
            raise ValueError("sample count, rating groups or SHA-256 disagree with manifest")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = json.dumps(inspect_games(**args), indent=2) + "\n"
    if output:
        output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
