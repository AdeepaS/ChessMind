"""Acquire a bounded, stratified prefix of an official Lichess rated archive."""

import argparse
import io
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import zstandard

from scripts.inspect_lichess_games import inspect_games
from src.chessmind.ml.data.game_dataset_loader import GameFilters, GamePGNReader


DEFAULT_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
DEFAULT_GROUPS = ((800, 1200, 150), (1200, 1800, 200), (1800, 2400, 150))


class BudgetReached(Exception):
    pass


class BoundedSource:
    def __init__(self, source, max_bytes, deadline):
        self.source, self.max_bytes, self.deadline = source, max_bytes, deadline
        self.bytes_read = 0

    def read(self, size):
        if time.monotonic() >= self.deadline:
            raise BudgetReached("time_budget")
        if self.bytes_read >= self.max_bytes:
            raise BudgetReached("compressed_byte_budget")
        data = self.source.read(min(size, self.max_bytes - self.bytes_read))
        self.bytes_read += len(data)
        return data


def acquire_sample(output_dir=Path("data/raw/ml/games"), *, source_url=DEFAULT_URL,
                   groups=DEFAULT_GROUPS, filters=None, unique_players=True,
                   max_games=100_000, max_bytes=64 * 1024 * 1024,
                   timeout=20, total_timeout=300):
    if not re.fullmatch(r"https://database\.lichess\.org/standard/"
                        r"lichess_db_standard_rated_[0-9]{4}-[0-9]{2}\.pgn\.zst", source_url):
        raise ValueError("source must be an official standard rated monthly archive URL")
    groups = tuple(tuple(g) for g in groups)
    if not groups or any(len(g) != 3 or any(type(v) is not int for v in g) or
                         not 1 <= g[0] < g[1] <= 4001 or g[2] <= 0 for g in groups):
        raise ValueError("groups require integer LOW:HIGH_EXCLUSIVE:COUNT")
    ordered = sorted(groups)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("rating groups must not overlap")
    if any(type(v) is not int or v <= 0 for v in (max_games, max_bytes)):
        raise ValueError("resource budgets must be positive integers")
    if not all(math.isfinite(v) and v > 0 for v in (timeout, total_timeout)):
        raise ValueError("timeouts must be positive finite numbers")
    filters = filters or GameFilters(min_plies=20)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts = [0] * len(groups)
    selected_players, selection_counts, references = set(), Counter(), []
    requested = sum(g[2] for g in groups)
    deadline = time.monotonic() + total_timeout
    reason = "source_end_or_game_budget"
    request = Request(source_url, headers={"User-Agent": "ChessMind-human-game-sample",
                                          "Accept-Encoding": "identity"})
    with tempfile.TemporaryDirectory(prefix=".games-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        sample = stage / "lichess_game_sample.pgn"
        with urlopen(request, timeout=min(timeout, total_timeout)) as response:
            if response.status != 200 or response.geturl() != source_url:
                raise ValueError("unexpected archive response or redirect")
            download = BoundedSource(response, max_bytes, deadline)
            with zstandard.ZstdDecompressor(max_window_size=128 * 1024 * 1024).stream_reader(
                    download, read_size=16384) as decoded:
                with io.TextIOWrapper(decoded, encoding="utf-8-sig") as text:
                    reader = GamePGNReader(text, strict=False, max_games=max_games,
                                           source_archive=source_url, filters=filters)
                    with sample.open("w", encoding="utf-8", newline="\n") as target:
                        try:
                            for game in reader:
                                if time.monotonic() >= deadline:
                                    raise BudgetReached("time_budget")
                                group = next((i for i, (lo, hi, _) in enumerate(groups)
                                              if lo <= game.average_rating < hi), None)
                                if group is None:
                                    selection_counts["outside_rating_groups"] += 1
                                elif counts[group] == groups[group][2]:
                                    selection_counts["rating_group_full"] += 1
                                elif unique_players and selected_players.intersection(game.player_keys):
                                    selection_counts["repeat_player"] += 1
                                else:
                                    target.write(game.original_pgn.rstrip() + "\n\n")
                                    counts[group] += 1
                                    selected_players.update(game.player_keys)
                                    references.append({"game_id": game.game_id,
                                                       "archive_game_index": game.source_index})
                                if sum(counts) == requested:
                                    reason = "target_reached"
                                    break
                        except BudgetReached as exc:
                            reason = str(exc)
        # Reparse the closed file before publishing any artifact.
        report = inspect_games(sample, strict=True, filters=filters)
        if report["accepted_games"] != sum(counts):
            raise ValueError("saved sample failed count validation")
        manifest = {
            "source_url": source_url, "source_page": "https://database.lichess.org/",
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_file": sample.name, "requested_count": requested,
            "accepted_count": sum(counts), "complete": sum(counts) == requested,
            "stop_reason": reason,
            "rating_groups": [{"low": lo, "high": hi, "requested": n,
                               "accepted": counts[i], "shortage": n - counts[i]}
                              for i, (lo, hi, n) in enumerate(groups)],
            "filters": vars(filters), "unique_players_required": unique_players,
            "processed_games": reader.stats.processed,
            "eligible_games": reader.stats.accepted,
            "rejected_games": reader.stats.rejected,
            "rejection_counts": dict(reader.stats.rejection_counts),
            "selection_filter_counts": dict(selection_counts),
            "compressed_bytes_read": download.bytes_read,
            "budgets": {"compressed_bytes": max_bytes, "games": max_games,
                        "seconds": total_timeout, "socket_timeout": timeout,
                        "game_characters": reader.max_game_chars},
            "selection_method": "First eligible games per average-rating stratum in archive order",
            "limitations": ["Prefix ordering bias; not representative of the Lichess population.",
                            "BOT tags exclude declared bots only; human identity is not verified.",
                            "Player-disjoint selection limits recurring-player evidence.",
                            "The downloaded compressed prefix is not a complete archive."],
            "sha256": report["sha256"], "source_references": references,
        }
        (stage / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report = inspect_games(sample, strict=True, filters=filters,
                               manifest_path=stage / "source_manifest.json")
        (stage / "inspection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ml/games"))
    parser.add_argument("--group", action="append", help="LOW:HIGH_EXCLUSIVE:COUNT; repeat for each stratum")
    parser.add_argument("--count", type=int, help="single-group target; replaces default quotas")
    parser.add_argument("--min-rating", type=int, default=800)
    parser.add_argument("--max-rating", type=int, default=2400)
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--time-classes", nargs="+", default=["rapid", "classical"])
    parser.add_argument("--allow-repeat-players", action="store_true")
    parser.add_argument("--max-games", type=int, default=100_000)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--total-timeout", type=float, default=300)
    args = vars(parser.parse_args())
    try:
        group, count = args.pop("group"), args.pop("count")
        lo, hi = args.pop("min_rating"), args.pop("max_rating")
        if group and count is not None:
            raise ValueError("use --group or --count, not both")
        args["groups"] = (tuple(tuple(map(int, g.split(":"))) for g in group) if group else
                          ((lo, hi + 1, count),) if count is not None else DEFAULT_GROUPS)
        args["filters"] = GameFilters(time_classes=tuple(args.pop("time_classes")),
                                      min_rating=lo, max_rating=hi,
                                      min_plies=args.pop("min_plies"))
        args["unique_players"] = not args.pop("allow_repeat_players")
        manifest = acquire_sample(**args)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; no sample published.\n")
    except (OSError, ValueError, zstandard.ZstdError) as exc:
        parser.exit(1, f"Acquisition failed; no sample published: {exc}\n")
    print(json.dumps(manifest, indent=2))
    if not manifest["complete"]:
        parser.exit(2, "Validated partial sample published; see manifest shortages.\n")


if __name__ == "__main__":
    main()
