"""Read-only production-pipeline benchmark; writes only the requested JSON report.

Run from the project root:
  .venv/Scripts/python.exe -B tests/benchmark_engine_performance.py

Each scenario has one discarded warm-up and three measured repetitions. Timings
include instrumentation overhead. No options, search behavior, or ownership are
changed. Duplicate request counts are opportunities, not determinism guarantees.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import chess.engine

from src.chessmind.analysis.game_analyzer import analyze_game, calculate_game_statistics
from src.chessmind.engine.analyzer import STOCKFISH_PATH, serialize_evaluation
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.players import multi_game_analyzer
from src.chessmind.reports.player_report import generate_player_report
from src.chessmind.reports.weakness_report import generate_weakness_report


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class Instrumentation:
    def __init__(self, binary_hash):
        self.binary_hash = binary_hash
        self.engines = []
        self.engine_metadata = []
        self.requests = []
        self.game_times = []
        self.launch_times = []
        self.search_times = []
        self.quit_times = []
        self.game_index = 0
        self.launch = chess.engine.SimpleEngine.popen_uci

    def popen(self, *args, **kwargs):
        start = perf_counter()
        engine = self.launch(*args, **kwargs)
        self.launch_times.append(perf_counter() - start)
        self.engines.append(engine)
        self.engine_metadata.append({
            "id": dict(engine.id),
            "advertised_options": {name: asdict(option) for name, option in engine.options.items()},
        })
        observer = self

        class ObservedEngine:
            def analyse(self, board, limit, **search_kwargs):
                # Snapshot before the production analyzer mutates its board.
                request = {
                    "game_index": observer.game_index,
                    "fen": board.fen(),
                    "root_fen": board.root().fen(en_passant="fen"),
                    "moves": [move.uci() for move in board.move_stack],
                    "variant": type(board).uci_variant,
                    "chess960": board.chess960,
                    "limit": asdict(limit),
                    "search_kwargs": search_kwargs,
                }
                # Count actual attempted calls, including failures.
                observer.requests.append(request)
                started = perf_counter()
                info = engine.analyse(board, limit, **search_kwargs)
                observer.search_times.append(perf_counter() - started)
                request["configuration"] = dict(engine.protocol.config)
                request["engine_id"] = dict(engine.id)
                request["evaluation"] = serialize_evaluation(info["score"].pov(chess.WHITE))
                request["achieved_depth"] = info.get("depth")
                request["pv"] = [move.uci() for move in info.get("pv", [])]
                request["best_move"] = request["pv"][0] if request["pv"] else None
                return info

            def quit(self):
                started = perf_counter()
                try:
                    return engine.quit()
                finally:
                    observer.quit_times.append(perf_counter() - started)

        return ObservedEngine()

    def timed_game(self, game, depth=8):
        self.game_index += 1
        started = perf_counter()
        try:
            return analyze_game(game, depth=depth)
        finally:
            self.game_times.append(perf_counter() - started)

    def metrics(self):
        fens = Counter(request["fen"] for request in self.requests)
        keys = []
        for request in self.requests:
            key = {field: request[field] for field in (
                "root_fen", "moves", "variant", "chess960", "limit",
                "search_kwargs", "configuration", "engine_id",
            )}
            key["binary_sha256"] = self.binary_hash
            keys.append(canonical(key))
        histories = Counter(keys)

        def across_games(values):
            seen = {}
            duplicates = 0
            for value, request in zip(values, self.requests):
                game = request["game_index"]
                if value in seen and seen[value] != game:
                    duplicates += 1
                seen.setdefault(value, game)
            return duplicates

        return {
            "engine_launches": len(self.engines),
            "analyse_calls": len(self.requests),
            "distinct_current_fens": len(fens),
            "duplicate_current_fen_occurrences": len(self.requests) - len(fens),
            "distinct_history_aware_requests": len(histories),
            "duplicate_history_aware_requests": len(keys) - len(histories),
            "cross_game_duplicate_fen_occurrences": across_games([r["fen"] for r in self.requests]),
            "cross_game_duplicate_history_requests": across_games(keys),
            "quit_calls": len(self.quit_times),
            "engine_processes_exited": [engine.returncode.done() for engine in self.engines],
            "engine_exit_codes": [engine.returncode.result(timeout=10) for engine in self.engines],
            "engine_pids": [engine.transport.get_pid() for engine in self.engines],
        }


def validate_game(game, analysis, stats, requests, depth):
    board = game.board()
    expected = [board.fen()]
    moves = list(game.mainline_moves())
    for move in moves:
        board.push(move)
        expected.append(board.fen())
    check(len(analysis) == len(moves), "Ply count changed")
    check([r["fen"] for r in requests] == expected, "N+1 board order changed")
    check(all(r["limit"]["depth"] == depth for r in requests), "Depth changed")
    for index, move in enumerate(analysis):
        check(move["evaluation_before_detail"] == requests[index]["evaluation"], "Before score changed")
        check(move["evaluation_after_detail"] == requests[index + 1]["evaluation"], "After score changed")
        check(move["best_move"] == requests[index]["best_move"], "Best move changed")
        mate = any(move[field]["type"] == "mate" for field in (
            "evaluation_before_detail", "evaluation_after_detail"))
        check((move["centipawn_loss"] is None) == mate, "Mate entered CP loss")
    for request in requests:
        score = request["evaluation"]
        check(score["perspective"] == "white", "Score perspective changed")
        if score["type"] == "mate":
            check(score["cp"] is None and score["winner"] in ("white", "black"), "Invalid mate score")
    for color in ("white", "black"):
        data = stats[color]
        for subset in [data, *data["phases"].values()]:
            check(subset["moves"] == subset["cp_moves"] + subset["mate_moves"], "Denominator accounting changed")
            expected_acpl = (subset["total_centipawn_loss"] / subset["cp_moves"]
                             if subset["cp_moves"] else None)
            check(subset["acpl"] == expected_acpl, "ACPL denominator changed")
        check(sum(p["moves"] for p in data["phases"].values()) == data["moves"], "Phase accounting changed")
    if board.is_checkmate():
        final = requests[-1]["evaluation"]
        check(final["type"] == "mate" and final["mate"] == 0, "Terminal Mate(0) lost")
        check(final["winner"] == ("black" if board.turn else "white"), "Terminal winner changed")


def run_once(games, batch, depth, binary_hash):
    observer = Instrumentation(binary_hash)
    started = perf_counter()
    # Patching is scoped and restored even on exceptions. Owners still call quit.
    with patch.object(chess.engine.SimpleEngine, "popen_uci", observer.popen), \
            patch.object(multi_game_analyzer, "analyze_game", observer.timed_game):
        if batch:
            entries = [{"game_number": index, "game": game, "color": "white",
                        "opponent": game.headers.get("Black", "?"),
                        "result": game.headers.get("Result", "*")}
                       for index, game in enumerate(games, 1)]
            multi = multi_game_analyzer.analyze_player_games(entries, depth=depth)
            records = multi["games"]
        else:
            analysis = observer.timed_game(games[0], depth=depth)
            records = [{"analysis": analysis, "stats": calculate_game_statistics(analysis)}]
            multi = None
    elapsed = perf_counter() - started
    # Report generation, checks, duplicate counting, and disk I/O are not timed.
    metrics = observer.metrics()
    check(metrics["engine_launches"] == len(games), "Per-game launch ownership changed")
    check(metrics["quit_calls"] == len(games), "Missing normal cleanup")
    check(all(metrics["engine_processes_exited"]), "Engine process still running")
    check(all(code == 0 for code in metrics["engine_exit_codes"]), "Nonzero engine exit")
    correctness = {"games": [], "total_plies": 0}
    for index, (game, record) in enumerate(zip(games, records), 1):
        requests = [r for r in observer.requests if r["game_index"] == index]
        analysis, stats = record["analysis"], record["stats"]
        validate_game(game, analysis, stats, requests, depth)
        correctness["total_plies"] += len(analysis)
        correctness["games"].append({
            "headers": dict(game.headers), "total_plies": len(analysis),
            "classification_counts": dict(Counter(m["classification"] for m in analysis)),
            "mate_related_moves": sum(m["centipawn_loss"] is None for m in analysis),
            "analysis": analysis, "statistics": stats,
            "weakness_report": generate_weakness_report(analysis, stats),
            "position_evaluations": requests,
            "final_two_moves": analysis[-2:],
        })
    if multi is not None:
        # Historical games have different players. This is an explicitly synthetic
        # White-side cohort, used to exercise the real multi-game report interface.
        correctness["cohort_description"] = "Synthetic White-side cohort; not one real player's record"
        correctness["summary"] = multi["summary"]
        correctness["player_report"] = generate_player_report("Benchmark White-side cohort", multi)
    return {
        "timing": {"total_seconds": elapsed, "per_game_seconds": observer.game_times,
                   "launch_seconds": observer.launch_times,
                   "quit_seconds": observer.quit_times,
                   "search_seconds_total": sum(observer.search_times)},
        "operations": metrics, "correctness": correctness,
        "engine": {**observer.engine_metadata[0],
                   "effective_configuration": dict(observer.engines[0].protocol.config)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/day7_baseline.json")
    args = parser.parse_args()
    if args.depth < 1 or args.repetitions < 1:
        parser.error("depth and repetitions must be positive")
    os.chdir(ROOT)  # Match the project's relative Stockfish executable path.
    validation = load_multiple_pgn("data/raw/validation_games.pgn")
    samples = load_multiple_pgn("data/raw/sample_games.pgn")
    check([sum(1 for _ in g.mainline_moves()) for g in validation] == [87, 103, 82], "Historical workload changed")
    check([sum(1 for _ in g.mainline_moves()) for g in samples] == [6, 4, 4], "Sample workload changed")
    binary_hash = digest(STOCKFISH_PATH)
    revision = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    result = {
        "schema_version": 1,
        "environment": {"started_utc": datetime.now(timezone.utc).isoformat(),
                        "python": sys.version, "python_executable": sys.executable,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "logical_cpu_count": os.cpu_count(), "chess_version": chess.__version__,
                        "git_revision": revision, "depth": args.depth,
                        "stockfish_path": str(Path(STOCKFISH_PATH).resolve()),
                        "stockfish_sha256": binary_hash,
                        "dataset_sha256": {p: digest(p) for p in (
                            "data/raw/validation_games.pgn", "data/raw/sample_games.pgn")}},
        "methodology": {
            "warmups_per_scenario": 1, "measured_repetitions": args.repetitions,
            "repeated_game_copies": 3, "application_cache": False,
            "engine_options_overridden": False,
            "timing_scope": "Production analysis and statistics plus instrumentation; excludes report generation, checks and JSON writing",
            "per_game_scope": "analyze_game including process launch and quit; excludes statistics aggregation",
            "duplicate_scope": "Within each run; occurrences after first, not pair counts; no reuse between repetitions",
            "history_key": "binary hash + engine identity + effective UCI configuration + variant + Chess960 + root FEN + full move stack + search limit + kwargs",
            "limitations": ["No promise of deterministic engine evaluations or timings",
                            "Cache opportunity counts do not model warm transposition-table state",
                            "Synthetic White-side cohort used for historical multi-game reporting",
                            "Warm-up warms host resources; production still starts fresh engines per game"],
        },
        "scenarios": {},
    }
    workloads = [
        ("single_full_game", validation[:1], False),
        ("multiple_historical_games", validation, True),
        ("same_game_three_times", [validation[0]] * 3, True),
        ("cross_game_positions", samples, True),
    ]
    for name, games, batch in workloads:
        print(f"{name}: warm-up", flush=True)
        warmup = run_once(games, batch, args.depth, binary_hash)
        scenario = {"warmup": {k: warmup[k] for k in ("timing", "operations")},
                    "timing": [], "operations": [], "correctness": []}
        result["environment"]["engine"] = warmup["engine"]
        for repetition in range(args.repetitions):
            run = run_once(games, batch, args.depth, binary_hash)
            for key in ("timing", "operations", "correctness"):
                scenario[key].append(run[key])
            print(f"{name} repetition {repetition + 1}: {run['timing']['total_seconds']:.3f}s; "
                  f"launches={run['operations']['engine_launches']}, searches={run['operations']['analyse_calls']}", flush=True)
        scenario["median_total_seconds"] = statistics.median(t["total_seconds"] for t in scenario["timing"])
        scenario["median_per_game_seconds"] = [statistics.median(t["per_game_seconds"][i] for t in scenario["timing"])
                                                for i in range(len(games))]
        scenario["correctness_identical_across_repetitions"] = all(
            canonical(c) == canonical(scenario["correctness"][0]) for c in scenario["correctness"])
        result["scenarios"][name] = scenario
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Baseline saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
