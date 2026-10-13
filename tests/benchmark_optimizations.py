"""Day 7 Step 5B: four-mode depth-8 benchmark; production code is unchanged.

Run: .venv/Scripts/python.exe -B tests/benchmark_optimizations.py
Writes only data/reports/day7_optimization_benchmark.json.
"""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
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

from benchmark_engine_performance import digest, validate_game, check
from src.chessmind.engine import analyzer, evaluation_cache_keys as keys
from src.chessmind.engine.evaluation_cache import EvaluationCache
from src.chessmind.analysis import game_analyzer
from src.chessmind.players import multi_game_analyzer as multi
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.reports.player_report import generate_player_report

MODES = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
SCHEDULE = ["ABCD", "BCDA", "CDAB", "DABC"]
DEPTH = 8


class Observer:
    def __init__(self, cache):
        self.cache = cache
        self.engines, self.logical, self.games = [], [], []
        self.searches = self.shutdowns = self.boundaries = self.bypassed = 0
        self.bypass_reasons = Counter()
        self.timings = {}
        self.game_index = -1

    def timed(self, name, function):
        def call(*args, **kwargs):
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            except keys.CacheIdentityUnavailable as exc:
                self.bypass_reasons[str(exc)] += 1
                raise
            finally:
                self.timings.setdefault(name, []).append(perf_counter() - start)
        return call

    def instrument(self, stack):
        launch = chess.engine.SimpleEngine.popen_uci
        analyse_board, analyse_game = game_analyzer.analyze_board, multi.analyze_game

        def popen(*args, **kwargs):
            engine = self.timed("launch", launch)(*args, **kwargs)
            self.engines.append(engine)
            analyse, quit_engine, send = engine.analyse, engine.quit, engine.protocol.send_line
            def search(*args, **kwargs):
                self.searches += 1
                return self.timed("search", analyse)(*args, **kwargs)
            def quit():
                self.shutdowns += 1
                return self.timed("shutdown", quit_engine)()
            def send_line(line):
                if line == "ucinewgame":
                    self.boundaries += 1
                send(line)
            engine.analyse, engine.quit, engine.protocol.send_line = search, quit, send_line
            return engine

        def board(engine, board, depth=15, **kwargs):
            prior = self.cache.info() if self.cache is not None else None
            result = analyse_board(engine, board, depth=depth, **kwargs)
            if prior is not None:
                after = self.cache.info()
                if (prior.hits, prior.misses) == (after.hits, after.misses):
                    self.bypassed += 1
            self.logical.append({"game_index": self.game_index, "fen": board.fen(),
                                 "limit": {"depth": depth}, "evaluation": result["evaluation"],
                                 "best_move": result["best_move"].uci() if result["best_move"] else None,
                                 "pv": [move.uci() for move in result["pv"]]})
            return result

        def game(*args, **kwargs):
            self.game_index += 1
            return self.timed("game", analyse_game)(*args, **kwargs)

        stack.enter_context(patch.object(chess.engine.SimpleEngine, "popen_uci", popen))
        stack.enter_context(patch.object(game_analyzer, "analyze_board", board))
        stack.enter_context(patch.object(multi, "analyze_game", game))
        for module, name, label in (
            (keys, "_fingerprint", "fingerprint"),
            (analyzer, "create_cache_context", "context"),
            (analyzer, "snapshot_engine_metadata", "metadata"),
            (analyzer, "build_cache_key", "key"),
        ):
            stack.enter_context(patch.object(module, name, self.timed(label, getattr(module, name))))
        if self.cache is not None:
            for name in ("get", "put"):
                stack.enter_context(patch.object(self.cache, name, self.timed(name, getattr(self.cache, name))))


def dictionary_paths(value, path=""):
    """Dictionary key structure only; scalar values may legitimately differ."""
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(path + "/" + key)
            found.update(dictionary_paths(child, path + "/" + key))
    elif isinstance(value, list):
        for child in value:
            found.update(dictionary_paths(child, path + "[]"))
    return found


def run(games, mode, baseline):
    caching, reuse = MODES[mode]
    cache = EvaluationCache() if caching else None
    observer = Observer(cache)
    entries = [{"game_number": i, "game": game, "color": "white",
                "opponent": game.headers.get("Black", "?"), "result": game.headers.get("Result", "*")}
               for i, game in enumerate(games, 1)]
    with ExitStack() as stack:
        observer.instrument(stack)
        started = perf_counter()
        result = multi.analyze_player_games(entries, depth=DEPTH, cache=cache, reuse_engine=reuse)
        elapsed = perf_counter() - started
    report = generate_player_report("Benchmark White-side cohort", result)
    exits = [engine.returncode.result(timeout=10) for engine in observer.engines]
    check(exits == [0] * len(observer.engines), "Unclean engine exit")
    check(observer.shutdowns == len(observer.engines) == (1 if reuse else 3), "Ownership mismatch")
    for engine in observer.engines:
        check(engine.protocol.config["Threads"] == 1 and engine.protocol.config["Hash"] == 16,
              "Search settings changed")
    mate_zero = 0
    for i, (game, record, saved) in enumerate(zip(games, result["games"], baseline["games"])):
        requests = [r for r in observer.logical if r["game_index"] == i]
        validate_game(game, record["analysis"], record["stats"], requests, DEPTH)
        check(dictionary_paths(record["stats"]) == dictionary_paths(saved["statistics"]), "Stats schema changed")
        for row, old in zip(record["analysis"], saved["analysis"]):
            check(row.keys() == old.keys(), "Move schema changed")
            check(row["phase"] == old["phase"], "Phase assignment changed")
        for request in requests:
            board = chess.Board(request["fen"])
            if request["best_move"] is not None:
                check(chess.Move.from_uci(request["best_move"]) in board.legal_moves, "Illegal best move")
            for move in request["pv"]:
                parsed = chess.Move.from_uci(move)
                check(parsed in board.legal_moves, "Illegal PV")
                board.push(parsed)
            score = request["evaluation"]
            check(set(score) == {"type", "cp", "mate", "winner", "perspective"}, "Score schema changed")
            if score["type"] == "cp":
                check(type(score["cp"]) is int and score["mate"] is None and score["winner"] is None, "Invalid CP")
            else:
                check(score["type"] == "mate" and type(score["mate"]) is int, "Invalid mate type")
                if score["mate"]:
                    check(score["winner"] == ("white" if score["mate"] > 0 else "black"), "Invalid mate winner")
                else:
                    mate_zero += 1
    # Optional report lists can be empty; compare fixed report structure separately.
    check(report.keys() == baseline["player_report"].keys(), "Player report schema changed")
    check(dictionary_paths(report["overall_performance"]) ==
          dictionary_paths(baseline["player_report"]["overall_performance"]), "Overall schema changed")
    for actual, saved in zip(report["games"], baseline["player_report"]["games"]):
        check(actual.keys() == saved.keys(), "Game report schema changed")
    counters = asdict(cache.info()) if cache is not None else {"hits": 0, "misses": 0, "evictions": 0, "size": 0}
    check(len(observer.logical) == sum(len(list(g.mainline_moves())) + 1 for g in games), "Logical count changed")
    check(observer.searches + counters["hits"] == len(observer.logical), "Search accounting mismatch")
    check(not observer.bypassed and not observer.bypass_reasons, "Unexpected cache identity bypass")
    counts = {"logical_requests": len(observer.logical), "actual_searches": observer.searches,
              "cache": counters, "launches": len(observer.engines), "shutdowns": observer.shutdowns,
              "boundaries": observer.boundaries, "fingerprints": len(observer.timings.get("fingerprint", [])),
              "identity_contexts": len(observer.timings.get("context", [])),
              "bypassed_requests": observer.bypassed, "cache_disabled_requests": 0 if caching else len(observer.logical),
              "identity_bypass_reasons": dict(observer.bypass_reasons), "exit_codes": exits,
              "process_ids": [e.transport.get_pid() for e in observer.engines]}
    return {"runtime_seconds": elapsed, "timing_components_seconds": observer.timings,
            "counts": counts, "verification": {"passed": True, "mate_zero_requests": mate_zero},
            "engine": {"id": dict(observer.engines[0].protocol.id), "configuration": dict(observer.engines[0].protocol.config)},
            "outputs": result, "player_report": report}


def differences(actual, reference):
    pairs = [(a, b) for ag, bg in zip(actual["games"], reference["games"])
             for a, b in zip(ag["analysis"], bg["analysis"])]
    return {"numeric_or_mate_rows": sum(any(a[k] != b[k] for k in
            ("evaluation_before_detail", "evaluation_after_detail")) for a, b in pairs),
            "classification_rows": sum(a["classification"] != b["classification"] for a, b in pairs),
            "best_move_rows": sum(a["best_move"] != b["best_move"] for a, b in pairs),
            "phase_rows": sum(a["phase"] != b["phase"] for a, b in pairs),
            "statistics_games": sum(a["stats"] != b["stats"] for a, b in zip(actual["games"], reference["games"]))}


def main():
    os.chdir(ROOT)
    baseline_path = ROOT / "data/reports/day7_baseline.json"
    baseline_digest = digest(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    protected = {str(p): digest(p) for p in (ROOT / "src").rglob("*.py")}
    historical = load_multiple_pgn("data/raw/validation_games.pgn")
    samples = load_multiple_pgn("data/raw/sample_games.pgn")
    check([len(list(g.mainline_moves())) for g in historical] == [87, 103, 82], "Historical input changed")
    check([len(list(g.mainline_moves())) for g in samples] == [6, 4, 4], "Sample input changed")
    output = ROOT / "data/reports/day7_optimization_benchmark.json"
    result = {"environment": {"started_utc": datetime.now(timezone.utc).isoformat(),
              "client_timezone": "Asia/Colombo", "python": sys.version, "platform": platform.platform(),
              "processor": platform.processor(), "cpu_count": os.cpu_count(), "chess_version": chess.__version__,
              "git_revision": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], text=True).strip(),
              "executable": str(Path(analyzer.STOCKFISH_PATH).resolve()), "executable_sha256": digest(analyzer.STOCKFISH_PATH),
              "dataset_sha256": {p: digest(p) for p in ("data/raw/validation_games.pgn", "data/raw/sample_games.pgn")}},
              "methodology": {"depth": DEPTH, "threads": 1, "hash_mib": 16, "warmups_per_mode_per_workload": 1,
              "measured_repetitions": 4, "order": SCHEDULE, "modes": MODES,
              "timing_scope": "Production batch with instrumentation, including startup, identity, searches, stats and shutdown; excludes report generation, verification and disk writes",
              "game_timing_scope": "analyze_game only; shared startup/context/shutdown occurs outside per-game times",
              "overhead_note": "Context includes initial metadata and fingerprints; components overlap. Context + per-request metadata + key + get + put is an approximate cache overhead; instrumentation also costs time.",
              "fresh_cache_and_processes_per_run": True, "outliers_excluded": False,
              "cohort": "Synthetic White-side cohort, not a real single-player history"}, "workloads": {}}
    for name, games in (("same_game_three_times", [historical[0]] * 3),
                        ("multiple_historical_games", historical), ("cross_game_positions", samples)):
        reference = baseline["scenarios"][name]["correctness"][0]
        scenario = {"plies": [len(list(g.mainline_moves())) for g in games],
                    "headers": [dict(g.headers) for g in games], "modes": {}}
        for mode in MODES:
            print(f"{name} {mode} warm-up", flush=True)
            warmup = run(games, mode, reference)
            scenario["modes"][mode] = {"warmup": {k: v for k, v in warmup.items() if k not in ("outputs", "player_report")}, "runs": []}
        for repetition, order in enumerate(SCHEDULE):
            for mode in order:
                trial = run(games, mode, reference)
                scenario["modes"][mode]["runs"].append(trial)
                print(f"{name} {mode} run {repetition+1}: {trial['runtime_seconds']:.3f}s {trial['counts']}", flush=True)
        original_runs = scenario["modes"]["A"]["runs"]
        original_median = statistics.median(r["runtime_seconds"] for r in original_runs)
        for mode, data in scenario["modes"].items():
            times = [r["runtime_seconds"] for r in data["runs"]]
            median = statistics.median(times)
            data["summary"] = {"raw_seconds": times, "median_seconds": median, "min_seconds": min(times),
                "max_seconds": max(times), "stdev_seconds": statistics.stdev(times),
                "improvement_percent_vs_A": 100 * (original_median - median) / original_median,
                "search_reduction_percent_vs_A": 100 * (1 - statistics.median(r["counts"]["actual_searches"] for r in data["runs"]) / original_runs[0]["counts"]["actual_searches"]),
                "startup_reduction_percent_vs_A": 100 * (1 - data["runs"][0]["counts"]["launches"] / 3),
                "median_per_game_seconds": [statistics.median(r["timing_components_seconds"]["game"][i] for r in data["runs"]) for i in range(3)],
                "median_fingerprint_seconds": statistics.median(sum(r["timing_components_seconds"].get("fingerprint", [])) for r in data["runs"])}
            for i, trial in enumerate(data["runs"]):
                trial["comparison_to_A_same_repetition"] = differences(trial["outputs"], original_runs[i]["outputs"])
                trial["comparison_to_first_same_mode"] = differences(trial["outputs"], data["runs"][0]["outputs"])
        result["workloads"][name] = scenario
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    check(digest(baseline_path) == baseline_digest, "Baseline changed")
    check(all(digest(p) == value for p, value in protected.items()), "Production source changed")
    result["verification"] = {"baseline_unchanged": True, "production_sources_unchanged": True,
                              "all_created_processes_exited": True, "baseline_sha256": baseline_digest}
    result["environment"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
