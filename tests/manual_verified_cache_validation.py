"""Step 4D.2 real-engine validation; no engine reuse or baseline writes.

Run from the repository root:
  .venv/Scripts/python.exe -B tests/manual_verified_cache_validation.py

Prints JSON measurements for one disabled/enabled depth-8 trial. Timing includes
instrumentation; independent Stockfish searches need not return identical scores.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import chess.engine

from src.chessmind.engine import analyzer, evaluation_cache_keys as keys
from src.chessmind.engine.evaluation_cache import EvaluationCache
from src.chessmind.engine.embedded_resource_attestations import VERIFIED_EMBEDDED_RESOURCES
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.players.multi_game_analyzer import analyze_player_games


def measure(workload, cache, expected_requests):
    engines, fingerprint_times = [], []
    launch, fingerprint = chess.engine.SimpleEngine.popen_uci, keys._fingerprint

    def observed_launch(*args, **kwargs):
        engine = launch(*args, **kwargs)
        engines.append(engine)
        engine.analyse = Mock(wraps=engine.analyse)
        engine.quit = Mock(wraps=engine.quit)
        return engine

    def observed_fingerprint(*args, **kwargs):
        started = perf_counter()
        try:
            return fingerprint(*args, **kwargs)
        finally:
            fingerprint_times.append(perf_counter() - started)

    started = perf_counter()
    with patch.object(chess.engine.SimpleEngine, "popen_uci", observed_launch), \
         patch.object(keys, "_fingerprint", observed_fingerprint):
        result = workload(cache)
    elapsed = perf_counter() - started
    shutdowns = sum(e.quit.call_count for e in engines)
    assert shutdowns == len(engines)
    assert all(e.returncode.result(timeout=10) == 0 for e in engines)
    searches = sum(e.analyse.call_count for e in engines)
    info = asdict(cache.info()) if cache is not None else None
    assert searches + (cache.info().hits if cache is not None else 0) == expected_requests
    return result, {
        "depth": 8, "logical_requests": expected_requests,
        "runtime_seconds": elapsed, "launches": len(engines),
        "clean_shutdowns": shutdowns, "actual_searches": searches,
        "searches_per_process": [e.analyse.call_count for e in engines],
        "cache": info, "fingerprint_calls": len(fingerprint_times),
        "fingerprint_seconds": sum(fingerprint_times),
    }


def main():
    baseline = ROOT / "data/reports/day7_baseline.json"
    baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
    binary = ROOT / analyzer.STOCKFISH_PATH
    record = VERIFIED_EMBEDDED_RESOURCES[0]
    assert keys._fingerprint(binary)[1] == record.executable_sha256
    game = load_multiple_pgn(ROOT / "data/raw/validation_games.pgn")[0]
    assert len(list(game.mainline_moves())) == 87
    entries = [{"game_number": i, "game": game, "color": "white",
                "opponent": game.headers.get("Black", "?"),
                "result": game.headers.get("Result", "*")} for i in range(1, 4)]

    def positions(cache):
        return [analyzer.analyze_position(chess.STARTING_FEN, depth=8, cache=cache)
                for _ in range(2)]

    result, position_report = measure(positions, EvaluationCache(), 2)
    assert result[0] == result[1]
    assert position_report["actual_searches"] == 1
    assert position_report["launches"] == position_report["clean_shutdowns"] == 2
    assert (position_report["cache"]["misses"], position_report["cache"]["hits"]) == (1, 1)
    print("POSITION " + json.dumps(position_report), flush=True)

    def repeated(cache):
        return analyze_player_games(entries, depth=8, cache=cache)

    uncached, disabled = measure(repeated, None, 264)
    print("DISABLED " + json.dumps(disabled), flush=True)
    cached, enabled = measure(repeated, EvaluationCache(), 264)
    assert disabled["actual_searches"] == 264
    assert enabled["actual_searches"] == 88
    assert disabled["launches"] == enabled["launches"] == 3
    assert (enabled["cache"]["misses"], enabled["cache"]["hits"]) == (88, 176)
    assert enabled["fingerprint_calls"] == 3 and disabled["fingerprint_calls"] == 0
    assert all(g["analysis"] == cached["games"][0]["analysis"] and
               g["stats"] == cached["games"][0]["stats"] for g in cached["games"])
    for output in (uncached, cached):
        for g in output["games"]:
            assert len(g["analysis"]) == 87
            for row in g["analysis"]:
                for field in ("evaluation_before_detail", "evaluation_after_detail"):
                    value = row[field]
                    assert value["perspective"] == "white"
                    if value["type"] == "mate":
                        assert value["cp"] is None and value["winner"] in ("white", "black")
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == baseline_hash
    print("ENABLED " + json.dumps(enabled), flush=True)
    print("COMPARISON " + json.dumps({
        "independent_runs_equal": uncached == cached,
        "analysis_rows_different": sum(a != b for ug, cg in zip(uncached["games"], cached["games"])
                                       for a, b in zip(ug["analysis"], cg["analysis"])),
        "statistics_equal": [u["stats"] == c["stats"] for u, c in zip(uncached["games"], cached["games"])],
        "cached_replays_identical": True, "baseline_unchanged": True,
    }), flush=True)


if __name__ == "__main__":
    main()
