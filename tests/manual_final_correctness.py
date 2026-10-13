"""Step 6 validation and cache provenance audit; never writes benchmark files.

Run from the repository root with .venv/Scripts/python.exe -B.
Writes day7_final_correctness.json and day7_classification_differences.csv.
"""

from collections import Counter
from copy import deepcopy
import csv
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import benchmark_optimizations as benchmark
from benchmark_engine_performance import digest
import chess
import chess.engine
from src.chessmind.engine import analyzer
from src.chessmind.engine.evaluation_cache import EvaluationCache
from src.chessmind.analysis import game_analyzer
from src.chessmind.players import multi_game_analyzer as multi
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn

ROOT = Path(__file__).resolve().parents[1]
AUDITS = []


class AuditCache(EvaluationCache):
    def __init__(self):
        super().__init__()
        self.origins = {}

    def put(self, key, result):
        super().put(key, result)
        self.origins[key] = (deepcopy(result), AUDITS[-1].game_index)

    def get(self, key):
        result = super().get(key)
        if result is not None:
            expected, origin = self.origins[key]
            assert result == expected and result is not expected
            audit = AUDITS[-1]
            audit.hits.append({"origin_game": origin + 1, "game": audit.game_index + 1,
                               "root_fen": key.search.root_fen,
                               "moves": list(key.search.moves), "fen": key.search.current_fen,
                               "evaluation": result["evaluation"]})
            audit.last_hit = result
        return result


class AuditObserver(benchmark.Observer):
    def __init__(self, cache):
        super().__init__(cache)
        self.hits, self.native, self.provenance, self.signals = [], [], [], []
        self.tokens = {}
        AUDITS.append(self)

    def instrument(self, stack):
        super().instrument(stack)
        analyse = chess.engine.SimpleEngine.analyse
        board_call = game_analyzer.analyze_board
        boundary = chess.engine.UciProtocol._ucinewgame

        def native(engine, board, limit, **kwargs):
            token = kwargs.get("game")
            if token is not None:
                if self.game_index in self.tokens:
                    assert self.tokens[self.game_index] is token
                else:
                    assert all(other is not token for other in self.tokens.values())
                    self.tokens[self.game_index] = token
            value = analyse(engine, board, limit, **kwargs)
            self.native.append({"game": self.game_index + 1, "moves": [m.uci() for m in board.move_stack],
                                "evaluation": analyzer.serialize_evaluation(value["score"].pov(chess.WHITE)),
                                "pv": [m.uci() for m in value.get("pv", [])]})
            return value

        def newgame(protocol):
            if self.game_index in self.tokens:
                assert protocol.game is self.tokens[self.game_index]
            self.signals.append({"game": self.game_index + 1, "search_number": len(self.native) + 1})
            return boundary(protocol)

        def board(engine, board, depth=15, **kwargs):
            count = len(self.native)
            value = board_call(engine, board, depth=depth, **kwargs)
            if len(self.native) == count:
                assert value == self.last_hit
                assert self.hits[-1]["moves"] == [m.uci() for m in board.move_stack]
                assert self.hits[-1]["fen"] == board.fen(en_passant="fen")
                source = "verified_cache_hit"
            else:
                assert len(self.native) == count + 1
                assert value["evaluation"] == self.native[-1]["evaluation"]
                assert [m.uci() for m in value["pv"]] == self.native[-1]["pv"]
                source = "fresh_search"
            self.provenance.append({"game": self.game_index + 1, "ply": len(board.move_stack),
                                    "source": source, "evaluation": value["evaluation"]})
            return value

        stack.enter_context(patch.object(chess.engine.SimpleEngine, "analyse", native))
        stack.enter_context(patch.object(chess.engine.UciProtocol, "_ucinewgame", newgame))
        stack.enter_context(patch.object(game_analyzer, "analyze_board", board))


def classification_changes(original, optimized):
    changes = []
    good, bad = {"Excellent", "Good"}, {"Mistake", "Blunder"}
    for index, (a, b) in enumerate(zip(original["games"], optimized["games"]), 1):
        for x, y in zip(a["analysis"], b["analysis"]):
            if x["classification"] == y["classification"]:
                continue
            old, new = x["classification"], y["classification"]
            severity = ("good_to_serious" if old in good and new in bad else
                        "serious_to_good" if old in bad and new in good else "other")
            losses = (x["centipawn_loss"], y["centipawn_loss"])
            distances = [min(abs(loss - threshold) for threshold in (20, 50, 100, 200))
                         if loss is not None else None for loss in losses]
            changes.append({"game": index, "ply": x["ply"], "move_number": x["move_number"],
                            "color": x["color"], "san": x["san"], "severity": severity,
                            "original": {k: x[k] for k in ("classification", "centipawn_loss", "evaluation_before_detail", "evaluation_after_detail", "mate_quality")},
                            "optimized": {k: y[k] for k in ("classification", "centipawn_loss", "evaluation_before_detail", "evaluation_after_detail", "mate_quality")},
                            "distance_to_nearest_cp_threshold": distances,
                            "both_within_10cp_of_threshold": all(d is not None and d <= 10 for d in distances),
                            "either_within_10cp_of_threshold": any(d is not None and d <= 10 for d in distances)})
    return changes


def main():
    protected = [ROOT / "data/reports/day7_baseline.json", ROOT / "data/reports/day7_optimization_benchmark.json"]
    protected += list((ROOT / "src").rglob("*.py"))
    hashes = {str(p): digest(p) for p in protected}
    saved = json.loads(protected[1].read_text())
    baseline = json.loads(protected[0].read_text())
    historical = load_multiple_pgn(ROOT / "data/raw/validation_games.pgn")
    report = {"depth": 8, "four_modes": {}, "classification_comparisons": {}, "signatures": {}}
    live = {}
    with patch.object(benchmark, "Observer", AuditObserver), patch.object(benchmark, "EvaluationCache", AuditCache):
        for workload, games in (("multiple_historical_games", historical), ("same_game_three_times", [historical[0]] * 3)):
            live[workload] = {}
            for mode in benchmark.MODES:
                result = benchmark.run(games, mode, baseline["scenarios"][workload]["correctness"][0])
                audit = AUDITS[-1]
                assert len(result["outputs"]["games"]) == len(games)
                for i, (record, game) in enumerate(zip(result["outputs"]["games"], games), 1):
                    assert (record["game_number"], record["color"], record["opponent"], record["result"]) == (
                        i, "white", game.headers.get("Black", "?"), game.headers.get("Result", "*"))
                live[workload][mode] = result
                report["four_modes"][workload + "/" + mode] = {
                    "counts": result["counts"], "verification": result["verification"],
                    "cache_hit_origins": audit.hits, "request_provenance": audit.provenance,
                    "boundary_signals": audit.signals, "native_searches_verified": len(audit.native),
                    "compared_with_step5b": benchmark.differences(result["outputs"], saved["workloads"][workload]["modes"][mode]["runs"][0]["outputs"])}
                print(workload, mode, result["counts"], flush=True)
    # Controlled replay: replace only subsequent initial-position searches with
    # the first game's native result. No EvaluationCache or key lookup is used.
    actual_board = game_analyzer.analyze_board
    initial = []
    def root_replay(engine, board, depth=15, **kwargs):
        if not board.move_stack and initial:
            return deepcopy(initial[0])
        result = actual_board(engine, board, depth=depth, **kwargs)
        if not board.move_stack:
            initial.append(deepcopy(result))
        return result
    entries = [{"game_number": i, "game": game, "color": "white", "opponent": game.headers.get("Black", "?"),
                "result": game.headers.get("Result", "*")} for i, game in enumerate(historical, 1)]
    with patch.object(game_analyzer, "analyze_board", root_replay):
        control = multi.analyze_player_games(entries, depth=8)
    report["initial_search_omission_control"] = benchmark.differences(
        control, live["multiple_historical_games"]["B"]["outputs"])
    for mode in ("B", "D"):
        modes = saved["workloads"]["multiple_historical_games"]["modes"]
        a, b = modes["A"]["runs"][0], modes[mode]["runs"][0]
        changes = classification_changes(a["outputs"], b["outputs"])
        report["classification_comparisons"]["A_vs_" + mode] = {
            "changes": changes, "severity_counts": dict(Counter(c["severity"] for c in changes)),
            "both_near_10cp": sum(c["both_within_10cp_of_threshold"] for c in changes),
            "either_near_10cp": sum(c["either_within_10cp_of_threshold"] for c in changes),
            "transitions": dict(Counter(c["original"]["classification"] + " -> " + c["optimized"]["classification"] for c in changes)),
            "statistics": [{"game": i, "original": x["stats"], "optimized": y["stats"]}
                           for i, (x, y) in enumerate(zip(a["outputs"]["games"], b["outputs"]["games"]), 1)],
            "original_player_report": a["player_report"], "optimized_player_report": b["player_report"]}
    for function in (analyzer.analyze_board, analyzer.analyze_position, game_analyzer.analyze_game, multi.analyze_player_games):
        report["signatures"][function.__name__] = str(inspect.signature(function))
    assert all(digest(path) == value for path, value in hashes.items())
    report["production_and_benchmarks_unchanged"] = True
    (ROOT / "data/reports/day7_final_correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "data/reports/day7_classification_differences.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "game", "ply", "move_number", "color", "san", "severity",
                         "original_class", "optimized_class", "original_loss", "optimized_loss",
                         "original_before", "original_after", "optimized_before", "optimized_after"])
        for comparison, data in report["classification_comparisons"].items():
            for c in data["changes"]:
                a, b = c["original"], c["optimized"]
                writer.writerow([comparison, c["game"], c["ply"], c["move_number"], c["color"], c["san"], c["severity"],
                                 a["classification"], b["classification"], a["centipawn_loss"], b["centipawn_loss"],
                                 *[json.dumps(v[k]) for v in (a, b) for k in ("evaluation_before_detail", "evaluation_after_detail")]])
    print('CONTROL',report["initial_search_omission_control"])
    for name, data in report["classification_comparisons"].items():
        print(name, data["severity_counts"], 'near thresholds', data["both_near_10cp"],data["either_near_10cp"])


if __name__ == "__main__":
    main()
