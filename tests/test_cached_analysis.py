"""Opt-in integration, deterministic UCI doubles, and a small real-engine smoke."""

from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
from unittest.mock import Mock

import chess
import chess.engine
import chess.pgn
import pytest

from src.chessmind.engine import analyzer as engine_module
from src.chessmind.engine.evaluation_cache import EvaluationCache
from src.chessmind.engine import evaluation_cache_keys as keys
from src.chessmind.engine.embedded_resource_attestations import VERIFIED_EMBEDDED_RESOURCES
from src.chessmind.analysis import game_analyzer
from src.chessmind.players.multi_game_analyzer import analyze_player_games
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.reports.player_report import generate_player_report
from src.chessmind.reports.weakness_report import generate_weakness_report

ROOT = Path(__file__).resolve().parents[1]


def metadata():
    option = chess.engine.Option("Threads", "spin", 1, 1, 1024, None)
    return keys.metadata_from_values({"name": "deterministic fixture"},
                                    {"Threads": option}, {"Threads": 1}, {"Threads": 1})


def response(board, limit):
    move = next(iter(board.legal_moves), None)
    return {"score": chess.engine.PovScore(chess.engine.Cp(len(board.move_stack) * 10), chess.WHITE),
            "depth": limit.depth, "pv": [move] if move else []}


@pytest.fixture
def engines(monkeypatch, tmp_path):
    binary = tmp_path / "fixture-engine"
    binary.write_bytes(b"engine fixture")
    created = []
    def launch(*args, **kwargs):
        engine = Mock()
        engine.analyse.side_effect = response
        created.append(engine)
        return engine
    factory = Mock(side_effect=launch)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", factory)
    monkeypatch.setattr(engine_module, "STOCKFISH_PATH", str(binary))
    monkeypatch.setattr(game_analyzer, "STOCKFISH_PATH", str(binary))
    snapshot = Mock(return_value=metadata())
    monkeypatch.setattr(engine_module, "snapshot_engine_metadata", snapshot)
    fingerprint = Mock(wraps=keys._fingerprint)
    monkeypatch.setattr(keys, "_fingerprint", fingerprint)
    return created, factory, snapshot, fingerprint, binary


def borrowed(engines):
    _, factory, _, _, binary = engines
    engine = factory()
    return engine, engine_module.create_cache_context(engine, binary)


def evaluate(engine, context, cache, board=None, depth=8):
    return engine_module.analyze_board(engine, chess.Board() if board is None else board,
                                      depth, cache=cache, cache_context=context)


def game():
    return chess.pgn.read_game(io.StringIO("1. e4 e5 2. Nf3 Nc6 *"))


def entries(games):
    return [{"game_number": i, "game": g, "color": "white",
             "opponent": g.headers.get("Black", "?"), "result": g.headers.get("Result", "*")}
            for i, g in enumerate(games, 1)]


def test_miss_hit_depth_and_no_per_position_hash(engines):
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    first = evaluate(engine, context, cache)
    assert evaluate(engine, context, cache) == first
    evaluate(engine, context, cache, depth=9)
    assert engine.analyse.call_count == 2
    assert (cache.info().hits, cache.info().misses) == (1, 2)
    assert engines[3].call_count == 1


def test_history_separation_and_independent_replay(engines):
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(move)
    evaluate(engine, context, cache, board)
    evaluate(engine, context, cache, chess.Board(board.fen()))
    evaluate(engine, context, cache, board.copy())
    assert engine.analyse.call_count == 2
    assert cache.info().hits == 1


@pytest.mark.parametrize("cache", [None, EvaluationCache(0)])
def test_disabled_cache_has_no_metadata_work(engines, cache):
    game_analyzer.analyze_game(game(), depth=8, cache=cache)
    engine = engines[0][0]
    assert engine.analyse.call_count == 5
    engine.quit.assert_called_once()
    engines[2].assert_not_called()
    engines[3].assert_not_called()


def test_missing_or_wrong_engine_context_bypasses(engines):
    engine, context = borrowed(engines)
    other = engines[1]()
    cache = EvaluationCache()
    evaluate(engine, None, cache)
    evaluate(other, context, cache)
    assert cache.info().size == cache.info().misses == 0
    assert engine.analyse.call_count == other.analyse.call_count == 1


def test_unavailable_identity_disabled_once_per_game(engines):
    engines[2].side_effect = keys.CacheIdentityUnavailable("unavailable")
    cache = EvaluationCache()
    game_analyzer.analyze_game(game(), cache=cache)
    assert engines[2].call_count == 1
    assert engines[0][0].analyse.call_count == 5
    engines[0][0].quit.assert_called_once()
    assert cache.info().misses == 0


def network_metadata(value):
    base = metadata()
    return replace(base, reported_id=(("name", "Stockfish 19"),),
                   options=base.options + (keys.OptionSpec("evalfile", "string", value, None, None, ()),),
                   config=base.config + (("evalfile", value),),
                   target_config=base.target_config + (("evalfile", value),))


def test_failed_attestation_keeps_normal_analysis(engines):
    engines[2].return_value = network_metadata(VERIFIED_EMBEDDED_RESOURCES[0].value)
    cache = EvaluationCache()
    result = engine_module.analyze_position(chess.STARTING_FEN, cache=cache)
    assert result["evaluation"]["perspective"] == "white"
    assert engines[0][0].analyse.call_count == 1
    engines[0][0].quit.assert_called_once()
    assert cache.info().hits == cache.info().misses == cache.info().size == 0


def test_resource_change_after_attested_context_cannot_hit(engines):
    record = VERIFIED_EMBEDDED_RESOURCES[0]
    engines[2].return_value = network_metadata(record.value)
    engines[3].return_value = ("audited.exe", record.executable_sha256)
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    evaluate(engine, context, cache)
    evaluate(engine, context, cache)
    engines[2].return_value = network_metadata("custom.nnue")
    evaluate(engine, context, cache)
    assert engine.analyse.call_count == 2
    assert (cache.info().misses, cache.info().hits) == (1, 1)
    engines[3].assert_called_once()


@pytest.mark.parametrize("failure", [RuntimeError("bug"), chess.engine.EngineError("search failed")])
def test_search_errors_propagate_and_are_not_cached(engines, failure):
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    engine.analyse.side_effect = failure
    with pytest.raises(type(failure)):
        evaluate(engine, context, cache)
    assert cache.info().size == 0
    engine.analyse.side_effect = response
    evaluate(engine, context, cache)
    assert engine.analyse.call_count == 2


@pytest.mark.parametrize("field,value", [("lowerbound", True), ("upperbound", True),
    ("partial", True), ("incomplete", True), ("complete", False), ("pv", []), ("depth", None)])
def test_ineligible_results_returned_but_not_cached(engines, field, value):
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    engine.analyse.side_effect = lambda board, limit: {**response(board, limit), field: value}
    first = evaluate(engine, context, cache)
    assert evaluate(engine, context, cache) == first
    assert engine.analyse.call_count == 2
    assert cache.info().size == 0


@pytest.mark.parametrize("score,winner", [(chess.engine.Mate(0), "black"),
                                         (chess.engine.MateGiven, "white")])
def test_mate_zero_and_moves_preserved_and_copied(engines, score, winner):
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    engine.analyse.side_effect = lambda b, l: {**response(b, l), "score": chess.engine.PovScore(score, chess.WHITE)}
    first = evaluate(engine, context, cache)
    expected = deepcopy(first)
    assert first["evaluation"]["winner"] == winner
    assert first["evaluation"]["mate"] == 0 and first["evaluation"]["cp"] is None
    first["evaluation"]["winner"] = "corrupted"
    first["pv"].clear()
    second = evaluate(engine, context, cache)
    assert second == expected
    assert isinstance(second["best_move"], chess.Move)
    second["best_move"].to_square = chess.A1
    assert evaluate(engine, context, cache) == expected
    assert engine.analyse.call_count == 1


def test_pipeline_reports_and_cross_game_mutation_isolation(engines):
    games = [game(), game(), game()]
    uncached = analyze_player_games(entries(games), depth=8)
    cache = EvaluationCache()
    cached = analyze_player_games(entries(games), depth=8, cache=cache)
    assert cached == uncached
    assert generate_player_report("fixture", cached) == generate_player_report("fixture", uncached)
    assert (cache.info().misses, cache.info().hits) == (5, 10)
    assert [e.analyse.call_count for e in engines[0]] == [5, 5, 5, 5, 0, 0]
    for engine in engines[0]:
        engine.quit.assert_called_once()
    cached["games"][0]["analysis"][0]["evaluation_before_detail"]["cp"] = 999
    assert cached["games"][1]["analysis"] == uncached["games"][1]["analysis"]
    assert game_analyzer.analyze_game(game(), depth=8, cache=cache) == uncached["games"][0]["analysis"]


def test_zero_move_and_empty_batch(engines):
    cache = EvaluationCache()
    assert analyze_player_games([], cache=cache)["games"] == []
    engines[1].assert_not_called()
    for _ in range(2):
        assert game_analyzer.analyze_game(chess.pgn.Game(), cache=cache) == []
    assert [e.analyse.call_count for e in engines[0]] == [1, 0]
    for engine in engines[0]:
        engine.quit.assert_called_once()


@pytest.mark.parametrize("api", ["game", "position"])
@pytest.mark.parametrize("failure_stage", ["identity", "search"])
def test_owners_cleanup_on_unexpected_failures(engines, api, failure_stage):
    if failure_stage == "identity":
        engines[2].side_effect = RuntimeError("unexpected metadata bug")
    else:
        launch = engines[1].side_effect
        def failing(*args, **kwargs):
            engine = launch(*args, **kwargs)
            engine.analyse.side_effect = chess.engine.EngineError("search failed")
            return engine
        engines[1].side_effect = failing
    with pytest.raises((RuntimeError, chess.engine.EngineError)):
        if api == "game":
            game_analyzer.analyze_game(game(), cache=EvaluationCache())
        else:
            engine_module.analyze_position(chess.STARTING_FEN, cache=EvaluationCache())
    engines[0][0].quit.assert_called_once()


def test_metadata_failure_during_request_only_bypasses_expected_exception(engines):
    engine, context = borrowed(engines)
    cache = EvaluationCache()
    engines[2].side_effect = keys.CacheIdentityUnavailable("unknown")
    evaluate(engine, context, cache)
    engines[2].side_effect = RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):
        evaluate(engine, context, cache)
    assert engine.analyse.call_count == 1


def test_position_api_shared_cache_still_launches_each_time(engines):
    cache = EvaluationCache()
    first = engine_module.analyze_position(chess.STARTING_FEN, cache=cache)
    assert engine_module.analyze_position(chess.STARTING_FEN, cache=cache) == first
    assert [e.analyse.call_count for e in engines[0]] == [1, 0]
    assert engines[3].call_count == 2
    for engine in engines[0]:
        engine.quit.assert_called_once()


def native_response(record):
    detail = record["evaluation"]
    if detail["type"] == "cp":
        score = chess.engine.Cp(detail["cp"])
    elif detail["mate"] == 0:
        score = chess.engine.MateGiven if detail["winner"] == "white" else chess.engine.Mate(0)
    else:
        score = chess.engine.Mate(detail["mate"])
    return {"score": chess.engine.PovScore(score, chess.WHITE),
            "depth": record["achieved_depth"],
            "pv": [chess.Move.from_uci(m) for m in record["pv"]]}


def test_baseline_replay_repeated_game_264_requests(engines):
    baseline = json.loads((ROOT / "data/reports/day7_baseline.json").read_text())
    saved = baseline["scenarios"]["same_game_three_times"]["correctness"][0]
    positions = {tuple(r["moves"]): native_response(r)
                 for r in saved["games"][0]["position_evaluations"]}
    launch = engines[1].side_effect
    def replay_launch(*args, **kwargs):
        engine = launch(*args, **kwargs)
        engine.analyse.side_effect = lambda b, l: deepcopy(positions[tuple(m.uci() for m in b.move_stack)])
        return engine
    engines[1].side_effect = replay_launch
    historical = load_multiple_pgn(ROOT / "data/raw/validation_games.pgn")[0]
    cache = EvaluationCache()
    actual = analyze_player_games(entries([historical] * 3), depth=8, cache=cache)
    assert (cache.info().misses, cache.info().hits) == (88, 176)
    assert cache.info().misses + cache.info().hits == 264
    assert engines[1].call_count == engines[3].call_count == 3
    assert [e.analyse.call_count for e in engines[0]] == [88, 0, 0]
    for engine, got, expected in zip(engines[0], actual["games"], saved["games"]):
        engine.quit.assert_called_once()
        assert got["analysis"] == expected["analysis"]
        assert got["stats"] == expected["statistics"]
        assert generate_weakness_report(got["analysis"], got["stats"]) == expected["weakness_report"]
    assert actual["summary"] == saved["summary"]
    assert generate_player_report("Benchmark White-side cohort", actual) == saved["player_report"]


def test_baseline_historical_mate_and_phase_replay(engines):
    baseline = json.loads((ROOT / "data/reports/day7_baseline.json").read_text())
    saved = baseline["scenarios"]["multiple_historical_games"]["correctness"][0]
    games = load_multiple_pgn(ROOT / "data/raw/validation_games.pgn")
    launch = engines[1].side_effect
    for historical, expected in zip(games, saved["games"]):
        positions = {tuple(r["moves"]): native_response(r) for r in expected["position_evaluations"]}
        def replay_launch(*args, **kwargs):
            engine = launch(*args, **kwargs)
            engine.analyse.side_effect = lambda b, l: deepcopy(positions[tuple(m.uci() for m in b.move_stack)])
            return engine
        engines[1].side_effect = replay_launch
        cache = EvaluationCache()
        first = game_analyzer.analyze_game(historical, depth=8, cache=cache)
        second = game_analyzer.analyze_game(historical, depth=8, cache=cache)
        assert first == second == expected["analysis"]
        assert game_analyzer.calculate_game_statistics(second) == expected["statistics"]
    assert second[-1]["evaluation_after_detail"] == {
        "type": "mate", "cp": None, "mate": 0, "winner": "black", "perspective": "white"}
    for engine in engines[0]:
        engine.quit.assert_called_once()


@pytest.mark.skipif(not (ROOT / engine_module.STOCKFISH_PATH).is_file(), reason="Project Stockfish unavailable")
def test_real_stockfish_smoke(monkeypatch):
    # Require hits for the audited binary; retain safe fallback for other builds.
    monkeypatch.chdir(ROOT)
    launch = chess.engine.SimpleEngine.popen_uci
    created = []
    def observed(*args, **kwargs):
        engine = launch(*args, **kwargs)
        engine.analyse = Mock(wraps=engine.analyse)
        engine.quit = Mock(wraps=engine.quit)
        created.append(engine)
        return engine
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", observed)
    cache = EvaluationCache()
    for _ in range(2):
        result = engine_module.analyze_position(chess.STARTING_FEN, depth=2, cache=cache)
        assert result["evaluation"]["perspective"] == "white"
        assert result["best_move"] in chess.Board().legal_moves
    assert len(created) == 2
    for engine in created:
        engine.quit.assert_called_once()
        assert engine.returncode.result(timeout=10) == 0
    assert sum(e.analyse.call_count for e in created) + cache.info().hits == 2
    _, digest = keys._fingerprint(ROOT / engine_module.STOCKFISH_PATH)
    if digest == VERIFIED_EMBEDDED_RESOURCES[0].executable_sha256:
        assert (cache.info().misses, cache.info().hits) == (1, 1)
        assert [e.analyse.call_count for e in created] == [1, 0]
