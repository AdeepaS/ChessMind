"""Sequential ownership, cache lifetime, and python-chess game boundaries."""

import asyncio
from dataclasses import replace
import io
from pathlib import Path
from unittest.mock import Mock

import chess
import chess.engine
import chess.pgn
import pytest

from src.chessmind.engine import analyzer, evaluation_cache_keys as keys
from src.chessmind.engine.evaluation_cache import EvaluationCache
from src.chessmind.analysis import game_analyzer
from src.chessmind.players import multi_game_analyzer as multi
from src.chessmind.reports.player_report import generate_player_report


def game(pgn="1. e4 e5 *"):
    return chess.pgn.read_game(io.StringIO(pgn))


def entries(*games):
    return [{"game_number": i, "game": g, "color": "white",
             "opponent": "fixture", "result": g.headers.get("Result", "*")}
            for i, g in enumerate(games, 1)]


@pytest.fixture
def engines(monkeypatch, tmp_path):
    binary = tmp_path / "engine"
    binary.write_bytes(b"lifecycle fixture")
    option = chess.engine.Option("Threads", "spin", 1, 1, 16, None)
    metadata = keys.metadata_from_values({"name": "fixture"}, {"Threads": option},
                                         {"Threads": 1}, {"Threads": 1})
    created = []

    def launch(*args, **kwargs):
        engine = Mock()
        def respond(board, limit, **kwargs):
            score = chess.engine.Mate(0) if board.is_checkmate() else chess.engine.Cp(0)
            return {"score": chess.engine.PovScore(score, board.turn), "depth": limit.depth,
                    "pv": list(board.legal_moves)[:1]}
        engine.analyse.side_effect = respond
        created.append(engine)
        return engine

    factory = Mock(side_effect=launch)
    snapshot = Mock(return_value=metadata)
    fingerprint = Mock(wraps=keys._fingerprint)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", factory)
    monkeypatch.setattr(analyzer, "STOCKFISH_PATH", str(binary))
    monkeypatch.setattr(game_analyzer, "STOCKFISH_PATH", str(binary))
    monkeypatch.setattr(analyzer, "snapshot_engine_metadata", snapshot)
    monkeypatch.setattr(keys, "_fingerprint", fingerprint)
    return created, factory, snapshot, fingerprint


def test_default_keeps_per_game_ownership_and_legacy_search_calls(engines):
    result = multi.analyze_player_games(entries(game(), game()))
    assert len(result["games"]) == 2
    assert engines[1].call_count == 2
    for engine in engines[0]:
        engine.quit.assert_called_once()
        assert engine.analyse.call_count == 3
        assert all(not call.kwargs for call in engine.analyse.call_args_list)


@pytest.mark.parametrize("cache", [None, EvaluationCache(0)])
def test_shared_ownership_tokens_and_disabled_metadata(engines, cache):
    multi.analyze_player_games(entries(game(), game()), cache=cache, reuse_engine=True)
    engines[1].assert_called_once()
    engine = engines[0][0]
    engine.quit.assert_called_once()
    tokens = [call.kwargs["game"] for call in engine.analyse.call_args_list]
    assert len(tokens) == 6
    assert all(t is tokens[0] for t in tokens[:3])
    assert all(t is tokens[3] for t in tokens[3:])
    assert tokens[0] is not tokens[3]
    engines[2].assert_not_called()
    engines[3].assert_not_called()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("as_iterator", [False, True])
def test_empty_input_never_launches(engines, reuse, as_iterator):
    result = multi.analyze_player_games(iter([]) if as_iterator else [], reuse_engine=reuse)
    assert result["games"] == []
    engines[1].assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_borrowed_engine_never_launches_or_closes(engines, fails):
    engine = engines[1]()
    engines[1].reset_mock()
    if fails:
        engine.analyse.side_effect = RuntimeError("search failed")
        with pytest.raises(RuntimeError, match="search failed"):
            game_analyzer.analyze_game(game(), engine=engine)
    else:
        assert len(game_analyzer.analyze_game(game(), engine=engine)) == 2
    engine.quit.assert_not_called()
    engines[1].assert_not_called()


@pytest.mark.parametrize("failure", ["startup", "identity", "search", "shutdown"])
def test_owner_propagates_errors_without_retry(engines, failure):
    error = RuntimeError(failure)
    if failure == "startup":
        engines[1].side_effect = error
    elif failure == "identity":
        engines[2].side_effect = error
    else:
        launch = engines[1].side_effect
        def failing(*args, **kwargs):
            engine = launch(*args, **kwargs)
            getattr(engine, "analyse" if failure == "search" else "quit").side_effect = error
            return engine
        engines[1].side_effect = failing
    with pytest.raises(RuntimeError, match=failure):
        multi.analyze_player_games(entries(game(), game()), cache=EvaluationCache(), reuse_engine=True)
    engines[1].assert_called_once()
    for engine in engines[0]:
        engine.quit.assert_called_once()


def test_shared_cache_once_all_hit_game_then_mixed_hits(engines):
    cache = EvaluationCache()
    result = multi.analyze_player_games(entries(game(), game(), game("1. e4 c5 *")),
                                       cache=cache, reuse_engine=True)
    engine = engines[0][0]
    assert engine.analyse.call_count == 4
    assert (cache.info().hits, cache.info().misses) == (5, 4)
    engines[3].assert_called_once()
    assert engines[2].call_count == 10  # Initial identity plus all nine requests.
    calls = engine.analyse.call_args_list
    assert calls[0].kwargs["game"] is calls[2].kwargs["game"]
    assert calls[3].kwargs["game"] is not calls[0].kwargs["game"]
    assert result["games"][0]["analysis"] == result["games"][1]["analysis"]
    assert result["games"][0]["stats"] == result["games"][1]["stats"]
    # Different game tokens still hit identical request keys; outputs are copies.
    result["games"][0]["analysis"][0]["evaluation_before_detail"]["cp"] = 999
    assert result["games"][1]["analysis"][0]["evaluation_before_detail"]["cp"] == 0
    engine.quit.assert_called_once()


def test_unsupported_identity_attempted_only_once(engines):
    engines[2].side_effect = keys.CacheIdentityUnavailable("unidentified resource")
    cache = EvaluationCache()
    multi.analyze_player_games(entries(game(), game()), cache=cache, reuse_engine=True)
    engines[2].assert_called_once()
    assert engines[0][0].analyse.call_count == 6
    assert cache.info().hits == cache.info().misses == 0
    engines[0][0].quit.assert_called_once()


@pytest.mark.parametrize("change", ["threads", "resource"])
def test_changed_configuration_cannot_hit_shared_identity(engines, change):
    original = engines[2].return_value
    changed = replace(original, target_config=(("threads", 2),)) if change == "threads" else replace(
        original, options=original.options + (keys.OptionSpec("evalfile", "string", "new.nnue", None, None, ()),),
        config=original.config + (("evalfile", "new.nnue"),))
    engines[2].side_effect = [original] * 4 + [changed] * 3
    cache = EvaluationCache()
    multi.analyze_player_games(entries(game(), game()), cache=cache, reuse_engine=True)
    assert engines[0][0].analyse.call_count == 6
    assert cache.info().hits == 0
    engines[3].assert_called_once()


def test_mates_phases_statistics_and_reports_match_default(engines):
    games = entries(game("1. f3 e5 2. g4 Qh4# 0-1"), game("1. f3 e5 2. g4 Qh4# 0-1"))
    expected = multi.analyze_player_games(games)
    cache = EvaluationCache()
    actual = multi.analyze_player_games(games, cache=cache, reuse_engine=True)
    assert actual == expected
    assert generate_player_report("fixture", actual) == generate_player_report("fixture", expected)
    last = actual["games"][1]["analysis"][-1]
    assert last["evaluation_after_detail"] == {
        "type": "mate", "cp": None, "mate": 0, "winner": "black", "perspective": "white"}
    assert last["centipawn_loss"] is None
    stats = actual["games"][1]["stats"]["black"]
    assert stats["cp_moves"] == 1 and stats["mate_moves"] == 1
    assert all(row["phase"] == "opening" for row in actual["games"][1]["analysis"])
    assert (cache.info().hits, cache.info().misses) == (5, 5)


def test_default_supports_old_monkeypatched_game_signature(monkeypatch):
    def legacy(game, depth=12):
        return []
    monkeypatch.setattr(multi, "analyze_game", legacy)
    assert multi.analyze_player_games(entries(game()))["games"][0]["analysis"] == []


def test_python_chess_repeated_and_changed_game_tokens():
    async def run():
        protocol = chess.engine.UciProtocol()
        transport = chess.engine.MockTransport(protocol)
        transport.expect("uci", ["id name boundary fixture", "uciok"])
        await protocol.initialize()
        first, second = object(), object()
        for token, boundary in ((first, True), (first, False), (second, True)):
            if boundary:
                transport.expect("ucinewgame")
                transport.expect("isready", ["readyok"])
            transport.expect("position startpos")
            transport.expect("go depth 1", ["info depth 1 score cp 0 pv e2e4", "bestmove e2e4"])
            await protocol.analyse(chess.Board(), chess.engine.Limit(depth=1), game=token)
        transport.assert_done()
    asyncio.run(run())


@pytest.mark.skipif(not Path(analyzer.STOCKFISH_PATH).is_file(), reason="Stockfish unavailable")
@pytest.mark.parametrize("fully_cached_middle", [False, True])
def test_real_shared_engine_smoke_and_boundary_after_hits(monkeypatch, fully_cached_middle):
    created, commands = [], []
    launch = chess.engine.SimpleEngine.popen_uci
    def observed(*args, **kwargs):
        engine = launch(*args, **kwargs)
        created.append(engine)
        engine.analyse = Mock(wraps=engine.analyse)
        engine.quit = Mock(wraps=engine.quit)
        send = engine.protocol.send_line
        def send_line(line):
            commands.append(line)
            send(line)
        engine.protocol.send_line = send_line
        return engine
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", observed)
    cache = EvaluationCache()
    games = [game(), game("1. e4 c5 *")]
    if fully_cached_middle:
        games.insert(1, game())
    result = multi.analyze_player_games(entries(*games),
                                       depth=2, cache=cache, reuse_engine=True)
    assert len(created) == 1
    engine = created[0]
    engine.quit.assert_called_once()
    assert engine.returncode.result(timeout=10) == 0
    assert commands.count("ucinewgame") == (2 if cache.info().hits else len(games))
    assert all(len(g["analysis"]) == 2 for g in result["games"])
    assert all(row["evaluation_after_detail"]["perspective"] == "white"
               for g in result["games"] for row in g["analysis"])
    assert engine.analyse.call_count + cache.info().hits == len(games) * 3
    if cache.info().hits:
        assert (cache.info().misses, cache.info().hits) == (4, 5 if fully_cached_middle else 2)
        # python-chess emits the next game's boundary after two cache hits.
        boundary = [i for i, command in enumerate(commands) if command == "ucinewgame"][1]
        assert commands[boundary + 1] == "isready"
        assert commands[boundary + 2].endswith("moves e2e4 c7c5")
