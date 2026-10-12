"""Missing integration regressions identified by the Day 6 Step 4 audit.

Existing mate and phase unit policies are reused, not redefined here.
Constructed move records test aggregation; engine tests mock only the UCI boundary.
"""
import copy
import io
import json
from unittest.mock import Mock

import chess
import chess.engine
import chess.pgn
import pytest

from src.chessmind.analysis.game_analyzer import analyze_game, calculate_game_statistics
from src.chessmind.analysis.weakness_analyzer import detect_weakest_phase, find_worst_moves
from src.chessmind.engine.analyzer import analyze_position
from src.chessmind.pgn.parser import load_pgn
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.players.player_identifier import find_player_games
from src.chessmind.players import multi_game_analyzer
from src.chessmind.players.player_statistics import aggregate_player_statistics
from src.chessmind.players.recurring_weakness_analyzer import detect_recurring_weaknesses
from src.chessmind.reports.weakness_report import generate_weakness_report
from src.chessmind.reports.player_report import generate_player_report, save_player_report


def move_record(color, phase, loss, classification, number=1):
    """Aggregation input, intentionally independent of quality calculation."""
    return {
        "color": color, "phase": phase, "centipawn_loss": loss,
        "classification": classification, "move_number": number, "san": "e4",
        "best_move": None, "fen_before": chess.STARTING_FEN,
        "mate_quality": ("lost_forced_mate" if classification == "Blunder"
                         else "maintained_forced_mate") if loss is None else None,
    }


def game_record(moves, color="white", number=1, result="1-0"):
    return {
        "game_number": number, "color": color, "opponent": "Opponent",
        "result": result, "analysis": moves, "stats": calculate_game_statistics(moves),
    }


@pytest.fixture
def mixed_moves():
    return [
        move_record("white", "opening", 10, "Excellent"),
        move_record("black", "opening", 200, "Mistake"),
        move_record("white", "opening", 0, "Excellent"),
        move_record("white", "middlegame", 60, "Inaccuracy"),
        move_record("black", "middlegame", None, "Blunder"),
        move_record("white", "middlegame", None, "Blunder"),
        move_record("black", "endgame", None, "Good"),
        move_record("white", "endgame", None, "Good"),
    ]


def test_two_color_phase_and_classification_accounting(mixed_moves):
    original = copy.deepcopy(mixed_moves)
    stats = calculate_game_statistics(mixed_moves)
    assert mixed_moves == original
    expected = {"white": (5, 3, 2, 70, 70 / 3), "black": (3, 1, 2, 200, 200)}
    for color, (moves, cp, mate, loss, acpl) in expected.items():
        data = stats[color]
        assert (data["moves"], data["cp_moves"], data["mate_moves"],
                data["total_centipawn_loss"]) == (moves, cp, mate, loss)
        assert data["acpl"] == pytest.approx(acpl)
        assert data["mate_errors"] == 1
        assert sum(data[key] for key in
                   ("excellent", "good", "inaccuracy", "mistake", "blunder")) == moves
        for key in ("moves", "cp_moves", "mate_moves", "mate_errors", "total_centipawn_loss"):
            assert sum(p[key] for p in data["phases"].values()) == data[key]
        assert data["phases"]["endgame"]["acpl"] is None
    assert stats["white"]["phases"]["opening"]["acpl"] == 5
    assert stats["white"]["phases"]["middlegame"]["acpl"] == 60
    assert stats["black"]["phases"]["middlegame"]["acpl"] is None
    assert stats["white"]["excellent"] == 2
    assert stats["black"]["mistake"] == 1
    assert stats["white"]["good"] == stats["black"]["good"] == 1


def test_weighted_player_pipeline_with_color_changes_and_zero_cp_games(monkeypatch):
    games = [chess.pgn.Game() for _ in range(5)]
    colors = ["white", "black", "white", "black", "white"]
    results = ["1-0", "0-1", "0-1", "1/2-1/2", "*"]
    batches = [
        [move_record("white", "opening", 10, "Excellent") for _ in range(10)],
        [move_record("black", "opening", 60, "Inaccuracy") for _ in range(5)],
        [move_record("white", "middlegame", None, "Blunder")],
        [move_record("black", "endgame", None, "Good")],
        [],
    ]
    for game, color, result in zip(games, colors, results):
        game.headers.update(White="Target" if color == "white" else "Opponent",
                            Black="Target" if color == "black" else "Opponent",
                            Result=result)
    # Distractor statistics must never enter the target player's aggregate.
    batches[0].append(move_record("black", "opening", 900, "Blunder"))
    fake = Mock(side_effect=copy.deepcopy(batches))
    monkeypatch.setattr(multi_game_analyzer, "analyze_game", fake)
    selected = find_player_games(games, "target")
    multi = multi_game_analyzer.analyze_player_games(selected)
    summary = multi["summary"]
    assert (summary["total_games"], summary["wins"], summary["losses"],
            summary["draws"], summary["unfinished"]) == (5, 2, 1, 1, 1)
    assert (summary["total_moves"], summary["cp_moves"], summary["mate_moves"]) == (17, 15, 2)
    assert summary["total_centipawn_loss"] == 400
    assert summary["overall_acpl"] == round(400 / 15, 2)
    assert summary["overall_acpl"] != 35
    assert summary["classifications"] == {
        "excellent": 10, "good": 1, "inaccuracy": 5, "mistake": 0, "blunder": 1}
    assert summary["phase_statistics"]["opening"]["acpl"] == 26.67
    assert summary["phase_statistics"]["middlegame"]["acpl"] is None
    report = generate_player_report("Target", multi)
    assert [g["color"] for g in report["games"]] == colors
    assert [g["result"] for g in report["games"]] == results
    assert [g["moves_analyzed"] for g in report["games"]] == [10, 5, 1, 1, 0]
    assert [g["acpl"] for g in report["games"]] == [10, 60, None, None, None]
    assert report["recurring_weaknesses"]["detected_phases"] == []
    assert fake.call_count == 5


@pytest.mark.parametrize("has_error,game_count,recurring", [
    (False, 3, False), (True, 2, False), (True, 3, True),
])
def test_all_mate_reports_null_acpl_and_evidence_thresholds(
        has_error, game_count, recurring, tmp_path):
    moves = [move_record("black", "endgame", None, "Blunder" if has_error else "Good")]
    records = [game_record(moves, "black", n, "0-1") for n in range(game_count)]
    multi = {"games": records, "summary": aggregate_player_statistics(records)}
    single = generate_weakness_report(moves, records[0]["stats"])
    report = generate_player_report("Target", multi)
    assert single["black"]["overall_acpl"] is None
    assert single["white"]["overall_acpl"] is None
    assert single["white"]["worst_moves"] == []
    assert bool(single["black"]["worst_moves"]) == has_error
    assert multi["summary"]["overall_acpl"] is None
    assert multi["summary"]["total_centipawn_loss"] == 0
    assert multi["summary"]["cp_moves"] == 0
    assert multi["summary"]["mate_moves"] == game_count
    assert all(g["acpl"] is None and g["cp_moves"] == 0 for g in report["games"])
    detail = report["recurring_weaknesses"]["phase_details"]["endgame"]
    assert detail["is_recurring_weakness"] is recurring
    assert detail["eligible_games"] == (game_count if has_error else 0)
    assert detail["game_acpls"] == ([None] * game_count if has_error else [])
    assert detail["overall_acpl"] is None
    assert json.loads(json.dumps(single, allow_nan=False)) == single
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    path = save_player_report(report, tmp_path / "report.json")
    assert json.loads(path.read_text()) == report
    assert '"acpl": null' in path.read_text()


def test_recurring_frequency_counts_games_not_weighted_acpl():
    # One long weak game dominates aggregate ACPL, but two short good games
    # mean only 1/3 eligible games are weak: no recurring weakness at 60%.
    records = [game_record([
        move_record("white", "opening", loss, "Inaccuracy" if loss else "Excellent")
        for _ in range(count)], number=index)
        for index, (count, loss) in enumerate([(100, 100), (5, 0), (5, 0)])]
    multi = {"games": records, "summary": aggregate_player_statistics(records)}
    assert multi["summary"]["phase_statistics"]["opening"]["acpl"] == 90.91
    weakness = detect_recurring_weaknesses(multi)["opening"]
    assert (weakness["eligible_games"], weakness["weak_games"]) == (3, 1)
    assert weakness["weak_frequency"] == 0.33
    assert not weakness["is_recurring_weakness"]


def test_worst_move_ties_are_stable_and_do_not_mutate_input():
    moves = [
        move_record("black", "endgame", 500, "Blunder", 1),
        move_record("black", "endgame", None, "Blunder", 2),
        move_record("black", "endgame", None, "Blunder", 3),
        move_record("black", "endgame", 500, "Blunder", 4),
    ]
    original = copy.deepcopy(moves)
    weaknesses = detect_weakest_phase(calculate_game_statistics(moves))
    result = find_worst_moves(moves, weaknesses, top_n=4)
    assert [m["move_number"] for m in result["black"]] == [2, 3, 1, 4]
    assert result == find_worst_moves(moves, weaknesses, top_n=4)
    assert moves == original


@pytest.fixture
def engine_factory(monkeypatch):
    engines = []
    snapshots = []

    def create(*args):
        engine = Mock()
        seen = []

        def analyse(board, limit):
            # Snapshot immediately; the real analyzer mutates its board later.
            seen.append((board.fen(), limit.depth))
            return {"score": chess.engine.PovScore(
                chess.engine.Cp(-10 * len(board.move_stack)), chess.WHITE)}

        engine.analyse.side_effect = analyse
        engines.append(engine)
        snapshots.append(seen)
        return engine

    factory = Mock(side_effect=create)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", factory)
    return factory, engines, snapshots


def test_repeated_analysis_preserves_board_order_and_reuses_adjacent_scores(engine_factory):
    factory, engines, snapshots = engine_factory
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 2. Nf3 Nc6 *"))
    original = str(game)
    first = analyze_game(game, depth=7)
    second = analyze_game(game, depth=7)
    assert first == second
    assert str(game) == original
    board = game.board()
    expected_fens = [board.fen()]
    for move in game.mainline_moves():
        board.push(move)
        expected_fens.append(board.fen())
    assert factory.call_count == 2
    assert engines[0] is not engines[1]
    for engine, seen in zip(engines, snapshots):
        assert seen == [(fen, 7) for fen in expected_fens]
        assert engine.analyse.call_count == 5
        engine.quit.assert_called_once()
    for index, move in enumerate(first):
        assert move["fen_before"] == expected_fens[index]
        assert move["fen_after"] == expected_fens[index + 1]
        if index:
            assert move["evaluation_before_detail"] == first[index - 1]["evaluation_after_detail"]


def test_capture_causing_endgame_belongs_to_pre_capture_phase(engine_factory):
    board = chess.Board("r3k2r/pp3ppp/5b2/8/8/2B5/PP3PPP/R3K2R w KQkq - 0 25")
    assert board.is_valid()
    board.push_san("Bxf6")  # 26 -> 23 non-pawn units, queenless.
    board.push_san("gxf6")
    game = chess.pgn.Game.from_board(board)
    analysis = analyze_game(game)
    assert [(m["san"], m["color"], m["phase"]) for m in analysis] == [
        ("Bxf6", "white", "middlegame"), ("gxf6", "black", "endgame")]
    stats = calculate_game_statistics(analysis)
    assert stats["white"]["phases"]["middlegame"]["moves"] == 1
    assert stats["black"]["phases"]["endgame"]["moves"] == 1


def test_midgame_engine_error_closes_resource_and_next_run_is_clean(engine_factory):
    _, engines, _ = engine_factory
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 *"))
    original = str(game)
    # First engine fails after the initial and first post-move evaluations.
    factory = engine_factory[0]
    create = factory.side_effect

    def create_failing(*args):
        engine = create(*args)
        normal = engine.analyse.side_effect

        def fail_later(board, limit):
            if len(board.move_stack) == 2:
                raise RuntimeError("midgame failure")
            return normal(board, limit)

        engine.analyse.side_effect = fail_later
        return engine

    factory.side_effect = create_failing
    with pytest.raises(RuntimeError, match="midgame failure"):
        analyze_game(game)
    engines[0].quit.assert_called_once()
    factory.side_effect = create
    assert len(analyze_game(game)) == 2
    engines[1].quit.assert_called_once()
    assert str(game) == original


@pytest.mark.parametrize("fails", [False, True])
def test_standalone_position_api_closes_engine(monkeypatch, fails):
    engine = Mock()
    if fails:
        engine.analyse.side_effect = RuntimeError("position failure")
    else:
        engine.analyse.return_value = {
            "score": chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE)}
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", Mock(return_value=engine))
    if fails:
        with pytest.raises(RuntimeError, match="position failure"):
            analyze_position(chess.STARTING_FEN)
    else:
        assert analyze_position(chess.STARTING_FEN)["evaluation"]["cp"] == 35
    engine.quit.assert_called_once()


def test_empty_pipeline_and_zero_move_game_reports(engine_factory):
    factory, engines, _ = engine_factory
    empty = multi_game_analyzer.analyze_player_games([])
    factory.assert_not_called()
    report = generate_player_report("Nobody", empty)
    assert report["games"] == []
    assert report["overall_performance"]["overall_acpl"] is None
    assert report["recurring_weaknesses"]["detected_phases"] == []
    assert analyze_game(chess.pgn.Game()) == []
    assert engines[0].analyse.call_count == 1  # Existing N+1 contract, N=0.
    engines[0].quit.assert_called_once()
    single = generate_weakness_report([], calculate_game_statistics([]))
    for color in ("white", "black"):
        assert single[color]["overall_acpl"] is None
        assert single[color]["weakest_phase"] is None
        assert single[color]["worst_moves"] == []
    assert json.loads(json.dumps(single, allow_nan=False)) == single


def test_single_game_loader_empty_contract(tmp_path):
    path = tmp_path / "empty.pgn"
    path.write_text("")
    with pytest.raises(ValueError, match="No chess game found"):
        load_pgn(path)


def test_second_malformed_game_rejects_partial_multi_game_result(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text('[Event "Valid"]\n\n1. e4 e5 *\n\n'
                    '[Event "Invalid"]\n\n1. e4 e5 2. Ke3 *\n')
    with pytest.raises(ValueError, match="Invalid PGN game 2"):
        load_multiple_pgn(path)


@pytest.mark.parametrize("movetext", [
    '[Event "Illegal move"]\n\n1. e4 e5 2. Ke3 *\n',
    # Truncated promotion: python-chess recognizes a8 but reports the missing
    # promotion piece as an error. An ordinary unfinished game is valid below.
    '[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]\n\n1. a8=',
])
def test_single_game_loader_rejects_reported_parse_errors(tmp_path, movetext, engine_factory):
    parsed = chess.pgn.read_game(io.StringIO(movetext))
    assert parsed.errors  # Ensure the fixture exercises an actual parser error.
    path = tmp_path / "invalid.pgn"
    path.write_text(movetext, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid PGN game 1"):
        analyze_game(load_pgn(path))
    engine_factory[0].assert_not_called()


@pytest.mark.parametrize("movetext,result,plies,terminal", [
    ("1. e4 e5 2. Nf3", "*", 3, False),
    ("1. e4 e5 1-0", "1-0", 2, False),  # Resignation need not be checkmate.
    ("1. f3 e5 2. g4 Qh4# 0-1", "0-1", 4, True),
])
def test_single_game_loader_preserves_valid_games(tmp_path, movetext, result, plies, terminal):
    path = tmp_path / "valid.pgn"
    path.write_text(f'[Event "Valid"]\n[Result "{result}"]\n\n{movetext}\n\n'
                    '[Event "Second game"]\n\n1. d4 *\n', encoding="utf-8")
    game = load_pgn(path)
    assert isinstance(game, chess.pgn.Game)
    assert not game.errors
    assert game.headers["Event"] == "Valid"  # Still returns only the first game.
    assert game.headers["Result"] == result
    assert len(list(game.mainline_moves())) == plies
    assert game.end().board().is_game_over() is terminal


def test_single_game_loader_to_analysis_and_statistics(tmp_path, engine_factory):
    path = tmp_path / "game.pgn"
    path.write_text('[White "Alice"]\n[Black "Bob"]\n\n1. e4 e5 *\n', encoding="utf-8")
    game = load_pgn(path)
    analysis = analyze_game(game, depth=8)
    assert [m["san"] for m in analysis] == ["e4", "e5"]
    stats = calculate_game_statistics(analysis)
    assert stats["white"]["moves"] == stats["black"]["moves"] == 1
    assert stats["white"]["acpl"] == 10
    assert stats["black"]["acpl"] == 0
    factory, engines, snapshots = engine_factory
    assert factory.call_count == 1
    assert len(snapshots[0]) == 3
    engines[0].quit.assert_called_once()
