"""Deterministic regressions for native scores, mate policy, and reporting."""
import io
import json
from unittest.mock import Mock

import chess
import chess.engine
import chess.pgn
import pytest

from src.chessmind.engine.analyzer import analyze_board, serialize_evaluation
from src.chessmind.analysis import game_analyzer
from src.chessmind.analysis.game_analyzer import evaluate_move_quality, calculate_game_statistics
from src.chessmind.analysis.weakness_analyzer import detect_weakest_phase, find_worst_moves
from src.chessmind.players import multi_game_analyzer
from src.chessmind.players.player_statistics import aggregate_player_statistics
from src.chessmind.reports.weakness_report import generate_weakness_report
from src.chessmind.reports.player_report import generate_player_report

Cp = chess.engine.Cp
Mate = chess.engine.Mate


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("before,after,loss,classification", [
    (100, 20, 80, "Inaccuracy"), (0, 0, 0, "Excellent"),
    (20, 100, 0, "Excellent"), (100, -50, 150, "Mistake"),
    (300, 0, 300, "Blunder"),
])
def test_cp_quality(color, before, after, loss, classification):
    sign = 1 if color == chess.WHITE else -1
    result = evaluate_move_quality(
        serialize_evaluation(Cp(sign * before)),
        serialize_evaluation(Cp(sign * after)), color)
    assert result == {"centipawn_loss": loss, "classification": classification,
                      "mate_quality": None}


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("before,after,label,classification", [
    (Mate(3), Mate(5), "slower_mate", "Inaccuracy"),
    (Mate(5), Mate(3), "maintained_forced_mate", "Good"),
    (Mate(3), Mate(2), "maintained_forced_mate", "Good"),
    (Mate(3), Mate(3), "slower_mate", "Inaccuracy"),
    (Mate(-3), Mate(-5), "remains_forced_loss", "Good"),
    (Mate(-5), Mate(-3), "faster_defeat", "Inaccuracy"),
    (Mate(-3), Mate(-3), "remains_forced_loss", "Good"),
    (Mate(3), Cp(400), "lost_forced_mate", "Blunder"),
    (Mate(-3), Cp(-400), "escaped_forced_mate", "Good"),
    (Cp(-100), Mate(-3), "allowed_forced_mate", "Blunder"),
    (Cp(100), Mate(3), "achieved_forced_mate", "Good"),
    (Mate(3), Mate(-3), "reversed_to_losing_mate", "Blunder"),
    (Mate(-3), Mate(3), "reversed_to_winning_mate", "Good"),
    (Mate(1), chess.engine.MateGiven, "delivered_checkmate", "Good"),
    (Mate(-1), Mate(0), "checkmated", "Inaccuracy"),
    (Cp(-100), Mate(0), "allowed_forced_mate", "Blunder"),
])
def test_mate_quality_symmetry(color, before, after, label, classification):
    if color == chess.BLACK:
        before, after = -before, -after
    result = evaluate_move_quality(
        serialize_evaluation(before), serialize_evaluation(after), color)
    assert result == {"centipawn_loss": None, "classification": classification,
                      "mate_quality": label}


@pytest.mark.parametrize("score,winner", [
    (Mate(0), "black"), (chess.engine.MateGiven, "white"),
])
def test_zero_mate_preserves_winner(score, winner):
    data = serialize_evaluation(score)
    assert data == {"type": "mate", "cp": None, "mate": 0,
                    "winner": winner, "perspective": "white"}
    assert json.loads(json.dumps(data)) == data


def test_engine_normalizes_black_pov_and_keeps_legacy_projection():
    engine = Mock()
    engine.analyse.return_value = {
        "score": chess.engine.PovScore(Mate(3), chess.BLACK), "depth": 12}
    result = analyze_board(engine, chess.Board(), depth=12)
    assert result["score_cp"] == -99997
    assert result["evaluation"] == serialize_evaluation(Mate(-3))
    assert engine.analyse.call_count == 1


def test_real_pipeline_n_plus_one_terminal_and_json(monkeypatch):
    game = chess.pgn.read_game(io.StringIO("1. f3 e5 2. g4 Qh4# 0-1"))
    scores = [Cp(0), Cp(-20), Cp(-20), Mate(-1), Mate(0)]
    engine = Mock()
    engine.analyse.side_effect = [
        {"score": chess.engine.PovScore(score, chess.WHITE)} for score in scores]
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", Mock(return_value=engine))
    multi = multi_game_analyzer.analyze_player_games([{
        "game": game, "game_number": 1, "color": "black",
        "opponent": "White", "result": "0-1",
    }])
    analysis = multi["games"][0]["analysis"]
    assert engine.analyse.call_count == 5
    engine.quit.assert_called_once()
    assert analysis[-1]["san"] == "Qh4#"
    assert analysis[-1]["evaluation_after_detail"]["winner"] == "black"
    assert analysis[-1]["evaluation_after_detail"]["mate"] == 0
    assert analysis[-1]["mate_quality"] == "delivered_checkmate"
    report = generate_player_report("Black", multi)
    assert report["games"][0]["mate_transitions"][-1]["evaluation_after_detail"]["mate"] == 0
    assert json.loads(json.dumps(report)) == report


def test_engine_closed_on_analysis_failure(monkeypatch):
    engine = Mock()
    engine.analyse.side_effect = RuntimeError("search failed")
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", Mock(return_value=engine))
    with pytest.raises(RuntimeError, match="search failed"):
        game_analyzer.analyze_game(chess.pgn.Game())
    engine.quit.assert_called_once()


def make_move(before, after, phase="middlegame"):
    before, after = serialize_evaluation(before), serialize_evaluation(after)
    return {
        "color": "white", "phase": phase, "move_number": 1, "san": "e4",
        "best_move": "e2e4", "fen_before": chess.STARTING_FEN,
        "evaluation_before_detail": before, "evaluation_after_detail": after,
        **evaluate_move_quality(before, after, chess.WHITE),
    }


def test_aggregate_denominators_weakness_and_reports():
    moves = [make_move(Cp(100), Cp(40), "opening"),
             make_move(Mate(3), Cp(400))]
    stats = calculate_game_statistics(moves)
    assert stats["white"]["moves"] == 2
    assert stats["white"]["cp_moves"] == 1
    assert stats["white"]["mate_moves"] == 1
    assert stats["white"]["total_centipawn_loss"] == 60
    assert stats["white"]["acpl"] == 60
    assert stats["white"]["phases"]["middlegame"]["acpl"] is None
    assert stats["black"]["acpl"] is None
    weaknesses = detect_weakest_phase(stats)
    assert weaknesses["white"]["phase"] == "middlegame"
    assert find_worst_moves(moves, weaknesses)["white"][0]["mate_quality"] == "lost_forced_mate"
    single = generate_weakness_report(moves, stats)
    assert single["white"]["worst_moves"][0]["centipawn_loss"] is None
    assert json.loads(json.dumps(single)) == single
    records = [{
        "game_number": i, "color": "white", "result": "1-0",
        "opponent": "Opponent", "stats": stats, "analysis": moves,
    } for i in range(3)]
    summary = aggregate_player_statistics(records)
    assert summary["cp_moves"] == 3
    assert summary["total_moves"] == 6
    assert summary["overall_acpl"] == 60
    report = generate_player_report("White", {"summary": summary, "games": records})
    assert "middlegame" in report["recurring_weaknesses"]["detected_phases"]
    assert json.loads(json.dumps(report)) == report


def test_only_mate_statistics_and_worst_move_order():
    moves = [make_move(Mate(3), Mate(2)), make_move(Mate(3), Cp(100)),
             make_move(Cp(500), Cp(0)), make_move(Mate(3), Mate(5))]
    stats = calculate_game_statistics(moves[:2])
    assert stats["white"]["acpl"] is None
    assert stats["white"]["total_centipawn_loss"] == 0
    ranked = find_worst_moves(moves, {"white": {"phase": "middlegame"}})["white"]
    assert ranked == [moves[1], moves[2], moves[3], moves[0]]
    record = {"stats": stats, "color": "white", "result": "1-0"}
    assert aggregate_player_statistics([record])["overall_acpl"] is None

def test_large_genuine_cp_is_not_inferred_as_mate():
    result = evaluate_move_quality(
        serialize_evaluation(Cp(99999)), serialize_evaluation(Cp(99900)), chess.WHITE)
    assert result["centipawn_loss"] == 99
    assert result["mate_quality"] is None


def test_weighted_cp_counts_and_mate_errors_outrank_good_moves():
    first = [make_move(Cp(100), Cp(40)), make_move(Mate(3), Mate(5))]
    second = [make_move(Cp(100), Cp(80)), make_move(Cp(0), Cp(0))]
    records = [{"stats": calculate_game_statistics(moves), "color": "white", "result": "1-0"}
               for moves in (first, second)]
    summary = aggregate_player_statistics(records)
    assert summary["cp_moves"] == 3
    assert summary["total_moves"] == 4
    assert summary["overall_acpl"] == 26.67
    assert summary["phase_statistics"]["middlegame"]["acpl"] == 26.67
    ranked = find_worst_moves(
        first + second, {"white": {"phase": "middlegame"}})["white"]
    assert ranked.index(first[1]) < ranked.index(second[0])
