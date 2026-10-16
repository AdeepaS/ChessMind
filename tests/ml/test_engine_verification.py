"""Existing engine adapter, CP signs, PV validation and terminal mate semantics."""

from unittest.mock import Mock

import chess
import chess.engine
import pytest

from src.chessmind.engine.analyzer import serialize_evaluation
from src.chessmind.ml.labeling import engine_verification as verification


def response(score, pv, depth=12):
    return {"evaluation": serialize_evaluation(score), "pv": [chess.Move.from_uci(m) for m in pv], "depth": depth}


@pytest.mark.parametrize("black", [False, True])
def test_cp_signs_borrowed_engine_and_disagreement(labeling_case, monkeypatch, black):
    c, _, _ = labeling_case(fen=chess.Board().fen(), move="e2e4", after_pv=())
    if black:
        board = chess.Board()
        board.push_uci("e2e4")
        c, _, _ = labeling_case(fen=board.fen(), move="e7e5", after=chess.engine.Cp(500), after_pv=())
    rows = [response(chess.engine.Cp(0), (c["played_move_uci"],)),
            response(chess.engine.Cp(10 if black else -10), ("g1f3",) if black else ("e7e5",))]
    analyze = Mock(side_effect=rows)
    monkeypatch.setattr(verification, "analyze_board", analyze)
    engine, cache, context = Mock(), Mock(), Mock()
    result = verification.verify_candidate(c, engine, cache=cache, cache_context=context)
    assert result.status == "verified" and result.centipawn_loss == 10
    assert result.significant_status_changed and result.classification_changed
    assert result.original_cp_loss == 500 and result.cp_loss_change == -490
    assert analyze.call_count == 2
    assert analyze.call_args_list[0].kwargs["game"] is analyze.call_args_list[1].kwargs["game"]
    assert analyze.call_args.kwargs["cache_context"] is context
    engine.quit.assert_not_called()
    engine.configure.assert_not_called()


def test_mate_zero_preserved(labeling_case, monkeypatch):
    c, _, _ = labeling_case(fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g7",
                            before=chess.engine.Mate(1), after=chess.engine.MateGiven, after_pv=())
    monkeypatch.setattr(verification, "analyze_board", Mock(side_effect=[
        response(chess.engine.Mate(1), ("g6g7",)), response(chess.engine.MateGiven, (), 0)]))
    result = verification.verify_candidate(c, Mock())
    assert result.status == "verified"
    assert result.after.mate == 0 and result.after.winner == "white"
    assert result.centipawn_loss is None and result.mate_quality == "delivered_checkmate"


def test_black_mate_zero_winner_preserved(labeling_case, monkeypatch):
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1").mirror()
    c, _, _ = labeling_case(fen=board.fen(), move="g3g2", before=chess.engine.Mate(-1),
                            after=chess.engine.Mate(0), after_pv=())
    monkeypatch.setattr(verification, "analyze_board", Mock(side_effect=[
        response(chess.engine.Mate(-1), ("g3g2",)), response(chess.engine.Mate(0), (), 0)]))
    result = verification.verify_candidate(c, Mock())
    assert result.status == "verified" and result.after.winner == "black" and result.after.mate == 0


@pytest.mark.parametrize("failure", [RuntimeError("search failed"), ValueError("bad result")])
def test_explicit_failure(labeling_case, monkeypatch, failure):
    c, _, _ = labeling_case()
    monkeypatch.setattr(verification, "analyze_board", Mock(side_effect=failure))
    result = verification.verify_candidate(c, Mock())
    assert result.status == "failed" and result.reason
    assert result.before is None and result.centipawn_loss is None


@pytest.mark.parametrize("result", [response(chess.engine.Cp(0), ("a1a8",)),
    response(chess.engine.Cp(0), ()), response(chess.engine.Cp(0), ("c2b4",), 8),
    response(chess.engine.MateGiven, ("c2b4",))])
def test_bad_or_shallow_continuations_rejected(labeling_case, monkeypatch, result):
    c, _, _ = labeling_case()
    monkeypatch.setattr(verification, "analyze_board", Mock(return_value=result))
    assert verification.verify_candidate(c, Mock()).status == "failed"
