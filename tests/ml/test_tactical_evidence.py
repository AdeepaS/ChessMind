"""Narrow legal capture/mating evidence, with explicit counterexamples."""

from dataclasses import replace

import chess
import chess.engine

from src.chessmind.ml.labeling.tactical_evidence import capture_evidence, missed_mate_evidence


def test_hanging_capture_pv_material_support(labeling_case):
    c, f, v = labeling_case()
    e = capture_evidence(c, f, v)
    assert e.supported and e.captures_moved_piece
    assert not e.victim_defended and not e.previously_attacked
    assert e.pv_material_gain == e.minimum_gain_after_immediate_recapture == 3


def test_defended_capture_recovers_material(labeling_case):
    # c3 pawn can recapture the rook on b4: losing a knight does not win the exchange.
    c, f, v = labeling_case(fen="1r5k/8/8/8/8/2P5/2N5/7K w - - 0 1")
    e = capture_evidence(c, f, v)
    assert e.victim_defended and not e.supported
    assert e.minimum_gain_after_immediate_recapture == -2
    assert e.reason == "material_recovered_or_compensated"


def test_pinned_illegal_capture(labeling_case):
    # e7 knight geometrically attacks c6 but cannot expose its king to e1 rook.
    c, f, v = labeling_case(fen="4k3/1B2n3/8/8/8/8/8/K3R3 w - - 0 1",
                            move="b7c6", after_pv=("e7c6",))
    e = capture_evidence(c, f, v)
    assert not e.supported and e.reason == "pv_does_not_start_with_legal_capture"


def test_already_attacked_abstains(labeling_case):
    # A rook moving along the already attacked b-file does not establish a new threat.
    c, f, v = labeling_case(fen="1r5k/8/8/8/8/8/1R6/7K w - - 0 1", move="b2b4")
    assert capture_evidence(c, f, v).reason == "victim_already_geometrically_attacked"


def test_low_loss_and_short_pv_not_proof(labeling_case):
    c, f, v = labeling_case(after=chess.engine.Cp(-10))
    assert capture_evidence(c, f, v).reason == "evaluation_does_not_support_significant_error"
    c, f, v = labeling_case(after_pv=("b8b4",))
    assert capture_evidence(c, f, v).reason == "continuation_too_short_for_material_check"


def test_no_capture_and_compensation_in_pv(labeling_case):
    c, f, v = labeling_case(after_pv=("h8g8",))
    assert not capture_evidence(c, f, v).supported
    c, f, v = labeling_case(fen="1r5k/8/8/8/8/2P5/2N5/7K w - - 0 1",
                            after_pv=("b8b4", "c3b4", "h8g8", "h1g1"))
    assert capture_evidence(c, f, v).pv_material_gain == -2


def test_material_earned_by_played_capture_counts_as_compensation(labeling_case):
    # Nxb4 wins a queen before ...Rxb4: the rook's immediate capture is not a net gain.
    c, f, v = labeling_case(fen="1r5k/8/8/8/1q6/8/2N5/7K w - - 0 1")
    evidence = capture_evidence(c, f, v)
    assert evidence.pv_material_gain == 3
    assert not evidence.supported and evidence.reason == "material_recovered_or_compensated"


def test_missed_mate_concrete_alternative(labeling_case):
    c, _, v = labeling_case(fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Cp(500), before_pv=("g6g7",), after_pv=("h8h7",))
    e = missed_mate_evidence(c, v)
    assert e.supported and e.continuation == ("g6g7",)
    assert not missed_mate_evidence(c, replace(v, before_pv=("f7f6",))).supported


def test_retained_mate_and_opponent_mate_not_missed(labeling_case):
    c, _, v = labeling_case(before=chess.engine.Mate(2), after=chess.engine.Mate(5))
    assert missed_mate_evidence(c, v).reason == "forced_mate_retained_or_delayed"
    c, _, v = labeling_case(before=chess.engine.Cp(0), after=chess.engine.Mate(-3))
    assert missed_mate_evidence(c, v).reason == "no_verified_mover_mate_before"


def test_black_capture_symmetry(labeling_case):
    original = chess.Board("1r5k/8/8/8/8/8/2N5/7K w - - 0 1")
    mirror_move = lambda u: chess.Move(chess.square_mirror(chess.Move.from_uci(u).from_square),
                                      chess.square_mirror(chess.Move.from_uci(u).to_square)).uci()
    c, f, v = labeling_case(fen=original.mirror().fen(), move=mirror_move("c2b4"),
        after=chess.engine.Cp(500), after_pv=tuple(map(mirror_move, ("b8b4", "h1g1", "b4b5", "g1f1"))))
    assert capture_evidence(c, f, v).supported
