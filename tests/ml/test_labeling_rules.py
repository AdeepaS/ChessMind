"""Proposals are narrow, reproducible and distinct from expert review."""

from dataclasses import replace

import chess.engine
import pytest

from src.chessmind.ml.labeling.rules import propose_labels
from src.chessmind.ml.labeling.schema import VerificationResult
from src.chessmind.ml.labeling.taxonomy import MistakeConcept


def test_hanging_proposal_and_determinism(labeling_case):
    c, f, v = labeling_case()
    result = propose_labels(c, f, v)
    assert result == propose_labels(c, f, v)
    proposals, evidence, status, _ = result
    assert status == "proposed" and len(proposals) == 1
    assert proposals[0].concept == MistakeConcept.HANGING_PIECE
    assert proposals[0].review_status == "unreviewed"
    assert all(p.evidence for p in proposals)
    assert evidence[0].evidence_type == "allowed_immediate_tactical_capture"


@pytest.mark.parametrize("black", [False, True])
@pytest.mark.parametrize("cp", [500, 0, -500])
def test_mate_to_cp_requires_review(labeling_case, black, cp):
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    c, f, v = labeling_case(fen=board.mirror().fen() if black else board.fen(),
        move="g3g4" if black else "g6g5", before=chess.engine.Mate(-1 if black else 1),
        after=chess.engine.Cp(-cp if black else cp),
        before_pv=("g3g2",) if black else ("g6g7",), after_pv=("h1h2",) if black else ("h8h7",))
    proposals, evidence, status, reasons = propose_labels(c, f, v)
    assert proposals == () and status == "requires_review"
    assert reasons == ("post_move_forced_mate_unconfirmed",)
    assert not evidence[1].supported and evidence[1].continuation == v.before_pv
    assert v.classification == "Blunder" and v.mate_quality == "lost_forced_mate"


def test_reversed_to_opponent_mate_still_proposed(labeling_case):
    # Synthetic engine scores exercise the unchanged reversal branch.
    c, f, v = labeling_case(fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Mate(-3), before_pv=("g6g7",), after_pv=("h8h7",))
    assert [p.concept for p in propose_labels(c, f, v)[0]] == [MistakeConcept.MISSED_MATE]


@pytest.mark.parametrize("pv,reason", [((), "no_distinct_mating_alternative"),
    (("g6g5",), "no_distinct_mating_alternative"), (("f7f6",), "mating_pv_not_complete")])
def test_unproven_alternative_does_not_trigger_cp_review(labeling_case, pv, reason):
    c, f, v = labeling_case(fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Cp(500), before_pv=pv, after_pv=("h8h7",))
    proposals, evidence, status, _ = propose_labels(c, f, v)
    assert proposals == () and status == "insufficient_evidence"
    assert evidence[1].reason == reason


def test_failure_never_falls_back_and_controls_abstain(labeling_case):
    c, f, _ = labeling_case()
    for status in ("failed", "skipped"):
        result = propose_labels(c, f, VerificationResult(status, reason="fixture"))
        assert result[0] == () and result[2] == "insufficient_evidence"
    c, f, v = labeling_case(after=chess.engine.Cp(-10))
    assert propose_labels(c, f, v)[0] == ()
    assert propose_labels(c, f, v)[2] == "conflicting_evidence"
    c["selection_type"] = "LOW_ERROR_CONTROL"
    assert propose_labels(c, f, v)[2] == "insufficient_evidence"


def test_features_alone_and_retained_mate_do_not_label(labeling_case):
    c, f, v = labeling_case(after_pv=("h8g8",))
    f["features"]["after"]["white"]["attacked_undefended_pieces"] = 100
    assert propose_labels(c, f, v)[0] == ()
    c, f, v = labeling_case(before=chess.engine.Mate(2), after=chess.engine.Mate(5))
    result = propose_labels(c, f, v)
    assert result[0] == () and result[2] == "requires_review"
