"""Proposals are narrow, reproducible and distinct from expert review."""

from dataclasses import replace

import chess.engine

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


def test_missed_mate_only(labeling_case):
    c, f, v = labeling_case(fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Cp(500), before_pv=("g6g7",), after_pv=("h8h7",))
    assert [p.concept for p in propose_labels(c, f, v)[0]] == [MistakeConcept.MISSED_MATE]


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
