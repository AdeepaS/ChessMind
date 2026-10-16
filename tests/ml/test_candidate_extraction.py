"""Offline selection policy and immutable evidence tests."""

import copy
import json
from dataclasses import FrozenInstanceError, replace

import chess.engine
import pytest

from src.chessmind.ml.data.candidate_extraction import extract_candidates
from src.chessmind.ml.data.candidate_schema import ExtractionConfig, candidate_id, candidate_json


@pytest.mark.parametrize("loss,kind,classification", [
    (0, "LOW_ERROR_CONTROL", "Excellent"), (20, "LOW_ERROR_CONTROL", "Excellent"),
    (21, None, "Good"), (99, None, "Inaccuracy"),
    (100, "SIGNIFICANT_CP_ERROR", "Inaccuracy"),
    (101, "SIGNIFICANT_CP_ERROR", "Mistake"), (201, "SIGNIFICANT_CP_ERROR", "Blunder")])
def test_thresholds_preserve_classification(candidate_fixture, loss, kind, classification):
    game, rows = candidate_fixture([chess.engine.Cp(0), chess.engine.Cp(-loss)], count=1)
    found = extract_candidates(rows, game)
    if kind is None:
        assert found == ()
    else:
        assert len(found) == 1
        assert found[0].selection_type == kind
        assert found[0].existing_classification == classification


def test_controls_determinism_seeds_order_and_nonmutation(candidate_fixture):
    game, rows = candidate_fixture()
    original = copy.deepcopy(rows)
    first = extract_candidates(rows, game)
    assert first == extract_candidates(rows, game)
    other = extract_candidates(rows, game, config=ExtractionConfig(random_seed=99))
    assert {r.ply_index for r in first} != {r.ply_index for r in other}
    assert len(first) == 5
    assert [r.ply_index for r in first] == sorted(r.ply_index for r in first)
    assert rows == original
    with pytest.raises(FrozenInstanceError):
        first[0].score_before.cp = 12


def test_overlapping_thresholds_have_single_record_per_ply(candidate_fixture):
    game, rows = candidate_fixture()
    selected = extract_candidates(rows, game, config=ExtractionConfig(cp_loss_threshold=0))
    assert len(selected) == len({r.candidate_id for r in selected}) == 12
    assert {r.selection_type for r in selected} == {"SIGNIFICANT_CP_ERROR"}


def test_black_white_fens_identity_and_missing_best(candidate_fixture):
    game, rows = candidate_fixture([chess.engine.Cp(0), chess.engine.Cp(-100), chess.engine.Cp(0)], count=2)
    rows[0].pop("best_move")
    selected = extract_candidates(rows, game)
    assert [r.player_identifier for r in selected] == ["lichess:alice", "lichess:bob"]
    assert [r.centipawn_loss for r in selected] == [100, 100]
    assert selected[0].best_move_uci is None
    assert selected[0].analysis_profile.configured_depth is None
    assert selected[0].analysis_profile.engine_id is None
    for row, record in zip(rows, selected):
        assert record.fen_before == row["fen_before"]
        assert record.fen_after == row["fen_after"]
        assert record.full_move_number == 1
        assert record.source_game_id == game.game_id
        assert record.candidate_id == candidate_id("lichess", game.game_id, row["ply"])
        assert candidate_json(record) == candidate_json(record)
        assert "label" not in json.loads(candidate_json(record))
    assert candidate_id("lichess", "abcdefgh", 1) != candidate_id("elsewhere", "abcdefgh", 1)
    assert candidate_id("lichess", "abcdefgh", 1) != candidate_id("lichess", "abcdefgh", 2)


@pytest.mark.parametrize("before,after,status,kind", [
    (chess.engine.Mate(3), chess.engine.Cp(200), "harmful", "MATE_RELATED"),
    (chess.engine.Cp(0), chess.engine.Mate(-3), "harmful", "MATE_RELATED"),
    (chess.engine.Mate(3), chess.engine.Mate(-3), "harmful", "MATE_RELATED"),
    (chess.engine.Mate(-3), chess.engine.Cp(-200), "beneficial", "MATE_RELATED"),
    (chess.engine.Cp(0), chess.engine.Mate(3), "beneficial", "MATE_RELATED"),
    (chess.engine.Mate(-3), chess.engine.Mate(3), "beneficial", "MATE_RELATED"),
    (chess.engine.Mate(1), chess.engine.MateGiven, "beneficial", "MATE_RELATED"),
    (chess.engine.Mate(3), chess.engine.Mate(2), None, None),
    (chess.engine.Mate(-3), chess.engine.Mate(-3), None, None),
    (chess.engine.Mate(3), chess.engine.Mate(4), "ambiguous", "AMBIGUOUS"),
    (chess.engine.Mate(-3), chess.engine.Mate(-2), "ambiguous", "AMBIGUOUS"),
])
@pytest.mark.parametrize("black", [False, True])
def test_mate_transitions(candidate_fixture, before, after, status, kind, black):
    scores = [before, after]
    if black:
        scores = [-before, -before, -after]
    game, rows = candidate_fixture(scores, count=2 if black else 1)
    result = [r for r in extract_candidates(rows, game) if r.ply_index == (2 if black else 1)]
    if kind is None:
        assert result == []
    else:
        assert result[0].selection_type == kind
        assert result[0].mate_transition_status == status
        assert result[0].centipawn_loss is None
    excluded = extract_candidates(rows, game, config=ExtractionConfig(include_mate_candidates=False))
    assert not any(r.ply_index == (2 if black else 1) for r in excluded)


def test_undefined_loss_and_unknown_mate_are_explicit(candidate_fixture):
    game, rows = candidate_fixture(count=1)
    rows[0]["centipawn_loss"] = None
    assert extract_candidates(rows, game)[0].selection_reason == "undefined_cp_loss"
    game, rows = candidate_fixture([chess.engine.Cp(0), chess.engine.Mate(3)], count=1)
    rows[0]["mate_quality"] = None
    assert extract_candidates(rows, game)[0].selection_type == "AMBIGUOUS"


def test_empty_and_no_controls(candidate_fixture):
    game, rows = candidate_fixture()
    assert extract_candidates(rows, game, config=ExtractionConfig(max_control_moves_per_game=0)) == ()
    empty = replace(game, move_count=0, uci_moves=())
    assert extract_candidates([], empty) == ()


@pytest.mark.parametrize("field,value", [
    ("ply", 2), ("color", "black"), ("fen_before", "bad"), ("fen_after", "bad"),
    ("uci", "d2d4"), ("san", "d4"), ("best_move", "a1a8"), ("phase", "unknown"),
    ("centipawn_loss", float("nan")), ("centipawn_loss", -1),
    ("classification", "invented"), ("evaluation_before_detail", {"type": "cp", "cp": 0}),
])
def test_malformed_input(candidate_fixture, field, value):
    game, rows = candidate_fixture(count=1)
    rows[0][field] = value
    with pytest.raises(ValueError):
        extract_candidates(rows, game)


def test_duplicate_plies_incomplete_and_synthetic_mate_cp_rejected(candidate_fixture):
    game, rows = candidate_fixture(count=2)
    with pytest.raises(ValueError):
        extract_candidates(rows[:1], game)
    rows[1]["ply"] = 1
    with pytest.raises(ValueError):
        extract_candidates(rows, game)
    game, rows = candidate_fixture([chess.engine.Cp(0), chess.engine.Mate(3)], count=1)
    rows[0]["centipawn_loss"] = 100000
    with pytest.raises(ValueError, match="mate transition"):
        extract_candidates(rows, game)


@pytest.mark.parametrize("options", [{"cp_loss_threshold": -1}, {"control_cp_loss_max": float("inf")},
    {"max_control_moves_per_game": True}, {"random_seed": "42"}, {"include_mate_candidates": 1}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        ExtractionConfig(**options)
