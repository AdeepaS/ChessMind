"""Validation and multi-label behavior of the initial annotation vocabulary."""
from dataclasses import FrozenInstanceError

import pytest

from src.chessmind.ml.labeling.taxonomy import (
    CONCEPT_TO_CATEGORY,
    GamePhase,
    MistakeCategory,
    MistakeConcept,
    MoveLabels,
    category_for_concept,
)


EXPECTED_CATEGORIES = {
    "tactical": {"hanging_piece", "missed_fork", "missed_pin", "missed_skewer", "missed_mate"},
    "positional": {"king_safety", "pawn_structure", "piece_activity", "weak_squares"},
    "opening_knowledge": {"development", "center_control"},
    "endgame_technique": {"king_activity", "pawn_endgame", "rook_endgame"},
}


def test_stable_identifiers_and_complete_mapping():
    assert {phase.value for phase in GamePhase} == {"opening", "middlegame", "endgame"}
    assert {category.value for category in MistakeCategory} == set(EXPECTED_CATEGORIES)
    assert {concept.value for concept in MistakeConcept} == set().union(*EXPECTED_CATEGORIES.values())
    assert set(CONCEPT_TO_CATEGORY) == set(MistakeConcept)


@pytest.mark.parametrize("category,concepts", EXPECTED_CATEGORIES.items())
def test_explicit_mapping(category, concepts):
    for concept in concepts:
        assert category_for_concept(concept) is MistakeCategory(category)
        assert category_for_concept(MistakeConcept(concept)) is MistakeCategory(category)


@pytest.mark.parametrize("phase", list(GamePhase))
def test_empty_labels(phase):
    labels = MoveLabels(phase.value)
    assert labels.phase is phase
    assert labels.concepts == frozenset()
    assert labels.categories == frozenset()
    assert labels == MoveLabels(phase, [])


def test_multiple_concepts_in_one_category_and_duplicates():
    labels = MoveLabels("middlegame", ["missed_fork", MistakeConcept.MISSED_PIN, "missed_fork"])
    assert labels.concepts == frozenset({MistakeConcept.MISSED_FORK, MistakeConcept.MISSED_PIN})
    assert labels.categories == frozenset({MistakeCategory.TACTICAL})


@pytest.mark.parametrize("phase", list(GamePhase))
def test_multiple_categories_are_independent_of_phase(phase):
    labels = MoveLabels(phase, ["hanging_piece", "king_safety", "development", "rook_endgame"])
    assert labels.phase is phase
    assert len(labels.concepts) == 4
    assert labels.categories == frozenset(MistakeCategory)


@pytest.mark.parametrize("unknown", ["unknown", "Missed_Fork", "", "tactical", "Blunder", None, 1])
def test_unknown_concepts_rejected(unknown):
    with pytest.raises(ValueError):
        category_for_concept(unknown)
    with pytest.raises(ValueError):
        MoveLabels("opening", ["development", unknown])


@pytest.mark.parametrize("unknown", ["unknown", "Opening", "", "tactical", None])
def test_unknown_phases_rejected(unknown):
    with pytest.raises(ValueError):
        MoveLabels(unknown)


@pytest.mark.parametrize("unknown", ["unknown", "opening", "hanging_piece", "Blunder"])
def test_unknown_categories_rejected(unknown):
    with pytest.raises(ValueError):
        MistakeCategory(unknown)


@pytest.mark.parametrize("invalid", ["missed_fork", b"missed_fork", None, 1])
def test_invalid_concept_containers(invalid):
    with pytest.raises(TypeError):
        MoveLabels("opening", invalid)


def test_labels_and_mapping_cannot_be_mutated():
    concepts = ["missed_mate"]
    labels = MoveLabels("endgame", concepts)
    concepts.append("king_activity")
    assert labels.concepts == frozenset({MistakeConcept.MISSED_MATE})
    with pytest.raises(FrozenInstanceError):
        labels.phase = GamePhase.OPENING
    with pytest.raises(FrozenInstanceError):
        labels.categories = frozenset({MistakeCategory.POSITIONAL})
    with pytest.raises(TypeError):
        CONCEPT_TO_CATEGORY[MistakeConcept.MISSED_MATE] = MistakeCategory.POSITIONAL
