"""Concept labels, independent of move severity and phase detection.

These identifiers describe annotations; they do not detect mistakes. Existing
centipawn and mate classification policies remain in the analysis package.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Iterable, Mapping


class GamePhase(str, Enum):
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


class MistakeCategory(str, Enum):
    TACTICAL = "tactical"
    POSITIONAL = "positional"
    OPENING_KNOWLEDGE = "opening_knowledge"
    ENDGAME_TECHNIQUE = "endgame_technique"


class MistakeConcept(str, Enum):
    HANGING_PIECE = "hanging_piece"
    MISSED_FORK = "missed_fork"
    MISSED_PIN = "missed_pin"
    MISSED_SKEWER = "missed_skewer"
    MISSED_MATE = "missed_mate"
    KING_SAFETY = "king_safety"
    PAWN_STRUCTURE = "pawn_structure"
    PIECE_ACTIVITY = "piece_activity"
    WEAK_SQUARES = "weak_squares"
    DEVELOPMENT = "development"
    CENTER_CONTROL = "center_control"
    KING_ACTIVITY = "king_activity"
    PAWN_ENDGAME = "pawn_endgame"
    ROOK_ENDGAME = "rook_endgame"


CONCEPT_TO_CATEGORY: Mapping[MistakeConcept, MistakeCategory] = MappingProxyType({
    MistakeConcept.HANGING_PIECE: MistakeCategory.TACTICAL,
    MistakeConcept.MISSED_FORK: MistakeCategory.TACTICAL,
    MistakeConcept.MISSED_PIN: MistakeCategory.TACTICAL,
    MistakeConcept.MISSED_SKEWER: MistakeCategory.TACTICAL,
    MistakeConcept.MISSED_MATE: MistakeCategory.TACTICAL,
    MistakeConcept.KING_SAFETY: MistakeCategory.POSITIONAL,
    MistakeConcept.PAWN_STRUCTURE: MistakeCategory.POSITIONAL,
    MistakeConcept.PIECE_ACTIVITY: MistakeCategory.POSITIONAL,
    MistakeConcept.WEAK_SQUARES: MistakeCategory.POSITIONAL,
    MistakeConcept.DEVELOPMENT: MistakeCategory.OPENING_KNOWLEDGE,
    MistakeConcept.CENTER_CONTROL: MistakeCategory.OPENING_KNOWLEDGE,
    MistakeConcept.KING_ACTIVITY: MistakeCategory.ENDGAME_TECHNIQUE,
    MistakeConcept.PAWN_ENDGAME: MistakeCategory.ENDGAME_TECHNIQUE,
    MistakeConcept.ROOK_ENDGAME: MistakeCategory.ENDGAME_TECHNIQUE,
})


def category_for_concept(concept: MistakeConcept | str) -> MistakeCategory:
    """Return the explicit category, raising ValueError for unknown concepts."""
    return CONCEPT_TO_CATEGORY[MistakeConcept(concept)]


@dataclass(frozen=True, init=False)
class MoveLabels:
    """Immutable labels for one move, with exactly one explicit game phase.

    Accept enum members or their exact, case-sensitive string values. Unknown
    identifiers raise ValueError. Concepts must be an iterable, not a bare
    string; invalid containers raise TypeError. Duplicates collapse into a set.
    Categories are derived, so they cannot conflict with concepts. Empty
    concepts produce empty categories and mean unannotated, not error-free.
    Phase never constrains concepts (e.g. a fork can occur in an endgame).
    Severity is deliberately not part of this record.
    """

    phase: GamePhase
    concepts: FrozenSet[MistakeConcept]

    def __init__(
        self,
        phase: GamePhase | str,
        concepts: Iterable[MistakeConcept | str] = (),
    ) -> None:
        if isinstance(concepts, (str, bytes)):
            raise TypeError("concepts must be an iterable of identifiers, not a string")
        object.__setattr__(self, "phase", GamePhase(phase))
        object.__setattr__(self, "concepts", frozenset(MistakeConcept(c) for c in concepts))

    @property
    def categories(self) -> FrozenSet[MistakeCategory]:
        """Zero or more unique categories implied by the annotated concepts."""
        return frozenset(CONCEPT_TO_CATEGORY[c] for c in self.concepts)
