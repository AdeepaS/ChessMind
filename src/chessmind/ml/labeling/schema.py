"""Immutable provisional proposals; expert review is a separate state."""

import json
from dataclasses import asdict, dataclass
from typing import Literal

from ..data.candidate_schema import AnalysisProfile, ScoreEvidence
from .taxonomy import MistakeConcept


LABEL_SCHEMA_VERSION = "mistake_proposals_v1"
RULE_VERSION = "conservative_tactics_v1"


@dataclass(frozen=True)
class LabelingConfig:
    max_candidates: int = 24
    verification_depth: int = 12
    verify: bool = True
    threads: int = 1
    hash_mb: int = 64
    strict: bool = True

    def __post_init__(self):
        for name, low, high in (("max_candidates", 0, 1000), ("verification_depth", 1, 30),
                                ("threads", 1, 4), ("hash_mb", 1, 512)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        if type(self.verify) is not bool or type(self.strict) is not bool:
            raise ValueError("verify and strict must be bool")


@dataclass(frozen=True)
class VerificationResult:
    status: Literal["verified", "failed", "skipped"]
    reason: str | None = None
    before: ScoreEvidence | None = None
    after: ScoreEvidence | None = None
    before_pv: tuple[str, ...] = ()
    after_pv: tuple[str, ...] = ()
    before_depth: int | None = None
    after_depth: int | None = None
    centipawn_loss: float | None = None
    classification: str | None = None
    mate_quality: str | None = None
    significant_error_supported: bool | None = None
    original_cp_loss: float | None = None
    original_classification: str | None = None
    original_mate_quality: str | None = None
    cp_loss_change: float | None = None
    classification_changed: bool | None = None
    significant_status_changed: bool | None = None
    materially_changed: bool | None = None
    profile: AnalysisProfile | None = None


@dataclass(frozen=True)
class TacticalEvidence:
    evidence_type: str
    supported: bool
    reason: str
    continuation: tuple[str, ...] = ()
    capture_uci: str | None = None
    victim_square: str | None = None
    captures_moved_piece: bool | None = None
    victim_defended: bool | None = None
    previously_attacked: bool | None = None
    pv_material_gain: int | None = None  # Opponent-relative pawn units, from post-move board.
    minimum_gain_after_immediate_recapture: int | None = None
    attacked_undefended_delta: int | None = None


@dataclass(frozen=True)
class LabelProposal:
    concept: MistakeConcept
    evidence: tuple[TacticalEvidence, ...]
    status: str = "proposed"
    review_status: str = "unreviewed"
    rule_version: str = RULE_VERSION


@dataclass(frozen=True)
class LabelRecord:
    candidate_id: str
    source_game_id: str
    player_identifier: str
    ply_index: int
    candidate_schema_version: str
    feature_schema_version: str
    original_selection_type: str
    verification: VerificationResult
    evidence: tuple[TacticalEvidence, ...]
    proposals: tuple[LabelProposal, ...]
    proposal_status: Literal["proposed", "insufficient_evidence", "conflicting_evidence", "requires_review"]
    reasons: tuple[str, ...]
    candidate_input_sha256: str
    feature_input_sha256: str
    feature_record_sha256: str
    review_status: str = "unreviewed"
    rule_version: str = RULE_VERSION
    schema_version: str = LABEL_SCHEMA_VERSION


def label_json(record: LabelRecord) -> str:
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
