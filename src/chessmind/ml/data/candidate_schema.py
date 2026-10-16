"""Immutable evidence records. Candidates are not verified mistake labels."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal


SCHEMA_VERSION = "candidate_moves_v1"
Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class ExtractionConfig:
    cp_loss_threshold: float = 100
    control_cp_loss_max: float = 20
    max_control_moves_per_game: int = 5
    random_seed: int = 42
    include_mate_candidates: bool = True

    def __post_init__(self):
        for value in (self.cp_loss_threshold, self.control_cp_loss_max):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("CP thresholds must be finite nonnegative numbers")
        if type(self.max_control_moves_per_game) is not int or self.max_control_moves_per_game < 0:
            raise ValueError("max_control_moves_per_game must be a nonnegative integer")
        if type(self.random_seed) is not int or type(self.include_mate_candidates) is not bool:
            raise ValueError("seed must be an integer and include_mate_candidates a bool")


@dataclass(frozen=True)
class ScoreEvidence:
    type: Literal["cp", "mate"]
    cp: int | None
    mate: int | None
    winner: Literal["white", "black"] | None
    perspective: str = "white"


@dataclass(frozen=True)
class AnalysisProfile:
    configured_depth: int | None = None
    achieved_depth: int | None = None  # Not returned by analyze_game.
    engine_id: tuple[tuple[str, str], ...] | None = None
    engine_binary_sha256: str | None = None
    engine_options: tuple[tuple[str, Scalar], ...] | None = None
    engine_resources: tuple[tuple[str, str, str, str], ...] | None = None
    chess_version: str | None = None

    def __post_init__(self):
        if self.configured_depth is not None and (
                type(self.configured_depth) is not int or not 1 <= self.configured_depth <= 128):
            raise ValueError("depth must be an integer in 1..128")


@dataclass(frozen=True)
class SourceEvidence:
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    source_archive: str | None = None
    archive_game_index: int | None = None


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    source_game_id: str
    source: str
    player_identifier: str
    player_color: Literal["white", "black"]
    ply_index: int
    full_move_number: int
    fen_before: str
    fen_after: str
    played_move_uci: str
    played_move_san: str
    best_move_uci: str | None
    game_phase: str
    score_before: ScoreEvidence
    score_after: ScoreEvidence
    centipawn_loss: float | None
    existing_classification: str
    mate_quality: str | None
    mate_transition_status: str | None
    selection_type: Literal["SIGNIFICANT_CP_ERROR", "LOW_ERROR_CONTROL", "MATE_RELATED", "AMBIGUOUS"]
    selection_reason: str
    extraction_configuration: ExtractionConfig
    analysis_profile: AnalysisProfile
    source_evidence: SourceEvidence
    white_rating: int | None
    black_rating: int | None
    time_control: str | None
    utc_date: str | None
    utc_time: str | None
    selection_provenance: str = "existing analyze_game results; per-game SHA-256-ranked controls"
    schema_version: str = SCHEMA_VERSION


def candidate_id(source: str, game_id: str, ply: int) -> str:
    identity = json.dumps([SCHEMA_VERSION, source, game_id, ply], separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def candidate_json(record: CandidateRecord) -> str:
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
