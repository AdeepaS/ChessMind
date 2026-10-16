"""Feature units: pawn material units, counts, zero-based files/ranks and booleans."""

import json
from dataclasses import asdict, dataclass

from ..data.candidate_schema import SourceEvidence


FEATURE_SCHEMA_VERSION = "chess_features_v1"


@dataclass(frozen=True)
class ColorFeatures:
    pawns: int
    knights: int
    bishops: int
    rooks: int
    queens: int
    kings: int
    material: int
    non_pawn_material: int
    bishop_pair: bool
    attacked_pieces: int
    defended_pieces: int
    undefended_pieces: int
    attacked_undefended_pieces: int
    geometrically_attacked_squares: int
    doubled_excess_pawns: int
    isolated_pawns: int
    passed_pawns: int
    king_square: str
    king_file: int
    king_rank: int
    kingside_castling_right: bool
    queenside_castling_right: bool
    pawn_shield: int
    attacked_king_zone_squares: int


@dataclass(frozen=True)
class PositionFeatures:
    white: ColorFeatures
    black: ColorFeatures
    material_balance: int  # White minus Black, heuristic pawn units.
    side_to_move: str
    side_to_move_in_check: bool
    side_to_move_legal_moves: int


@dataclass(frozen=True)
class MoveContext:
    moving_color: str
    piece_type: str
    source_square: str
    target_square: str
    played_move_uci: str
    is_capture: bool
    is_en_passant: bool
    is_castling: bool
    promotion_piece: str | None
    gives_check: bool


@dataclass(frozen=True)
class NumericDelta:
    feature: str  # Dotted PositionFeatures field path; explicit color retained.
    change: int   # after - before; booleans mapped to 0/1 first.


@dataclass(frozen=True)
class MoveFeatures:
    before: PositionFeatures
    after: PositionFeatures
    context: MoveContext
    deltas: tuple[NumericDelta, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class CandidateFeatureRecord:
    candidate_id: str
    source_game_id: str
    source: str
    player_identifier: str
    ply_index: int
    candidate_schema_version: str
    source_evidence: SourceEvidence
    features: MoveFeatures
    schema_version: str = FEATURE_SCHEMA_VERSION


def feature_json(record: CandidateFeatureRecord) -> str:
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def numeric_position_values(position: PositionFeatures) -> dict[str, int]:
    result = {}
    def visit(value, prefix):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                visit(item, name)
            elif type(item) in (int, bool):
                result[name] = int(item)
    visit(asdict(position), "")
    return result
