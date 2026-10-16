"""Position/move/candidate adapters, independent of engine analysis."""

from collections.abc import Mapping
from dataclasses import asdict, fields

import chess

from ..data.candidate_schema import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION, CandidateRecord, SourceEvidence, candidate_id,
)
from .king_safety import king_safety_features
from .material import material_features
from .pawn_structure import pawn_structure_features
from .piece_safety import piece_safety_features
from .schema import (CandidateFeatureRecord, ColorFeatures, MoveContext, MoveFeatures,
                     NumericDelta, PositionFeatures, numeric_position_values)


class FeatureValidationError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _validate_board(board):
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise FeatureValidationError("invalid_standard_position")


def extract_position_features(board: chess.Board) -> PositionFeatures:
    _validate_board(board)
    def side(color):
        return ColorFeatures(**material_features(board, color),
                             **piece_safety_features(board, color),
                             **pawn_structure_features(board, color),
                             **king_safety_features(board, color))
    white, black = side(chess.WHITE), side(chess.BLACK)
    return PositionFeatures(white, black, white.material - black.material,
                            "white" if board.turn else "black", board.is_check(),
                            board.legal_moves.count())


def extract_move_features(board: chess.Board, move: chess.Move) -> MoveFeatures:
    _validate_board(board)
    if not isinstance(move, chess.Move) or not board.is_legal(move):
        raise FeatureValidationError("illegal_move")
    context = MoveContext("white" if board.turn else "black",
                          chess.piece_name(board.piece_type_at(move.from_square)),
                          chess.square_name(move.from_square), chess.square_name(move.to_square),
                          move.uci(), board.is_capture(move), board.is_en_passant(move),
                          board.is_castling(move), chess.piece_name(move.promotion) if move.promotion else None,
                          board.gives_check(move))
    before = extract_position_features(board)
    after_board = board.copy(stack=False)
    after_board.push(move)
    after = extract_position_features(after_board)
    old, new = numeric_position_values(before), numeric_position_values(after)
    deltas = tuple(NumericDelta(name, new[name] - old[name]) for name in sorted(old))
    return MoveFeatures(before, after, context, deltas)


def extract_candidate_features(candidate: Mapping | CandidateRecord) -> CandidateFeatureRecord:
    if isinstance(candidate, CandidateRecord):
        candidate = asdict(candidate)
    if not isinstance(candidate, Mapping):
        raise FeatureValidationError("invalid_candidate_object")
    required = ("candidate_id", "source_game_id", "source", "player_identifier", "player_color",
                "ply_index", "full_move_number", "fen_before", "fen_after", "played_move_uci", "schema_version")
    if any(key not in candidate for key in required):
        raise FeatureValidationError("missing_fields")
    if any(not isinstance(candidate[key], str) or not candidate[key] for key in required
           if key not in ("ply_index", "full_move_number")):
        raise FeatureValidationError("invalid_field_type")
    if candidate["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise FeatureValidationError("unsupported_candidate_schema")
    if any(type(candidate[key]) is not int or candidate[key] < 1 for key in ("ply_index", "full_move_number")):
        raise FeatureValidationError("invalid_ply")
    if candidate["candidate_id"] != candidate_id(candidate["source"], candidate["source_game_id"], candidate["ply_index"]):
        raise FeatureValidationError("candidate_id_mismatch")
    try:
        if len(candidate["fen_before"].split()) != 6 or len(candidate["fen_after"].split()) != 6:
            raise ValueError("expected six FEN fields")
        board = chess.Board(candidate["fen_before"])
        expected_after = chess.Board(candidate["fen_after"])
        move = chess.Move.from_uci(candidate["played_move_uci"])
    except ValueError as exc:
        raise FeatureValidationError("invalid_fen_or_uci") from exc
    _validate_board(board)
    _validate_board(expected_after)
    color = "white" if board.turn else "black"
    if (candidate["player_color"] != color or candidate["full_move_number"] != board.fullmove_number
            or candidate["ply_index"] != board.ply() + 1
            or not candidate["player_identifier"].startswith(candidate["source"] + ":")):
        raise FeatureValidationError("position_identity_mismatch")
    features = extract_move_features(board, move)
    board_after = board.copy(stack=False)
    board_after.push(move)
    if board_after.fen() != candidate["fen_after"]:
        raise FeatureValidationError("after_fen_mismatch")
    evidence = candidate.get("source_evidence")
    evidence = {} if evidence is None else evidence
    if not isinstance(evidence, Mapping) or set(evidence) - {f.name for f in fields(SourceEvidence)}:
        raise FeatureValidationError("invalid_source_evidence")
    for key, value in evidence.items():
        valid = value is None or (type(value) is int and value > 0 if key == "archive_game_index"
                                  else isinstance(value, str))
        if not valid:
            raise FeatureValidationError("invalid_source_evidence")
    return CandidateFeatureRecord(candidate["candidate_id"], candidate["source_game_id"], candidate["source"],
                                  candidate["player_identifier"], candidate["ply_index"], candidate["schema_version"],
                                  SourceEvidence(**evidence), features)
