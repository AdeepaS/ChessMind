"""Simple material counts, reusing established non-pawn heuristic values."""

import chess

from ...analysis.game_phase import PIECE_VALUES


def material_features(board: chess.Board, color: chess.Color) -> dict:
    counts = {kind: len(board.pieces(kind, color)) for kind in chess.PIECE_TYPES}
    non_pawn = sum(counts[kind] * value for kind, value in PIECE_VALUES.items())
    return {"pawns": counts[chess.PAWN], "knights": counts[chess.KNIGHT],
            "bishops": counts[chess.BISHOP], "rooks": counts[chess.ROOK],
            "queens": counts[chess.QUEEN], "kings": counts[chess.KING],
            "material": non_pawn + counts[chess.PAWN], "non_pawn_material": non_pawn,
            "bishop_pair": counts[chess.BISHOP] >= 2}
