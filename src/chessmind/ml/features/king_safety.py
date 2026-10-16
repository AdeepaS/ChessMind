"""Local king geometry, not a comprehensive king-safety evaluation."""

import chess


def king_safety_features(board: chess.Board, color: chess.Color) -> dict:
    king = board.king(color)
    if king is None:
        raise ValueError("position must contain both kings")
    file, rank = chess.square_file(king), chess.square_rank(king)
    ahead = rank + (1 if color else -1)
    shield = sum(board.piece_at(chess.square(f, ahead)) == chess.Piece(chess.PAWN, color)
                 for f in range(max(0, file - 1), min(7, file + 1) + 1)) if 0 <= ahead < 8 else 0
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[king]) | chess.SquareSet([king])
    return {"king_square": chess.square_name(king), "king_file": file, "king_rank": rank,
            "kingside_castling_right": board.has_kingside_castling_rights(color),
            "queenside_castling_right": board.has_queenside_castling_rights(color),
            "pawn_shield": shield,
            "attacked_king_zone_squares": sum(board.is_attacked_by(not color, sq) for sq in zone)}
