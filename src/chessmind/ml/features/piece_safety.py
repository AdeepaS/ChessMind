"""Geometric indicators include pinned attackers; they are not threat labels."""

import chess


def piece_safety_features(board: chess.Board, color: chess.Color) -> dict:
    squares = chess.SquareSet(board.occupied_co[color] & ~board.kings)
    attacked = {sq for sq in squares if board.is_attacked_by(not color, sq)}
    defended = {sq for sq in squares if board.is_attacked_by(color, sq)}
    coverage = chess.SquareSet()
    for square in chess.SquareSet(board.occupied_co[color]):
        coverage |= board.attacks(square)
    return {"attacked_pieces": len(attacked), "defended_pieces": len(defended),
            "undefended_pieces": len(squares) - len(defended),
            "attacked_undefended_pieces": len(attacked - defended),
            "geometrically_attacked_squares": len(coverage)}
