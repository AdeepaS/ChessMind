"""Doubled = excess on each file; isolated/passed count individual pawns."""

import chess


def pawn_structure_features(board: chess.Board, color: chess.Color) -> dict:
    pawns = board.pieces(chess.PAWN, color)
    enemy = board.pieces(chess.PAWN, not color)
    files = [sum(chess.square_file(sq) == f for sq in pawns) for f in range(8)]
    isolated = passed = 0
    for square in pawns:
        file, rank = chess.square_file(square), chess.square_rank(square)
        isolated += not any(files[f] for f in (file - 1, file + 1) if 0 <= f < 8)
        passed += not any(abs(chess.square_file(other) - file) <= 1 and
                          (chess.square_rank(other) > rank if color else chess.square_rank(other) < rank)
                          for other in enemy)
    return {"doubled_excess_pawns": sum(max(0, count - 1) for count in files),
            "isolated_pawns": isolated, "passed_pawns": passed}
