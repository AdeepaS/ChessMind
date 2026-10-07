import chess


PIECE_VALUES = {
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def calculate_non_pawn_material(board):
    """Calculate total non-pawn material remaining on the board."""

    total = 0

    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))

        total += (white_count + black_count) * value

    return total


def detect_game_phase(board, move_number):
    """Detect whether the current position is opening, middlegame, or endgame."""

    if move_number <= 10:
        return "opening"

    non_pawn_material = calculate_non_pawn_material(board)

    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))

    total_queens = white_queens + black_queens

    if total_queens == 0 and non_pawn_material <= 24:
        return "endgame"

    if non_pawn_material <= 14:
        return "endgame"

    return "middlegame"

