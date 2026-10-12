import chess


# Material units across both colors; pawns and kings do not contribute.
PIECE_VALUES = {
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Heuristics, not experimentally calibrated boundaries.
QUEENLESS_ENDGAME_MATERIAL = 24  # Up to four rooks and a minor piece.
QUEEN_ENDGAME_SUPPORT = 10      # At most two rooks' worth besides queens.
QUEEN_ENDGAME_MATERIAL = 28     # Two queens plus two rooks; caps promotions.
OPENING_MIN_MATERIAL = 44      # Full starting non-queen material (62 - 18).
OPENING_MIN_HOME_MINORS = 3    # At least three of the eight minors still home.
OPENING_BASE_MOVES = 6
OPENING_MOVES_PER_HOME_MINOR = 2  # More unfinished development extends opening.
PHASE_ORDER = ("opening", "middlegame", "endgame")
HOME_MINORS = (
    (chess.B1, chess.KNIGHT, chess.WHITE),
    (chess.G1, chess.KNIGHT, chess.WHITE),
    (chess.C1, chess.BISHOP, chess.WHITE),
    (chess.F1, chess.BISHOP, chess.WHITE),
    (chess.B8, chess.KNIGHT, chess.BLACK),
    (chess.G8, chess.KNIGHT, chess.BLACK),
    (chess.C8, chess.BISHOP, chess.BLACK),
    (chess.F8, chess.BISHOP, chess.BLACK),
)


def _material_counts(board):
    """Inspect the four non-pawn piece types once, using bitboard counts."""
    return {
        piece_type: sum(len(board.pieces(piece_type, color))
                        for color in (chess.WHITE, chess.BLACK))
        for piece_type in PIECE_VALUES
    }


def calculate_non_pawn_material(board):
    """Calculate total non-pawn material remaining on the board."""

    return sum(count * PIECE_VALUES[piece_type]
               for piece_type, count in _material_counts(board).items())


def detect_game_phase(board, move_number, previous_phase=None):
    """Classify an orthodox-chess position without search or hidden state.

    Simplified material takes precedence over move number. Queenless boards
    allow up to 24 units; with queens, require <=10 supporting units and <=28
    total. Queen exchange alone therefore does not force an endgame.

    Opening requires >=44 units and >=3 correctly colored minors on their
    original squares. Its move window is 6 + 2 * home_minors: three home minors
    allow opening through move 12, four through 14, eight through 22. This is a
    development-dependent progression signal rather than a fixed move-10 cut.
    Castling rights are deliberately not used: they can survive deep into play.

    previous_phase is optional for compatible two-argument position queries.
    Supply the previous returned label for forward-only game progression,
    including promotions or pieces returning home. The board is never changed.
    """
    if previous_phase is not None and previous_phase not in PHASE_ORDER:
        raise ValueError(f"Unknown previous phase: {previous_phase}")

    counts = _material_counts(board)
    material = sum(count * PIECE_VALUES[piece_type]
                   for piece_type, count in counts.items())
    queens = counts[chess.QUEEN]
    support = material - queens * PIECE_VALUES[chess.QUEEN]

    if ((queens == 0 and material <= QUEENLESS_ENDGAME_MATERIAL)
            or (queens > 0 and support <= QUEEN_ENDGAME_SUPPORT
                and material <= QUEEN_ENDGAME_MATERIAL)):
        phase = "endgame"
    else:
        home_minors = sum(board.piece_at(square) == chess.Piece(piece_type, color)
                          for square, piece_type, color in HOME_MINORS)
        opening_window = OPENING_BASE_MOVES + OPENING_MOVES_PER_HOME_MINOR * home_minors
        if (material >= OPENING_MIN_MATERIAL
                and home_minors >= OPENING_MIN_HOME_MINORS
                and move_number <= opening_window):
            phase = "opening"
        else:
            phase = "middlegame"

    if previous_phase is not None:
        return PHASE_ORDER[max(PHASE_ORDER.index(previous_phase), PHASE_ORDER.index(phase))]
    return phase

