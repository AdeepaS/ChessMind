import chess
import chess.engine


STOCKFISH_PATH = "tools/stockfish/stockfish.exe"


def serialize_evaluation(score):
    """Serialize a native score already normalized to White's perspective.

    winner disambiguates Mate(0) (White lost) from MateGiven (White won),
    whose mate() values are both zero. No synthetic CP enters this structure.
    """
    is_mate = score.is_mate()
    return {
        "type": "mate" if is_mate else "cp",
        "cp": score.score(),
        "mate": score.mate(),
        "winner": ("white" if score > chess.engine.Cp(0) else "black")
        if is_mate else None,
        "perspective": "white",
    }


def analyze_board(engine, board, depth=15):
    """
    Analyze a chess.Board position using an already-running Stockfish engine.

    Evaluation convention:
        Positive score -> White is better
        Negative score -> Black is better

    evaluation contains genuine CP or signed mate information.
    score_cp is the legacy synthetic numeric projection, retained for display
    compatibility only; it must not be used to calculate mate-related loss.
    """

    info = engine.analyse(
        board,
        chess.engine.Limit(depth=depth)
    )

    # Always evaluate from White's perspective
    score = info["score"].pov(chess.WHITE)

    # Convert score to centipawns.
    # Mate scores are mapped to a very large centipawn value.
    score_cp = score.score(mate_score=100000)

    best_move = info["pv"][0] if info.get("pv") else None

    return {
        "score_cp": score_cp,
        "evaluation": serialize_evaluation(score),
        "best_move": best_move,
        "depth": info.get("depth"),
        "pv": info.get("pv", [])
    }


def analyze_position(fen, depth=15):
    """
    Analyze a FEN position.

    Useful for testing individual positions.
    """

    board = chess.Board(fen)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        return analyze_board(engine, board, depth)

    finally:
        engine.quit()


if __name__ == "__main__":

    board = chess.Board()

    result = analyze_position(board.fen())

    print("Position:")
    print(board)
    print()

    print("Evaluation:", result["score_cp"], "cp")
    print("Best move:", result["best_move"])
    print("Depth:", result["depth"])
    print("PV:", result["pv"])
