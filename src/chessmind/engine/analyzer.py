import chess
import chess.engine


STOCKFISH_PATH = "tools/stockfish/stockfish.exe"


def analyze_position(fen, depth=15):
    """Analyze a chess position using Stockfish."""

    board = chess.Board(fen)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth)
        )

        score = info["score"].pov(board.turn)

        best_move = info["pv"][0] if info.get("pv") else None

        return {
            "score": score,
            "best_move": best_move,
            "depth": info.get("depth"),
            "pv": info.get("pv", [])
        }

    finally:
        engine.quit()


if __name__ == "__main__":

    board = chess.Board()

    result = analyze_position(board.fen())

    print("Position:")
    print(board)
    print()

    print("Evaluation:", result["score"])
    print("Best move:", result["best_move"])
    print("Depth:", result["depth"])
    print("PV:", result["pv"])