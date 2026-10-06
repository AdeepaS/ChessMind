import chess
import chess.engine

from src.chessmind.engine.analyzer import (
    STOCKFISH_PATH,
    analyze_board
)


def calculate_centipawn_loss(before_score, after_score, color):
    """
    Calculate centipawn loss for the player who made the move.

    Scores are always from White's perspective.
    """

    if color == chess.WHITE:
        loss = before_score - after_score
    else:
        loss = after_score - before_score

    # A move can sometimes appear better than Stockfish's previous estimate
    # because of search-depth differences.
    # Centipawn loss should never be negative.
    return max(0, loss)


def classify_move(cp_loss):
    """
    Basic move classification based on centipawn loss.
    """

    if cp_loss <= 20:
        return "Excellent"
    elif cp_loss <= 50:
        return "Good"
    elif cp_loss <= 100:
        return "Inaccuracy"
    elif cp_loss <= 200:
        return "Mistake"
    else:
        return "Blunder"

def analyze_game(game, depth=15):
    """
    Analyze every move of a PGN game with Stockfish.

    Optimized version:
    - Analyze the initial position once.
    - After each move, analyze the resulting position.
    - Reuse that result as the "before" analysis of the next move.
    """

    board = game.board()
    results = []

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        # Analyze the initial position only once
        before_analysis = analyze_board(
            engine,
            board,
            depth=depth
        )

        for ply, move in enumerate(game.mainline_moves(), start=1):

            moving_color = board.turn
            move_number = board.fullmove_number
            san = board.san(move)

            # Reuse the already-calculated evaluation
            before_score = before_analysis["score_cp"]
            best_move = before_analysis["best_move"]

            # Play the actual move
            board.push(move)

            # Analyze the new position only once
            after_analysis = analyze_board(
                engine,
                board,
                depth=depth
            )

            after_score = after_analysis["score_cp"]

            cp_loss = calculate_centipawn_loss(
                before_score,
                after_score,
                moving_color
            )

            classification = classify_move(cp_loss)

            results.append({
                "ply": ply,
                "move_number": move_number,
                "color": "white" if moving_color == chess.WHITE else "black",
                "san": san,
                "uci": move.uci(),
                "evaluation_before": before_score,
                "evaluation_after": after_score,
                "best_move": (
                    best_move.uci()
                    if best_move is not None
                    else None
                ),
                "centipawn_loss": cp_loss,
                "classification": classification
            })

            # Key optimization:
            # this position is the "before" position for the next move
            before_analysis = after_analysis

    finally:
        engine.quit()

    return results


def calculate_game_statistics(results):
    """
    Calculate game-level statistics separately for White and Black.
    """

    stats = {
        "white": {
            "moves": 0,
            "total_centipawn_loss": 0,
            "excellent": 0,
            "good": 0,
            "inaccuracy": 0,
            "mistake": 0,
            "blunder": 0,
        },
        "black": {
            "moves": 0,
            "total_centipawn_loss": 0,
            "excellent": 0,
            "good": 0,
            "inaccuracy": 0,
            "mistake": 0,
            "blunder": 0,
        }
    }

    for move in results:
        color = move["color"]
        cp_loss = move["centipawn_loss"]
        classification = move["classification"].lower()

        stats[color]["moves"] += 1
        stats[color]["total_centipawn_loss"] += cp_loss

        if classification == "excellent":
            stats[color]["excellent"] += 1
        elif classification == "good":
            stats[color]["good"] += 1
        elif classification == "inaccuracy":
            stats[color]["inaccuracy"] += 1
        elif classification == "mistake":
            stats[color]["mistake"] += 1
        elif classification == "blunder":
            stats[color]["blunder"] += 1

    for color in ["white", "black"]:

        move_count = stats[color]["moves"]

        if move_count > 0:
            stats[color]["acpl"] = (
                stats[color]["total_centipawn_loss"]
                / move_count
            )
        else:
            stats[color]["acpl"] = 0

    return stats


if __name__ == "__main__":

    from src.chessmind.pgn.parser import load_pgn

    game = load_pgn("data/raw/sample_game.pgn")

    analysis = analyze_game(game, depth=12)

    for move in analysis:

        prefix = (
            f'{move["move_number"]}.'
            if move["color"] == "white"
            else f'{move["move_number"]}...'
        )

        print(
            f'{prefix} {move["san"]} | '
            f'Before: {move["evaluation_before"]} | '
            f'After: {move["evaluation_after"]} | '
            f'Loss: {move["centipawn_loss"]} cp | '
            f'Best: {move["best_move"]} | '
            f'{move["classification"]}'
        )
        
        
    stats = calculate_game_statistics(analysis)

    print("\n===== GAME SUMMARY =====")

    for color in ["white", "black"]:

        player_stats = stats[color]

        print(f"\n{color.upper()}")

        print("Moves:", player_stats["moves"])
        print(f'ACPL: {player_stats["acpl"]:.2f}')

        print("Excellent:", player_stats["excellent"])
        print("Good:", player_stats["good"])
        print("Inaccuracies:", player_stats["inaccuracy"])
        print("Mistakes:", player_stats["mistake"])
        print("Blunders:", player_stats["blunder"])