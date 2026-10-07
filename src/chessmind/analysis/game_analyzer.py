import chess
import chess.engine

from src.chessmind.analysis.game_phase import detect_game_phase
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

            # Detect the game phase before the move is played
            phase = detect_game_phase(board, move_number)

            # Store the position before the move
            fen_before = board.fen()

            # Play the actual move
            board.push(move)

            # Store the resulting position
            fen_after = board.fen()

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
                "color": (
                    "white"
                    if moving_color == chess.WHITE
                    else "black"
                ),
                "san": san,
                "uci": move.uci(),

                "phase": phase,

                "fen_before": fen_before,
                "fen_after": fen_after,

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

    Also calculate statistics for each game phase:
    - Opening
    - Middlegame
    - Endgame
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

            "phases": {
                "opening": {
                    "moves": 0,
                    "total_centipawn_loss": 0
                },
                "middlegame": {
                    "moves": 0,
                    "total_centipawn_loss": 0
                },
                "endgame": {
                    "moves": 0,
                    "total_centipawn_loss": 0
                }
            }
        },

        "black": {
            "moves": 0,
            "total_centipawn_loss": 0,

            "excellent": 0,
            "good": 0,
            "inaccuracy": 0,
            "mistake": 0,
            "blunder": 0,

            "phases": {
                "opening": {
                    "moves": 0,
                    "total_centipawn_loss": 0
                },
                "middlegame": {
                    "moves": 0,
                    "total_centipawn_loss": 0
                },
                "endgame": {
                    "moves": 0,
                    "total_centipawn_loss": 0
                }
            }
        }
    }

    # ---------------------------------------
    # Process every analyzed move
    # ---------------------------------------

    for move in results:

        color = move["color"]
        cp_loss = move["centipawn_loss"]
        classification = move["classification"].lower()
        phase = move["phase"]

        # Overall statistics
        stats[color]["moves"] += 1
        stats[color]["total_centipawn_loss"] += cp_loss

        # Phase statistics
        stats[color]["phases"][phase]["moves"] += 1

        stats[color]["phases"][phase][
            "total_centipawn_loss"
        ] += cp_loss

        # Move classification statistics
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

    # ---------------------------------------
    # Calculate ACPL values
    # ---------------------------------------

    for color in ["white", "black"]:

        move_count = stats[color]["moves"]

        # Overall ACPL
        if move_count > 0:
            stats[color]["acpl"] = (
                stats[color]["total_centipawn_loss"]
                / move_count
            )
        else:
            stats[color]["acpl"] = 0

        # Phase ACPL
        for phase in [
            "opening",
            "middlegame",
            "endgame"
        ]:

            phase_stats = stats[color]["phases"][phase]

            phase_move_count = phase_stats["moves"]

            if phase_move_count > 0:

                phase_stats["acpl"] = (
                    phase_stats["total_centipawn_loss"]
                    / phase_move_count
                )

            else:
                phase_stats["acpl"] = 0

    return stats


if __name__ == "__main__":

    from src.chessmind.pgn.parser import load_pgn

    game = load_pgn(
        "data/raw/sample_game.pgn"
    )

    analysis = analyze_game(
        game,
        depth=12
    )

    # ---------------------------------------
    # Print move-by-move analysis
    # ---------------------------------------

    for move in analysis:

        prefix = (
            f'{move["move_number"]}.'
            if move["color"] == "white"
            else f'{move["move_number"]}...'
        )

        print(
            f'{prefix} {move["san"]} | '
            f'Phase: {move["phase"]} | '
            f'Before: {move["evaluation_before"]} | '
            f'After: {move["evaluation_after"]} | '
            f'Loss: {move["centipawn_loss"]} cp | '
            f'Best: {move["best_move"]} | '
            f'{move["classification"]}'
        )

    # ---------------------------------------
    # Calculate game statistics
    # ---------------------------------------

    stats = calculate_game_statistics(
        analysis
    )

    print("\n===== GAME SUMMARY =====")

    for color in ["white", "black"]:

        player_stats = stats[color]

        print(
            f"\n{color.upper()}"
        )

        # Overall statistics
        print(
            "Moves:",
            player_stats["moves"]
        )

        print(
            f'ACPL: '
            f'{player_stats["acpl"]:.2f}'
        )

        print(
            "Excellent:",
            player_stats["excellent"]
        )

        print(
            "Good:",
            player_stats["good"]
        )

        print(
            "Inaccuracies:",
            player_stats["inaccuracy"]
        )

        print(
            "Mistakes:",
            player_stats["mistake"]
        )

        print(
            "Blunders:",
            player_stats["blunder"]
        )

        # -----------------------------------
        # Phase statistics
        # -----------------------------------

        print("\nPhase Statistics:")

        for phase in [
            "opening",
            "middlegame",
            "endgame"
        ]:

            phase_stats = (
                player_stats["phases"][phase]
            )

            print(
                f'{phase.capitalize()} | '
                f'Moves: {phase_stats["moves"]} | '
                f'ACPL: {phase_stats["acpl"]:.2f}'
            )