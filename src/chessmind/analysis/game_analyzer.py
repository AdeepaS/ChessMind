import chess
import chess.engine

from src.chessmind.analysis.weakness_analyzer import detect_weakest_phase
from src.chessmind.analysis.game_phase import detect_game_phase
from src.chessmind.reports.weakness_report import generate_weakness_report

from src.chessmind.engine.analyzer import (
    STOCKFISH_PATH,
    analyze_board
)

from src.chessmind.analysis.weakness_analyzer import (
    detect_weakest_phase,
    find_worst_moves
)


def calculate_centipawn_loss(before_score, after_score, color):
    """
    Calculate centipawn loss for the player who made the move.

    Scores must be genuine CP values, always from White's perspective.
    For structured evaluations involving mate, use evaluate_move_quality().
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


def evaluate_move_quality(before, after, color):
    """Compare structured White-POV evaluations from the mover's perspective.

    CP/CP keeps the existing thresholds. Any mate endpoint makes CP loss
    undefined. Lost/allowed/reversed winning mate is a Blunder. Slower winning
    mate or faster losing mate is an Inaccuracy. Other mate outcomes are Good,
    with a separate quality label (not a claim of CP precision).

    On the winner's move the expected distance decreases by one; on the
    loser's move it stays constant. Distance comparisons are search heuristics.
    """
    if before["type"] == after["type"] == "cp":
        loss = calculate_centipawn_loss(before["cp"], after["cp"], color)
        return {"centipawn_loss": loss, "classification": classify_move(loss),
                "mate_quality": None}

    mover = "white" if color == chess.WHITE else "black"
    before_mate = before["type"] == "mate"
    after_mate = after["type"] == "mate"
    won_before = before_mate and before["winner"] == mover
    won_after = after_mate and after["winner"] == mover
    classification = "Good"
    if before_mate and not after_mate:
        quality = "lost_forced_mate" if won_before else "escaped_forced_mate"
        if won_before:
            classification = "Blunder"
    elif not before_mate and after_mate:
        quality = "achieved_forced_mate" if won_after else "allowed_forced_mate"
        if not won_after:
            classification = "Blunder"
    elif won_before != won_after:
        quality = "reversed_to_winning_mate" if won_after else "reversed_to_losing_mate"
        if not won_after:
            classification = "Blunder"
    elif after["mate"] == 0:
        quality = "delivered_checkmate" if won_after else "checkmated"
        if not won_after:
            classification = "Inaccuracy"
    else:
        expected = max(0, abs(before["mate"]) - (1 if won_before else 0))
        distance = abs(after["mate"])
        if won_after:
            quality = "slower_mate" if distance > expected else "maintained_forced_mate"
        else:
            quality = "faster_defeat" if distance < expected else "remains_forced_loss"
        if quality in ("slower_mate", "faster_defeat"):
            classification = "Inaccuracy"
    return {"centipawn_loss": None, "classification": classification,
            "mate_quality": quality}


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

            quality = evaluate_move_quality(
                before_analysis["evaluation"],
                after_analysis["evaluation"],
                moving_color
            )

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
                "evaluation_before_detail": before_analysis["evaluation"],
                "evaluation_after_detail": after_analysis["evaluation"],

                "best_move": (
                    best_move.uci()
                    if best_move is not None
                    else None
                ),

                **quality,
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

    ACPL = total genuine CP loss / cp_moves; None when cp_moves is zero.
    moves and classification counts include every move. mate_moves/mate_errors
    separately expose excluded transitions and heuristic mate-quality errors.

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

    # CP-comparable counts are explicit; mate moves remain in moves/classifications.
    for player in stats.values():
        for data in [player, *player["phases"].values()]:
            data.update(cp_moves=0, mate_moves=0, mate_errors=0)

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

        # Phase statistics
        stats[color]["phases"][phase]["moves"] += 1

        for data in (stats[color], stats[color]["phases"][phase]):
            if cp_loss is not None:
                data["total_centipawn_loss"] += cp_loss
                data["cp_moves"] += 1
            else:
                data["mate_moves"] += 1
                if classification in ("inaccuracy", "mistake", "blunder"):
                    data["mate_errors"] += 1

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

        move_count = stats[color]["cp_moves"]

        # Overall ACPL
        if move_count > 0:
            stats[color]["acpl"] = (
                stats[color]["total_centipawn_loss"]
                / move_count
            )
        else:
            stats[color]["acpl"] = None

        # Phase ACPL
        for phase in [
            "opening",
            "middlegame",
            "endgame"
        ]:

            phase_stats = stats[color]["phases"][phase]

            phase_move_count = phase_stats["cp_moves"]

            if phase_move_count > 0:

                phase_stats["acpl"] = (
                    phase_stats["total_centipawn_loss"]
                    / phase_move_count
                )

            else:
                phase_stats["acpl"] = None

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
    
    print("\n===== GAME PHASE TRANSITIONS =====")

    previous_phase = None

    for move in analysis:
        current_phase = move["phase"]

        if current_phase != previous_phase:
            print(
                f'Move {move["move_number"]} '
                f'({move["color"]}): '
                f'{current_phase.upper()}'
            )

            previous_phase = current_phase

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
            f'{player_stats["acpl"]}'
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
                f'ACPL: {phase_stats["acpl"]}'
            )
            
    weaknesses = detect_weakest_phase(stats)

    print("\n===== WEAKEST PHASE ANALYSIS =====")

    for color, weakness in weaknesses.items():
        print(f"\n{color.upper()}")

        if weakness is None:
            print("Insufficient moves for phase analysis.")
            continue

        print(f'Weakest phase: {weakness["phase"]}')
        print(f'ACPL: {weakness["acpl"]}')
        print(f'Moves analyzed: {weakness["moves"]}')
 
    
    worst_moves = find_worst_moves(
        analysis,
        weaknesses,
        top_n=5
    )

    print("\n===== WORST MOVES IN WEAKEST PHASE =====")

    for color, moves in worst_moves.items():
        print(f"\n{color.upper()}")

        if not moves:
            print("No moves available.")
            continue

        for move in moves:
            prefix = (
                f'{move["move_number"]}.'
                if move["color"] == "white"
                else f'{move["move_number"]}...'
            )

            print(
                f'{prefix} {move["san"]} | '
                f'{move["classification"]} | '
                f'Loss: {move["centipawn_loss"]} cp | '
                f'Best: {move["best_move"]}'
            )
    
    import json

    report = generate_weakness_report(
        analysis,
        stats,
        top_n=5
    )

    print("\n===== CHESSMIND WEAKNESS REPORT =====")

    print(json.dumps(report, indent=4))
