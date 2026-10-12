"""Day 6, Step 1B: report existing Stockfish results without changing them.

Run: .venv/Scripts/python.exe tests/manual_stockfish_validation.py --depth 12
Exit status is nonzero if any diagnostic check or game analysis fails.
Mate evaluations are checked directly; ACPL uses only CP-comparable moves.
"""

import argparse
import math
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chess

from src.chessmind.analysis.game_analyzer import (
    analyze_game,
    calculate_game_statistics,
    classify_move,
)
from src.chessmind.engine.analyzer import STOCKFISH_PATH
from src.chessmind.analysis.weakness_analyzer import move_severity
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.pgn.parser import get_game_info, get_moves_with_notation


PHASES = ("opening", "middlegame", "endgame")
CLASSIFICATIONS = ("excellent", "good", "inaccuracy", "mistake", "blunder")
EXPECTED_GAMES = ((87, "1-0"), (103, "1-0"), (82, "0-1"))
def format_acpl(value):
    return "N/A" if value is None else f"{value:.6f}"


def move_label(move):
    dots = "." if move["color"] == "white" else "..."
    return f'{move["move_number"]}{dots} {move["san"]} (ply {move["ply"]})'


def print_statistics(analysis, stats):
    print("\nPhase Transitions (position before the listed move):")
    previous_phase = None
    for move in analysis:
        if move["phase"] != previous_phase:
            print(f'  {move["phase"].title()}: {move_label(move)}')
            previous_phase = move["phase"]
    for color in ("white", "black"):
        player = stats[color]
        print(f"\n{color.title()} Statistics:")
        print(f'  Total moves: {player["moves"]}')
        print(f'  Total CP loss: {player["total_centipawn_loss"]}')
        print(f'  ACPL: {format_acpl(player["acpl"])}')
        for classification in CLASSIFICATIONS:
            print(f"  {classification.title()}: {player[classification]}")
        print(f'  ACPL denominator: {player["cp_moves"]}; mate moves: {player["mate_moves"]}')
        print("  Phase Statistics:")
        for phase in PHASES:
            data = player["phases"][phase]
            print(f'    {phase.title()}: moves={data["moves"]}, '
                  f'ACPL={format_acpl(data["acpl"])}')
        print("  Worst Moves (all phases):")
        # find_worst_moves() only ranks a player's weakest phase, so it cannot
        # supply this whole-game view. Rank the existing fields directly.
        moves = [move for move in analysis if move["color"] == color]
        for move in sorted(moves, key=move_severity,
                           reverse=True)[:5]:
            print(f'    {move_label(move)} | {move["phase"]} | '
                  f'{move["centipawn_loss"]} cp | {move["classification"]}')


def print_mate_diagnostics(analysis):
    print("\nMate Score Diagnostics:")
    print("  Structured evaluations use White POV; winner disambiguates mate zero.")
    print("  Legacy numeric projections are display-only, never ACPL inputs.")
    positions = []
    if analysis:
        positions.append(("Initial position", analysis[0]["evaluation_before_detail"]))
    positions.extend((f"After {move_label(m)}", m["evaluation_after_detail"])
                     for m in analysis)
    mates = [(label, score) for label, score in positions if score["type"] == "mate"]
    for label, score in mates:
        print(f'  {label}: mate={score["mate"]}, winner={score["winner"]}')
    if not mates:
        print("  No forced-mate evaluations reported.")
    for move in analysis:
        if move["mate_quality"] is not None:
            print(f'  {move_label(move)}: {move["mate_quality"]} | '
                  f'{move["classification"]} | CP loss={move["centipawn_loss"]}')
    if analysis and chess.Board(analysis[-1]["fen_after"]).is_checkmate():
        print(f'  Final checkmate FEN; evaluation={analysis[-1]["evaluation_after_detail"]}')
    observations = [
        f'{move_label(m)}: {m["mate_quality"]} ({m["classification"]}); CP loss undefined'
        for m in analysis if m["mate_quality"] is not None
        and m["classification"] in ("Inaccuracy", "Mistake", "Blunder")
    ]
    return len(mates), observations


def validate_game(analysis, stats, expected_moves, check):
    print("\nValidation:")
    check(len(analysis) == len(expected_moves),
          f"Analyzed plies={len(analysis)}, PGN plies={len(expected_moves)}")
    identity_fields = ("ply", "move_number", "color", "san", "uci")
    check(len(analysis) == len(expected_moves) and all(
        all(actual[key] == expected[key] for key in identity_fields)
        for actual, expected in zip(analysis, expected_moves)),
        "Returned moves match PGN order, notation, and colors")
    invalid_losses = [(move_label(m), m["centipawn_loss"]) for m in analysis
                      if m["centipawn_loss"] is not None and (
                          not math.isfinite(m["centipawn_loss"]) or m["centipawn_loss"] < 0)]
    check(not invalid_losses, f"Finite nonnegative CP losses; invalid={invalid_losses}")
    mismatches = [move_label(m) for m in analysis
                  if m["centipawn_loss"] is not None
                  and m["classification"] != classify_move(m["centipawn_loss"])]
    check(not mismatches, f"Existing classification thresholds; mismatches={mismatches}")
    invalid_mate_losses = []
    invalid_scores = []
    for move in analysis:
        before = move["evaluation_before_detail"]
        after = move["evaluation_after_detail"]
        mate_move = before["type"] == "mate" or after["type"] == "mate"
        if (move["centipawn_loss"] is None) != mate_move:
            invalid_mate_losses.append(move_label(move))
        for score in (before, after):
            valid = score["perspective"] == "white" and (
                (score["type"] == "cp" and score["cp"] is not None
                 and score["mate"] is None and score["winner"] is None)
                or (score["type"] == "mate" and score["cp"] is None
                    and isinstance(score["mate"], int) and score["winner"] in ("white", "black"))
            )
            if not valid:
                invalid_scores.append(move_label(move))
    check(not invalid_mate_losses,
          f"CP loss undefined exactly for mate transitions; invalid={invalid_mate_losses}")
    check(not invalid_scores, f"Structured evaluations consistent; invalid={invalid_scores}")
    if analysis and chess.Board(analysis[-1]["fen_after"]).is_checkmate():
        board = chess.Board(analysis[-1]["fen_after"])
        score = analysis[-1]["evaluation_after_detail"]
        winner = "black" if board.turn == chess.WHITE else "white"
        check(score["type"] == "mate" and score["mate"] == 0 and score["winner"] == winner,
              "Terminal mate zero and winner match final board")
    for color in ("white", "black"):
        player = stats[color]
        expected_count = sum(m["color"] == color for m in expected_moves)
        cp_count = sum(m["color"] == color and m["centipawn_loss"] is not None
                       for m in analysis)
        check(player["cp_moves"] == cp_count,
              f'{color}: ACPL denominator={player["cp_moves"]}, CP-comparable moves={cp_count}')
        check(player["moves"] == expected_count,
              f'{color}: moves={player["moves"]}, expected={expected_count}')
        classified = sum(player[key] for key in CLASSIFICATIONS)
        check(classified == player["moves"],
              f'{color}: classification count={classified}, moves={player["moves"]}')
        phase_count = sum(player["phases"][p]["moves"] for p in PHASES)
        check(phase_count == player["moves"],
              f'{color}: phase count={phase_count}, moves={player["moves"]}')
        for label, data in [(color, player)] + [
            (f"{color}/{p}", player["phases"][p]) for p in PHASES
        ]:
            expected_acpl = (data["total_centipawn_loss"] / data["cp_moves"]
                             if data["cp_moves"] else None)
            check(data["cp_moves"] + data["mate_moves"] == data["moves"],
                  f"{label}: CP plus mate counts match moves")
            check((data["acpl"] is None if expected_acpl is None else
                   data["acpl"] is not None and math.isclose(
                       data["acpl"], expected_acpl, rel_tol=1e-9, abs_tol=1e-9)),
                  f'{label}: ACPL={format_acpl(data["acpl"])}, expected={format_acpl(expected_acpl)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=12,
                        help="Stockfish analysis depth (default: 12)")
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be a positive integer")

    checks = []
    observations = []

    def check(condition, message):
        checks.append(bool(condition))
        print(f"  {'PASS' if condition else 'FAIL'}: {message}", flush=True)

    print("=" * 72)
    print("CHESSMIND - DAY 6 STOCKFISH VALIDATION")
    print(f"Depth: {args.depth}")
    print("=" * 72, flush=True)
    processed = 0
    mate_candidates = 0
    original_cwd = Path.cwd()
    try:
        # The existing engine path is relative; preserve its integration while
        # allowing this script to be invoked from outside the repository.
        os.chdir(PROJECT_ROOT)
        games = load_multiple_pgn(PROJECT_ROOT / "data/raw/validation_games.pgn")
        check(len(games) == 3, f"Dataset contains {len(games)} games; expected 3")
        if not Path(STOCKFISH_PATH).is_file():
            raise FileNotFoundError(f"Stockfish executable missing: {STOCKFISH_PATH}")
        for index, game in enumerate(games, start=1):
            print("\n" + "=" * 72)
            info = get_game_info(game)
            print(f'GAME {index}: {info["white"]} vs {info["black"]}')
            print(f'White: {info["white"]}\nBlack: {info["black"]}')
            print(f'Result: {info["result"]}')
            try:
                moves = get_moves_with_notation(game)
                print(f"Total plies: {len(moves)}", flush=True)
                if index <= len(EXPECTED_GAMES):
                    expected = EXPECTED_GAMES[index - 1]
                    check((len(moves), info["result"]) == expected,
                          f"Historical fixture plies/result: expected {expected}")
                print("Running existing analyzer (N+1 positions, one engine per game)...",
                      flush=True)
                # analyze_game owns its engine and closes it in finally.
                analysis = analyze_game(game, depth=args.depth)
                stats = calculate_game_statistics(analysis)
                print_statistics(analysis, stats)
                candidates, anomalies = print_mate_diagnostics(analysis)
                mate_candidates += candidates
                observations.extend(f"Game {index}: {item}" for item in anomalies)
                validate_game(analysis, stats, moves, check)
                processed += 1
            except Exception as exc:
                check(False, f"Game {index} failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        check(False, f"Validation could not complete: {type(exc).__name__}: {exc}")
    finally:
        os.chdir(original_cwd)

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    check(processed == 3, f"Games successfully processed: {processed}/3")
    print(f"Checks: {sum(checks)} passed, {len(checks) - sum(checks)} failed")
    print(f"Confirmed mate positions: {mate_candidates}")
    print(f"Mate-related quality observations: {len(observations)}")
    for observation in observations:
        print(f"  {observation}")
    print("ACPL excludes mate transitions; mate quality uses the documented heuristic policy.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
