"""Validate the longer PGN games for ChessMind Day 6, Step 1.

Run from the project root:
    python tests/manual_day6_validation.py
"""

from pathlib import Path

import chess.pgn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PGN_PATH = PROJECT_ROOT / "data" / "raw" / "validation_games.pgn"


def validate_games() -> None:
    if not PGN_PATH.is_file():
        raise FileNotFoundError(f"PGN file not found: {PGN_PATH}")

    count = 0
    with PGN_PATH.open("r", encoding="utf-8-sig") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            count += 1
            if game.errors:
                raise ValueError(f"Game {count} contains PGN parsing errors: {game.errors}")

            board = game.board()
            plies = 0
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move in game {count}: {move.uci()}")
                board.push(move)
                plies += 1

            if plies == 0:
                raise ValueError(f"Game {count} has no moves")

            print(f"\nGame {count}")
            print(f"White: {game.headers.get('White', 'Unknown')}")
            print(f"Black: {game.headers.get('Black', 'Unknown')}")
            print(f"Result: {game.headers.get('Result', '*')}")
            print(f"Last move number: {(plies + 1) // 2}")
            print(f"Total plies: {plies}")
            print(f"Board game-over state: {board.is_game_over()}")
            print(f"Final FEN: {board.fen()}")

    if count != 3:
        raise AssertionError(f"Expected 3 games, found {count}")

    print(f"\nPASS: All {count} PGNs parsed with legal moves and no reported errors.")


if __name__ == "__main__":
    validate_games()
