
from pathlib import Path
import chess.pgn


def load_multiple_pgn(file_path):
    """
    Load all chess games from a PGN file.

    Args:
        file_path (str | Path): Path to the PGN file.

    Returns:
        list[chess.pgn.Game]: Parsed chess games.
    """

    games = []

    with open(file_path, "r", encoding="utf-8-sig") as pgn_file:

        while True:
            game = chess.pgn.read_game(pgn_file)

            # None indicates end of file
            if game is None:
                break

            # Do not silently accept malformed games
            if game.errors:
                raise ValueError(
                    f"Invalid PGN game {len(games) + 1}: "
                    f"{game.errors}"
                )

            games.append(game)

    return games


if __name__ == "__main__":

    file_path = "data/raw/sample_games.pgn"

    games = load_multiple_pgn(file_path)

    print(f"Total games loaded: {len(games)}")

    for index, game in enumerate(games, start=1):

        headers = game.headers

        print(f"\nGame {index}")
        print(f"White: {headers.get('White', 'Unknown')}")
        print(f"Black: {headers.get('Black', 'Unknown')}")
        print(f"Result: {headers.get('Result', '*')}")
        print(f"Moves: {sum(1 for _ in game.mainline_moves())}")
