
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.analysis.player_identifier import find_player_games


def main():
    games = load_multiple_pgn("data/raw/sample_games.pgn")

    player_games = find_player_games(games, "adeepa")

    print(f"Games found: {len(player_games)}")

    for entry in player_games:
        print(
            f"Game {entry['game_number']} | "
            f"Color: {entry['color']} | "
            f"Opponent: {entry['opponent']} | "
            f"Result: {entry['result']}"
        )


if __name__ == "__main__":
    main()
