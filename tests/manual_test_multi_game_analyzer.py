
import json

from src.chessmind.pgn.multi_game_parser import (
    load_multiple_pgn,
)

from src.chessmind.analysis.player_identifier import (
    find_player_games,
)

from src.chessmind.analysis.multi_game_analyzer import (
    analyze_player_games,
)


def main():

    # Step 1: Load PGN games
    games = load_multiple_pgn("data/raw/sample_games.pgn")

    # Step 2: Identify target player
    player_games = find_player_games(games, "Adeepa")

    print(f"Found {len(player_games)} games")

    # Step 3: Analyze each game using Stockfish
    report = analyze_player_games(
        player_games,
        depth=8,
    )

    # Step 4: Print aggregated statistics
    summary = report["summary"]

    print("\n===== PLAYER PERFORMANCE =====")

    print(json.dumps(summary, indent=4))

    # Basic integration checks
    assert summary["total_games"] == 3
    assert summary["wins"] == 2
    assert summary["draws"] == 1
    assert summary["losses"] == 0
    assert summary["total_moves"] == 7

    phase_move_total = sum(
        phase["total_moves"]
        for phase in summary["phase_statistics"].values()
    )

    assert phase_move_total == summary["total_moves"]

    assert sum(
        summary["classifications"].values()
    ) == summary["total_moves"]

    print("\nMulti-game integration checks passed!")


if __name__ == "__main__":
    main()
