
import json

from src.chessmind.pgn.multi_game_parser import (
    load_multiple_pgn,
)

from src.chessmind.players.player_identifier import (
    find_player_games,
)

from src.chessmind.players.multi_game_analyzer import (
    analyze_player_games,
)

from src.chessmind.reports.player_report import (
    generate_player_report,
    save_player_report,
)


def main():
    player_name = "Adeepa"

    # Step 1: Load games
    games = load_multiple_pgn("data/raw/sample_games.pgn")

    # Step 2: Identify player
    player_games = find_player_games(games, player_name)

    # Step 3: Perform multi-game Stockfish analysis
    analysis = analyze_player_games(
        player_games,
        depth=8,
    )

    # Step 4: Generate structured report
    report = generate_player_report(
        player_name,
        analysis,
    )

    # Step 5: Save JSON
    output_path = save_player_report(
        report,
        "data/reports/player_report.json",
    )

    print("\n===== PLAYER REPORT =====")
    print(json.dumps(report, indent=4))

    print(f"\nReport saved to: {output_path}")

    # Validate report structure
    assert report["player"]["name"] == player_name
    assert report["overall_performance"]["total_games"] == 3
    assert len(report["games"]) == 3

    # Short sample games do not provide sufficient
    # evidence for recurring weakness detection.
    assert (
        report["recurring_weaknesses"]["detected_phases"]
        == []
    )

    with open(output_path, encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report

    print("\nPlayer report checks passed!")


if __name__ == "__main__":
    main()
