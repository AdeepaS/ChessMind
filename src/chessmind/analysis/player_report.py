
import json

from pathlib import Path

from src.chessmind.analysis.recurring_weakness_analyzer import (
    detect_recurring_weaknesses,
)


def generate_player_report(
    player_name,
    multi_game_report,
    min_games=3,
    min_moves_per_game=5,
    acpl_threshold=50,
    frequency_threshold=0.6,
):
    """
    Generate a structured multi-game player performance report.

    Reuses previously calculated game analyses and statistics.
    Does not perform additional Stockfish evaluations.
    """

    summary = multi_game_report["summary"]
    games = multi_game_report["games"]

    weaknesses = detect_recurring_weaknesses(
        multi_game_report,
        min_games=min_games,
        min_moves_per_game=min_moves_per_game,
        acpl_threshold=acpl_threshold,
        frequency_threshold=frequency_threshold,
    )

    game_summaries = []

    for entry in games:
        color = entry["color"]
        player_stats = entry["stats"][color]

        game_summaries.append({
            "game_number": entry["game_number"],
            "color": color,
            "opponent": entry["opponent"],
            "result": entry["result"],
            "moves_analyzed": player_stats["moves"],
            "acpl": round(player_stats["acpl"], 2),
            "phase_acpl": {
                phase: (
                    round(phase_stats["acpl"], 2)
                    if phase_stats["moves"] > 0
                    else None
                )
                for phase, phase_stats
                in player_stats["phases"].items()
            },
        })

    recurring_phases = [
        phase
        for phase, data in weaknesses.items()
        if data["is_recurring_weakness"]
    ]

    return {
        "player": {
            "name": player_name,
        },
        "overall_performance": summary,
        "recurring_weaknesses": {
            "detected_phases": recurring_phases,
            "phase_details": weaknesses,
        },
        "games": game_summaries,
    }


def save_player_report(report, output_path):
    """
    Save the generated report as a JSON file.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return path
