"""Run from the project root: python tests/manual_test_player_statistics.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chessmind.analysis.game_analyzer import calculate_game_statistics, classify_move
from src.chessmind.players.player_statistics import (
    aggregate_player_statistics,
)


def make_stats(color, moves, acpl):
    # Uniform synthetic CP losses yield the requested mean without Stockfish.
    return calculate_game_statistics([
        {
            "color": color,
            "phase": "opening",
            "centipawn_loss": acpl,
            "classification": classify_move(acpl),
        }
        for _ in range(moves)
    ])


def make_record(color, result, moves, acpl):
    return {
        "color": color,
        "result": result,
        "stats": make_stats(color, moves, acpl),
    }


analyzed_games = [
    make_record("white", "1-0", 40, 25),
    make_record("black", "0-1", 20, 60),
    make_record("white", "1/2-1/2", 60, 30),
]

summary = aggregate_player_statistics(analyzed_games)

assert summary["total_games"] == 3
assert summary["wins"] == 2
assert summary["draws"] == 1
assert summary["losses"] == 0
assert summary["total_moves"] == 120
assert summary["overall_acpl"] == 33.33
assert summary["phase_statistics"]["opening"]["acpl"] == 33.33

print(summary)
print("\nAll aggregation checks passed!")
