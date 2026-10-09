
from src.chessmind.analysis.player_statistics import (
    aggregate_player_statistics,
)


def make_stats(moves, acpl):
    return {
        "total_moves": moves,
        "overall_acpl": acpl,
        "phase_statistics": {
            "opening": {
                "total_moves": moves,
                "acpl": acpl,
            },
            "middlegame": {
                "total_moves": 0,
                "acpl": None,
            },
            "endgame": {
                "total_moves": 0,
                "acpl": None,
            },
        },
    }


def make_record(color, result, moves, acpl):
    return {
        "color": color,
        "result": result,
        "stats": {
            color: make_stats(moves, acpl),
        },
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
