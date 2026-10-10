
from src.chessmind.players.recurring_weakness_analyzer import (
    detect_recurring_weaknesses,
)


def make_game(opening_acpl, middlegame_acpl, endgame_acpl):
    phase_values = {
        "opening": opening_acpl,
        "middlegame": middlegame_acpl,
        "endgame": endgame_acpl,
    }

    return {
        "color": "white",
        "stats": {
            "white": {
                "phases": {
                    phase: {
                        "moves": 10,
                        "acpl": acpl,
                    }
                    for phase, acpl in phase_values.items()
                }
            }
        },
    }


games = [
    make_game(15, 65, 25),
    make_game(20, 80, 30),
    make_game(12, 55, 20),
    make_game(18, 70, 40),
    make_game(25, 75, 35),
]

report = {
    "games": games,
    "summary": {
        "phase_statistics": {
            "opening": {"acpl": 18},
            "middlegame": {"acpl": 69},
            "endgame": {"acpl": 30},
        }
    },
}

weaknesses = detect_recurring_weaknesses(report)

for phase, data in weaknesses.items():
    print(
        f"{phase.capitalize()} | "
        f"Eligible: {data['eligible_games']} | "
        f"Weak: {data['weak_games']} | "
        f"Frequency: {data['weak_frequency']:.0%} | "
        f"Recurring: {data['is_recurring_weakness']}"
    )

assert weaknesses["middlegame"]["is_recurring_weakness"]
assert not weaknesses["opening"]["is_recurring_weakness"]
assert not weaknesses["endgame"]["is_recurring_weakness"]

print("\nRecurring weakness detection checks passed!")
