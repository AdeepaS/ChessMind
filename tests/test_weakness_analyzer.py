
from src.chessmind.analysis.weakness_analyzer import (
    detect_weakest_phase,
    find_worst_moves,
)


def test_detect_weakest_phase():
    stats = {
        "white": {
            "phases": {
                "opening": {"moves": 10, "acpl": 20},
                "middlegame": {"moves": 15, "acpl": 65},
                "endgame": {"moves": 12, "acpl": 30},
            }
        },
        "black": {
            "phases": {
                "opening": {"moves": 10, "acpl": 40},
                "middlegame": {"moves": 15, "acpl": 25},
                "endgame": {"moves": 12, "acpl": 15},
            }
        },
    }

    result = detect_weakest_phase(stats)

    assert result["white"]["phase"] == "middlegame"
    assert result["black"]["phase"] == "opening"


def test_find_worst_moves():
    analysis = [
        {
            "color": "white",
            "phase": "middlegame",
            "centipawn_loss": 150,
            "san": "Nf3",
        },
        {
            "color": "white",
            "phase": "middlegame",
            "centipawn_loss": 300,
            "san": "Qh5",
        },
        {
            "color": "white",
            "phase": "opening",
            "centipawn_loss": 400,
            "san": "e4",
        },
    ]

    weaknesses = {
        "white": {"phase": "middlegame"},
        "black": None,
    }

    result = find_worst_moves(
        analysis,
        weaknesses,
        top_n=2
    )

    assert len(result["white"]) == 2
    assert result["white"][0]["san"] == "Qh5"
    assert result["white"][1]["san"] == "Nf3"
    assert result["black"] == []
