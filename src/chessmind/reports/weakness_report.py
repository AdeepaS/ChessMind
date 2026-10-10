
from src.chessmind.analysis.weakness_analyzer import (
    detect_weakest_phase,
    find_worst_moves,
)


def generate_weakness_report(analysis, stats, top_n=5):
    """
    Generate structured weakness reports for
    White and Black.
    """

    weaknesses = detect_weakest_phase(stats)

    worst_moves = find_worst_moves(
        analysis,
        weaknesses,
        top_n=top_n
    )

    report = {}

    for color in ["white", "black"]:
        weakness = weaknesses[color]

        report[color] = {
            "total_moves": stats[color]["moves"],
            "overall_acpl": round(stats[color]["acpl"], 2),
            "weakest_phase": weakness,
            "phase_statistics": {
                phase: {
                    "moves": data["moves"],
                    "acpl": round(data["acpl"], 2),
                }
                for phase, data in stats[color]["phases"].items()
            },
            "worst_moves": [
                {
                    "move_number": move["move_number"],
                    "san": move["san"],
                    "classification": move["classification"],
                    "centipawn_loss": move["centipawn_loss"],
                    "best_move": move["best_move"],
                    "fen_before": move["fen_before"],
                }
                for move in worst_moves[color]
            ],
        }

    return report
