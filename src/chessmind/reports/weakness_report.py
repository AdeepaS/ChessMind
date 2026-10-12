
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
            "overall_acpl": (round(stats[color]["acpl"], 2) if stats[color]["acpl"] is not None else None),
            "cp_moves": stats[color].get("cp_moves", stats[color]["moves"]),
            "mate_moves": stats[color].get("mate_moves", 0),
            "mate_errors": stats[color].get("mate_errors", 0),
            "weakest_phase": weakness,
            "phase_statistics": {
                phase: {
                    "moves": data["moves"],
                    "acpl": round(data["acpl"], 2) if data["acpl"] is not None else None,
                    "cp_moves": data.get("cp_moves", data["moves"]),
                    "mate_moves": data.get("mate_moves", 0),
                    "mate_errors": data.get("mate_errors", 0),
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
                    "mate_quality": move.get("mate_quality"),
                    "evaluation_before_detail": move.get("evaluation_before_detail"),
                    "evaluation_after_detail": move.get("evaluation_after_detail"),
                }
                for move in worst_moves[color]
            ],
        }

    return report
