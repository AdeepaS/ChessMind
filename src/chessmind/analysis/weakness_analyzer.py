
def move_severity(move):
    """Rank by classification severity, then mate blunders, then genuine CP loss.

    The tuple is an ordering policy, never a synthetic centipawn value.
    Legacy moves without classification retain ordinary CP threshold ordering.
    """
    loss = move["centipawn_loss"]
    severity = {"Excellent": 0, "Good": 1, "Inaccuracy": 2, "Mistake": 3, "Blunder": 4}
    classification = move.get("classification")
    if classification is None and loss is not None:
        level = sum(loss > threshold for threshold in (20, 50, 100, 200))
    else:
        level = severity.get(classification, 0)
    return (level, int(loss is None and classification == "Blunder"),
            loss if loss is not None else 0)


def detect_weakest_phase(stats, min_moves=5):
    """
    Prioritize mate-error count, then highest defined CP-only ACPL.
    A mate error is eligible even below min_moves; otherwise require min_moves.
    Mate-only phases without errors provide no evidence of a weakness.
    """

    results = {}

    for color in ["white", "black"]:
        phase_stats = stats[color]["phases"]

        eligible_phases = {
            phase: data
            for phase, data in phase_stats.items()
            if (data.get("mate_errors", 0) > 0 or
                (data["moves"] >= min_moves and data["acpl"] is not None))
        }

        if not eligible_phases:
            results[color] = None
            continue

        weakest_phase = max(
            eligible_phases,
            key=lambda phase: (
                eligible_phases[phase].get("mate_errors", 0),
                eligible_phases[phase]["acpl"]
                if eligible_phases[phase]["acpl"] is not None else -1,
            )
        )

        results[color] = {
            "phase": weakest_phase,
            "acpl": eligible_phases[weakest_phase]["acpl"],
            "moves": eligible_phases[weakest_phase]["moves"],
            "mate_errors": eligible_phases[weakest_phase].get("mate_errors", 0),
        }

    return results



def find_worst_moves(analysis, weaknesses, top_n=5):
    """
    Find the most severe moves (mate-aware ordering)
    within each player's weakest game phase.
    """

    results = {}

    for color in ["white", "black"]:
        weakness = weaknesses.get(color)

        if weakness is None:
            results[color] = []
            continue

        weakest_phase = weakness["phase"]

        # Filter moves by player and phase
        phase_moves = [
            move for move in analysis
            if move["color"] == color
            and move["phase"] == weakest_phase
        ]

        # Sort by mate-aware severity (worst first)
        sorted_moves = sorted(
            phase_moves,
            key=move_severity,
            reverse=True
        )

        results[color] = sorted_moves[:top_n]

    return results
