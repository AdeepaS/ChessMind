
def detect_weakest_phase(stats, min_moves=5):
    """
    Identify the phase with the highest ACPL
    for each player.

    Only consider phases containing at least
    min_moves analyzed moves.
    """

    results = {}

    for color in ["white", "black"]:
        phase_stats = stats[color]["phases"]

        eligible_phases = {
            phase: data
            for phase, data in phase_stats.items()
            if data["moves"] >= min_moves
        }

        if not eligible_phases:
            results[color] = None
            continue

        weakest_phase = max(
            eligible_phases,
            key=lambda phase: eligible_phases[phase]["acpl"]
        )

        results[color] = {
            "phase": weakest_phase,
            "acpl": eligible_phases[weakest_phase]["acpl"],
            "moves": eligible_phases[weakest_phase]["moves"]
        }

    return results



def find_worst_moves(analysis, weaknesses, top_n=5):
    """
    Find the highest centipawn-loss moves
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

        # Sort by centipawn loss (highest first)
        sorted_moves = sorted(
            phase_moves,
            key=lambda move: move["centipawn_loss"],
            reverse=True
        )

        results[color] = sorted_moves[:top_n]

    return results
