
from src.chessmind.analysis.player_statistics import PHASES


def detect_recurring_weaknesses(
    multi_game_report,
    min_games=3,
    min_moves_per_game=5,
    acpl_threshold=50,
    frequency_threshold=0.6,
):
    """
    Detect phases where a player repeatedly performs poorly.

    A phase is recurring weakness when:
    - At least min_games contain sufficient moves in the phase.
    - Phase ACPL exceeds acpl_threshold in a sufficient
      fraction of eligible games.

    Thresholds are heuristic and configurable.
    """

    if min_games < 1 or min_moves_per_game < 1:
        raise ValueError("Minimum counts must be positive.")

    if acpl_threshold < 0:
        raise ValueError("ACPL threshold cannot be negative.")

    if not 0 <= frequency_threshold <= 1:
        raise ValueError("Frequency threshold must be 0 to 1.")

    summary = multi_game_report["summary"]
    games = multi_game_report["games"]

    weaknesses = {}

    for phase in PHASES:

        eligible_games = 0
        weak_games = 0
        phase_acpls = []

        for game_record in games:

            color = game_record["color"]

            phase_stats = (
                game_record["stats"][color]["phases"][phase]
            )

            moves = phase_stats["moves"]

            if moves < min_moves_per_game:
                continue

            eligible_games += 1
            acpl = phase_stats["acpl"]
            phase_acpls.append(acpl)

            if acpl > acpl_threshold:
                weak_games += 1

        frequency = (
            weak_games / eligible_games
            if eligible_games > 0
            else 0.0
        )

        is_recurring = (
            eligible_games >= min_games
            and frequency >= frequency_threshold
            and weak_games > 0
        )

        overall_phase = summary["phase_statistics"][phase]

        weaknesses[phase] = {
            "eligible_games": eligible_games,
            "weak_games": weak_games,
            "weak_frequency": round(frequency, 2),
            "overall_acpl": overall_phase["acpl"],
            "game_acpls": phase_acpls,
            "is_recurring_weakness": is_recurring,
        }

    return weaknesses
