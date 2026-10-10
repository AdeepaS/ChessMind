
PHASES = ("opening", "middlegame", "endgame")

CLASSIFICATIONS = (
    "excellent",
    "good",
    "inaccuracy",
    "mistake",
    "blunder",
)


def aggregate_player_statistics(analyzed_games):
    """
    Aggregate the target player's statistics across games.

    Uses raw centipawn-loss totals to calculate accurate
    overall and phase ACPL values.
    """

    summary = {
        "total_games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "unfinished": 0,
        "total_moves": 0,
        "total_centipawn_loss": 0,
        "overall_acpl": None,
        "classifications": {
            name: 0 for name in CLASSIFICATIONS
        },
        "phase_statistics": {
            phase: {
                "total_moves": 0,
                "total_centipawn_loss": 0,
                "acpl": None,
            }
            for phase in PHASES
        },
    }

    for record in analyzed_games:

        color = record["color"]
        result = record["result"]

        if color not in ("white", "black"):
            raise ValueError(f"Invalid color: {color}")

        player_stats = record["stats"][color]

        # Game results from the player's perspective
        summary["total_games"] += 1

        if result == "1/2-1/2":
            summary["draws"] += 1

        elif result == "1-0":
            if color == "white":
                summary["wins"] += 1
            else:
                summary["losses"] += 1

        elif result == "0-1":
            if color == "black":
                summary["wins"] += 1
            else:
                summary["losses"] += 1

        elif result == "*":
            summary["unfinished"] += 1

        else:
            raise ValueError(f"Unknown result: {result}")

        # Aggregate raw statistics
        summary["total_moves"] += player_stats["moves"]

        summary["total_centipawn_loss"] += (
            player_stats["total_centipawn_loss"]
        )

        # Aggregate classifications
        for classification in CLASSIFICATIONS:
            summary["classifications"][classification] += (
                player_stats[classification]
            )

        # Aggregate individual phases
        for phase in PHASES:

            phase_stats = player_stats["phases"][phase]
            target = summary["phase_statistics"][phase]

            target["total_moves"] += phase_stats["moves"]

            target["total_centipawn_loss"] += (
                phase_stats["total_centipawn_loss"]
            )

    # Overall ACPL
    if summary["total_moves"] > 0:
        summary["overall_acpl"] = round(
            summary["total_centipawn_loss"]
            / summary["total_moves"],
            2,
        )

    # Phase ACPL
    for phase in PHASES:

        phase_stats = summary["phase_statistics"][phase]

        if phase_stats["total_moves"] > 0:
            phase_stats["acpl"] = round(
                phase_stats["total_centipawn_loss"]
                / phase_stats["total_moves"],
                2,
            )

    return summary
