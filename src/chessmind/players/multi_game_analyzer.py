
from src.chessmind.analysis.game_analyzer import (
    analyze_game,
    calculate_game_statistics,
)

from src.chessmind.players.player_statistics import (
    aggregate_player_statistics,
)


def analyze_player_games(player_games, depth=12, *, cache=None):
    """
    Analyze multiple games belonging to a player.

    Args:
        player_games: Output from find_player_games().
        depth: Stockfish search depth.

    Returns:
        A dictionary containing individual game analyses
        and aggregated player statistics.
    """

    analyzed_games = []
    cache_arguments = ({"cache": cache}
                       if cache is not None and cache.info().max_entries > 0 else {})

    for entry in player_games:

        game_number = entry["game_number"]
        game = entry["game"]

        print(f"Analyzing game {game_number}...")

        # Reuse existing single-game analysis
        analysis = analyze_game(game, depth=depth, **cache_arguments)

        # Reuse existing game statistics
        stats = calculate_game_statistics(analysis)

        analyzed_games.append({
            "game_number": game_number,
            "color": entry["color"],
            "opponent": entry["opponent"],
            "result": entry["result"],
            "analysis": analysis,
            "stats": stats,
        })

    summary = aggregate_player_statistics(analyzed_games)

    return {
        "summary": summary,
        "games": analyzed_games,
    }
