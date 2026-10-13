
import chess.engine

from src.chessmind.engine import analyzer as engine_analyzer
from src.chessmind.analysis.game_analyzer import (
    analyze_game,
    calculate_game_statistics,
)

from src.chessmind.players.player_statistics import (
    aggregate_player_statistics,
)


def analyze_player_games(player_games, depth=12, *, cache=None, reuse_engine=False):
    """
    Analyze multiple games belonging to a player.

    Args:
        player_games: Output from find_player_games().
        depth: Stockfish search depth.
        reuse_engine: Own one engine for the nonempty batch, used sequentially.
            Cache identity is constructed once; the caller owns the cache.

    Returns:
        A dictionary containing individual game analyses
        and aggregated player statistics.
    """

    analyzed_games = []
    cache_arguments = ({"cache": cache}
                       if cache is not None and cache.info().max_entries > 0 else {})

    engine = None
    try:
        for entry in player_games:
            # Launch lazily so empty iterables also avoid starting a process.
            if reuse_engine and engine is None:
                engine = chess.engine.SimpleEngine.popen_uci(engine_analyzer.STOCKFISH_PATH)
                cache_arguments = engine_analyzer.prepare_cache_arguments(
                    engine, cache, engine_analyzer.STOCKFISH_PATH)
                cache_arguments["engine"] = engine

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
    finally:
        if engine is not None:
            engine.quit()

    summary = aggregate_player_statistics(analyzed_games)

    return {
        "summary": summary,
        "games": analyzed_games,
    }
