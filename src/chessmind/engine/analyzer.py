import chess
import chess.engine
from dataclasses import dataclass

from src.chessmind.engine.evaluation_cache_keys import (
    CacheIdentityUnavailable, EngineIdentity, build_cache_key,
    build_engine_identity, snapshot_engine_metadata,
)


STOCKFISH_PATH = "tools/stockfish/stockfish.exe"


@dataclass(frozen=True)
class EngineCacheContext:
    """Identity bound to one borrowed engine; never owns or closes that engine."""

    engine: object
    identity: EngineIdentity


def create_cache_context(engine, executable_path, *, resource_paths=None,
                         embedded_resources=()):
    """Build once per engine, between searches, within the owner's cleanup scope.

    Resource declarations have the same explicit-attestation contract as
    build_engine_identity(). Files must remain unchanged for this engine lifetime.
    Callers must serialize metadata reads, configuration, and searches.
    """
    metadata = snapshot_engine_metadata(engine)
    identity = build_engine_identity(executable_path, metadata,
                                     resource_paths=resource_paths,
                                     embedded_resources=embedded_resources)
    return EngineCacheContext(engine, identity)


def prepare_cache_arguments(engine, cache, executable_path):
    """Opt-in owner setup; unsupported identity disables caching for this engine."""
    if cache is None or cache.info().max_entries == 0:
        return {}
    try:
        context = create_cache_context(engine, executable_path)
    except CacheIdentityUnavailable:
        return {}
    return {"cache": cache, "cache_context": context}


def serialize_evaluation(score):
    """Serialize a native score already normalized to White's perspective.

    winner disambiguates Mate(0) (White lost) from MateGiven (White won),
    whose mate() values are both zero. No synthetic CP enters this structure.
    """
    is_mate = score.is_mate()
    return {
        "type": "mate" if is_mate else "cp",
        "cp": score.score(),
        "mate": score.mate(),
        "winner": ("white" if score > chess.engine.Cp(0) else "black")
        if is_mate else None,
        "perspective": "white",
    }


def analyze_board(engine, board, depth=15, *, cache=None, cache_context=None, game=None):
    """
    Analyze a chess.Board position using an already-running Stockfish engine.

    Evaluation convention:
        Positive score -> White is better
        Negative score -> Black is better

    evaluation contains genuine CP or signed mate information.
    score_cp is the legacy synthetic numeric projection, retained for display
    compatibility only; it must not be used to calculate mate-related loss.

    Cache use requires a context created for this exact engine. Missing or
    unsupported identity falls back to a normal search. Calls on an engine must
    be sequential, including external configuration changes.
    game is a python-chess session token, forwarded only on actual searches and
    deliberately excluded from cache identity.
    """

    limit = chess.engine.Limit(depth=depth)
    key = None
    if (cache is not None and cache.info().max_entries > 0
            and isinstance(cache_context, EngineCacheContext)
            and cache_context.engine is engine):
        try:
            metadata = snapshot_engine_metadata(engine)
            key = build_cache_key(cache_context.identity, metadata, board, limit)
        except CacheIdentityUnavailable:
            pass
        if key is not None:
            cached = cache.get(key)
            if cached is not None:
                return cached

    info = engine.analyse(
        board,
        limit,
        **({"game": game} if game is not None else {}),
    )

    # Always evaluate from White's perspective
    score = info["score"].pov(chess.WHITE)

    # Convert score to centipawns.
    # Mate scores are mapped to a very large centipawn value.
    score_cp = score.score(mate_score=100000)

    best_move = info["pv"][0] if info.get("pv") else None

    result = {
        "score_cp": score_cp,
        "evaluation": serialize_evaluation(score),
        "best_move": best_move,
        "depth": info.get("depth"),
        "pv": info.get("pv", [])
    }
    # analyse() normally returns a completed search. Bound/explicitly partial
    # responses are never reusable. Missing depth/PV is conservatively bypassed
    # except at terminal positions, where no PV is expected.
    if key is not None:
        complete = (not any(info.get(flag) for flag in
                            ("lowerbound", "upperbound", "partial", "incomplete"))
                    and info.get("complete") is not False)
        sufficient = (board.is_checkmate() or board.is_stalemate()
                      or (info.get("depth") is not None and bool(info.get("pv"))))
        if complete and sufficient:
            cache.put(key, result)
    return result


def analyze_position(fen, depth=15, *, cache=None):
    """
    Analyze a FEN position.

    Useful for testing individual positions.
    """

    board = chess.Board(fen)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        cache_arguments = prepare_cache_arguments(engine, cache, STOCKFISH_PATH)
        return analyze_board(engine, board, depth, **cache_arguments)

    finally:
        engine.quit()


if __name__ == "__main__":

    board = chess.Board()

    result = analyze_position(board.fen())

    print("Position:")
    print(board)
    print()

    print("Evaluation:", result["score_cp"], "cp")
    print("Best move:", result["best_move"])
    print("Depth:", result["depth"])
    print("PV:", result["pv"])
