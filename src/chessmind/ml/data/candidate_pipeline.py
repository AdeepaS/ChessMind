"""Sequential candidate batches using the existing analyzer and cache contracts."""

import hashlib
import io
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from ...analysis.game_analyzer import analyze_game
from ...engine import analyzer as engine_api
from ...engine.evaluation_cache import EvaluationCache
from ...engine.evaluation_cache_keys import CacheIdentityUnavailable, snapshot_engine_metadata
from .candidate_extraction import extract_candidates
from .candidate_schema import (
    SCHEMA_VERSION, AnalysisProfile, ExtractionConfig, SourceEvidence, candidate_json,
)
from .game_dataset_loader import open_games


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _profile(engine, depth, binary_path, context):
    """Read observed metadata; unavailable values remain explicit nulls."""
    identity = context.identity if context is not None else None
    try:
        metadata = snapshot_engine_metadata(engine)
    except CacheIdentityUnavailable:
        metadata = None
    digest = identity.executable_sha256 if identity else (
        file_sha256(binary_path) if binary_path is not None else None)
    reported = metadata.reported_id if metadata else tuple(sorted(engine.id.items()))
    return AnalysisProfile(
        configured_depth=depth, engine_id=reported, engine_binary_sha256=digest,
        engine_options=metadata.config if metadata else None,
        engine_resources=identity.resources if identity else None, chess_version=chess.__version__)


def run_candidate_pipeline(input_path, output_path, *, source_manifest_path=None,
                           max_games=10, depth=8, config=None, engine=None, cache=None,
                           cache_context=None, engine_path=None, progress=None):
    """Publish a new output directory containing JSONL and its manifest.

    A supplied engine is borrowed, never configured or closed. Its caller must
    provide a cache context to enable caching. Owned batches start one engine,
    configure Threads=1/Hash=64 and prepare the existing cache once. Failed games
    are recorded and skipped; engine termination stops the batch. Setup, I/O,
    shutdown and interruption errors abort publication. No retries or workers.
    """
    if type(max_games) is not int or max_games < 0:
        raise ValueError("max_games must be a nonnegative integer")
    if type(depth) is not int:
        raise ValueError("depth must be an integer in 1..128")
    AnalysisProfile(configured_depth=depth)
    config = config or ExtractionConfig()
    input_path, output_path = Path(input_path), Path(output_path)
    output_dir = output_path.parent
    if output_dir.exists():
        raise FileExistsError("output parent must be a new directory for atomic bundle publication")
    if output_path.name == "candidate_manifest.json":
        raise ValueError("output filename conflicts with manifest")
    started = time.perf_counter()
    input_digest = file_sha256(input_path)
    if source_manifest_path is None:
        adjacent = input_path.with_name("source_manifest.json")
        source_manifest_path = adjacent if adjacent.is_file() else None
    manifest_digest, archive, references = None, None, {}
    if source_manifest_path is not None:
        source_manifest_path = Path(source_manifest_path)
        manifest_digest = file_sha256(source_manifest_path)
        original = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if original.get("sha256") != input_digest:
            raise ValueError("input SHA-256 disagrees with source manifest")
        archive = original.get("source_url")
        for ref in original.get("source_references", []):
            if ref["game_id"] in references:
                raise ValueError("duplicate game reference in source manifest")
            references[ref["game_id"]] = ref["archive_game_index"]
    source = SourceEvidence(str(input_path), input_digest,
                            str(source_manifest_path) if source_manifest_path else None,
                            manifest_digest, archive)
    owns_engine = engine is None
    if not owns_engine and cache_context is not None and cache_context.engine is not engine:
        raise ValueError("cache context belongs to a different engine")
    binary_path = Path(engine_path or engine_api.STOCKFISH_PATH) if owns_engine else (
        Path(engine_path) if engine_path else None)
    cache = cache if cache is not None else EvaluationCache(4096)
    cache_before = asdict(cache.info())
    profile = None
    counts, statuses = Counter(), Counter()
    failures, successful, seen_ids = [], [], set()
    moves_analyzed = 0
    cache_arguments = {}
    if not owns_engine and cache_context is not None:
        cache_arguments = {"cache": cache, "cache_context": cache_context}
    stop_reason = "input_end_or_game_limit"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidates-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        target = stage / output_path.name
        try:
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                with open_games(input_path, strict=False, max_records=max_games) as reader:
                    for game in reader:
                        if engine is None:
                            if not binary_path.is_file():
                                raise FileNotFoundError(f"Stockfish binary not found: {binary_path}")
                            engine = chess.engine.SimpleEngine.popen_uci(str(binary_path))
                            engine.configure({"Threads": 1, "Hash": 64})
                            cache_arguments = engine_api.prepare_cache_arguments(engine, cache, binary_path)
                        if profile is None:
                            profile = _profile(engine, depth, binary_path,
                                               cache_arguments.get("cache_context"))
                        if progress:
                            progress(f"Analyzing game {reader.stats.accepted}/{max_games}: {game.game_id}")
                        game_start = time.perf_counter()
                        try:
                            parsed = chess.pgn.read_game(io.StringIO(game.original_pgn))
                            analysis = analyze_game(parsed, depth=depth, engine=engine, **cache_arguments)
                            evidence = replace(source, source_archive=archive or game.source_archive,
                                               archive_game_index=references.get(game.game_id))
                            records = extract_candidates(analysis, game, config=config,
                                                         profile=profile, source=evidence)
                            ids = {r.candidate_id for r in records}
                            if len(ids) != len(records) or ids & seen_ids:
                                raise ValueError("duplicate candidate identifier")
                            lines = [candidate_json(r) + "\n" for r in records]
                        except (ValueError, TypeError, KeyError, RuntimeError, chess.engine.EngineError,
                                TimeoutError) as exc:
                            failures.append({"game_id": game.game_id, "error_type": type(exc).__name__,
                                             "message": str(exc)})
                            if progress:
                                progress(f"Failed {game.game_id}: {type(exc).__name__}: {exc}")
                            if isinstance(exc, chess.engine.EngineTerminatedError):
                                stop_reason = "engine_terminated"
                                break
                            continue
                        # Writes remain outside game-error handling: I/O failure aborts publication.
                        stream.writelines(lines)
                        seen_ids.update(ids)
                        counts.update(r.selection_type for r in records)
                        statuses.update(r.mate_transition_status for r in records if r.mate_transition_status)
                        moves_analyzed += len(analysis)
                        successful.append({"game_id": game.game_id, "plies": len(analysis),
                                           "candidates": len(records),
                                           "elapsed_seconds": time.perf_counter() - game_start})
                    load_stats = asdict(reader.stats)
                    # dataclasses.asdict reconstructs Counter from (key, value)
                    # pairs as keys; explicitly serialize this mapping instead.
                    load_stats["rejection_counts"] = dict(reader.stats.rejection_counts)
        finally:
            if owns_engine and engine is not None:
                engine.quit()
        elapsed = time.perf_counter() - started
        cache_after = asdict(cache.info())
        manifest = {
            "schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_path), "input_sha256": input_digest,
            "source_manifest_path": source.manifest_path, "source_manifest_sha256": manifest_digest,
            "output_file": output_path.name, "output_sha256": file_sha256(target),
            "selection_configuration": asdict(config),
            "analysis_profile": asdict(profile) if profile else None,
            "configured_depth": depth, "requested_games": max_games,
            "games_analyzed": len(successful), "games_failed": len(failures),
            "games_attempted": len(successful) + len(failures),
            "total_moves_analyzed": moves_analyzed, "candidate_count": sum(counts.values()),
            "selection_counts": {k: counts[k] for k in
                                 ("SIGNIFICANT_CP_ERROR", "LOW_ERROR_CONTROL", "MATE_RELATED", "AMBIGUOUS")},
            "mate_transition_counts": dict(statuses), "loader_stats": load_stats,
            "failures": failures, "games": successful, "stop_reason": stop_reason,
            "complete_requested_batch": len(successful) == max_games,
            "elapsed_seconds": elapsed,
            "average_seconds_per_successful_game": elapsed / len(successful) if successful else None,
            "cache": {"before": cache_before, "after": cache_after,
                      "delta": {k: cache_after[k] - cache_before[k] for k in ("hits", "misses", "evictions")},
                      "context_available": "cache_context" in cache_arguments},
            "limitations": ["Candidates are not verified tactical or positional labels.",
                            "Configured depth is not measured achieved depth; low-depth scores are provisional.",
                            "First eligible input games and capped controls introduce selection bias.",
                            "Reproducible JSONL requires identical underlying engine results and metadata."],
        }
        (stage / "candidate_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.rename(stage, output_dir)
    return manifest
