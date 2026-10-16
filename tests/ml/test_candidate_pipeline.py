"""Offline batch ownership, deterministic output, failure and provenance tests."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import chess
import chess.engine
import pytest

from src.chessmind.engine import analyzer as engine_api
from src.chessmind.engine.evaluation_cache import EvaluationCache
from src.chessmind.engine.evaluation_cache_keys import CacheIdentityUnavailable
from src.chessmind.ml.data import candidate_pipeline as pipeline


@pytest.fixture
def batch(tmp_path, monkeypatch, candidate_fixture):
    game, rows = candidate_fixture(count=2)
    other, _ = candidate_fixture(count=2, game_id="ijklmnop")
    source = tmp_path / "input.pgn"
    source.write_text(game.original_pgn + "\n\n" + other.original_pgn, encoding="utf-8")
    binary = tmp_path / "stockfish.exe"
    binary.write_bytes(b"synthetic engine")
    engine = Mock()
    engine.id = {"name": "offline fixture"}
    launch = Mock(return_value=engine)
    analyze = Mock(return_value=rows)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", launch)
    monkeypatch.setattr(pipeline, "analyze_game", analyze)
    monkeypatch.setattr(engine_api, "prepare_cache_arguments", Mock(return_value={}))
    monkeypatch.setattr(pipeline, "snapshot_engine_metadata", Mock(side_effect=CacheIdentityUnavailable("fixture")))
    return source, binary, engine, launch, analyze


def test_owned_one_engine_manifest_counts_hash_and_deterministic_jsonl(batch, tmp_path):
    source, binary, engine, launch, analyze = batch
    out = tmp_path / "first/candidate_moves.jsonl"
    manifest = pipeline.run_candidate_pipeline(source, out, engine_path=binary, max_games=2)
    launch.assert_called_once_with(str(binary))
    engine.configure.assert_called_once_with({"Threads": 1, "Hash": 64})
    engine.quit.assert_called_once()
    assert analyze.call_count == 2
    assert all(c.kwargs["engine"] is engine for c in analyze.call_args_list)
    assert manifest["games_analyzed"] == 2 and manifest["total_moves_analyzed"] == 4
    assert manifest["candidate_count"] == 4 and manifest["games_failed"] == 0
    assert manifest["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len({r["candidate_id"] for r in records}) == 4
    assert records[0]["analysis_profile"]["engine_options"] is None
    assert records[0]["analysis_profile"]["achieved_depth"] is None
    assert records[0]["analysis_profile"]["engine_binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    other = tmp_path / "second/candidate_moves.jsonl"
    pipeline.run_candidate_pipeline(source, other, engine_path=binary, max_games=2)
    assert out.read_bytes() == other.read_bytes()
    assert json.loads(out.with_name("candidate_manifest.json").read_text())["candidate_count"] == 4


def test_borrowed_engine_and_cache_context(batch, tmp_path):
    source, _, engine, launch, analyze = batch
    cache = EvaluationCache()
    identity = Mock(executable_sha256="fixture digest", resources=())
    context = engine_api.EngineCacheContext(engine, identity)
    pipeline.run_candidate_pipeline(source, tmp_path / "borrowed/moves.jsonl", max_games=1,
                                    engine=engine, cache=cache, cache_context=context)
    launch.assert_not_called()
    engine.quit.assert_not_called()
    engine.configure.assert_not_called()
    assert analyze.call_args.kwargs["cache"] is cache
    assert analyze.call_args.kwargs["cache_context"] is context
    assert cache.info().hits == 0


def test_borrowed_without_context_caching_disabled(batch, tmp_path):
    source, _, engine, _, analyze = batch
    manifest = pipeline.run_candidate_pipeline(source, tmp_path / "borrowed/moves.jsonl", engine=engine)
    assert "cache" not in analyze.call_args.kwargs
    assert manifest["cache"]["context_available"] is False
    assert manifest["analysis_profile"]["engine_binary_sha256"] is None


@pytest.mark.parametrize("failure", [ValueError("bad game"), RuntimeError("search failed"), chess.engine.EngineTerminatedError("died")])
def test_failures_recorded_without_partial_game(batch, tmp_path, failure):
    source, binary, engine, _, analyze = batch
    rows = analyze.return_value
    analyze.side_effect = [failure, rows]
    out = tmp_path / "failed/moves.jsonl"
    manifest = pipeline.run_candidate_pipeline(source, out, engine_path=binary, max_games=2)
    assert manifest["games_failed"] == 1
    expected = 0 if isinstance(failure, chess.engine.EngineTerminatedError) else 1
    assert manifest["games_analyzed"] == expected
    assert len(out.read_text().splitlines()) == expected * 2
    assert not manifest["complete_requested_batch"]
    engine.quit.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["analysis", "startup", "shutdown", "serialization"])
def test_fatal_cleanup(batch, tmp_path, monkeypatch, failure_stage):
    source, binary, engine, launch, analyze = batch
    if failure_stage == "analysis":
        analyze.side_effect = KeyboardInterrupt()
        error = KeyboardInterrupt
    elif failure_stage == "startup":
        launch.side_effect = OSError("start failed")
        error = OSError
    elif failure_stage == "shutdown":
        engine.quit.side_effect = OSError("quit failed")
        error = OSError
    else:
        monkeypatch.setattr(pipeline, "candidate_json", Mock(side_effect=OSError("disk failure")))
        error = OSError
    with pytest.raises(error):
        pipeline.run_candidate_pipeline(source, tmp_path / "aborted/moves.jsonl", engine_path=binary)
    assert not (tmp_path / "aborted").exists()
    assert not list(tmp_path.glob(".candidates-*"))
    if failure_stage != "startup":
        engine.quit.assert_called_once()


def test_source_manifest_identity_and_tampering(batch, tmp_path):
    source, binary, _, launch, _ = batch
    manifest_path = source.with_name("source_manifest.json")
    manifest_path.write_text(json.dumps({"sha256": pipeline.file_sha256(source), "source_url": "official archive",
        "source_references": [{"game_id": "abcdefgh", "archive_game_index": 90}]}))
    out = tmp_path / "verified/moves.jsonl"
    result = pipeline.run_candidate_pipeline(source, out, engine_path=binary, max_games=1)
    first = json.loads(out.read_text().splitlines()[0])
    assert first["source_evidence"]["archive_game_index"] == 90
    assert result["source_manifest_sha256"] == pipeline.file_sha256(manifest_path)
    launch.reset_mock()
    source.write_text("changed")
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline.run_candidate_pipeline(source, tmp_path / "tampered/moves.jsonl", engine_path=binary)
    launch.assert_not_called()


def test_duplicates_rejections_zero_and_empty(batch, tmp_path):
    source, binary, _, launch, analyze = batch
    original = source.read_text()
    source.write_text(original + "\n\n" + original)
    result = pipeline.run_candidate_pipeline(source, tmp_path / "duplicates/moves.jsonl", engine_path=binary)
    assert result["loader_stats"]["rejection_counts"] == {"duplicate_id": 2}
    assert result["games_analyzed"] == 2
    launch.reset_mock()
    pipeline.run_candidate_pipeline(source, tmp_path / "zero/moves.jsonl", max_games=0)
    source.write_text("")
    manifest = pipeline.run_candidate_pipeline(source, tmp_path / "empty/moves.jsonl")
    assert manifest["candidate_count"] == 0 and manifest["analysis_profile"] is None
    launch.assert_not_called()


@pytest.mark.parametrize("options", [{"max_games": -1}, {"max_games": True}, {"depth": 0}, {"depth": 1.5}])
def test_invalid_limits(batch, tmp_path, options):
    with pytest.raises(ValueError):
        pipeline.run_candidate_pipeline(batch[0], tmp_path / "bad/moves.jsonl", **options)


def test_existing_output_preserved(batch, tmp_path):
    out = tmp_path / "existing/moves.jsonl"
    out.parent.mkdir()
    out.write_text("keep")
    with pytest.raises(FileExistsError):
        pipeline.run_candidate_pipeline(batch[0], out)
    assert out.read_text() == "keep"


@pytest.mark.skipif(not Path(engine_api.STOCKFISH_PATH).is_file(), reason="Stockfish unavailable")
def test_real_candidate_engine_smoke(tmp_path, candidate_fixture, monkeypatch):
    game, _ = candidate_fixture(count=2)
    source = tmp_path / "smoke.pgn"
    source.write_text(game.original_pgn)
    created = []
    original = chess.engine.SimpleEngine.popen_uci
    def observed(*args, **kwargs):
        engine = original(*args, **kwargs)
        engine.analyse = Mock(wraps=engine.analyse)
        engine.quit = Mock(wraps=engine.quit)
        created.append(engine)
        return engine
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", observed)
    out = tmp_path / "smoke/moves.jsonl"
    manifest = pipeline.run_candidate_pipeline(source, out, depth=2, max_games=1)
    assert manifest["games_analyzed"] == 1 and manifest["games_failed"] == 0
    assert manifest["total_moves_analyzed"] == 2
    assert created[0].analyse.call_count + manifest["cache"]["delta"]["hits"] == 3
    created[0].quit.assert_called_once()
    assert created[0].returncode.result(timeout=10) == 0
