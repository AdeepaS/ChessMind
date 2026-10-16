"""Candidate adapter and safe streaming publication; no Stockfish needed."""

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from src.chessmind.ml.data.candidate_extraction import extract_candidates
from src.chessmind.ml.features.extractor import extract_candidate_features
from src.chessmind.ml.features.pipeline import generate_features
from src.chessmind.ml.features.schema import FEATURE_SCHEMA_VERSION, feature_json


@pytest.fixture
def candidates(candidate_fixture):
    game, rows = candidate_fixture(count=2)
    return [asdict(r) for r in extract_candidates(rows, game)]


def write_source(tmp_path, rows):
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_linkage_schema_and_feature_count(candidates):
    original = copy.deepcopy(candidates[0])
    f = extract_candidate_features(original)
    assert f.candidate_id == original["candidate_id"]
    assert f.source_game_id == original["source_game_id"]
    assert f.source_evidence.dataset_sha256 is None
    assert f.schema_version == f.features.schema_version == FEATURE_SCHEMA_VERSION
    assert len(f.features.deltas) == 49
    assert len(asdict(f.features.context)) == 10
    assert feature_json(f) == feature_json(extract_candidate_features(original))
    assert original == candidates[0]


def test_existing_typed_candidate_supported(candidate_fixture):
    game, analysis = candidate_fixture(count=1)
    record, = extract_candidates(analysis, game)
    assert extract_candidate_features(record) == extract_candidate_features(asdict(record))


@pytest.mark.parametrize("field,value,reason", [
    ("fen_after", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 1", "after_fen_mismatch"),
    ("fen_before", "bad", "invalid_fen_or_uci"),
    ("played_move_uci", "e2e5", "illegal_move"),
    ("played_move_uci", "bad", "invalid_fen_or_uci"),
    ("candidate_id", "wrong", "candidate_id_mismatch"),
    ("player_color", "black", "position_identity_mismatch"),
    ("ply_index", True, "invalid_ply"),
    ("schema_version", "future", "unsupported_candidate_schema"),
    ("source_evidence", [], "invalid_source_evidence"),
    ("source_evidence", {"archive_game_index": "1"}, "invalid_source_evidence"),
])
def test_candidate_rejections(candidates, field, value, reason):
    candidates[0][field] = value
    with pytest.raises(ValueError, match=reason):
        extract_candidate_features(candidates[0])


def test_missing_fields(candidates):
    del candidates[0]["fen_after"]
    with pytest.raises(ValueError, match="missing_fields"):
        extract_candidate_features(candidates[0])


def test_manifest_counts_hashes_determinism_and_source_unchanged(tmp_path, candidates):
    source = write_source(tmp_path, candidates)
    original = source.read_bytes()
    source_manifest = tmp_path / "candidate_manifest.json"
    source_manifest.write_text(json.dumps({"output_sha256": hashlib.sha256(original).hexdigest(),
                                          "candidate_count": 2, "schema_version": "candidate_moves_v1"}))
    out = tmp_path / "one/features.jsonl"
    manifest = generate_features(source, out)
    assert (manifest["total_processed"], manifest["accepted"], manifest["rejected"]) == (2, 2, 0)
    assert manifest["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["source_provenance"]["sha256"] == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    assert source.read_bytes() == original
    other = tmp_path / "two/features.jsonl"
    generate_features(source, other)
    assert out.read_bytes() == other.read_bytes()
    assert len({json.loads(line)["candidate_id"] for line in out.read_text().splitlines()}) == 2


def test_strict_permissive_and_duplicate_ids(tmp_path, candidates):
    source = write_source(tmp_path, [candidates[0], candidates[0], {}, candidates[1]])
    source.write_text(source.read_text() + "invalid JSON\n")
    out = tmp_path / "strict/features.jsonl"
    with pytest.raises(ValueError, match="duplicate_candidate_id"):
        generate_features(source, out)
    assert not out.exists() and not out.with_name("feature_manifest.json").exists()
    out = tmp_path / "permissive/features.jsonl"
    manifest = generate_features(source, out, strict=False)
    assert manifest["accepted"] == 2 and manifest["rejected"] == 3 and manifest["total_processed"] == 5
    assert manifest["rejection_counts"] == {"duplicate_candidate_id": 1, "missing_fields": 1, "invalid_json": 1}


@pytest.mark.parametrize("raw,reason", [("[]\n", "invalid_candidate_object"),
    ('{"x":1,"x":2}\n', "invalid_json"), ('{"x":NaN}\n', "invalid_json")])
def test_malformed_json_objects(tmp_path, raw, reason):
    source = tmp_path / "input.jsonl"
    source.write_text(raw)
    result = generate_features(source, tmp_path / "out/features.jsonl", strict=False)
    assert result["rejection_counts"] == {reason: 1}


@pytest.mark.parametrize("raw", ["", "\n  \n"])
def test_empty(tmp_path, raw):
    source = tmp_path / "input.jsonl"
    source.write_text(raw)
    out = tmp_path / "out/features.jsonl"
    result = generate_features(source, out)
    assert result["total_processed"] == result["accepted"] == 0
    assert out.read_bytes() == b""


def test_line_budget_and_no_overwrite(tmp_path, candidates):
    source = write_source(tmp_path, candidates)
    out = tmp_path / "out/features.jsonl"
    result = generate_features(source, out, max_line_chars=30, strict=False)
    assert result["rejection_counts"] == {"line_too_large": 2}
    with pytest.raises(FileExistsError):
        generate_features(source, out)
    with pytest.raises(ValueError):
        generate_features(source, source)


def test_hash_mismatch_aborts(tmp_path, candidates):
    source = write_source(tmp_path, candidates)
    (tmp_path / "candidate_manifest.json").write_text('{"output_sha256":"wrong"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_features(source, tmp_path / "out/features.jsonl")


def test_publication_failure_rolls_back(tmp_path, candidates, monkeypatch):
    from src.chessmind.ml.features import pipeline
    source = write_source(tmp_path, candidates)
    real_link = pipeline.os.link
    def fail_manifest(src, dst):
        if dst.name == "feature_manifest.json":
            raise OSError("publication failed")
        real_link(src, dst)
    monkeypatch.setattr(pipeline.os, "link", fail_manifest)
    out = tmp_path / "out/features.jsonl"
    with pytest.raises(OSError):
        generate_features(source, out)
    assert not out.exists()
    assert list(out.parent.iterdir()) == []
