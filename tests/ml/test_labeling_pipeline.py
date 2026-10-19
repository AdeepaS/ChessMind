"""Bounded joins, engine ownership, deterministic output and safe publication."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import chess.engine
import pytest

from src.chessmind.engine import analyzer as engine_api
from src.chessmind.ml.data.candidate_schema import AnalysisProfile
from src.chessmind.ml.labeling import pipeline
from src.chessmind.ml.labeling.schema import LabelingConfig, VerificationResult
from src.chessmind.ml.labeling import engine_verification
from src.chessmind.engine.analyzer import serialize_evaluation


@pytest.fixture
def labeling_inputs(tmp_path, labeling_case, monkeypatch):
    candidate, feature, verified = labeling_case()
    candidates, features = tmp_path / "candidates.jsonl", tmp_path / "features.jsonl"
    candidates.write_text(json.dumps(candidate) + "\n")
    features.write_text(json.dumps(feature) + "\n")
    binary = tmp_path / "engine.exe"
    binary.write_bytes(b"fixture")
    engine = Mock()
    launch = Mock(return_value=engine)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", launch)
    monkeypatch.setattr(engine_api, "prepare_cache_arguments", Mock(return_value={}))
    monkeypatch.setattr(pipeline, "_profile", Mock(return_value=AnalysisProfile(configured_depth=12)))
    monkeypatch.setattr(pipeline, "verify_candidate", Mock(return_value=verified))
    return candidates, features, binary, engine, launch


def test_counts_hashes_and_deterministic_output(labeling_inputs, tmp_path):
    c, f, binary, engine, launch = labeling_inputs
    first = pipeline.run_labeling_pipeline(c, f, tmp_path / "one", engine_path=binary)
    assert first["processed"] == first["proposal_count"] == 1
    assert first["abstentions"] == first["verification_failures"] == 0
    assert first["concept_counts"] == {"hanging_piece": 1}
    assert pipeline.file_sha256(tmp_path / "one/mistake_label_proposals.jsonl") == first["output_sha256"]
    assert pipeline.file_sha256(tmp_path / "one/label_quality_report.json") == first["quality_report_sha256"]
    launch.assert_called_once_with(str(binary))
    engine.configure.assert_called_once_with({"Threads": 1, "Hash": 64})
    engine.quit.assert_called_once()
    pipeline.run_labeling_pipeline(c, f, tmp_path / "two", engine_path=binary)
    assert (tmp_path / "one/mistake_label_proposals.jsonl").read_bytes() == (tmp_path / "two/mistake_label_proposals.jsonl").read_bytes()


@pytest.mark.parametrize("black", [False, True])
def test_cp_mate_review_preserves_verification_and_provenance(tmp_path, labeling_case, monkeypatch, black):
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    c, f, _ = labeling_case(fen=board.mirror().fen() if black else board.fen(),
        move="g3g4" if black else "g6g5")
    # Original CP classification differs from the new mate-to-CP verification.
    before = chess.engine.Mate(-1 if black else 1)
    after = chess.engine.Cp(-500 if black else 500)
    before_pv = ("g3g2",) if black else ("g6g7",)
    after_pv = ("h1h2",) if black else ("h8h7",)
    responses = [{"evaluation": serialize_evaluation(score), "depth": 12,
                  "pv": [chess.Move.from_uci(u) for u in pv]}
                 for score, pv in [(before, before_pv), (after, after_pv)]]
    monkeypatch.setattr(engine_verification, "analyze_board", Mock(side_effect=responses))
    monkeypatch.setattr(pipeline, "_profile", Mock(return_value=AnalysisProfile(configured_depth=12)))
    # Ensure the historical classification differs for both mover colors.
    c["existing_classification"] = "Excellent"
    c["centipawn_loss"] = 0
    from src.chessmind.ml.features.extractor import extract_candidate_features
    f = asdict(extract_candidate_features(c))
    candidates, features = tmp_path / "candidates.jsonl", tmp_path / "features.jsonl"
    candidates.write_text(json.dumps(c))
    features.write_text(json.dumps(f))
    original_bytes = candidates.read_bytes(), features.read_bytes()
    out = tmp_path / "review"
    manifest = pipeline.run_labeling_pipeline(candidates, features, out, engine=Mock())
    report = json.loads((out / "label_quality_report.json").read_text())
    row = json.loads((out / "mistake_label_proposals.jsonl").read_text())
    assert report["status_counts"] == {"requires_review": 1}
    assert report["proposal_count"] == 0 and report["abstentions"] == 1
    assert report["verification_failures"] == 0 and report["classification_changes"] == 1
    assert row["candidate_id"] == c["candidate_id"] and row["review_status"] == "unreviewed"
    assert row["proposals"] == []
    assert row["reasons"] == ["post_move_forced_mate_unconfirmed"]
    assert row["evidence"][1]["continuation"] == list(before_pv)
    verified = row["verification"]
    assert verified["before"]["mate"] == (-1 if black else 1)
    assert verified["after"]["cp"] == (-500 if black else 500)
    assert verified["after_pv"] == list(after_pv)
    assert verified["original_classification"] == "Excellent"
    assert verified["classification"] == "Blunder" and verified["classification_changed"]
    assert verified["mate_quality"] == "lost_forced_mate"
    assert report["comparisons"][0]["original_classification"] == "Excellent"
    assert report["comparisons"][0]["classification"] == "Blunder"
    assert row["candidate_input_sha256"] == manifest["candidate_sha256"] == pipeline.file_sha256(candidates)
    assert row["feature_input_sha256"] == manifest["feature_sha256"] == pipeline.file_sha256(features)
    assert row["rule_version"] == manifest["rule_version"] == "conservative_tactics_v2"
    assert (candidates.read_bytes(), features.read_bytes()) == original_bytes


@pytest.mark.parametrize("problem", ["missing", "mismatch", "duplicate_candidate", "duplicate_feature", "bad_json"])
def test_strict_and_permissive_inputs(labeling_inputs, tmp_path, problem):
    c, f, binary, _, launch = labeling_inputs
    if problem == "missing":
        f.write_text("")
    elif problem == "mismatch":
        row = json.loads(f.read_text())
        row["schema_version"] = "wrong"
        f.write_text(json.dumps(row))
    elif problem == "duplicate_candidate":
        c.write_text(c.read_text() * 2)
    elif problem == "duplicate_feature":
        f.write_text(f.read_text() * 2)
    else:
        c.write_text("invalid json\n")
    with pytest.raises(ValueError):
        pipeline.run_labeling_pipeline(c, f, tmp_path / "strict", engine_path=binary)
    launch.assert_not_called()
    result = pipeline.run_labeling_pipeline(c, f, tmp_path / "permissive", engine_path=binary,
                                            config=LabelingConfig(strict=False))
    assert result["input_diagnostics"] and result["processed"] == 0
    assert not (tmp_path / "strict").exists()


def test_feature_tampering_not_just_id_check(labeling_inputs, tmp_path):
    c, f, binary, _, _ = labeling_inputs
    feature = json.loads(f.read_text())
    feature["features"]["before"]["white"]["material"] += 10
    f.write_text(json.dumps(feature))
    with pytest.raises(ValueError, match="contents"):
        pipeline.run_labeling_pipeline(c, f, tmp_path / "out", engine_path=binary)


def test_skipped_and_failed_verification_never_labels(labeling_inputs, tmp_path, monkeypatch):
    c, f, binary, engine, launch = labeling_inputs
    result = pipeline.run_labeling_pipeline(c, f, tmp_path / "skipped", config=LabelingConfig(verify=False))
    assert result["abstentions"] == 1 and result["proposal_count"] == 0
    launch.assert_not_called()
    monkeypatch.setattr(pipeline, "verify_candidate", Mock(return_value=VerificationResult("failed", reason="fixture")))
    result = pipeline.run_labeling_pipeline(c, f, tmp_path / "failed", engine_path=binary)
    assert result["verification_failures"] == result["abstentions"] == 1
    engine.quit.assert_called_once()


def test_empty_zero_and_borrowed_ownership(labeling_inputs, tmp_path):
    c, f, _, engine, launch = labeling_inputs
    pipeline.run_labeling_pipeline(c, f, tmp_path / "zero", config=LabelingConfig(max_candidates=0))
    launch.assert_not_called()
    pipeline.run_labeling_pipeline(c, f, tmp_path / "borrowed", engine=engine)
    engine.quit.assert_not_called()
    engine.configure.assert_not_called()
    c.write_text("")
    f.write_text("")
    assert pipeline.run_labeling_pipeline(c, f, tmp_path / "empty")["processed"] == 0
    launch.assert_not_called()


def test_borrowed_cache_context_forwarded(labeling_inputs, tmp_path):
    from src.chessmind.engine.evaluation_cache import EvaluationCache
    c, f, _, engine, launch = labeling_inputs
    cache = EvaluationCache()
    context = engine_api.EngineCacheContext(engine, Mock())
    pipeline.run_labeling_pipeline(c, f, tmp_path / "borrowed_cache", engine=engine,
                                   cache=cache, cache_context=context)
    assert pipeline.verify_candidate.call_args.kwargs["cache"] is cache
    assert pipeline.verify_candidate.call_args.kwargs["cache_context"] is context
    engine.quit.assert_not_called()
    engine.configure.assert_not_called()
    launch.assert_not_called()


def test_feature_boolean_type_mismatch(labeling_inputs, tmp_path):
    c, f, _, _, _ = labeling_inputs
    row = json.loads(f.read_text())
    row["features"]["before"]["white"]["bishop_pair"] = 0
    f.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="contents"):
        pipeline.run_labeling_pipeline(c, f, tmp_path / "out")


def test_startup_failure_and_interrupt(labeling_inputs, tmp_path, monkeypatch):
    c, f, binary, engine, launch = labeling_inputs
    launch.side_effect = OSError("startup")
    result = pipeline.run_labeling_pipeline(c, f, tmp_path / "startup", engine_path=binary)
    assert result["verification_failures"] == 1 and result["proposal_count"] == 0
    launch.side_effect = None
    monkeypatch.setattr(pipeline, "verify_candidate", Mock(side_effect=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_labeling_pipeline(c, f, tmp_path / "interrupted", engine_path=binary)
    assert not (tmp_path / "interrupted").exists()
    engine.quit.assert_called_once()


def test_input_manifest_hash_checked(labeling_inputs, tmp_path):
    c, f, _, _, launch = labeling_inputs
    (tmp_path / "candidate_manifest.json").write_text('{"output_sha256":"wrong"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.run_labeling_pipeline(c, f, tmp_path / "out")
    launch.assert_not_called()


def test_selection_mixture_and_order():
    joined = [(i, {"selection_type": kind}, {}) for i, kind in enumerate(
        ["SIGNIFICANT_CP_ERROR"] * 20 + ["LOW_ERROR_CONTROL"] * 10 + ["MATE_RELATED"] * 8 + ["AMBIGUOUS"] * 3)]
    chosen = pipeline._select(joined, 24)
    assert len(chosen) == 24 and [x[0] for x in chosen] == sorted(x[0] for x in chosen)
    assert {x[1]["selection_type"] for x in chosen} == set(pipeline.SELECTION_TYPES)


@pytest.mark.parametrize("kwargs", [{"max_candidates": -1}, {"verification_depth": 0}, {"threads": 5}, {"hash_mb": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LabelingConfig(**kwargs)


@pytest.mark.skipif(not Path(engine_api.STOCKFISH_PATH).is_file(), reason="Stockfish unavailable")
def test_real_verification_smoke(tmp_path, labeling_case):
    c, f, _ = labeling_case()
    candidates, features = tmp_path / "candidates.jsonl", tmp_path / "features.jsonl"
    candidates.write_text(json.dumps(c))
    features.write_text(json.dumps(f))
    result = pipeline.run_labeling_pipeline(candidates, features, tmp_path / "out",
                                            config=LabelingConfig(max_candidates=1, verification_depth=2))
    assert result["processed"] == 1 and result["verification_failures"] == 0
    assert result["engine_profile"]["engine_id"]
    assert result["cache_after"]["hits"] + result["cache_after"]["misses"] in (0, 2)
