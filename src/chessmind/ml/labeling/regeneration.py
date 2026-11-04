"""Guarded C3 regeneration. Historical inputs are immutable; no review authority.

The caller must keep inputs and the destination parent quiescent during execution.
Freshness comes from an owned engine and disabled evaluation cache, not JSON flags.
"""

import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine

from ...engine.evaluation_cache import EvaluationCache
from ...engine import analyzer as engine_api
from ..data.candidate_pipeline import file_sha256
from ..data.dataset_readiness import (
    HISTORICAL_RULE_VERSION, _json, _records, _require, inspect_training_readiness, report_json,
)
from . import pipeline
from .adjudication import record_sha256
from .engine_verification import _result
from .schema import LABEL_SCHEMA_VERSION, RULE_VERSION, LabelingConfig


COHORT_SIZE = 24
ENGINE_SHA256 = "45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0"
CONFIG = LabelingConfig(max_candidates=24, verification_depth=12, threads=1, hash_mb=64)


def _destination(path, protected):
    path = Path(path)
    resolved = path.resolve()
    for source in protected.values():
        directory = source.resolve().parent
        _require(not (resolved == directory or resolved in directory.parents
                      or directory in resolved.parents), "unsafe output directory alias")
    if path.exists() or path.is_symlink():
        raise FileExistsError("output directory already exists")
    _require(path.parent.is_dir(), "output parent must already exist")
    return resolved


def _unchanged(paths, hashes):
    _require(all(file_sha256(path) == hashes[name] for name, path in paths.items()),
             "protected input changed during regeneration")


def _inspect(paths):
    report = inspect_training_readiness(**paths)
    _require(report["integrity_valid"], "; ".join(report["integrity_errors"]))
    _require(report["eligible_labels"]["total"] == 0 and not report["training_ready"],
             "evidence generation cannot authorize labels")
    return report


def _profile_valid(profile, binary_hash):
    _require(isinstance(profile, dict), "missing engine profile")
    _require(profile["engine_binary_sha256"] == binary_hash
             and dict(profile["engine_id"])["name"] == "Stockfish 19"
             and profile["configured_depth"] == 12, "engine identity/depth mismatch")
    options = dict(profile["engine_options"])
    for name, expected in (("threads", 1), ("hash", 64), ("multipv", 1),
                           ("uci_limitstrength", False), ("skill level", 20)):
        _require(type(options.get(name)) is type(expected) and options[name] == expected,
                 "engine configuration mismatch: " + name)
    _require(bool(profile["engine_resources"]) and bool(profile["chess_version"]),
             "missing engine resource provenance")


def _revision():
    root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def regenerate_evidence(*, candidate_path, feature_path, evidence_path,
                        source_dataset_path, source_manifest_path, output_dir,
                        engine_path, progress=None):
    """Generate only this 24-position cohort; publish after all checks succeed.

    All three adjacent manifests and the historical quality report are mandatory.
    No borrowed engine, cached verification, review store or approval input exists.
    """
    _require(RULE_VERSION == "conservative_tactics_v2", "unexpected current rule version")
    paths = {name: Path(value) for name, value in dict(
        candidate_path=candidate_path, feature_path=feature_path, evidence_path=evidence_path,
        source_dataset_path=source_dataset_path, source_manifest_path=source_manifest_path).items()}
    for kind, base in (("candidate", "candidate"), ("feature", "feature"), ("labeling", "evidence")):
        paths[kind + "_manifest_path"] = paths[base + "_path"].with_name(kind + "_manifest.json")
    protected = dict(paths, quality_report=paths["labeling_manifest_path"].with_name("label_quality_report.json"),
                     engine=Path(engine_path))
    destination = _destination(output_dir, protected)
    hashes = {name: file_sha256(path) for name, path in protected.items()}
    _require(hashes["engine"] == ENGINE_SHA256, "unexpected Stockfish executable hash")
    historical_report = _inspect(paths)
    historical_manifest = _json(paths["labeling_manifest_path"].read_bytes())
    _require(historical_manifest.get("quality_report_sha256") == hashes["quality_report"],
             "historical quality report binding missing")
    historical = _records(paths["evidence_path"])
    candidates = _records(paths["candidate_path"])
    features = _records(paths["feature_path"])
    cohort = list(historical)
    _require(len(cohort) == COHORT_SIZE, "expected exactly 24 historical candidates")
    _require(all(row["rule_version"] == HISTORICAL_RULE_VERSION
                 and row["verification"]["status"] == "verified" for row in historical.values()),
             "historical cohort must contain only verified v1 evidence")
    for identity, row in historical.items():
        for endpoint in ("before", "after"):
            verification = row["verification"]
            _result(chess.Board(candidates[identity]["fen_" + endpoint]), {
                "evaluation": verification[endpoint], "depth": verification[endpoint + "_depth"],
                "pv": [chess.Move.from_uci(move) for move in verification[endpoint + "_pv"]],
            }, CONFIG.verification_depth)
    selected = pipeline._select([(i, c, features[key]) for i, (key, c) in enumerate(candidates.items())], 24)
    _require([c["candidate_id"] for _, c, _ in selected] == cohort, "historical cohort selection drift")
    _unchanged(protected, hashes)
    # Same-volume staging; the existing generator publishes only inside this private directory.
    with tempfile.TemporaryDirectory(prefix=".c3-stage-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bundle"
        engine = chess.engine.SimpleEngine.popen_uci(str(protected["engine"]))
        try:
            # MultiPV is managed by python-chess analyse(), whose default is one.
            engine.configure({"Threads": 1, "Hash": 64})
            cache = EvaluationCache(0)
            context = engine_api.create_cache_context(engine, protected["engine"])
            _profile_valid(asdict(pipeline._profile(engine, 12, protected["engine"], context)), hashes["engine"])
            _unchanged(protected, hashes)
            manifest = pipeline.run_labeling_pipeline(
                paths["candidate_path"], paths["feature_path"], stage, config=CONFIG,
                engine=engine, engine_path=protected["engine"], cache=cache,
                cache_context=context, progress=progress)
        finally:
            engine.quit()
        _require(manifest["verification_failures"] == 0 and not manifest["input_diagnostics"],
                 "new verification failed; bundle not accepted")
        _profile_valid(manifest["engine_profile"], hashes["engine"])
        new_paths = dict(paths, evidence_path=stage / "mistake_label_proposals.jsonl",
                         labeling_manifest_path=stage / "labeling_manifest.json")
        report = _inspect(new_paths)
        rows = _records(new_paths["evidence_path"])
        _require(list(rows) == cohort, "generated cohort/order mismatch")
        for identity, row in rows.items():
            _require(row["rule_version"] == RULE_VERSION and row["schema_version"] == LABEL_SCHEMA_VERSION,
                     "generated version mismatch")
            verification = row["verification"]
            _require(verification["status"] == "verified", "incomplete verification")
            _profile_valid(verification["profile"], hashes["engine"])
            _require(record_sha256(verification["profile"]) == record_sha256(manifest["engine_profile"]),
                     "record engine profile mismatch")
            candidate = candidates[identity]
            for original, source in (("original_cp_loss", "centipawn_loss"),
                                     ("original_classification", "existing_classification"),
                                     ("original_mate_quality", "mate_quality")):
                _require(verification[original] == candidate.get(source), "original verification value mismatch")
            for endpoint in ("before", "after"):
                _result(chess.Board(candidate["fen_" + endpoint]), {
                    "evaluation": verification[endpoint], "depth": verification[endpoint + "_depth"],
                    "pv": [chess.Move.from_uci(move) for move in verification[endpoint + "_pv"]],
                }, CONFIG.verification_depth)
        audit = dict(schema_version="c3_regeneration_audit_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                     code_revision=_revision(), rule_version=RULE_VERSION, configuration=asdict(CONFIG),
                     implementation_sha256={str(path.relative_to(Path(__file__).resolve().parents[4])): file_sha256(path)
                                            for path in Path(__file__).resolve().parents[1].rglob("*.py")},
                     engine_sha256=hashes["engine"], engine_profile=manifest["engine_profile"],
                     protected_sha256=hashes, historical_artifact_sha256=historical_report["artifact_sha256"],
                     selected_candidate_ids=cohort, cohort_sha256=record_sha256(cohort),
                     generated_artifact_sha256=report["artifact_sha256"],
                     acceptance=dict(verified=24, integrity_valid=True, eligible_labels=0,
                                     fresh_engine=True, evaluation_cache_enabled=False))
        (stage / "regeneration_audit.json").write_text(report_json(audit), encoding="utf-8")
        _unchanged(protected, hashes)
        _require(_destination(output_dir, protected) == destination, "output path changed")
        os.rename(stage, destination)
    return audit
