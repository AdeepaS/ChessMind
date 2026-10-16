"""Bounded, sequential labeling pilot with validated candidate/feature joins."""

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import chess.engine

from ...engine import analyzer as engine_api
from ...engine.evaluation_cache import EvaluationCache
from ..data.candidate_extraction import _score
from ..data.candidate_pipeline import _profile, file_sha256
from ..data.candidate_schema import ExtractionConfig
from ..features.extractor import extract_candidate_features
from ..features.pipeline import _invalid_constant, _unique_object
from ..features.schema import feature_json
from .engine_verification import verify_candidate
from .rules import propose_labels
from .schema import LABEL_SCHEMA_VERSION, RULE_VERSION, LabelingConfig, LabelRecord, VerificationResult, label_json


SELECTION_TYPES = ("SIGNIFICANT_CP_ERROR", "LOW_ERROR_CONTROL", "MATE_RELATED", "AMBIGUOUS")


def _read_records(path, strict, diagnostics):
    """Small-artifact bound: 32 MiB, 10,000 records, 2 MiB per physical line."""
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("input exceeds 32 MiB pilot artifact budget")
    records, seen = {}, set()
    with path.open(encoding="utf-8") as stream:
        for index in range(10001):
            line = stream.readline(2_000_001)
            if not line:
                break
            if index == 10000 or len(line) > 2_000_000:
                raise ValueError("input line/record resource budget exceeded")
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str) or not row["candidate_id"]:
                    raise ValueError("missing candidate identity")
                identity = row["candidate_id"]
                if identity in seen:
                    records.pop(identity, None)  # Neither ambiguous copy is eligible.
                    raise ValueError("duplicate candidate ID")
                seen.add(identity)
                records[identity] = (index, row)
            except (ValueError, RecursionError) as exc:
                if strict:
                    raise ValueError(f"{path}, line {index + 1}: {exc}") from exc
                diagnostics.append({"path": str(path), "line": index + 1, "reason": str(exc)})
    return records


def _manifest_identity(path, expected_hash, *, candidate_hash=None):
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("output_sha256") != expected_hash:
        raise ValueError(f"manifest hash mismatch: {path}")
    if candidate_hash is not None and manifest.get("input_sha256") != candidate_hash:
        raise ValueError("features were generated from a different candidate dataset")
    return {"path": str(path), "sha256": file_sha256(path), "schema_version": manifest.get("schema_version")}


def _validate_candidate(candidate):
    expected = json.loads(feature_json(extract_candidate_features(candidate)))
    if candidate.get("selection_type") not in SELECTION_TYPES:
        raise ValueError("unknown original selection type")
    ExtractionConfig(**candidate["extraction_configuration"])
    _score(candidate["score_before"])
    _score(candidate["score_after"])
    loss = candidate["centipawn_loss"]
    if loss is not None and (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0):
        raise ValueError("invalid original CP loss")
    if candidate["existing_classification"] not in ("Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"):
        raise ValueError("invalid original classification")
    return expected


def _select(joined, limit):
    groups = [[item for item in joined if item[1]["selection_type"] == kind] for kind in SELECTION_TYPES]
    chosen = []
    for offset in range(limit):
        for group in groups:
            if offset < len(group) and len(chosen) < limit:
                chosen.append(group[offset])
    return sorted(chosen, key=lambda item: item[0])


def run_labeling_pipeline(candidate_path, feature_path, output_dir, *, config=None,
                          engine=None, cache=None, cache_context=None, engine_path=None, progress=None):
    config = config or LabelingConfig()
    candidate_path, feature_path, output_dir = map(Path, (candidate_path, feature_path, output_dir))
    if output_dir.exists():
        raise FileExistsError("choose a new output directory for atomic publication")
    started = time.perf_counter()
    candidate_hash, feature_hash = file_sha256(candidate_path), file_sha256(feature_path)
    provenance = {
        "candidate_manifest": _manifest_identity(candidate_path.with_name("candidate_manifest.json"), candidate_hash),
        "feature_manifest": _manifest_identity(feature_path.with_name("feature_manifest.json"), feature_hash,
                                                candidate_hash=candidate_hash),
    }
    diagnostics = []
    candidates = _read_records(candidate_path, config.strict, diagnostics)
    features = _read_records(feature_path, config.strict, diagnostics)
    joined = []
    for identity, (index, candidate) in candidates.items():
        try:
            if identity not in features:
                raise ValueError("missing feature record")
            feature = features[identity][1]
            expected = _validate_candidate(candidate)
            if json.dumps(expected, sort_keys=True) != json.dumps(feature, sort_keys=True):
                raise ValueError("feature contents/schema/source linkage mismatch")
            joined.append((index, candidate, feature))
        except (ValueError, TypeError, KeyError) as exc:
            if config.strict:
                raise ValueError(f"candidate {identity}: {exc}") from exc
            diagnostics.append({"candidate_id": identity, "reason": str(exc)})
    extras = sorted(set(features) - set(candidates))
    if extras and config.strict:
        raise ValueError("feature IDs have no matching candidates")
    diagnostics.extend({"candidate_id": identity, "reason": "unmatched feature ID"} for identity in extras)
    selected = _select(joined, config.max_candidates)
    owns_engine = engine is None
    if cache_context is not None and (engine is None or cache_context.engine is not engine):
        raise ValueError("cache context must belong to the borrowed engine")
    binary = Path(engine_path or engine_api.STOCKFISH_PATH) if owns_engine else (Path(engine_path) if engine_path else None)
    cache = cache if cache is not None else EvaluationCache(4096)
    cache_before = asdict(cache.info())
    cache_args = {"cache": cache, "cache_context": cache_context} if cache_context is not None else {}
    profile, setup_failure = None, None
    records = []
    try:
        if selected and config.verify:
            try:
                if owns_engine:
                    if not binary.is_file():
                        raise FileNotFoundError(f"Stockfish not found: {binary}")
                    engine = chess.engine.SimpleEngine.popen_uci(str(binary))
                    engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
                    cache_args = engine_api.prepare_cache_arguments(engine, cache, binary)
                profile = _profile(engine, config.verification_depth, binary, cache_args.get("cache_context"))
            except (OSError, ValueError, RuntimeError, chess.engine.EngineError) as exc:
                setup_failure = f"{type(exc).__name__}: {exc}"
        for number, (_, candidate, feature) in enumerate(selected, 1):
            if progress:
                progress(f"Verifying {number}/{len(selected)}: {candidate['source_game_id']} ply {candidate['ply_index']}")
            if not config.verify or setup_failure:
                verified = VerificationResult("failed" if setup_failure else "skipped",
                    reason=setup_failure or "verification_disabled",
                    original_cp_loss=candidate["centipawn_loss"],
                    original_classification=candidate["existing_classification"],
                    original_mate_quality=candidate.get("mate_quality"), profile=profile)
            else:
                verified = verify_candidate(candidate, engine, depth=config.verification_depth,
                                            profile=profile, **cache_args)
            proposals, evidence, status, reasons = propose_labels(candidate, feature, verified)
            feature_digest = hashlib.sha256(json.dumps(feature, sort_keys=True, separators=(",", ":"),
                                                        ensure_ascii=False).encode()).hexdigest()
            records.append(LabelRecord(candidate["candidate_id"], candidate["source_game_id"],
                candidate["player_identifier"], candidate["ply_index"], candidate["schema_version"],
                feature["schema_version"], candidate["selection_type"], verified, evidence, proposals,
                status, reasons, candidate_hash, feature_hash, feature_digest))
    finally:
        if owns_engine and engine is not None:
            engine.quit()
    if file_sha256(candidate_path) != candidate_hash or file_sha256(feature_path) != feature_hash:
        raise ValueError("input changed during labeling")
    counts = Counter(p.concept.value for record in records for p in record.proposals)
    abstentions = [record for record in records if not record.proposals]
    report = {
        "processed": len(records), "records_with_proposals": len(records) - len(abstentions),
        "proposal_count": sum(counts.values()), "abstentions": len(abstentions),
        "concept_counts": dict(counts), "status_counts": dict(Counter(r.proposal_status for r in records)),
        "abstention_reasons": dict(Counter(reason for r in abstentions for reason in r.reasons)),
        "verification_failures": sum(r.verification.status == "failed" for r in records),
        "verification_skipped": sum(r.verification.status == "skipped" for r in records),
        "selected_types": dict(Counter(r.original_selection_type for r in records)),
        "classification_changes": sum(r.verification.classification_changed is True for r in records),
        "significant_status_changes": sum(r.verification.significant_status_changed is True for r in records),
        "material_changes": sum(r.verification.materially_changed is True for r in records),
        "comparisons": [{"candidate_id": r.candidate_id, "game_id": r.source_game_id, "ply": r.ply_index,
            **{k: getattr(r.verification, k) for k in (
                "status", "original_cp_loss", "centipawn_loss", "original_classification", "classification",
                "original_mate_quality", "mate_quality", "significant_error_supported", "cp_loss_change",
                "classification_changed", "significant_status_changed", "materially_changed")}} for r in records],
        "expert_review": "All outputs unreviewed; no independent expert labels or accuracy estimates available.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".labels-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        output = stage / "mistake_label_proposals.jsonl"
        output.write_text("".join(label_json(r) + "\n" for r in records), encoding="utf-8", newline="\n")
        report_path = stage / "label_quality_report.json"
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": LABEL_SCHEMA_VERSION, "rule_version": RULE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_input": str(candidate_path), "feature_input": str(feature_path),
            "candidate_sha256": candidate_hash, "feature_sha256": feature_hash,
            "input_manifests": provenance, "configuration": asdict(config),
            "engine_profile": asdict(profile) if profile else None,
            "input_candidates": len(candidates), "input_features": len(features),
            "valid_joined_records": len(joined), "input_diagnostics": diagnostics,
            "selection_method": "round-robin original selection types, then restore input ordering",
            "output_file": output.name, "output_sha256": file_sha256(output),
            "quality_report_sha256": file_sha256(report_path),
            "elapsed_seconds": time.perf_counter() - started,
            "cache_before": cache_before, "cache_after": asdict(cache.info()),
            **{key: report[key] for key in ("processed", "records_with_proposals", "proposal_count", "abstentions",
                  "concept_counts", "verification_failures", "classification_changes", "significant_status_changes")},
        }
        (stage / "labeling_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                                     encoding="utf-8")
        os.rename(stage, output_dir)
    return manifest
