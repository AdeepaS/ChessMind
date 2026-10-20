"""Read-only pilot artifact inspection; training eligibility remains in adjudication."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

import chess

from .candidate_schema import SCHEMA_VERSION, AnalysisProfile, ScoreEvidence
from .candidate_extraction import _score
from ...analysis.game_analyzer import evaluate_move_quality
from ..features.pipeline import _invalid_constant, _unique_object
from ..features.schema import FEATURE_SCHEMA_VERSION
from ..labeling.adjudication import (
    AdjudicationRecord, _digest, is_training_eligible, record_sha256,
    validate_adjudications,
)
from ..labeling.engine_verification import _result, replay_pv
from ..labeling.pipeline import _read_records, _validate_candidate
from ..labeling.rules import propose_labels
from ..labeling.schema import (
    LABEL_SCHEMA_VERSION, RULE_VERSION, LabelRecord, LabelProposal,
    TacticalEvidence, VerificationResult,
)
from ..labeling.taxonomy import MistakeConcept


HISTORICAL_RULE_VERSION = "conservative_tactics_v1"


def report_json(report):
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _shape(record, cls):
    _require(type(record) is dict and set(record) == {f.name for f in fields(cls)},
             f"invalid {cls.__name__} fields")


def _records(path):
    # Reuse the labeling pipeline's strict duplicate and resource-limit checks.
    return {key: row for key, (_, row) in _read_records(Path(path), True, []).items()}


def _manifest(path, expected, *, count_fields=()):
    if path is None:
        return None
    manifest = _json(Path(path).read_bytes())
    _require(type(manifest) is dict, "manifest must be an object")
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"{Path(path).name}: {key} mismatch")
    for key, value in count_fields:
        _require(type(manifest.get(key)) is int and manifest[key] == value,
                 f"{Path(path).name}: {key} mismatch")
    return manifest


def _evidence(row, candidate, feature, candidate_hash, feature_hash):
    _shape(row, LabelRecord)
    _require(row["schema_version"] == LABEL_SCHEMA_VERSION, "unsupported evidence schema")
    _require(row["rule_version"] in (RULE_VERSION, HISTORICAL_RULE_VERSION),
             "unsupported evidence rule version")
    for key in ("candidate_id", "source_game_id", "player_identifier", "ply_index"):
        _require(type(row[key]) is type(candidate[key]) and row[key] == candidate[key],
                 f"evidence {key} mismatch")
    for key, value in (
        ("candidate_schema_version", SCHEMA_VERSION),
        ("feature_schema_version", FEATURE_SCHEMA_VERSION),
        ("original_selection_type", candidate["selection_type"]),
        ("candidate_input_sha256", candidate_hash),
        ("feature_input_sha256", feature_hash),
        ("feature_record_sha256", record_sha256(feature)),
        ("review_status", "unreviewed"),
    ):
        _require(row[key] == value, f"evidence {key} mismatch")
    _require(row["proposal_status"] in (
        "proposed", "requires_review", "insufficient_evidence", "conflicting_evidence"),
        "invalid proposal status")
    _require(type(row["reasons"]) is list and row["reasons"]
             and all(type(x) is str and x.strip() for x in row["reasons"]),
             "invalid evidence reasons")
    verification = row["verification"]
    _shape(verification, VerificationResult)
    _require(verification["status"] in ("verified", "failed", "skipped"),
             "invalid verification status")
    for name in ("significant_error_supported", "classification_changed",
                 "significant_status_changed", "materially_changed"):
        _require(verification[name] is None or type(verification[name]) is bool,
                 "invalid verification boolean")
    for name in ("centipawn_loss", "original_cp_loss", "cp_loss_change"):
        value = verification[name]
        _require(value is None or (type(value) in (int, float) and math.isfinite(value)
                                  and (name == "cp_loss_change" or value >= 0)),
                 "invalid verification loss")
    for endpoint in ("before", "after"):
        _require(type(verification[endpoint + "_pv"]) is list, "invalid verification PV")
        if verification[endpoint] is not None:
            _shape(verification[endpoint], ScoreEvidence)
            _score(verification[endpoint])
    if verification["profile"] is not None:
        _shape(verification["profile"], AnalysisProfile)
        AnalysisProfile(**verification["profile"])
    if verification["status"] == "verified":
        for endpoint in ("before", "after"):
            depth = verification[endpoint + "_depth"]
            _require(type(depth) is int and depth >= 0, "invalid verification depth")
            _result(chess.Board(candidate["fen_" + endpoint]), {
                "evaluation": verification[endpoint], "depth": depth,
                "pv": [chess.Move.from_uci(x) for x in verification[endpoint + "_pv"]],
            }, 1)
        quality = evaluate_move_quality(verification["before"], verification["after"],
                                        candidate["player_color"] == "white")
        for name in ("centipawn_loss", "classification", "mate_quality"):
            _require(verification[name] == quality[name], "inconsistent verification quality")
        supported = (quality["centipawn_loss"] >= candidate["extraction_configuration"]["cp_loss_threshold"]
                     if quality["centipawn_loss"] is not None else None)
        _require(verification["significant_error_supported"] is supported,
                 "inconsistent verification significant support")
    else:
        _require(type(verification["reason"]) is str and verification["reason"].strip(),
                 "failed/skipped verification requires a reason")
        _require(not row["proposals"] and not row["evidence"]
                 and row["proposal_status"] == "insufficient_evidence",
                 "unverified evidence cannot support proposals")
    _require(type(row["evidence"]) is list and type(row["proposals"]) is list,
             "evidence and proposals must be lists")
    for item in row["evidence"]:
        _shape(item, TacticalEvidence)
        _require(type(item["supported"]) is bool and type(item["reason"]) is str
                 and item["reason"].strip(), "invalid tactical evidence")
        for name in ("captures_moved_piece", "victim_defended", "previously_attacked"):
            _require(item[name] is None or type(item[name]) is bool, "invalid tactical boolean")
        for name in ("pv_material_gain", "minimum_gain_after_immediate_recapture",
                     "attacked_undefended_delta"):
            _require(item[name] is None or type(item[name]) is int, "invalid tactical count")
        for name in ("capture_uci", "victim_square"):
            _require(item[name] is None or type(item[name]) is str, "invalid tactical square/move")
        endpoint = {"missed_forced_mate": "before",
                    "allowed_immediate_tactical_capture": "after"}.get(item["evidence_type"])
        _require(endpoint is not None and type(item["continuation"]) is list,
                 "unsupported tactical evidence")
        replay_pv(chess.Board(candidate["fen_" + endpoint]), item["continuation"])
    concepts = set()
    for proposal in row["proposals"]:
        _shape(proposal, LabelProposal)
        MistakeConcept(proposal["concept"])
        _require(proposal["concept"] not in concepts, "duplicate proposed concept")
        concepts.add(proposal["concept"])
        _require(proposal["rule_version"] == row["rule_version"]
                 and proposal["status"] == "proposed"
                 and proposal["review_status"] == "unreviewed"
                 and type(proposal["evidence"]) is list and proposal["evidence"]
                 and all(x in row["evidence"] and x["supported"] is True
                         for x in proposal["evidence"]), "invalid proposal evidence")
    _require(bool(row["proposals"]) == (row["proposal_status"] == "proposed"),
             "proposal status/count mismatch")
    if row["rule_version"] == RULE_VERSION:
        # Replay current rules from saved verification, never perform engine searches.
        values = dict(verification)
        for endpoint in ("before", "after"):
            if values[endpoint] is not None:
                values[endpoint] = ScoreEvidence(**values[endpoint])
        proposals, evidence, status, reasons = propose_labels(
            candidate, feature, VerificationResult(**values))
        expected = ([asdict(p) for p in proposals], [asdict(e) for e in evidence],
                    status, reasons)
        actual = (row["proposals"], row["evidence"], row["proposal_status"], row["reasons"])
        _require(json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True),
                 "saved evidence disagrees with current rules")


def inspect_training_readiness(candidate_path, feature_path, evidence_path, *,
                               source_dataset_path, source_manifest_path,
                               candidate_manifest_path=None, feature_manifest_path=None,
                               labeling_manifest_path=None, adjudications_path=None,
                               approvals_path=None):
    """Return deterministic JSON-compatible diagnostics, without writes or searches.

    Approvals are a trusted operator input: a JSON list of canonical review hashes,
    established independently of submitted annotations. They are never inferred.
    Missing adjudications/approvals are normal ineligibility, not corrupt inputs.
    Integrity failures invalidate the whole collection; partial counts are not used.
    """
    report = {
        "schema_version": "dataset_readiness_v1",
        "training_ready": False, "integrity_valid": False, "integrity_errors": [],
        "blocking_reasons": [], "counts": {}, "versions": {}, "artifact_sha256": {},
        "adjudication_status": {}, "eligible_labels": {"total": 0, "by_concept": {}},
        "readiness_scope": "Label eligibility only; does not certify coverage, splits or model readiness.",
    }
    try:
        _inspect(report, candidate_path, feature_path, evidence_path,
                 source_dataset_path, source_manifest_path, candidate_manifest_path,
                 feature_manifest_path, labeling_manifest_path, adjudications_path, approvals_path)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError,
            OSError) as exc:
        report.update(training_ready=False, integrity_valid=False,
                      integrity_errors=[f"{type(exc).__name__}: {exc}"],
                      blocking_reasons=["artifact_integrity_failure"], counts={}, versions={},
                      artifact_sha256={}, adjudication_status={},
                      eligible_labels={"total": 0, "by_concept": {}})
    return report


def _inspect(report, candidate_path, feature_path, evidence_path, source_dataset_path,
             source_manifest_path, candidate_manifest_path, feature_manifest_path,
             labeling_manifest_path, adjudications_path, approvals_path):
    paths = dict(candidates=Path(candidate_path), features=Path(feature_path),
                 evidence=Path(evidence_path), source_dataset=Path(source_dataset_path),
                 source_manifest=Path(source_manifest_path))
    for name, supplied, adjacent in (
        ("candidate_manifest", candidate_manifest_path, paths["candidates"].with_name("candidate_manifest.json")),
        ("feature_manifest", feature_manifest_path, paths["features"].with_name("feature_manifest.json")),
        ("labeling_manifest", labeling_manifest_path, paths["evidence"].with_name("labeling_manifest.json")),
    ):
        if supplied is not None or adjacent.exists():
            paths[name] = Path(supplied) if supplied is not None else adjacent
    for name, path in (("adjudications", adjudications_path), ("approvals", approvals_path)):
        if path is not None:
            paths[name] = Path(path)
    _require(len({p.resolve() for p in paths.values()}) == len(paths), "input paths must be distinct")
    # Bounded pilot inspector, consistent with existing labeling artifact budgets.
    for path in paths.values():
        _require(path.stat().st_size <= 32 * 1024 * 1024, "artifact exceeds 32 MiB budget")
    data = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: _hash(value) for name, value in data.items()}
    candidates, features, evidence = (_records(paths[name]) for name in
                                     ("candidates", "features", "evidence"))
    _require(candidates.keys() == features.keys(), "candidate/feature IDs differ")
    _require(evidence.keys() <= candidates.keys(), "evidence has unknown candidate IDs")
    source = _json(data["source_manifest"])
    _require(type(source) is dict and source["sha256"] == hashes["source_dataset"],
             "source dataset hash mismatch")
    _require(type(source["source_url"]) is str and source["source_url"].strip(),
             "missing source archive")
    refs = source["source_references"]
    _require(type(refs) is list, "source references must be a list")
    references = {}
    for ref in refs:
        _require(type(ref) is dict and type(ref.get("game_id")) is str
                 and ref["game_id"] and type(ref.get("archive_game_index")) is int
                 and ref["archive_game_index"] > 0, "invalid source reference")
        _require(ref["game_id"] not in references, "duplicate source game reference")
        references[ref["game_id"]] = ref["archive_game_index"]
    _require(len(set(references.values())) == len(references), "duplicate source archive index")
    if "accepted_count" in source:
        _require(type(source["accepted_count"]) is int and source["accepted_count"] == len(refs),
                 "source reference count mismatch")
    _manifest(paths.get("candidate_manifest"),
        {"schema_version": SCHEMA_VERSION, "output_sha256": hashes["candidates"],
         "input_sha256": hashes["source_dataset"], "source_manifest_sha256": hashes["source_manifest"]},
        count_fields=(("candidate_count", len(candidates)),))
    fm = _manifest(paths.get("feature_manifest"),
        {"schema_version": FEATURE_SCHEMA_VERSION, "output_sha256": hashes["features"],
         "input_sha256": hashes["candidates"], "feature_value_count": 163},
        count_fields=(("accepted", len(features)),))
    lm = _manifest(paths.get("labeling_manifest"),
        {"schema_version": LABEL_SCHEMA_VERSION, "output_sha256": hashes["evidence"],
         "candidate_sha256": hashes["candidates"], "feature_sha256": hashes["features"]},
        count_fields=(("processed", len(evidence)),))
    if fm and fm.get("source_provenance") is not None:
        provenance = fm["source_provenance"]
        for key, artifact in (("sha256", "candidate_manifest"), ("input_sha256", "source_dataset"),
                              ("source_manifest_sha256", "source_manifest")):
            _require(artifact in hashes and provenance[key] == hashes[artifact],
                     "feature manifest provenance mismatch")
        _require(provenance["schema_version"] == SCHEMA_VERSION,
                 "feature provenance schema mismatch")
    if lm:
        _require(lm["rule_version"] in (RULE_VERSION, HISTORICAL_RULE_VERSION),
                 "unsupported manifest rule version")
        for name, ref in lm.get("input_manifests", {}).items():
            _require(name in ("candidate_manifest", "feature_manifest"), "unknown input manifest")
            if ref is not None:
                _require(name in hashes and ref["sha256"] == hashes[name], "label manifest provenance mismatch")
                _require(ref["schema_version"] == (
                    SCHEMA_VERSION if name == "candidate_manifest" else FEATURE_SCHEMA_VERSION),
                    "label provenance schema mismatch")
        if "quality_report_sha256" in lm:
            quality_path = paths["labeling_manifest"].with_name("label_quality_report.json")
            _require(quality_path.stat().st_size <= 32 * 1024 * 1024, "quality report exceeds budget")
            quality_data = quality_path.read_bytes()
            _require(_hash(quality_data) == lm["quality_report_sha256"],
                     "label quality report hash mismatch")
            _require(type(_json(quality_data)) is dict, "quality report must be an object")
            paths["quality_report"] = quality_path
            hashes["quality_report"] = _hash(quality_data)
    for identity, candidate in candidates.items():
        _require(record_sha256(_validate_candidate(candidate)) == record_sha256(features[identity]),
                 "feature contents/schema/source linkage mismatch")
        provenance = candidate["source_evidence"]
        for key in ("dataset_path", "manifest_path"):
            _require(type(provenance[key]) is str and provenance[key].strip(), "missing source path")
        _require(provenance["dataset_sha256"] == hashes["source_dataset"]
                 and provenance["manifest_sha256"] == hashes["source_manifest"]
                 and provenance["source_archive"] == source["source_url"]
                 and candidate["source_game_id"] in references
                 and type(provenance["archive_game_index"]) is int
                 and provenance["archive_game_index"] == references[candidate["source_game_id"]],
                 "candidate source provenance mismatch")
    for identity, row in evidence.items():
        _evidence(row, candidates[identity], features[identity], hashes["candidates"], hashes["features"])
        if lm:
            _require(row["rule_version"] == lm["rule_version"], "evidence/manifest rule mismatch")
    if lm:
        for key, value in (
            ("input_candidates", len(candidates)), ("input_features", len(features)),
            ("valid_joined_records", len(candidates)),
            ("proposal_count", sum(len(e["proposals"]) for e in evidence.values())),
            ("records_with_proposals", sum(bool(e["proposals"]) for e in evidence.values())),
        ):
            if key in lm:
                _require(type(lm[key]) is int and lm[key] == value, f"label manifest {key} mismatch")
    reviews = []
    if adjudications_path is not None:
        lines = data["adjudications"].decode("utf-8").splitlines()
        _require(len(lines) <= 10000, "adjudication record budget exceeded")
        reviews = [AdjudicationRecord(**_json(line)) for line in lines if line.strip()]
    reviews = validate_adjudications(reviews)
    approvals = _json(data["approvals"]) if approvals_path is not None else []
    _require(type(approvals) is list, "approvals must be a JSON list")
    for digest in approvals:
        _digest(digest)
    _require(len(set(approvals)) == len(approvals), "duplicate approval hash")
    approvals = frozenset(approvals)
    eligible, decisions = Counter(), Counter()
    approved = finalized = 0
    review_hashes = set()
    for review in reviews:
        identity = review.candidate_id
        _require(identity in candidates and identity in evidence, "adjudication missing candidate/evidence")
        for record, digest in (
            (candidates[identity], review.candidate_record_sha256),
            (features[identity], review.feature_record_sha256),
            (evidence[identity], review.evidence_record_sha256),
        ):
            _require(record_sha256(record) == digest, "adjudication record hash mismatch")
        _require(review.source_dataset_sha256 == hashes["source_dataset"]
                 and review.source_manifest_sha256 == hashes["source_manifest"],
                 "adjudication source hash mismatch")
        digest = record_sha256(asdict(review))
        review_hashes.add(digest)
        approved += digest in approvals
        finalized += review.review_state == "finalized"
        decisions[review.decision] += 1
        if is_training_eligible(review, reviews=reviews, candidate=candidates[identity],
                feature=features[identity], evidence=evidence[identity],
                source_dataset=data["source_dataset"], source_manifest=data["source_manifest"],
                approved_review_hashes=approvals):
            eligible[(review.concept, review.decision)] += 1
    # Detect changes during inspection, including separately supplied trust inputs.
    _require(all(_hash(path.read_bytes()) == hashes[name] for name, path in paths.items()),
             "input changed during inspection")
    total = sum(eligible.values())
    blockers = []
    if not reviews:
        blockers.append("no_adjudications")
    if not finalized:
        blockers.append("no_finalized_adjudications")
    if not approved:
        blockers.append("no_separately_approved_adjudications")
    if any(row["rule_version"] == HISTORICAL_RULE_VERSION for row in evidence.values()):
        blockers.append("historical_evidence_ineligible")
    if not total:
        blockers.append("no_eligible_labels")
    statuses = Counter(row["proposal_status"] for row in evidence.values())
    report.update(
        integrity_valid=True, training_ready=total > 0, blocking_reasons=blockers if not total else [],
        artifact_sha256=hashes,
        counts={"candidates": len(candidates), "feature_complete_candidates": len(features),
                "candidate_games": len({c["source_game_id"] for c in candidates.values()}),
                "source_games": len(references), "evidence_records": len(evidence),
                "candidates_without_evidence": len(candidates) - len(evidence),
                "engine_verified_candidates": sum(e["verification"]["status"] == "verified" for e in evidence.values()),
                "verification_status": dict(Counter(e["verification"]["status"] for e in evidence.values())),
                "provisional_proposals": sum(len(e["proposals"]) for e in evidence.values()),
                "historical_evidence_records": sum(e["rule_version"] == HISTORICAL_RULE_VERSION
                                                   for e in evidence.values()),
                "proposal_status": {status: statuses[status] for status in (
                    "proposed", "requires_review", "insufficient_evidence", "conflicting_evidence")}},
        versions={name: dict(Counter(row[key] for row in rows)) for name, rows, key in (
            ("candidate_schema", candidates.values(), "schema_version"),
            ("feature_schema", features.values(), "schema_version"),
            ("evidence_schema", evidence.values(), "schema_version"),
            ("evidence_rules", evidence.values(), "rule_version"),
            ("adjudication_schema", [asdict(r) for r in reviews], "schema_version"),
            ("eligibility_policy", [asdict(r) for r in reviews], "eligibility_version"))},
        adjudication_status={"total": len(reviews), "finalized": finalized,
            "draft": len(reviews) - finalized, "decisions": dict(decisions),
            "approved": approved, "unapproved": len(reviews) - approved,
            "approval_hashes": len(approvals), "unused_approval_hashes": len(approvals - review_hashes),
            "ineligible": len(reviews) - total},
        eligible_labels={"total": total, "by_concept": {
            concept: {decision: eligible[(concept, decision)] for decision in ("positive", "negative")}
            for concept in sorted({concept for concept, _ in eligible})}},
    )
