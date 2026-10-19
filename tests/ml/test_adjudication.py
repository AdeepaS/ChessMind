"""Synthetic annotations only: no real candidates receive human labels here."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from src.chessmind.ml.features.extractor import extract_candidate_features
from src.chessmind.ml.labeling.adjudication import (
    AdjudicationRecord, is_training_eligible, record_sha256, validate_adjudications,
)
from src.chessmind.ml.labeling.rules import propose_labels
from src.chessmind.ml.labeling.schema import LabelRecord


@pytest.fixture
def adjudication_case(labeling_case):
    c, _, v = labeling_case(before_pv=("c2b4",))
    # Opaque synthetic source bytes; never published as a real game or annotation.
    source = b"synthetic source fixture"
    manifest = json.dumps({"sha256": hashlib.sha256(source).hexdigest(),
        "source_url": "fixture:archive", "source_references": [
            {"game_id": c["source_game_id"], "archive_game_index": 1}]}).encode()
    c["source_evidence"] = {"dataset_path": "fixture.pgn", "manifest_path": "fixture.json",
        "source_archive": "fixture:archive", "archive_game_index": 1,
        "dataset_sha256": hashlib.sha256(source).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
    f = json.loads(json.dumps(asdict(extract_candidate_features(c))))
    proposals, evidence, status, reasons = propose_labels(c, f, v)
    e = asdict(LabelRecord(c["candidate_id"], c["source_game_id"], c["player_identifier"],
        c["ply_index"], c["schema_version"], f["schema_version"], c["selection_type"],
        v, evidence, proposals, status, reasons, "a" * 64, "b" * 64, record_sha256(f)))
    e = json.loads(json.dumps(e))
    annotation = AdjudicationRecord(c["candidate_id"], "hanging_piece", "positive",
        "fixture:reviewer", "Synthetic fixture decision, not a real annotation.",
        "2026-10-19T12:00:00Z", record_sha256(c), record_sha256(f), record_sha256(e),
        hashlib.sha256(source).hexdigest(), hashlib.sha256(manifest).hexdigest(), "finalized")
    return annotation, dict(reviews=(annotation,), candidate=c, feature=f, evidence=e,
        source_dataset=source, source_manifest=manifest,
        approved_review_hashes=frozenset({record_sha256(asdict(annotation))}))


def approve(annotation, kwargs):
    kwargs["reviews"] = (annotation,)
    kwargs["approved_review_hashes"] = frozenset({record_sha256(asdict(annotation))})


@pytest.mark.parametrize("decision,expected", [("positive", True), ("negative", True), ("unresolved", False)])
def test_explicit_decisions(adjudication_case, decision, expected):
    a, kwargs = adjudication_case
    a = replace(a, decision=decision)
    approve(a, kwargs)
    assert is_training_eligible(a, **kwargs) is expected


@pytest.mark.parametrize("field,value", [
    ("concept", "unknown"), ("decision", "proposed"), ("review_state", "accepted"),
    ("reviewer_id", " "), ("rationale", ""), ("reviewed_at", "2026-10-19"),
    ("reviewed_at", "2026-02-30T00:00:00Z"), ("candidate_id", "a"),
    ("evidence_record_sha256", None), ("schema_version", "future"),
    ("eligibility_version", "future"), ("source_manifest_sha256", "G" * 64),
])
def test_strict_record_validation(adjudication_case, field, value):
    a, _ = adjudication_case
    with pytest.raises((ValueError, TypeError)):
        replace(a, **{field: value})


def test_draft_or_untrusted_review_fails(adjudication_case):
    a, kwargs = adjudication_case
    kwargs["approved_review_hashes"] = frozenset()
    assert not is_training_eligible(a, **kwargs)
    a = replace(a, review_state="draft")
    approve(a, kwargs)
    assert not is_training_eligible(a, **kwargs)


def test_approval_binds_exact_decision_and_reviewer(adjudication_case):
    a, kwargs = adjudication_case
    for changes in ({"decision": "negative"}, {"reviewer_id": "fixture:other"}):
        changed = replace(a, **changes)
        kwargs["reviews"] = (changed,)
        assert not is_training_eligible(changed, **kwargs)


@pytest.mark.parametrize("decision", ["positive", "negative", "unresolved"])
def test_duplicate_finalized_decisions_fail(adjudication_case, decision):
    a, kwargs = adjudication_case
    kwargs["reviews"] = (a, replace(a, decision=decision, reviewer_id="fixture:second"))
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        validate_adjudications(kwargs["reviews"])
    assert not is_training_eligible(a, **kwargs)


def test_concepts_remain_independent(adjudication_case):
    a, kwargs = adjudication_case
    other = replace(a, concept="missed_mate", decision="unresolved")
    kwargs["reviews"] = (a, other)
    assert is_training_eligible(a, **kwargs)
    assert not is_training_eligible(other, **kwargs)
    assert len(validate_adjudications((a, replace(a, review_state="draft")))) == 2


@pytest.mark.parametrize("artifact,field,value", [
    ("candidate", "candidate_id", "0" * 64),
    ("candidate", "schema_version", "future"),
    ("feature", "schema_version", "future"),
    ("feature", "candidate_id", "0" * 64),
    ("evidence", "rule_version", "conservative_tactics_v1"),
    ("evidence", "schema_version", "future"),
    ("evidence", "evidence", []),
    ("evidence", "evidence", [{}]),
    ("evidence", "verification", {"status": "unreviewed"}),
    ("evidence", "feature_record_sha256", "0" * 64),
])
def test_malformed_or_unsupported_evidence_even_when_approved(adjudication_case, artifact, field, value):
    a, kwargs = adjudication_case
    kwargs[artifact][field] = value
    a = replace(a, **{artifact + "_record_sha256": record_sha256(kwargs[artifact])})
    approve(a, kwargs)
    assert not is_training_eligible(a, **kwargs)


@pytest.mark.parametrize("key", ["dataset_sha256", "manifest_sha256", "dataset_path", "archive_game_index"])
def test_missing_provenance(adjudication_case, key):
    a, kwargs = adjudication_case
    del kwargs["candidate"]["source_evidence"][key]
    assert not is_training_eligible(a, **kwargs)


@pytest.mark.parametrize("key", ["source_dataset", "source_manifest"])
def test_actual_source_bytes_required(adjudication_case, key):
    a, kwargs = adjudication_case
    kwargs[key] += b"tampering"
    assert not is_training_eligible(a, **kwargs)


@pytest.mark.parametrize("field,value", [("dataset_path", ""), ("archive_game_index", True),
    ("dataset_sha256", "bad"), ("source_archive", "fixture:different")])
def test_invalid_provenance_with_rebound_record_hashes(adjudication_case, field, value):
    a, kwargs = adjudication_case
    kwargs["candidate"]["source_evidence"][field] = value
    kwargs["feature"]["source_evidence"][field] = value
    kwargs["evidence"]["feature_record_sha256"] = record_sha256(kwargs["feature"])
    a = replace(a, candidate_record_sha256=record_sha256(kwargs["candidate"]),
        feature_record_sha256=record_sha256(kwargs["feature"]),
        evidence_record_sha256=record_sha256(kwargs["evidence"]))
    approve(a, kwargs)
    assert not is_training_eligible(a, **kwargs)


def test_manifest_game_link_required_even_with_matching_hashes(adjudication_case):
    a, kwargs = adjudication_case
    manifest = json.loads(kwargs["source_manifest"])
    manifest["source_references"] = []
    kwargs["source_manifest"] = json.dumps(manifest).encode()
    digest = hashlib.sha256(kwargs["source_manifest"]).hexdigest()
    for key in ("candidate", "feature"):
        kwargs[key]["source_evidence"]["manifest_sha256"] = digest
    kwargs["evidence"]["feature_record_sha256"] = record_sha256(kwargs["feature"])
    a = replace(a, source_manifest_sha256=digest,
        candidate_record_sha256=record_sha256(kwargs["candidate"]),
        feature_record_sha256=record_sha256(kwargs["feature"]),
        evidence_record_sha256=record_sha256(kwargs["evidence"]))
    approve(a, kwargs)
    assert not is_training_eligible(a, **kwargs)


@pytest.mark.parametrize("status", ["proposed", "requires_review", "insufficient_evidence", "conflicting_evidence"])
def test_rule_records_never_become_annotations(adjudication_case, status):
    _, kwargs = adjudication_case
    proposal = deepcopy(kwargs["evidence"])
    proposal.update(proposal_status=status, rule_version="conservative_tactics_v1", proposals=[])
    kwargs["reviews"] = (proposal,)
    assert not is_training_eligible(proposal, **kwargs)
    assert not is_training_eligible(None, **{**kwargs, "reviews": ()})


def test_explicit_review_can_adjudicate_an_abstention(adjudication_case):
    a, kwargs = adjudication_case
    kwargs["evidence"].update(proposals=[], proposal_status="requires_review")
    a = replace(a, decision="negative", evidence_record_sha256=record_sha256(kwargs["evidence"]))
    approve(a, kwargs)
    assert is_training_eligible(a, **kwargs)


@pytest.mark.parametrize("problem", ["missing_input_hash", "malformed_input_hash",
    "missing_status", "invented_status", "pretend_reviewed", "nonterminal_mate_zero",
    "missing_continuation", "illegal_continuation", "unknown_evidence_type"])
def test_incomplete_evidence_fails_even_with_approval(adjudication_case, problem):
    a, kwargs = adjudication_case
    evidence = kwargs["evidence"]
    if problem == "missing_input_hash":
        del evidence["candidate_input_sha256"]
    elif problem == "malformed_input_hash":
        evidence["feature_input_sha256"] = "not-a-hash"
    elif problem == "missing_status":
        del evidence["proposal_status"]
    elif problem == "invented_status":
        evidence["proposal_status"] = "confirmed"
    elif problem == "pretend_reviewed":
        evidence["review_status"] = "approved"
    elif problem == "nonterminal_mate_zero":
        evidence["verification"]["before"] = {
            "type": "mate", "mate": 0, "cp": None, "winner": "black", "perspective": "white"}
    elif problem == "missing_continuation":
        del evidence["evidence"][0]["continuation"]
    elif problem == "illegal_continuation":
        evidence["evidence"][0]["continuation"] = ["a1a8"]
    else:
        evidence["evidence"][0]["evidence_type"] = "confirmed_by_caller"
    a = replace(a, evidence_record_sha256=record_sha256(evidence))
    approve(a, kwargs)
    assert not is_training_eligible(a, **kwargs)
