"""Synthetic in-memory reviews only; no real human decisions or approvals."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone

import pytest

from src.chessmind.ml.data.dataset_readiness import inspect_training_readiness
from src.chessmind.ml.features.extractor import extract_candidate_features
from src.chessmind.ml.labeling.adjudication import (
    is_training_eligible, record_sha256, validate_adjudications,
)
from src.chessmind.ml.labeling.review_workflow import (
    finalize_review, prepare_review, record_review, validate_review,
)
from src.chessmind.ml.labeling.rules import propose_labels
from src.chessmind.ml.labeling.schema import LabelRecord


def encoded(row):
    return (json.dumps(row, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def artifacts(labeling_case):
    c, _, v = labeling_case(before_pv=("c2b4",))
    source = b"synthetic source, not a real adjudicated game"
    manifest = encoded({"sha256": digest(source), "source_url": "fixture:archive",
        "accepted_count": 1, "source_references": [
            {"game_id": c["source_game_id"], "archive_game_index": 1}]})
    c["source_evidence"] = dict(dataset_path="fixture.pgn", manifest_path="source.json",
        source_archive="fixture:archive", archive_game_index=1,
        dataset_sha256=digest(source), manifest_sha256=digest(manifest))
    f = json.loads(json.dumps(asdict(extract_candidate_features(c))))
    proposals, evidence, status, reasons = propose_labels(c, f, v)
    e = json.loads(json.dumps(asdict(LabelRecord(
        c["candidate_id"], c["source_game_id"], c["player_identifier"], c["ply_index"],
        c["schema_version"], f["schema_version"], c["selection_type"], v, evidence,
        proposals, status, reasons, digest(encoded(c)), digest(encoded(f)), record_sha256(f)))))
    return dict(candidate=c, feature=f, evidence=e, source_dataset=source, source_manifest=manifest)


def draft(artifacts, **options):
    fields = dict(candidate_id=artifacts["candidate"]["candidate_id"], concept="hanging_piece",
                  decision="unresolved", reviewer_id="fixture:human",
                  rationale="Synthetic review; further defensive evidence needed.")
    fields.update(options)
    return prepare_review(**fields, **artifacts)


@pytest.mark.parametrize("decision", ["positive", "negative", "unresolved"])
def test_explicit_drafts(artifacts, decision):
    original = deepcopy(artifacts)
    before = datetime.now(timezone.utc)
    review = draft(artifacts, decision=decision)
    after = datetime.now(timezone.utc)
    assert before <= datetime.fromisoformat(review.reviewed_at.replace("Z", "+00:00")) <= after
    assert review.decision == decision and review.review_state == "draft"
    assert review.schema_version == "human_adjudication_v1"
    assert review.eligibility_version == "adjudicated_concepts_v1"
    for key in ("candidate", "feature", "evidence"):
        assert getattr(review, key + "_record_sha256") == record_sha256(artifacts[key])
    assert review.source_dataset_sha256 == digest(artifacts["source_dataset"])
    assert review.source_manifest_sha256 == digest(artifacts["source_manifest"])
    assert validate_review(review, **artifacts) is review
    assert artifacts == original


@pytest.mark.parametrize("decision", ["positive", "negative", "unresolved"])
def test_finalize_and_no_automatic_approval(artifacts, decision):
    review = draft(artifacts, decision=decision)
    submitted = record_review(review, reviews=(), **artifacts)
    result = finalize_review(review, reviews=submitted, **artifacts)
    final = result[0]
    assert submitted == (review,) and review.review_state == "draft"
    assert final == replace(review, review_state="finalized")
    assert validate_adjudications(result) == result
    assert not is_training_eligible(final, reviews=result, **artifacts)
    # Synthetic trust input only: demonstrates unchanged eligibility contract.
    approvals = frozenset({record_sha256(asdict(final))})
    assert is_training_eligible(final, reviews=result, approved_review_hashes=approvals,
                                **artifacts) is (decision != "unresolved")


@pytest.mark.parametrize("field,value", [
    ("reviewer_id", None), ("reviewer_id", ""), ("reviewer_id", " human "),
    ("rationale", None), ("rationale", ""), ("rationale", "   "),
    ("concept", "unknown"), ("concept", "king_safety"), ("concept", None),
    ("concept", []), ("decision", None), ("decision", "approve"),
    ("candidate_id", "0" * 64),
])
def test_invalid_human_input(artifacts, field, value):
    with pytest.raises(ValueError):
        draft(artifacts, **{field: value})


@pytest.mark.parametrize("missing", ["decision", "reviewer_id", "rationale"])
def test_required_human_fields(artifacts, missing):
    values = dict(candidate_id=artifacts["candidate"]["candidate_id"], concept="hanging_piece",
                  decision="positive", reviewer_id="fixture:human", rationale="Synthetic rationale")
    del values[missing]
    with pytest.raises(TypeError):
        prepare_review(**values, **artifacts)


@pytest.mark.parametrize("concept", ["hanging_piece", "missed_mate", "missed_fork", "missed_pin", "missed_skewer"])
def test_existing_tactical_concepts(artifacts, concept):
    assert draft(artifacts, concept=concept).concept == concept


@pytest.mark.parametrize("artifact,field,value", [
    ("candidate", "candidate_id", "0" * 64),
    ("feature", "candidate_id", "0" * 64),
    ("evidence", "candidate_id", "0" * 64),
    ("evidence", "feature_record_sha256", "0" * 64),
    ("evidence", "candidate_input_sha256", "bad"),
    ("evidence", "rule_version", "future"),
    ("evidence", "review_status", "approved"),
    ("evidence", "reasons", ["invented current-rule reason"]),
    ("evidence", "verification", {}),
])
def test_mismatched_or_malformed_records(artifacts, artifact, field, value):
    artifacts[artifact][field] = value
    with pytest.raises(ValueError):
        draft(artifacts)


@pytest.mark.parametrize("key,value", [
    ("candidate", None), ("feature", []), ("evidence", {}),
    ("source_dataset", "not bytes"), ("source_dataset", b"changed"),
    ("source_manifest", b"{}"), ("source_manifest", b"null"),
    ("source_manifest", b'{"sha256":0,"sha256":1}'),
])
def test_malformed_artifacts(artifacts, key, value):
    artifacts[key] = value
    with pytest.raises(ValueError):
        prepare_review(candidate_id="a" * 64, concept="hanging_piece", decision="unresolved",
                       reviewer_id="fixture:human", rationale="Synthetic rationale", **artifacts)


@pytest.mark.parametrize("field", ["candidate_record_sha256", "feature_record_sha256",
    "evidence_record_sha256", "source_dataset_sha256", "source_manifest_sha256"])
def test_stale_bindings_never_record_or_finalize(artifacts, field):
    review = replace(draft(artifacts), **{field: "0" * 64})
    for operation in (record_review, finalize_review):
        with pytest.raises(ValueError, match="stale"):
            operation(review, reviews=(review,) if operation is finalize_review else (), **artifacts)
    assert review.review_state == "draft"


def test_evidence_changed_after_submission(artifacts):
    review = draft(artifacts)
    submitted = record_review(review, reviews=(), **artifacts)
    # Still structurally valid evidence, but no longer what the reviewer examined.
    artifacts["evidence"]["verification"]["reason"] = "changed metadata"
    with pytest.raises(ValueError, match="stale"):
        finalize_review(review, reviews=submitted, **artifacts)
    assert submitted == (review,)


def test_duplicates_and_invalid_collection(artifacts):
    review = draft(artifacts)
    with pytest.raises(ValueError, match="duplicate submission"):
        record_review(review, reviews=(review,), **artifacts)
    with pytest.raises(ValueError, match="duplicate submission"):
        finalize_review(review, reviews=(review, review), **artifacts)
    with pytest.raises(ValueError):
        record_review(review, reviews=(asdict(review),), **artifacts)
    with pytest.raises(ValueError, match="submitted"):
        finalize_review(review, reviews=(), **artifacts)
    final = replace(review, review_state="finalized")
    with pytest.raises(ValueError, match="only drafts"):
        finalize_review(final, reviews=(final,), **artifacts)


@pytest.mark.parametrize("decision", ["positive", "negative"])
def test_finalized_conflicts_even_if_agree(artifacts, decision):
    review = draft(artifacts, decision=decision)
    existing = replace(review, decision="positive", reviewer_id="fixture:other", review_state="finalized")
    collection = record_review(review, reviews=(existing,), **artifacts)
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        finalize_review(review, reviews=collection, **artifacts)
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        record_review(replace(review, review_state="finalized"), reviews=(existing,), **artifacts)
    assert collection == (existing, review)


def test_independent_concepts_and_distinct_drafts(artifacts):
    first = draft(artifacts)
    second = replace(first, reviewer_id="fixture:other")
    other = draft(artifacts, concept="missed_mate")
    records = record_review(second, reviews=(first, other), **artifacts)
    records = finalize_review(first, reviews=records, **artifacts)
    records = finalize_review(other, reviews=records, **artifacts)
    assert [r.review_state for r in records] == ["finalized", "finalized", "draft"]


@pytest.mark.parametrize("historical", [False, True])
def test_readiness_compatibility_and_historical_restriction(artifacts, tmp_path, historical):
    if historical:
        artifacts["evidence"]["rule_version"] = "conservative_tactics_v1"
        for proposal in artifacts["evidence"]["proposals"]:
            proposal["rule_version"] = "conservative_tactics_v1"
    review = draft(artifacts, decision="positive")
    final, = finalize_review(review, reviews=record_review(review, reviews=(), **artifacts), **artifacts)
    paths = {}
    for key in ("candidate", "feature", "evidence", "source_dataset", "source_manifest"):
        paths[key + "_path"] = tmp_path / key
        value = artifacts[key]
        paths[key + "_path"].write_bytes(value if type(value) is bytes else encoded(value))
    paths["adjudications_path"] = tmp_path / "reviews.jsonl"
    paths["adjudications_path"].write_bytes(encoded(asdict(final)))
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["adjudication_status"]["approved"] == 0
    # External synthetic approval, never generated by the workflow.
    paths["approvals_path"] = tmp_path / "trusted.json"
    paths["approvals_path"].write_bytes(encoded([record_sha256(asdict(final))]))
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"]
    assert report["training_ready"] is (not historical)
    assert report["eligible_labels"]["total"] == int(not historical)
