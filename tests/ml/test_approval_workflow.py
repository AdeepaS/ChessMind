"""Deterministic synthetic human decisions/authorizations in temporary files only."""

import json
from dataclasses import asdict, replace

import pytest

from src.chessmind.ml.labeling import approval_workflow as workflow
from src.chessmind.ml.labeling.adjudication import record_sha256
from src.chessmind.ml.labeling.approval_store import load_approvals, save_approvals
from src.chessmind.ml.labeling.review_store import load_snapshot, save_reviews
from src.chessmind.ml.labeling.review_workflow import prepare_review, record_review, finalize_review
from src.chessmind.ml.data.dataset_readiness import inspect_training_readiness
from tests.ml.test_dataset_readiness import make_bundle


ACTION = dict(confirm=True, operator_id="fixture:independent",
              independent_justification="Synthetic independent examination of alternatives; fixture only.")


def make_case(tmp_path, labeling_case, *, decision="positive", case=None):
    paths, candidate, feature, evidence = make_bundle(
        tmp_path, case if case is not None else labeling_case(before_pv=("c2b4",)))
    artifacts = dict(candidate=candidate, feature=feature, evidence=evidence,
                     source_dataset=paths["source_dataset_path"].read_bytes(),
                     source_manifest=paths["source_manifest_path"].read_bytes())
    draft = prepare_review(candidate_id=candidate["candidate_id"], concept="hanging_piece",
        decision=decision, reviewer_id="fixture:annotator", rationale="Synthetic human decision.", **artifacts)
    draft = replace(draft, reviewed_at="2026-11-03T00:00:00Z")
    reviews = finalize_review(draft, reviews=record_review(draft, reviews=(), **artifacts), **artifacts)
    paths["adjudications_path"] = tmp_path / "reviews.jsonl"
    initial = save_reviews(paths["adjudications_path"], (), expected_digest=None)
    save_reviews(paths["adjudications_path"], reviews, expected_digest=initial.digest)
    registry = tmp_path / "approvals.json"
    save_approvals(registry, (), expected_digest=None)
    return paths, registry, reviews[0]


@pytest.fixture
def approval_case(tmp_path, labeling_case):
    return make_case(tmp_path, labeling_case)


def write_reviews(paths, reviews):
    path = paths["adjudications_path"]
    save_reviews(path, reviews, expected_digest=load_snapshot(path).digest)


def authorize(case, review=None, **overrides):
    paths, registry, original = case
    action = {**ACTION, **overrides}
    return workflow.approve_review(record_sha256(asdict(review or original)),
                                   approvals_path=registry, **paths, **action)


@pytest.mark.parametrize("decision", ["positive", "negative"])
def test_explicit_finalized_decisions(tmp_path, labeling_case, decision):
    case = make_case(tmp_path, labeling_case, decision=decision)
    paths, registry, review = case
    original = {p: p.read_bytes() for p in paths.values()}
    report = inspect_training_readiness(**paths, approvals_path=registry)
    assert report["integrity_valid"] and not report["training_ready"]
    result = authorize(case)
    digest = record_sha256(asdict(review))
    assert result["approved_review_hash"] == digest
    assert load_approvals(registry).hashes == (digest,)
    assert all(p.read_bytes() == value for p, value in original.items())
    report = inspect_training_readiness(**paths, approvals_path=registry)
    assert report["training_ready"] and report["eligible_labels"]["total"] == 1
    assert report["eligible_labels"]["by_concept"] == {"hanging_piece": {
        "positive": int(decision == "positive"), "negative": int(decision == "negative")}}


@pytest.mark.parametrize("changes", [{"review_state": "draft"}, {"decision": "unresolved"},
                                     {"concept": "king_safety"}])
def test_invalid_review_for_approval(approval_case, changes):
    paths, registry, review = approval_case
    changed = replace(review, **changes)
    write_reviews(paths, (changed,))
    before = registry.read_bytes()
    with pytest.raises(ValueError):
        authorize(approval_case, changed)
    assert registry.read_bytes() == before


@pytest.mark.parametrize("changes", [{"confirm": False}, {"confirm": 1}, {"operator_id": None},
    {"operator_id": "fixture:annotator"}, {"independent_justification": None},
    {"independent_justification": " "}])
def test_no_automatic_or_self_approval(approval_case, changes):
    with pytest.raises(ValueError):
        authorize(approval_case, **changes)
    assert load_approvals(approval_case[1]).hashes == ()


@pytest.mark.parametrize("digest", [None, "", "bad", "A" * 64, "0" * 64])
def test_invalid_or_missing_review_hash(approval_case, digest):
    paths, registry, _ = approval_case
    with pytest.raises(ValueError):
        workflow.approve_review(digest, approvals_path=registry, **paths, **ACTION)
    assert load_approvals(registry).hashes == ()


def test_modified_review_content_invalidates_requested_hash(approval_case):
    paths, registry, review = approval_case
    write_reviews(paths, (replace(review, rationale="Changed human rationale."),))
    with pytest.raises(ValueError, match="hash not found"):
        authorize(approval_case)
    assert load_approvals(registry).hashes == ()


@pytest.mark.parametrize("key", ["candidate_path", "feature_path", "evidence_path",
                                "source_dataset_path", "source_manifest_path"])
def test_changed_artifacts(approval_case, key):
    paths, registry, _ = approval_case
    paths[key].write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        authorize(approval_case)
    assert load_approvals(registry).hashes == ()


def test_historical_evidence_rejected_even_with_bound_review(approval_case):
    paths, registry, review = approval_case
    evidence = json.loads(paths["evidence_path"].read_bytes())
    evidence["rule_version"] = "conservative_tactics_v1"
    for proposal in evidence["proposals"]:
        proposal["rule_version"] = "conservative_tactics_v1"
    paths["evidence_path"].write_text(json.dumps(evidence), encoding="utf-8")
    review = replace(review, evidence_record_sha256=record_sha256(evidence))
    write_reviews(paths, (review,))
    with pytest.raises(ValueError, match="current"):
        authorize(approval_case, review)
    assert load_approvals(registry).hashes == ()
    assert not inspect_training_readiness(**paths, approvals_path=registry)["training_ready"]


@pytest.mark.parametrize("decision", ["negative", "unresolved"])
def test_contradictory_drafts_block_issuance(approval_case, decision):
    paths, registry, review = approval_case
    write_reviews(paths, (review, replace(review, review_state="draft", decision=decision,
                                         reviewer_id="fixture:other")))
    with pytest.raises(ValueError, match="human attention"):
        authorize(approval_case)
    assert load_approvals(registry).hashes == ()


def test_conflicting_finalized_collection_rejected(approval_case):
    paths, registry, review = approval_case
    rows = (review, replace(review, decision="negative", reviewer_id="fixture:other"))
    paths["adjudications_path"].write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        authorize(approval_case)
    assert load_approvals(registry).hashes == ()


def test_same_decision_draft_and_other_concept_do_not_infer_labels(approval_case):
    paths, registry, review = approval_case
    write_reviews(paths, (review, replace(review, review_state="draft", reviewer_id="fixture:other"),
                          replace(review, concept="missed_mate", decision="unresolved")))
    authorize(approval_case)
    report = inspect_training_readiness(**paths, approvals_path=registry)
    assert report["eligible_labels"]["by_concept"] == {"hanging_piece": {"positive": 1, "negative": 0}}


def test_ambiguous_mate_requires_independent_justification(tmp_path, labeling_case):
    import chess.engine
    case = make_case(tmp_path, labeling_case, decision="negative", case=labeling_case(
        fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Cp(500),
        before_pv=("g6g7",), after_pv=("h8h7",)))
    paths, registry, review = case
    review = replace(review, concept="missed_mate")
    write_reviews(paths, (review,))
    assert json.loads(paths["evidence_path"].read_bytes())["proposal_status"] == "requires_review"
    with pytest.raises(ValueError):
        authorize(case, review, independent_justification=None)
    assert load_approvals(registry).hashes == ()
    # Explicit synthetic independent decision only, never inferred from the abstention.
    authorize(case, review, independent_justification="Synthetic independent analysis supports this concept-specific negative.")
    assert inspect_training_readiness(**paths, approvals_path=registry)["eligible_labels"]["by_concept"] == {
        "missed_mate": {"positive": 0, "negative": 1}}


def test_recheck_detects_concurrent_review_change(approval_case, monkeypatch):
    paths, registry, review = approval_case
    original = workflow.save_approvals
    def changed(*args, **kwargs):
        write_reviews(paths, (review, replace(review, review_state="draft", decision="unresolved")))
        return original(*args, **kwargs)
    monkeypatch.setattr(workflow, "save_approvals", changed)
    with pytest.raises(ValueError, match="changed"):
        authorize(approval_case)
    assert load_approvals(registry).hashes == ()
    assert not list(registry.parent.glob("*.lock"))
    assert not list(registry.parent.glob(".*.tmp"))


def test_duplicate_approval_and_path_alias_rejected(approval_case):
    paths, registry, review = approval_case
    with pytest.raises(ValueError, match="separate"):
        workflow.approve_review(record_sha256(asdict(review)), approvals_path=paths["adjudications_path"],
                                **paths, **ACTION)
    authorize(approval_case)
    before = registry.read_bytes()
    with pytest.raises(ValueError, match="already approved"):
        authorize(approval_case)
    assert registry.read_bytes() == before


def test_workflow_preserves_existing_authorizations(approval_case):
    paths, registry, review = approval_case
    initial = load_approvals(registry)
    # Opaque previous authorization is preserved, never reinterpreted as this review.
    save_approvals(registry, ["a" * 64], expected_digest=initial.digest)
    authorize(approval_case)
    assert set(load_approvals(registry).hashes) == {"a" * 64, record_sha256(asdict(review))}


def test_unverified_evidence_cannot_be_approved(approval_case):
    from src.chessmind.ml.labeling.schema import VerificationResult
    paths, registry, review = approval_case
    evidence = json.loads(paths["evidence_path"].read_bytes())
    evidence.update(verification=asdict(VerificationResult("failed", reason="Synthetic verification failure")),
                    evidence=[], proposals=[], proposal_status="insufficient_evidence",
                    reasons=["verification_failed"])
    paths["evidence_path"].write_text(json.dumps(evidence), encoding="utf-8")
    review = replace(review, evidence_record_sha256=record_sha256(evidence))
    write_reviews(paths, (review,))
    with pytest.raises(ValueError, match="verified current"):
        authorize(approval_case, review)
    assert load_approvals(registry).hashes == ()


def test_concurrent_registry_update_is_preserved(approval_case, monkeypatch):
    _, registry, _ = approval_case
    original = workflow.save_approvals
    def competing_write(*args, **kwargs):
        snapshot = load_approvals(registry)
        original(registry, (*snapshot.hashes, "a" * 64), expected_digest=snapshot.digest)
        return original(*args, **kwargs)
    monkeypatch.setattr(workflow, "save_approvals", competing_write)
    with pytest.raises(ValueError, match="stale"):
        authorize(approval_case)
    assert load_approvals(registry).hashes == ("a" * 64,)
