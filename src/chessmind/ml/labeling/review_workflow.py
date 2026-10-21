"""In-memory human review operations; no I/O, searches, or approval authority.

Pass the complete review collection to recording/finalization. Returned tuples
replace that collection only on success; inputs are never mutated. Source bytes
are checked, but PGN positions are not re-proven. Upstream candidate/feature file
hashes remain references (complete files are not supplied to this record API).
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone

from ..data.dataset_readiness import _evidence, _hash, _json, _require
from .adjudication import (
    AdjudicationRecord, _digest, _text, record_sha256, validate_adjudications,
)
from .pipeline import _validate_candidate
from .taxonomy import MistakeCategory, category_for_concept


def _bindings(*, candidate, feature, evidence, source_dataset, source_manifest):
    """Reuse readiness record validation, independently of training eligibility."""
    try:
        _require(all(type(row) is dict for row in (candidate, feature, evidence)),
                 "candidate, feature and evidence must be dictionaries")
        _require(type(source_dataset) is bytes and type(source_manifest) is bytes,
                 "source dataset and manifest must be bytes")
        _require(record_sha256(_validate_candidate(candidate)) == record_sha256(feature),
                 "candidate/feature mismatch")
        for key in ("candidate_input_sha256", "feature_input_sha256"):
            _digest(evidence[key])
        _evidence(evidence, candidate, feature, evidence["candidate_input_sha256"],
                  evidence["feature_input_sha256"])
        dataset_hash, manifest_hash = _hash(source_dataset), _hash(source_manifest)
        manifest = _json(source_manifest)
        _require(type(manifest) is dict and manifest["sha256"] == dataset_hash,
                 "source dataset hash mismatch")
        provenance = candidate["source_evidence"]
        for key in ("dataset_path", "manifest_path", "source_archive"):
            _text(provenance[key])
        _require(provenance["dataset_sha256"] == dataset_hash
                 and provenance["manifest_sha256"] == manifest_hash
                 and provenance["source_archive"] == manifest["source_url"],
                 "source provenance mismatch")
        refs = manifest["source_references"]
        _require(type(refs) is list, "source references must be a list")
        games, indices = {}, set()
        for ref in refs:
            _require(type(ref) is dict and type(ref.get("game_id")) is str
                     and bool(ref["game_id"]) and type(ref.get("archive_game_index")) is int
                     and ref["archive_game_index"] > 0, "invalid source reference")
            _require(ref["game_id"] not in games and ref["archive_game_index"] not in indices,
                     "duplicate source reference")
            games[ref["game_id"]] = ref["archive_game_index"]
            indices.add(ref["archive_game_index"])
        if "accepted_count" in manifest:
            _require(type(manifest["accepted_count"]) is int
                     and manifest["accepted_count"] == len(refs), "source count mismatch")
        _require(type(provenance["archive_game_index"]) is int
                 and candidate["source_game_id"] in games
                 and games[candidate["source_game_id"]] == provenance["archive_game_index"],
                 "source game/index mismatch")
        return dict(candidate_id=candidate["candidate_id"],
                    candidate_record_sha256=record_sha256(candidate),
                    feature_record_sha256=record_sha256(feature),
                    evidence_record_sha256=record_sha256(evidence),
                    source_dataset_sha256=dataset_hash,
                    source_manifest_sha256=manifest_hash)
    except (TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed review artifacts") from exc


def _tactical(concept):
    _require(isinstance(concept, str)
             and category_for_concept(concept) == MistakeCategory.TACTICAL,
             "expected an existing tactical concept")


def _collection(reviews):
    records = validate_adjudications(reviews)
    hashes = [record_sha256(asdict(record)) for record in records]
    _require(len(hashes) == len(set(hashes)), "duplicate submission")
    return records


def prepare_review(*, candidate_id, concept, decision, reviewer_id, rationale,
                   candidate, feature, evidence, source_dataset, source_manifest):
    """Create a draft for an explicitly supplied human decision, timestamped now.

    This prepares a record only; use record_review to submit it to a collection.
    All taxonomy tactical concepts are accepted, without asserting detector support.
    Historical/failed evidence may be reviewed but gains no training eligibility.
    """
    _tactical(concept)
    bindings = _bindings(candidate=candidate, feature=feature, evidence=evidence,
                         source_dataset=source_dataset, source_manifest=source_manifest)
    _require(candidate_id == bindings["candidate_id"], "selected candidate mismatch")
    return AdjudicationRecord(**bindings, concept=concept, decision=decision,
        reviewer_id=reviewer_id, rationale=rationale,
        reviewed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


def validate_review(review, *, candidate, feature, evidence, source_dataset, source_manifest):
    """Return the unchanged record after schema and exact evidence-binding checks."""
    validate_adjudications((review,))
    _tactical(review.concept)
    bindings = _bindings(candidate=candidate, feature=feature, evidence=evidence,
                         source_dataset=source_dataset, source_manifest=source_manifest)
    _require(all(getattr(review, key) == value for key, value in bindings.items()),
             "stale or mismatched review bindings")
    return review


def record_review(review, *, reviews, **artifacts):
    """Return a new complete collection with one validated submission appended.

    Exact duplicate records are rejected, including drafts. Distinct drafts may
    coexist, as in the existing adjudication contract. No review is approved here.
    Other collection members receive schema/conflict checks, not artifact checks;
    use dataset readiness for whole-artifact-collection validation.
    """
    records = _collection(reviews)
    validate_review(review, **artifacts)
    return _collection((*records, review))


def finalize_review(review, *, reviews, **artifacts):
    """Explicitly replace a submitted draft in a new tuple, without approval.

    Preserve the original human decision timestamp. The exact draft must occur
    once in the complete collection. Stale evidence or finalized conflicts fail
    before any replacement is returned. Unresolved finalizations stay ineligible.
    """
    records = _collection(reviews)
    validate_review(review, **artifacts)
    _require(review.review_state == "draft", "only drafts can be finalized")
    _require(review in records, "draft must be submitted before finalization")
    finalized = replace(review, review_state="finalized")
    return _collection(tuple(finalized if row == review else row for row in records))
