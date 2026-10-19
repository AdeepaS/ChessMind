"""Human concept decisions; no inference from rule proposals or absent labels."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime

import chess

from ..features.pipeline import _invalid_constant, _unique_object
from .engine_verification import _result, replay_pv
from .pipeline import _validate_candidate
from .schema import LABEL_SCHEMA_VERSION, RULE_VERSION
from .taxonomy import MistakeConcept


ADJUDICATION_SCHEMA_VERSION = "human_adjudication_v1"
ELIGIBILITY_VERSION = "adjudicated_concepts_v1"


def _text(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("expected nonempty trimmed text")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256")


def record_sha256(record):
    """Hash canonical JSON, not the bytes of a possibly differently formatted file."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdjudicationRecord:
    candidate_id: str
    concept: str
    decision: str
    reviewer_id: str
    rationale: str
    reviewed_at: str
    candidate_record_sha256: str
    feature_record_sha256: str
    evidence_record_sha256: str
    source_dataset_sha256: str
    source_manifest_sha256: str
    review_state: str = "draft"
    schema_version: str = ADJUDICATION_SCHEMA_VERSION
    eligibility_version: str = ELIGIBILITY_VERSION

    def __post_init__(self):
        for name in ("candidate_id", "candidate_record_sha256", "feature_record_sha256",
                     "evidence_record_sha256", "source_dataset_sha256", "source_manifest_sha256"):
            _digest(getattr(self, name))
        if not isinstance(self.concept, str):
            raise ValueError("invalid concept")
        MistakeConcept(self.concept)
        if self.decision not in ("positive", "negative", "unresolved"):
            raise ValueError("invalid adjudication decision")
        if self.review_state not in ("draft", "finalized"):
            raise ValueError("invalid review state")
        if (self.schema_version != ADJUDICATION_SCHEMA_VERSION
                or self.eligibility_version != ELIGIBILITY_VERSION):
            raise ValueError("unsupported adjudication version")
        for value in (self.reviewer_id, self.rationale, self.reviewed_at):
            _text(value)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", self.reviewed_at) is None:
            raise ValueError("review timestamp must be UTC ISO-8601 ending in Z")
        datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))


def validate_adjudications(records):
    """Validate a complete review collection; no implicit duplicate resolution."""
    result, finalized = [], set()
    for record in records:
        if type(record) is not AdjudicationRecord:
            raise ValueError("expected AdjudicationRecord, not a rule proposal")
        # Revalidate even instances reconstructed by unconventional deserializers.
        record.__post_init__()
        key = (record.candidate_id, record.concept)
        if record.review_state == "finalized":
            if key in finalized:
                raise ValueError("duplicate or conflicting finalized candidate/concept review")
            finalized.add(key)
        result.append(record)
    return tuple(result)


def is_training_eligible(annotation, *, reviews, candidate, feature, evidence,
                         source_dataset, source_manifest, approved_review_hashes=frozenset()):
    """Fail closed. `reviews` must be the complete collection being considered.

    Approval hashes MUST come from a trusted independent-review authority outside
    the submitted records. A hash proves integrity, not human identity or expertise.
    This pure predicate performs no I/O, engine analysis, target creation or writes.
    """
    try:
        collection = validate_adjudications(reviews)
        if type(annotation) is not AdjudicationRecord or annotation not in collection:
            return False
        if annotation.review_state != "finalized" or annotation.decision == "unresolved":
            return False
        if type(approved_review_hashes) is not frozenset:
            return False
        for digest in approved_review_hashes:
            _digest(digest)
        if record_sha256(asdict(annotation)) not in approved_review_hashes:
            return False
        if any(type(value) is not dict for value in (candidate, feature, evidence)):
            return False
        expected = _validate_candidate(candidate)
        if record_sha256(expected) != record_sha256(feature):
            return False
        for record, digest in ((candidate, annotation.candidate_record_sha256),
                               (feature, annotation.feature_record_sha256),
                               (evidence, annotation.evidence_record_sha256)):
            if record_sha256(record) != digest or record["candidate_id"] != annotation.candidate_id:
                return False
        if (evidence["schema_version"] != LABEL_SCHEMA_VERSION
                or evidence["rule_version"] != RULE_VERSION
                or evidence["candidate_schema_version"] != candidate["schema_version"]
                or evidence["feature_schema_version"] != feature["schema_version"]
                or evidence["feature_record_sha256"] != annotation.feature_record_sha256):
            return False
        for key in ("source_game_id", "player_identifier", "ply_index"):
            if evidence[key] != candidate[key]:
                return False
        for key in ("candidate_input_sha256", "feature_input_sha256"):
            _digest(evidence[key])
        if evidence["proposal_status"] not in (
                "proposed", "requires_review", "insufficient_evidence", "conflicting_evidence"):
            return False
        if evidence["review_status"] != "unreviewed":
            return False
        if evidence["verification"]["status"] != "verified" or not evidence["evidence"]:
            return False
        for endpoint in ("before", "after"):
            verification = evidence["verification"]
            board = chess.Board(candidate["fen_" + endpoint])
            pv = verification[endpoint + "_pv"]
            depth = verification[endpoint + "_depth"]
            if type(pv) is not list or type(depth) is not int or depth < 0:
                return False
            # Reuse endpoint validation, including terminal Mate(0) consistency.
            _result(board, {"evaluation": verification[endpoint], "depth": depth,
                           "pv": [chess.Move.from_uci(uci) for uci in pv]}, 1)
        if type(evidence["evidence"]) is not list:
            return False
        for item in evidence["evidence"]:
            if type(item) is not dict or type(item.get("supported")) is not bool:
                return False
            _text(item["evidence_type"])
            _text(item["reason"])
            endpoint = {"missed_forced_mate": "before",
                        "allowed_immediate_tactical_capture": "after"}.get(item["evidence_type"])
            if endpoint is None or type(item.get("continuation")) is not list:
                return False
            replay_pv(chess.Board(candidate["fen_" + endpoint]), item["continuation"])
        # Source integrity is checked against actual bytes, never a supplied flag.
        if type(source_dataset) is not bytes or type(source_manifest) is not bytes:
            return False
        if (hashlib.sha256(source_dataset).hexdigest() != annotation.source_dataset_sha256
                or hashlib.sha256(source_manifest).hexdigest() != annotation.source_manifest_sha256):
            return False
        provenance = candidate["source_evidence"]
        for key in ("dataset_path", "manifest_path", "source_archive"):
            _text(provenance[key])
        if type(provenance["archive_game_index"]) is not int or provenance["archive_game_index"] < 1:
            return False
        if (provenance["dataset_sha256"] != annotation.source_dataset_sha256
                or provenance["manifest_sha256"] != annotation.source_manifest_sha256):
            return False
        manifest = json.loads(source_manifest, object_pairs_hook=_unique_object,
                              parse_constant=_invalid_constant)
        if manifest["sha256"] != annotation.source_dataset_sha256:
            return False
        if manifest["source_url"] != provenance["source_archive"]:
            return False
        if {"game_id": candidate["source_game_id"], "archive_game_index": provenance["archive_game_index"]} not in manifest["source_references"]:
            return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return False
