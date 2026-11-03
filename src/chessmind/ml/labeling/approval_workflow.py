"""Issue one exact-hash authorization after explicit independent operator action.

Identity and justification are assertions, not authentication or chess proof.
External authorization and a protected registry establish the trust boundary.
"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from ..data.dataset_readiness import inspect_training_readiness, _json
from .adjudication import AdjudicationRecord, _digest, _text, is_training_eligible, record_sha256
from .approval_store import load_approvals, save_approvals
from .review_store import decode_reviews
from .review_workflow import validate_review
from .schema import RULE_VERSION


@dataclass(frozen=True)
class ApprovalContext:
    review_hash: str
    review: AdjudicationRecord
    reviews: tuple
    artifacts: dict
    input_hashes: dict
    input_paths: dict


def load_approval_context(review_hash, *, adjudications_path, **artifact_paths):
    """Read and validate the complete bundle, without authorizing or writing.

    Only artifact options accepted by the readiness validator may be supplied;
    approvals are deliberately not accepted as part of the evidence bundle.
    """
    _digest(review_hash)
    if "approvals_path" in artifact_paths:
        raise ValueError("approvals are separate from review evidence")
    paths = {name: Path(artifact_paths[key]) for name, key in (
        ("candidates", "candidate_path"), ("features", "feature_path"),
        ("evidence", "evidence_path"), ("source_dataset", "source_dataset_path"),
        ("source_manifest", "source_manifest_path"))}
    paths["adjudications"] = Path(adjudications_path)
    for kind, base in (("candidate", "candidates"), ("feature", "features"), ("labeling", "evidence")):
        name = kind + "_manifest"
        supplied = artifact_paths.get(name + "_path")
        adjacent = paths[base].with_name(name + ".json")
        if supplied is not None or adjacent.exists():
            paths[name] = Path(supplied) if supplied is not None else adjacent
    report = inspect_training_readiness(**artifact_paths, adjudications_path=adjudications_path)
    if not report["integrity_valid"]:
        raise ValueError("; ".join(report["integrity_errors"]))
    if "quality_report" in report["artifact_sha256"]:
        paths["quality_report"] = paths["labeling_manifest"].with_name("label_quality_report.json")
    data = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            data[name] = stream.read(32 * 1024 * 1024 + 1)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    if hashes != report["artifact_sha256"]:
        raise ValueError("inputs changed during approval inspection")
    reviews = decode_reviews(data["adjudications"])
    records = {name: {row["candidate_id"]: row for row in
                     (_json(line) for line in data[name].decode("utf-8").splitlines() if line.strip())}
               for name in ("candidates", "features", "evidence")}
    bundles = {}
    for review in reviews:
        identity = review.candidate_id
        bundle = dict(candidate=records["candidates"][identity], feature=records["features"][identity],
                      evidence=records["evidence"][identity], source_dataset=data["source_dataset"],
                      source_manifest=data["source_manifest"])
        validate_review(review, **bundle)
        bundles[identity] = bundle
    matches = [r for r in reviews if record_sha256(asdict(r)) == review_hash]
    if len(matches) != 1:
        raise ValueError("exact review hash not found; review may have changed")
    review = matches[0]
    return ApprovalContext(review_hash, review, reviews, bundles[review.candidate_id], hashes, paths)


def validate_approval(context, *, confirm=False, operator_id=None, independent_justification=None):
    """Validate explicit authorization; does not persist or modify any record.

    For abstentions (including ambiguous mate-to-CP), the operator must justify
    the human decision independently. Nonempty text is mandatory but cannot prove
    its truth. A different identifier prevents obvious self-approval, not aliases.
    """
    if confirm is not True:
        raise ValueError("explicit independent approval confirmation required")
    _text(operator_id)
    _text(independent_justification)
    review = context.review
    if operator_id.casefold() == review.reviewer_id.casefold():
        raise ValueError("independent operator must differ from the annotating reviewer")
    if record_sha256(asdict(review)) != context.review_hash:
        raise ValueError("review hash changed")
    if review.review_state != "finalized":
        raise ValueError("only finalized reviews can be approved")
    if review.decision not in ("positive", "negative"):
        raise ValueError("unresolved decisions cannot be approved")
    for other in context.reviews:
        if ((other.candidate_id, other.concept) == (review.candidate_id, review.concept)
                and other.decision != review.decision):
            raise ValueError("contradictory or unresolved review requires human attention")
    validate_review(review, **context.artifacts)
    evidence = context.artifacts["evidence"]
    if evidence["rule_version"] != RULE_VERSION or evidence["verification"]["status"] != "verified":
        raise ValueError("approval requires verified current conservative_tactics_v2 evidence")
    # A hypothetical singleton checks eligibility for this explicit request only;
    # it does not create an approval registry or silently authorize other reviews.
    if not is_training_eligible(review, reviews=context.reviews,
            approved_review_hashes=frozenset({context.review_hash}), **context.artifacts):
        raise ValueError("review fails existing training-label eligibility checks")
    return context.review_hash


def approve_review(review_hash, *, approvals_path, adjudications_path, confirm=False,
                   operator_id=None, independent_justification=None, **artifact_paths):
    """Validate, recheck, and append precisely one authorization to an existing store."""
    context = load_approval_context(review_hash, adjudications_path=adjudications_path, **artifact_paths)
    action = dict(confirm=confirm, operator_id=operator_id,
                  independent_justification=independent_justification)
    validate_approval(context, **action)
    destination = Path(approvals_path).resolve()
    lock = destination.with_name(destination.name + ".lock")
    for path in context.input_paths.values():
        resolved = path.resolve()
        review_lock = resolved.with_name(resolved.name + ".lock")
        if destination in (resolved, review_lock) or lock == resolved:
            raise ValueError("approval registry/lock must be separate from reviews and artifacts")
        if destination.exists() and destination.samefile(path):
            raise ValueError("approval registry aliases an input artifact")
    snapshot = load_approvals(destination)
    if review_hash in snapshot.hashes:
        raise ValueError("review already approved")

    def recheck():
        current = load_approval_context(review_hash, adjudications_path=adjudications_path, **artifact_paths)
        if current.input_hashes != context.input_hashes:
            raise ValueError("review or evidence changed before approval persistence")
        validate_approval(current, **action)

    save_approvals(destination, (*snapshot.hashes, review_hash),
                   expected_digest=snapshot.digest, before_replace=recheck)
    return dict(approved_review_hash=review_hash, decision=context.review.decision,
                concept=context.review.concept, operator_id=operator_id,
                independent_justification=independent_justification)
