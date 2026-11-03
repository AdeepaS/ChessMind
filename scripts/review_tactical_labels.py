"""Human tactical reviews. No engine searches, implicit decisions, or approvals."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Support both repository-root module execution and the documented direct script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess

from src.chessmind.ml.data.dataset_readiness import inspect_training_readiness, _json
from src.chessmind.ml.labeling.adjudication import record_sha256
from src.chessmind.ml.labeling.review_store import load_snapshot, save_reviews
from src.chessmind.ml.labeling.review_workflow import (
    prepare_review, record_review, finalize_review, validate_review,
)
from src.chessmind.ml.labeling.taxonomy import MistakeConcept, MistakeCategory, category_for_concept


DEFAULTS = {
    "candidate_path": "data/processed/ml/candidate_moves.jsonl",
    "feature_path": "data/processed/ml/candidate_features.jsonl",
    "evidence_path": "data/labeled/ml/mistake_label_proposals.jsonl",
    "source_dataset_path": "data/raw/ml/games/lichess_game_sample.pgn",
    "source_manifest_path": "data/raw/ml/games/source_manifest.json",
    "candidate_manifest_path": None, "feature_manifest_path": None,
    "labeling_manifest_path": None,
}
NOTICE = ("Engine concepts are provisional, not ground truth. Finalized reviews are not "
          "independently approved. Historical v1 evidence is training-ineligible. A mate-to-CP "
          "transition alone does not prove a missed mate. Finite search and geometric features "
          "have limitations; examine alternatives and source context. No approvals are loaded or created.")


def _paths(args):
    paths = {name: getattr(args, name) for name in DEFAULTS}
    actual = {"candidates": paths["candidate_path"], "features": paths["feature_path"],
              "evidence": paths["evidence_path"], "source_dataset": paths["source_dataset_path"],
              "source_manifest": paths["source_manifest_path"]}
    for kind, base in (("candidate", "candidates"), ("feature", "features"), ("labeling", "evidence")):
        name = kind + "_manifest"
        supplied = paths[name + "_path"]
        adjacent = actual[base].with_name(name + ".json")
        if supplied is not None or adjacent.exists():
            actual[name] = supplied or adjacent
    return paths, actual


def _check_store_path(args, actual):
    if args.store is not None:
        store = args.store.resolve()
        lock = store.with_name(store.name + ".lock")
        if any(p.resolve() in (store, lock) for p in actual.values()):
            raise ValueError("store/lock must be distinct from source artifacts")


def _bundle(args):
    paths, actual = _paths(args)
    report = inspect_training_readiness(**paths)
    if not report["integrity_valid"]:
        raise ValueError("; ".join(report["integrity_errors"]))
    if "quality_report" in report["artifact_sha256"]:
        actual["quality_report"] = actual["labeling_manifest"].with_name("label_quality_report.json")
    _check_store_path(args, actual)
    data = {name: path.read_bytes() for name, path in actual.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    if hashes != report["artifact_sha256"]:
        raise ValueError("artifacts changed during loading")
    rows = {}
    for name in ("candidates", "features", "evidence"):
        values = [_json(line) for line in data[name].decode("utf-8").splitlines() if line.strip()]
        rows[name] = {row["candidate_id"]: row for row in values}
    return rows, data, hashes


def _artifacts(identity, rows, data):
    if identity not in rows["candidates"]:
        raise ValueError("unknown candidate ID")
    if identity not in rows["evidence"]:
        raise ValueError("candidate has no saved evidence; review unavailable")
    return dict(candidate=rows["candidates"][identity], feature=rows["features"][identity],
                evidence=rows["evidence"][identity], source_dataset=data["source_dataset"],
                source_manifest=data["source_manifest"])


def _reviews_for(identity, reviews):
    return [{"review_hash": record_sha256(asdict(r)), **asdict(r)}
            for r in reviews if r.candidate_id == identity]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="explicitly create an empty review store")
    init.add_argument("--store", type=Path, required=True)
    for name in ("list", "show", "review", "finalize"):
        sub = commands.add_parser(name)
        for key, default in DEFAULTS.items():
            sub.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
        sub.add_argument("--store", type=Path, required=name in ("review", "finalize"))
        if name != "list":
            sub.add_argument("--candidate-id", required=True)
        if name == "review":
            tactical = [c.value for c in MistakeConcept
                        if category_for_concept(c) == MistakeCategory.TACTICAL]
            sub.add_argument("--concept", required=True, choices=tactical)
            sub.add_argument("--decision", required=True, choices=("positive", "negative", "unresolved"))
            sub.add_argument("--reviewer-id", required=True)
            sub.add_argument("--rationale", required=True)
        if name == "finalize":
            sub.add_argument("--review-hash", required=True, help="exact draft hash from list/show/review")
    return parser


def _run(args):
    if args.command == "init":
        save_reviews(args.store, (), expected_digest=None)
        print("Initialized empty review store.")
        return
    rows, data, hashes = _bundle(args)
    snapshot = load_snapshot(args.store) if args.store is not None else None
    reviews = snapshot.reviews if snapshot else ()
    # Validate the complete store against current evidence, not only the selected row.
    for review in reviews:
        validate_review(review, **_artifacts(review.candidate_id, rows, data))
    if args.command == "list":
        result = []
        for identity, candidate in rows["candidates"].items():
            evidence = rows["evidence"].get(identity)
            human = _reviews_for(identity, reviews)
            statuses = sorted({r["review_state"] + ":" + r["decision"] for r in human})
            result.append(dict(candidate_id=identity, game=candidate["source_game_id"],
                move=chess.Board(candidate["fen_before"]).san(chess.Move.from_uci(candidate["played_move_uci"])),
                review_status=", ".join(statuses) if human else "unreviewed",
                evidence_status=evidence["proposal_status"] if evidence else "missing",
                rule_version=evidence["rule_version"] if evidence else None,
                proposed_concepts=[p["concept"] for p in evidence["proposals"] if p["status"] == "proposed"]
                                  if evidence else [], human_reviews=human))
        print(json.dumps({"notice": NOTICE, "candidates": result}, indent=2, sort_keys=True))
        return
    artifacts = _artifacts(args.candidate_id, rows, data)
    if args.command == "show":
        candidate = artifacts["candidate"]
        print("Before:\n" + str(chess.Board(candidate["fen_before"])))
        print("After:\n" + str(chess.Board(candidate["fen_after"])))
        print(json.dumps(dict(notice=NOTICE, candidate=candidate, evidence=artifacts["evidence"],
            artifact_sha256=hashes, record_sha256={k: record_sha256(artifacts[k])
                for k in ("candidate", "feature", "evidence")},
            human_reviews=_reviews_for(args.candidate_id, reviews)), indent=2, sort_keys=True))
        return
    if args.command == "review":
        draft = prepare_review(candidate_id=args.candidate_id, concept=args.concept,
            decision=args.decision, reviewer_id=args.reviewer_id, rationale=args.rationale, **artifacts)
        # A retried CLI submission has a new timestamp. Reject an otherwise identical
        # submission without changing Stage A's permission for distinct drafts.
        signature = lambda r: {k: v for k, v in asdict(r).items() if k not in ("reviewed_at", "review_state")}
        if any(signature(r) == signature(draft) for r in reviews):
            raise ValueError("duplicate submission (same decision, reviewer, rationale and evidence)")
        updated = record_review(draft, reviews=reviews, **artifacts)
        selected = draft
    else:
        matches = [r for r in reviews if r.candidate_id == args.candidate_id
                   and record_sha256(asdict(r)) == args.review_hash]
        if len(matches) != 1:
            raise ValueError("exact submitted draft not found")
        selected = matches[0]
        updated = finalize_review(selected, reviews=reviews, **artifacts)
        selected = updated[reviews.index(selected)]
    # Recheck all upstream inputs before committing; artifacts must remain immutable
    # during review operations (they do not participate in the store lock protocol).
    if _bundle(args)[2] != hashes:
        raise ValueError("artifacts changed before persistence")
    save_reviews(args.store, updated, expected_digest=snapshot.digest)
    print(json.dumps(dict(review_hash=record_sha256(asdict(selected)), review=asdict(selected),
                          notice=NOTICE), indent=2, sort_keys=True))


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        _run(args)
        return 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(f"Review error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected review failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
