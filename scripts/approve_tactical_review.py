"""Trusted-operator approval of one finalized tactical review; no authentication service."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_tactical_labels import DEFAULTS
from src.chessmind.ml.labeling.approval_store import load_approvals, save_approvals
from src.chessmind.ml.labeling.approval_workflow import load_approval_context, approve_review


NOTICE = ("Operator identity and justification are assertions, not authentication. Use a separately "
          "protected registry and independent authorization procedure. Registry hashes alone do not "
          "establish current eligibility, class coverage, or leakage-safe training readiness.")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "list", "inspect", "approve"):
        sub = commands.add_parser(name)
        if name != "inspect":
            sub.add_argument("--approvals-path", type=Path, required=True)
        if name in ("inspect", "approve"):
            sub.add_argument("--adjudications-path", type=Path, required=True)
            sub.add_argument("--review-hash", required=True)
            for key, default in DEFAULTS.items():
                sub.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
        if name == "approve":
            sub.add_argument("--confirm-independent-approval", action="store_true", required=True)
            sub.add_argument("--operator-id", required=True)
            sub.add_argument("--independent-justification", required=True,
                             help="independent basis for this decision; address any ambiguous/conflicting evidence")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            save_approvals(args.approvals_path, (), expected_digest=None)
            result = {"initialized": str(args.approvals_path)}
        elif args.command == "list":
            snapshot = load_approvals(args.approvals_path)
            result = {"approved_hashes": snapshot.hashes, "registry_sha256": snapshot.digest,
                      "eligibility": "not assessed; run dataset readiness with complete artifacts"}
        else:
            paths = {key: getattr(args, key) for key in DEFAULTS}
            paths["adjudications_path"] = args.adjudications_path
            if args.command == "inspect":
                context = load_approval_context(args.review_hash, **paths)
                result = dict(review_hash=context.review_hash, review=asdict(context.review),
                    related_reviews=[asdict(r) for r in context.reviews
                        if r.candidate_id == context.review.candidate_id and r.concept == context.review.concept],
                    candidate=context.artifacts["candidate"], evidence=context.artifacts["evidence"],
                    artifact_sha256=context.input_hashes, approval_issued=False)
            else:
                result = approve_review(args.review_hash, approvals_path=args.approvals_path,
                    confirm=args.confirm_independent_approval, operator_id=args.operator_id,
                    independent_justification=args.independent_justification, **paths)
        print(json.dumps({"notice": NOTICE, **result}, indent=2, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(f"Approval error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected approval failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
