"""Eligible fixtures are synthetic, never independently reviewed real examples."""

import hashlib
import json
from dataclasses import asdict

import pytest

from scripts.inspect_training_readiness import main
from src.chessmind.ml.data.dataset_readiness import inspect_training_readiness, report_json
from src.chessmind.ml.features.extractor import extract_candidate_features
from src.chessmind.ml.labeling.adjudication import AdjudicationRecord, record_sha256
from src.chessmind.ml.labeling.rules import propose_labels
from src.chessmind.ml.labeling.schema import LabelRecord


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_bundle(tmp_path, case):
    c, _, v = case
    paths = {name: tmp_path / filename for name, filename in (
        ("candidate_path", "candidates.jsonl"), ("feature_path", "features.jsonl"),
        ("evidence_path", "evidence.jsonl"), ("source_dataset_path", "source.pgn"),
        ("source_manifest_path", "source.json"),
    )}
    paths["source_dataset_path"].write_bytes(b"synthetic fixture source; not a real PGN")
    write_json(paths["source_manifest_path"], {
        "sha256": digest(paths["source_dataset_path"]), "source_url": "fixture:archive",
        "accepted_count": 1, "source_references": [{"game_id": c["source_game_id"], "archive_game_index": 1}],
    })
    c["source_evidence"] = {
        "dataset_path": "fixture.pgn", "manifest_path": "fixture.json", "source_archive": "fixture:archive",
        "archive_game_index": 1, "dataset_sha256": digest(paths["source_dataset_path"]),
        "manifest_sha256": digest(paths["source_manifest_path"]),
    }
    f = json.loads(json.dumps(asdict(extract_candidate_features(c))))
    write_rows(paths["candidate_path"], [c])
    write_rows(paths["feature_path"], [f])
    proposals, evidence, status, reasons = propose_labels(c, f, v)
    e = json.loads(json.dumps(asdict(LabelRecord(
        c["candidate_id"], c["source_game_id"], c["player_identifier"], c["ply_index"],
        c["schema_version"], f["schema_version"], c["selection_type"], v, evidence, proposals,
        status, reasons, digest(paths["candidate_path"]), digest(paths["feature_path"]), record_sha256(f),
    ))))
    write_rows(paths["evidence_path"], [e])
    return paths, c, f, e


@pytest.fixture
def bundle(tmp_path, labeling_case):
    return make_bundle(tmp_path, labeling_case(before_pv=("c2b4",)))


def annotate(bundle, *, decision="positive", approved=True, state="finalized", concept="hanging_piece"):
    paths, c, f, e = bundle
    annotation = asdict(AdjudicationRecord(
        c["candidate_id"], concept, decision, "fixture:reviewer",
        "Synthetic test only, not an independently reviewed example.", "2026-10-20T00:00:00Z",
        record_sha256(c), record_sha256(f), record_sha256(e),
        digest(paths["source_dataset_path"]), digest(paths["source_manifest_path"]), state,
    ))
    paths["adjudications_path"] = paths["candidate_path"].with_name("reviews.jsonl")
    paths["approvals_path"] = paths["candidate_path"].with_name("approvals.json")
    write_rows(paths["adjudications_path"], [annotation])
    write_json(paths["approvals_path"], [record_sha256(annotation)] if approved else [])
    return annotation


def assert_invalid(paths, reason):
    report = inspect_training_readiness(**paths)
    assert not report["integrity_valid"] and not report["training_ready"]
    assert report["eligible_labels"]["total"] == 0
    assert report["blocking_reasons"] == ["artifact_integrity_failure"]
    assert reason in report["integrity_errors"][0]


def test_valid_zero_eligible_and_determinism(bundle):
    paths, *_ = bundle
    first = inspect_training_readiness(**paths)
    assert first["integrity_valid"] and not first["training_ready"]
    assert first["counts"]["engine_verified_candidates"] == 1
    assert first["counts"]["feature_complete_candidates"] == 1
    assert first["counts"]["provisional_proposals"] == 1
    assert first["eligible_labels"] == {"total": 0, "by_concept": {}}
    assert "no_adjudications" in first["blocking_reasons"]
    assert report_json(first) == report_json(inspect_training_readiness(**paths))


@pytest.mark.parametrize("decision", ["positive", "negative"])
def test_explicit_eligible_synthetic_labels(bundle, decision):
    paths, *_ = bundle
    annotate(bundle, decision=decision)
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and report["training_ready"]
    assert report["eligible_labels"]["total"] == 1
    assert report["eligible_labels"]["by_concept"] == {
        "hanging_piece": {"positive": int(decision == "positive"), "negative": int(decision == "negative")}}
    assert report["blocking_reasons"] == []


def test_negative_for_other_concept_does_not_label_unannotated_concepts(bundle):
    paths, *_ = bundle
    annotate(bundle, decision="negative", concept="missed_mate")
    report = inspect_training_readiness(**paths)
    assert report["eligible_labels"]["by_concept"] == {"missed_mate": {"positive": 0, "negative": 1}}


def test_requires_review_is_not_a_target_but_independent_review_can_adjudicate(tmp_path, labeling_case):
    import chess.engine
    bundle = make_bundle(tmp_path, labeling_case(
        fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Cp(500),
        before_pv=("g6g7",), after_pv=("h8h7",)))
    paths, *_ = bundle
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["counts"]["proposal_status"]["requires_review"] == 1
    assert report["counts"]["provisional_proposals"] == 0
    annotate(bundle, decision="negative", concept="missed_mate")
    report = inspect_training_readiness(**paths)
    assert report["training_ready"]
    assert report["eligible_labels"]["by_concept"] == {"missed_mate": {"positive": 0, "negative": 1}}


@pytest.mark.parametrize("options", [
    {"approved": False}, {"state": "draft"}, {"decision": "unresolved"},
])
def test_ordinary_ineligible_reviews(bundle, options):
    paths, *_ = bundle
    annotate(bundle, **options)
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["adjudication_status"]["ineligible"] == 1
    assert "no_eligible_labels" in report["blocking_reasons"]


@pytest.mark.parametrize("raw", ['{"candidate_id":', 'null\n', '{"candidate_id":"x","candidate_id":"y"}',
                                  '{"candidate_id":"x","bad":NaN}'])
def test_malformed_json(bundle, raw):
    paths, *_ = bundle
    paths["candidate_path"].write_text(raw, encoding="utf-8")
    assert_invalid(paths, "")


@pytest.mark.parametrize("artifact", ["candidate", "feature", "evidence"])
def test_duplicate_records(bundle, artifact):
    paths, c, f, e = bundle
    write_rows(paths[artifact + "_path"], [{"candidate": c, "feature": f, "evidence": e}[artifact]] * 2)
    assert_invalid(paths, "duplicate candidate ID")


@pytest.mark.parametrize("artifact", ["candidate", "feature"])
def test_missing_join(bundle, artifact):
    paths, *_ = bundle
    write_rows(paths[artifact + "_path"], [])
    assert_invalid(paths, "candidate/feature IDs differ")


def test_missing_evidence_is_normal_until_review_references_it(bundle):
    paths, *_ = bundle
    write_rows(paths["evidence_path"], [])
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and report["counts"]["candidates_without_evidence"] == 1
    annotate(bundle)
    assert_invalid(paths, "adjudication missing candidate/evidence")


@pytest.mark.parametrize("artifact,field,value,reason", [
    ("candidate", "schema_version", "future", "unsupported_candidate_schema"),
    ("feature", "schema_version", "future", "feature contents"),
    ("candidate", "candidate_id", "0" * 64, "candidate/feature IDs differ"),
    ("evidence", "schema_version", "future", "unsupported evidence schema"),
    ("evidence", "rule_version", "future", "unsupported evidence rule"),
    ("evidence", "source_game_id", "other", "evidence source_game_id mismatch"),
    ("evidence", "candidate_input_sha256", "0" * 64, "candidate_input_sha256 mismatch"),
    ("evidence", "feature_record_sha256", "0" * 64, "feature_record_sha256 mismatch"),
    ("evidence", "review_status", "approved", "review_status mismatch"),
])
def test_bad_schema_and_linkage(bundle, artifact, field, value, reason):
    paths, c, f, e = bundle
    row = {"candidate": c, "feature": f, "evidence": e}[artifact]
    row[field] = value
    write_rows(paths[artifact + "_path"], [row])
    assert_invalid(paths, reason)


def test_historical_evidence_never_eligible_even_with_approval(bundle):
    paths, _, _, e = bundle
    e["rule_version"] = "conservative_tactics_v1"
    for proposal in e["proposals"]:
        proposal["rule_version"] = e["rule_version"]
    write_rows(paths["evidence_path"], [e])
    annotate(bundle)
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and not report["training_ready"]
    assert "historical_evidence_ineligible" in report["blocking_reasons"]
    assert report["adjudication_status"]["approved"] == 1


def test_illegal_saved_verification(bundle):
    paths, _, _, e = bundle
    e["verification"]["after_pv"] = ["a1a8"]
    write_rows(paths["evidence_path"], [e])
    assert_invalid(paths, "illegal verification PV")


@pytest.mark.parametrize("field,value", [
    ("centipawn_loss", 0), ("classification", "Excellent"),
    ("significant_error_supported", False), ("significant_error_supported", 1),
])
def test_inconsistent_verification(bundle, field, value):
    paths, _, _, e = bundle
    e["verification"][field] = value
    write_rows(paths["evidence_path"], [e])
    assert_invalid(paths, "verification")


def test_missing_explicit_adjudication_file_is_invalid(bundle):
    paths, *_ = bundle
    paths["adjudications_path"] = paths["candidate_path"].with_name("missing.jsonl")
    assert_invalid(paths, "FileNotFoundError")


def test_failed_verification_is_valid_ineligible_data(bundle):
    from src.chessmind.ml.labeling.schema import VerificationResult
    paths, _, _, e = bundle
    e.update(verification=asdict(VerificationResult("failed", reason="synthetic failure")),
             evidence=[], proposals=[], proposal_status="insufficient_evidence",
             reasons=["verification_failed"])
    write_rows(paths["evidence_path"], [e])
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["counts"]["verification_status"] == {"failed": 1}


def test_tampered_tactical_evidence(bundle):
    paths, _, _, e = bundle
    e["evidence"][0]["reason"] = "invented reason"
    e["proposals"][0]["evidence"][0]["reason"] = "invented reason"
    write_rows(paths["evidence_path"], [e])
    assert_invalid(paths, "saved evidence disagrees")


@pytest.mark.parametrize("artifact", ["source_dataset", "source_manifest"])
def test_source_hash_tampering(bundle, artifact):
    paths, *_ = bundle
    with paths[artifact + "_path"].open("ab") as stream:
        stream.write(b" ")
    assert_invalid(paths, "hash mismatch" if artifact == "source_dataset" else "provenance mismatch")


def test_duplicate_source_references(bundle):
    paths, *_ = bundle
    source = json.loads(paths["source_manifest_path"].read_text())
    source["source_references"] *= 2
    write_json(paths["source_manifest_path"], source)
    assert_invalid(paths, "duplicate source game reference")


def test_conflicting_finalized_reviews(bundle):
    paths, *_ = bundle
    review = annotate(bundle)
    write_rows(paths["adjudications_path"], [review, {**review, "decision": "negative"}])
    assert_invalid(paths, "duplicate or conflicting")


def test_tampered_adjudication_reference(bundle):
    paths, *_ = bundle
    review = annotate(bundle)
    review["evidence_record_sha256"] = "0" * 64
    write_rows(paths["adjudications_path"], [review])
    assert_invalid(paths, "adjudication record hash mismatch")


@pytest.mark.parametrize("value", [True, {"approved": True}, ["bad"], ["a" * 64] * 2])
def test_malformed_approval_input(bundle, value):
    paths, *_ = bundle
    annotate(bundle)
    write_json(paths["approvals_path"], value)
    assert_invalid(paths, "")


def add_manifests(bundle):
    paths, *_ = bundle
    for name, manifest in (
        ("candidate", {"schema_version": "candidate_moves_v1", "candidate_count": 1,
            "input_sha256": digest(paths["source_dataset_path"]),
            "source_manifest_sha256": digest(paths["source_manifest_path"]),
            "output_sha256": digest(paths["candidate_path"])}),
        ("feature", {"schema_version": "chess_features_v1", "accepted": 1, "feature_value_count": 163,
            "input_sha256": digest(paths["candidate_path"]), "output_sha256": digest(paths["feature_path"])}),
        ("labeling", {"schema_version": "mistake_proposals_v1", "processed": 1,
            "rule_version": "conservative_tactics_v2", "candidate_sha256": digest(paths["candidate_path"]),
            "feature_sha256": digest(paths["feature_path"]), "output_sha256": digest(paths["evidence_path"])}),
    ):
        path = paths["candidate_path"].with_name(name + "_manifest.json")
        write_json(path, manifest)
        paths[name + "_manifest_path"] = path


@pytest.mark.parametrize("manifest,key,value", [
    ("candidate", "output_sha256", "0" * 64), ("candidate", "candidate_count", 2),
    ("feature", "feature_value_count", 162), ("feature", "input_sha256", "0" * 64),
    ("labeling", "schema_version", "future"), ("labeling", "processed", 5),
])
def test_manifest_mismatch(bundle, manifest, key, value):
    paths, *_ = bundle
    add_manifests(bundle)
    path = paths[manifest + "_manifest_path"]
    obj = json.loads(path.read_text())
    obj[key] = value
    write_json(path, obj)
    assert_invalid(paths, key + " mismatch")


def test_manifests_valid_and_auto_discovered(bundle):
    paths, *_ = bundle
    add_manifests(bundle)
    for name in ("candidate", "feature", "labeling"):
        del paths[name + "_manifest_path"]
    assert inspect_training_readiness(**paths)["integrity_valid"]


def test_empty_collection(bundle):
    paths, *_ = bundle
    for name in ("candidate", "feature", "evidence"):
        write_rows(paths[name + "_path"], [])
    report = inspect_training_readiness(**paths)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["counts"]["candidates"] == 0


def test_cli_read_only_and_valid_zero_exit(bundle, capsys):
    paths, *_ = bundle
    before = {p: p.read_bytes() for p in paths["candidate_path"].parent.iterdir()}
    args = [part for key, path in paths.items() for part in ("--" + key.replace("_", "-"), str(path))]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["integrity_valid"] and not report["training_ready"]
    assert before == {p: p.read_bytes() for p in paths["candidate_path"].parent.iterdir()}
    paths["feature_path"].unlink()
    assert main(args) == 2
    assert not json.loads(capsys.readouterr().out)["integrity_valid"]


def test_cli_unexpected_failure(bundle, monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("synthetic unexpected failure")
    monkeypatch.setattr("scripts.inspect_training_readiness.inspect_training_readiness", fail)
    assert main([]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "unexpected_failure"
