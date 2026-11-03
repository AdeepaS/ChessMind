"""CLI exercises use temporary synthetic artifacts, never production decisions."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from scripts import review_tactical_labels as cli
from src.chessmind.ml.data.dataset_readiness import inspect_training_readiness
from src.chessmind.ml.labeling.adjudication import record_sha256
from src.chessmind.ml.labeling.review_store import load_reviews
from src.chessmind.ml.labeling import review_store
from tests.ml.test_dataset_readiness import make_bundle


@pytest.fixture
def setup(tmp_path, labeling_case):
    paths, candidate, feature, evidence = make_bundle(tmp_path, labeling_case(before_pv=("c2b4",)))
    store = tmp_path / "reviews.jsonl"
    assert cli.main(["init", "--store", str(store)]) == 0
    options = ["--store", str(store)]
    for name, path in paths.items():
        options.extend(["--" + name.replace("_", "-"), str(path)])
    identity = ["--candidate-id", candidate["candidate_id"]]
    decision = ["--concept", "hanging_piece", "--decision", "unresolved",
                "--reviewer-id", "fixture:human", "--rationale", "Synthetic uncertain evidence."]
    return paths, store, options, identity, decision


def test_read_only_list_and_show(setup, capsys):
    paths, store, options, identity, _ = setup
    capsys.readouterr()
    before = {p: p.read_bytes() for p in [store, *paths.values()]}
    assert cli.main(["list", *options]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["candidates"][0]["review_status"] == "unreviewed"
    assert listing["candidates"][0]["human_reviews"] == []
    assert cli.main(["show", *options, *identity]) == 0
    shown = capsys.readouterr().out
    assert "Before:" in shown and "artifact_sha256" in shown and "not ground truth" in shown
    assert all(p.read_bytes() == value for p, value in before.items())


def test_submit_finalize_no_approval_and_readiness_unchanged(setup, capsys):
    paths, store, options, identity, decision = setup
    assert cli.main(["review", *options, *identity, *decision]) == 0
    review, = load_reviews(store)
    assert review.review_state == "draft" and review.decision == "unresolved"
    digest = record_sha256(asdict(review))
    assert cli.main(["list", *options]) == 0
    assert '"decision": "unresolved"' in capsys.readouterr().out
    assert cli.main(["finalize", *options, *identity, "--review-hash", digest]) == 0
    final, = load_reviews(store)
    assert final.review_state == "finalized" and final.reviewed_at == review.reviewed_at
    assert final.decision == "unresolved"
    report = inspect_training_readiness(**paths, adjudications_path=store)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["adjudication_status"]["approved"] == 0
    assert report["eligible_labels"]["total"] == 0
    before = store.read_bytes()
    assert cli.main(["finalize", *options, *identity, "--review-hash", digest]) == 2
    assert cli.main(["finalize", *options, *identity, "--review-hash",
                     record_sha256(asdict(final))]) == 2
    assert store.read_bytes() == before


@pytest.mark.parametrize("flag", ["--reviewer-id", "--rationale", "--decision", "--concept"])
def test_required_human_input(setup, flag):
    _, store, options, identity, decision = setup
    index = decision.index(flag)
    del decision[index:index + 2]
    with pytest.raises(SystemExit) as exc:
        cli.main(["review", *options, *identity, *decision])
    assert exc.value.code == 2 and store.read_bytes() == b""


@pytest.mark.parametrize("flag,value", [("--concept", "king_safety"), ("--concept", "invented"),
    ("--decision", "approve"), ("--reviewer-id", ""), ("--rationale", " ")])
def test_invalid_inputs(setup, flag, value):
    _, store, options, identity, decision = setup
    decision[decision.index(flag) + 1] = value
    try:
        result = cli.main(["review", *options, *identity, *decision])
    except SystemExit as exc:
        result = exc.code
    assert result == 2 and store.read_bytes() == b""


def test_unknown_candidate_duplicate_and_missing_draft(setup):
    _, store, options, identity, decision = setup
    assert cli.main(["show", *options, "--candidate-id", "0" * 64]) == 2
    assert cli.main(["finalize", *options, *identity, "--review-hash", "0" * 64]) == 2
    assert cli.main(["review", *options, *identity, *decision]) == 0
    before = store.read_bytes()
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert store.read_bytes() == before


@pytest.mark.parametrize("key", ["candidate_path", "feature_path", "source_dataset_path", "source_manifest_path"])
def test_changed_artifact_blocks_submission(setup, key):
    paths, store, options, identity, decision = setup
    paths[key].write_bytes(paths[key].read_bytes() + b" ")
    # Source manifest itself is bound by candidate provenance; candidate/feature
    # whitespace is bound by the evidence input hashes.
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert store.read_bytes() == b""


@pytest.mark.parametrize("mode", ["missing", "corrupt", "empty"])
def test_missing_or_corrupt_evidence(setup, mode):
    paths, store, options, identity, decision = setup
    if mode == "missing":
        paths["evidence_path"].unlink()
    else:
        paths["evidence_path"].write_bytes(b"{" if mode == "corrupt" else b"")
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert store.read_bytes() == b""


def test_stale_evidence_blocks_finalization(setup):
    paths, store, options, identity, decision = setup
    assert cli.main(["review", *options, *identity, *decision]) == 0
    before = store.read_bytes()
    digest = record_sha256(asdict(load_reviews(store)[0]))
    evidence = json.loads(paths["evidence_path"].read_text())
    evidence["verification"]["reason"] = "Changed metadata"
    paths["evidence_path"].write_text(json.dumps(evidence), encoding="utf-8")
    assert cli.main(["finalize", *options, *identity, "--review-hash", digest]) == 2
    assert store.read_bytes() == before


def test_conflicting_finalization_and_distinct_drafts(setup):
    _, store, options, identity, decision = setup
    assert cli.main(["review", *options, *identity, *decision]) == 0
    first = record_sha256(asdict(load_reviews(store)[0]))
    decision[decision.index("--reviewer-id") + 1] = "fixture:other"
    assert cli.main(["review", *options, *identity, *decision]) == 0
    second = record_sha256(asdict(load_reviews(store)[1]))
    assert cli.main(["finalize", *options, *identity, "--review-hash", first]) == 0
    before = store.read_bytes()
    assert cli.main(["finalize", *options, *identity, "--review-hash", second]) == 2
    assert store.read_bytes() == before


def test_corrupt_store_no_reinitialization(setup):
    _, store, options, identity, decision = setup
    store.write_bytes(b"broken")
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert cli.main(["init", "--store", str(store)]) == 2
    assert store.read_bytes() == b"broken"


def test_store_cannot_alias_artifacts(setup):
    paths, store, options, identity, decision = setup
    options[options.index("--store") + 1] = str(paths["candidate_path"])
    before = paths["candidate_path"].read_bytes()
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert paths["candidate_path"].read_bytes() == before


def test_changed_inputs_between_validation_and_save(setup, monkeypatch):
    paths, store, options, identity, decision = setup
    original = cli.record_review
    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        paths["source_dataset_path"].write_bytes(b"changed during operation")
        return result
    monkeypatch.setattr(cli, "record_review", change)
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert store.read_bytes() == b""


def test_concurrent_store_update_is_not_overwritten(setup, monkeypatch):
    _, store, options, identity, decision = setup
    original = cli.record_review
    committed = []
    def competing_commit(*args, **kwargs):
        result = original(*args, **kwargs)
        review_store.save_reviews(store, result,
                                 expected_digest=review_store.load_snapshot(store).digest)
        committed.append(store.read_bytes())
        return result
    monkeypatch.setattr(cli, "record_review", competing_commit)
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert store.read_bytes() == committed[0]


@pytest.mark.parametrize("historical", [True, False])
def test_positive_finalization_is_not_approval(setup, historical):
    paths, store, options, identity, decision = setup
    if historical:
        evidence = json.loads(paths["evidence_path"].read_text())
        evidence["rule_version"] = "conservative_tactics_v1"
        for proposal in evidence["proposals"]:
            proposal["rule_version"] = "conservative_tactics_v1"
        paths["evidence_path"].write_text(json.dumps(evidence), encoding="utf-8")
    decision[decision.index("--decision") + 1] = "positive"
    assert cli.main(["review", *options, *identity, *decision]) == 0
    review, = load_reviews(store)
    assert review.review_state == "draft"
    assert cli.main(["finalize", *options, *identity, "--review-hash",
                     record_sha256(asdict(review))]) == 0
    report = inspect_training_readiness(**paths, adjudications_path=store)
    assert report["integrity_valid"] and not report["training_ready"]
    assert report["adjudication_status"]["approved"] == 0
    if historical:
        assert "historical_evidence_ineligible" in report["blocking_reasons"]


def test_mate_to_cp_remains_requires_review(tmp_path, labeling_case, capsys):
    import chess.engine
    paths, candidate, _, _ = make_bundle(tmp_path, labeling_case(
        fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g5",
        before=chess.engine.Mate(1), after=chess.engine.Cp(500),
        before_pv=("g6g7",), after_pv=("h8h7",)))
    options = [part for key, path in paths.items()
               for part in ("--" + key.replace("_", "-"), str(path))]
    before = {path: path.read_bytes() for path in paths.values()}
    assert cli.main(["list", *options]) == 0
    row, = json.loads(capsys.readouterr().out)["candidates"]
    assert row["candidate_id"] == candidate["candidate_id"]
    assert row["evidence_status"] == "requires_review"
    assert row["proposed_concepts"] == [] and row["human_reviews"] == []
    assert all(path.read_bytes() == value for path, value in before.items())


def test_explicit_bad_manifest_blocks_all_operations(setup):
    paths, store, options, identity, decision = setup
    manifest = store.with_name("bad_manifest.json")
    manifest.write_text("{}", encoding="utf-8")
    options.extend(["--candidate-manifest-path", str(manifest)])
    assert cli.main(["review", *options, *identity, *decision]) == 2
    assert store.read_bytes() == b""


def test_missing_explicit_store_is_not_silently_empty(setup):
    _, store, options, _, _ = setup
    store.unlink()
    assert cli.main(["list", *options]) == 2
    assert not store.exists()


@pytest.mark.parametrize("module", [True, False])
def test_cli_entrypoint_help(module):
    command = (["-m", "scripts.review_tactical_labels"] if module
               else ["scripts/review_tactical_labels.py"])
    result = subprocess.run([sys.executable, "-B", *command, "list", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--candidate-path" in result.stdout
