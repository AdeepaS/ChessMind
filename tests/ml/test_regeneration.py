"""C3 uses synthetic bundles only. Never regenerate production data in tests."""

import copy
import json
from dataclasses import asdict
from unittest.mock import Mock

import chess
import chess.engine
import pytest

from src.chessmind.ml.data.candidate_schema import AnalysisProfile, candidate_id
from src.chessmind.ml.features.extractor import extract_candidate_features
from src.chessmind.ml.labeling import regeneration as regen, engine_verification
from src.chessmind.ml.labeling.adjudication import record_sha256
from src.chessmind.ml.labeling.rules import propose_labels
from src.chessmind.ml.labeling.schema import LabelRecord
from scripts import regenerate_tactical_evidence as cli


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rows(path, values):
    path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")


@pytest.fixture
def case(tmp_path, labeling_case, monkeypatch, request):
    for folder in ("source", "processed", "historical", "engine", "output"):
        (tmp_path / folder).mkdir()
    p = dict(candidate_path=tmp_path / "processed/candidate_moves.jsonl",
             feature_path=tmp_path / "processed/candidate_features.jsonl",
             evidence_path=tmp_path / "historical/mistake_label_proposals.jsonl",
             source_dataset_path=tmp_path / "source/sample.pgn",
             source_manifest_path=tmp_path / "source/source_manifest.json",
             engine_path=tmp_path / "engine/stockfish.exe", output_dir=tmp_path / "output/c3_v2")
    p["engine_path"].write_bytes(b"synthetic engine")
    monkeypatch.setattr(regen, "ENGINE_SHA256", regen.file_sha256(p["engine_path"]))
    p["source_dataset_path"].write_bytes(b"synthetic source")
    write(p["source_manifest_path"], dict(sha256=regen.file_sha256(p["source_dataset_path"]),
        source_url="fixture:source", accepted_count=24,
        source_references=[dict(game_id=f"game{i}", archive_game_index=i+1) for i in range(24)]))
    candidates, features, evidence = [], [], []
    for i in range(24):
        if getattr(request, "param", None) == "terminal":
            c, _, v = labeling_case(fen="7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="g6g7",
                before=chess.engine.Mate(1), after=chess.engine.MateGiven,
                before_pv=("g6g7",), after_pv=())
        else:
            c, _, v = labeling_case(before_pv=("c2b4",))
        c["source_game_id"] = f"game{i}"
        c["candidate_id"] = candidate_id(c["source"], c["source_game_id"], c["ply_index"])
        c["source_evidence"] = dict(dataset_path=str(p["source_dataset_path"]),
            manifest_path=str(p["source_manifest_path"]), source_archive="fixture:source", archive_game_index=i+1,
            dataset_sha256=regen.file_sha256(p["source_dataset_path"]),
            manifest_sha256=regen.file_sha256(p["source_manifest_path"]))
        candidates.append(c)
        features.append(json.loads(json.dumps(asdict(extract_candidate_features(c)))))
    rows(p["candidate_path"], candidates)
    rows(p["feature_path"], features)
    for c, f in zip(candidates, features):
        proposals, ev, status, reasons = propose_labels(c, f, v)
        e = json.loads(json.dumps(asdict(LabelRecord(c["candidate_id"], c["source_game_id"],
            c["player_identifier"], c["ply_index"], c["schema_version"], f["schema_version"],
            c["selection_type"], v, ev, proposals, status, reasons,
            regen.file_sha256(p["candidate_path"]), regen.file_sha256(p["feature_path"]), record_sha256(f)))))
        e["rule_version"] = regen.HISTORICAL_RULE_VERSION
        for prop in e["proposals"]:
            prop["rule_version"] = regen.HISTORICAL_RULE_VERSION
        evidence.append(e)
    rows(p["evidence_path"], evidence)
    write(tmp_path / "processed/candidate_manifest.json", dict(schema_version="candidate_moves_v1",
        output_sha256=regen.file_sha256(p["candidate_path"]), candidate_count=24,
        input_sha256=regen.file_sha256(p["source_dataset_path"]),
        source_manifest_sha256=regen.file_sha256(p["source_manifest_path"])))
    write(tmp_path / "processed/feature_manifest.json", dict(schema_version="chess_features_v1",
        output_sha256=regen.file_sha256(p["feature_path"]), accepted=24,
        input_sha256=regen.file_sha256(p["candidate_path"]), feature_value_count=163))
    write(tmp_path / "historical/label_quality_report.json", {})
    def bind():
        write(tmp_path / "historical/labeling_manifest.json", dict(schema_version="mistake_proposals_v1",
            rule_version=regen.HISTORICAL_RULE_VERSION, processed=len(evidence),
            candidate_sha256=regen.file_sha256(p["candidate_path"]),
            feature_sha256=regen.file_sha256(p["feature_path"]),
            output_sha256=regen.file_sha256(p["evidence_path"]),
            quality_report_sha256=regen.file_sha256(tmp_path / "historical/label_quality_report.json")))
    bind()
    profile = AnalysisProfile(configured_depth=12, engine_binary_sha256=regen.ENGINE_SHA256,
        engine_id=(("name", "Stockfish 19"),), chess_version="fixture",
        engine_options=(("threads", 1), ("hash", 64), ("multipv", 1), ("uci_limitstrength", False), ("skill level", 20)),
        engine_resources=(("evalfile", "fixture", "embedded", regen.ENGINE_SHA256),))
    monkeypatch.setattr(regen.pipeline, "_profile", Mock(return_value=profile))
    launch = Mock(return_value=Mock())
    monkeypatch.setattr(regen.engine_api, "create_cache_context", Mock(return_value=Mock(engine=launch.return_value)))
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", launch)
    # Exercise the real verifier, mocking only position analysis; new scores differ from v1.
    def analyze(engine, board, **kwargs):
        before = board.turn == chess.WHITE
        if getattr(request, "param", None) == "terminal":
            from src.chessmind.engine.analyzer import serialize_evaluation
            return dict(evaluation=serialize_evaluation(chess.engine.Mate(1) if before else chess.engine.MateGiven),
                        depth=12 if before else 0,
                        pv=[chess.Move.from_uci("g6g7")] if before else [])
        return dict(evaluation=dict(type="cp", cp=0 if before else -400, mate=None,
                                    winner=None, perspective="white"), depth=12,
                    pv=[chess.Move.from_uci(x) for x in (("c2b4",) if before else
                                                       ("b8b4", "h1g1", "b4b5", "g1f1"))])
    search = Mock(side_effect=analyze)
    monkeypatch.setattr(engine_verification, "analyze_board", search)
    return p, evidence, bind, launch, search


def test_success_fresh_and_preserves_inputs(case):
    p, _, _, launch, search = case
    before = {f: f.read_bytes() for f in p["output_dir"].parents[1].rglob("*") if f.is_file()}
    audit = regen.regenerate_evidence(**p)
    assert search.call_count == 48
    launch.return_value.configure.assert_called_once_with({"Threads": 1, "Hash": 64})
    launch.return_value.quit.assert_called_once()
    assert audit["acceptance"]["eligible_labels"] == 0
    assert len(audit["selected_candidate_ids"]) == 24
    assert all(f.read_bytes() == data for f, data in before.items())
    generated = regen._records(p["output_dir"] / "mistake_label_proposals.jsonl")
    assert all(r["verification"]["after"]["cp"] == -400 for r in generated.values())
    assert not list(p["output_dir"].parent.glob(".c3-stage-*"))


@pytest.mark.parametrize("problem", ["duplicate", "missing", "extra", "unknown", "order", "version", "failed"])
def test_bad_cohort_before_engine(case, problem):
    p, evidence, bind, launch, _ = case
    if problem == "duplicate": evidence.append(copy.deepcopy(evidence[0]))
    if problem == "missing": evidence.pop()
    if problem == "extra": evidence.append(dict(evidence[0], candidate_id="a" * 64))
    if problem == "unknown": evidence[0]["candidate_id"] = "b" * 64
    if problem == "order": evidence.reverse()
    if problem == "version": evidence[0]["rule_version"] = "unknown"
    if problem == "failed": evidence[0]["verification"]["status"] = "failed"
    rows(p["evidence_path"], evidence)
    bind()
    with pytest.raises(ValueError): regen.regenerate_evidence(**p)
    launch.assert_not_called()
    assert not p["output_dir"].exists()


@pytest.mark.parametrize("problem", ["startup", "partial", "depth", "illegal", "interrupt", "tamper", "replay"])
def test_failure_never_publishes(case, problem, monkeypatch):
    p, _, _, launch, search = case
    original = search.side_effect
    calls = 0
    def damaged(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 3:
            if problem == "partial": raise chess.engine.EngineError("failed search")
            if problem == "interrupt": raise KeyboardInterrupt()
            if problem == "depth": result["depth"] = 11
            if problem == "illegal": result["pv"] = [chess.Move.from_uci("a1a8")]
            if problem == "tamper": p["source_dataset_path"].write_bytes(b"external mutation")
        return result
    search.side_effect = damaged
    if problem == "startup": launch.side_effect = OSError("engine unavailable")
    if problem == "replay":
        real = regen.pipeline.run_labeling_pipeline
        def corrupt(*args, **kwargs):
            manifest = real(*args, **kwargs)
            path = args[2] / "mistake_label_proposals.jsonl"
            records = list(regen._records(path).values())
            records[0]["reasons"] = ["invented"]
            rows(path, records)
            manifest["output_sha256"] = regen.file_sha256(path)
            write(args[2] / "labeling_manifest.json", manifest)
            return manifest
        monkeypatch.setattr(regen.pipeline, "run_labeling_pipeline", corrupt)
    with pytest.raises((ValueError, OSError, KeyboardInterrupt)): regen.regenerate_evidence(**p)
    assert not p["output_dir"].exists()
    assert not list(p["output_dir"].parent.glob(".c3-stage-*"))


@pytest.mark.parametrize("target", ["source_dataset_path", "source_manifest_path", "candidate_manifest", "feature_manifest"])
def test_tampered_inputs_before_engine(case, target):
    p, _, _, launch, _ = case
    path = p[target] if target in p else p["candidate_path"].with_name(target + ".json")
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError): regen.regenerate_evidence(**p)
    launch.assert_not_called()


@pytest.mark.parametrize("destination", ["existing", "historical", "inside", "ancestor", "source"])
def test_unsafe_output(case, destination):
    p, _, _, launch, _ = case
    if destination == "existing": p["output_dir"].mkdir()
    if destination == "historical": p["output_dir"] = p["evidence_path"].parent
    if destination == "inside": p["output_dir"] = p["evidence_path"].parent / "new"
    if destination == "ancestor": p["output_dir"] = p["evidence_path"].parents[1]
    if destination == "source": p["output_dir"] = p["source_dataset_path"].parent
    with pytest.raises((ValueError, FileExistsError)): regen.regenerate_evidence(**p)
    launch.assert_not_called()


def test_cli_failure_and_interrupt(monkeypatch, capsys):
    for error, code in ((ValueError("bad cohort"), 2), (KeyboardInterrupt(), 130)):
        monkeypatch.setattr(cli, "regenerate_evidence", Mock(side_effect=error))
        assert cli.main([]) == code
        assert not capsys.readouterr().out


@pytest.mark.parametrize("case", ["terminal"], indirect=True)
def test_terminal_zero_depth_and_zero_proposals(case):
    p, _, _, _, search = case
    audit = regen.regenerate_evidence(**p)
    assert search.call_count == 48 and audit["acceptance"]["eligible_labels"] == 0
    generated = list(regen._records(p["output_dir"] / "mistake_label_proposals.jsonl").values())
    assert all(r["verification"]["after_depth"] == 0 and not r["proposals"] for r in generated)


@pytest.mark.parametrize("field,value", [("engine_binary_sha256", "0" * 64),
    ("engine_id", (("name", "Other engine"),)), ("configured_depth", 8),
    ("engine_options", (("threads", 2),)), ("engine_resources", None)])
def test_engine_profile_rejected_before_search(case, field, value, monkeypatch):
    from dataclasses import replace
    p, _, _, _, search = case
    profile = regen.pipeline._profile.return_value
    monkeypatch.setattr(regen.pipeline, "_profile", Mock(return_value=replace(profile, **{field: value})))
    with pytest.raises(ValueError): regen.regenerate_evidence(**p)
    search.assert_not_called()
    assert not p["output_dir"].exists()


def test_wrong_binary_before_launch(case):
    p, _, _, launch, _ = case
    p["engine_path"].write_bytes(b"different engine")
    with pytest.raises(ValueError, match="executable hash"): regen.regenerate_evidence(**p)
    launch.assert_not_called()


def test_destination_created_during_run_not_overwritten(case):
    p, _, _, _, search = case
    original = search.side_effect
    def racing(*args, **kwargs):
        p["output_dir"].mkdir(exist_ok=True)
        (p["output_dir"] / "sentinel").write_bytes(b"other writer")
        return original(*args, **kwargs)
    search.side_effect = racing
    with pytest.raises(FileExistsError): regen.regenerate_evidence(**p)
    assert list(p["output_dir"].iterdir()) == [p["output_dir"] / "sentinel"]


def test_resolved_alias_rejected(case, monkeypatch):
    from pathlib import Path
    p, _, _, launch, _ = case
    original = Path.resolve
    def alias(path, *args, **kwargs):
        if path == p["output_dir"]:
            return p["evidence_path"].parent
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", alias)
    with pytest.raises(ValueError, match="alias"): regen.regenerate_evidence(**p)
    launch.assert_not_called()
