"""Trusted-operator CLI tests; synthetic fixtures never touch production data."""

import json
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from scripts.approve_tactical_review import main
from src.chessmind.ml.labeling.adjudication import record_sha256
from src.chessmind.ml.labeling.approval_store import load_approvals
from tests.ml.test_approval_workflow import approval_case, write_reviews


def arguments(case):
    paths, registry, review = case
    return ["--review-hash", record_sha256(asdict(review)), *[
        part for name, path in paths.items() for part in ("--" + name.replace("_", "-"), str(path))]]


ACTION = ["--confirm-independent-approval", "--operator-id", "fixture:operator",
          "--independent-justification", "Synthetic independent examination of this exact decision."]


def test_init_list_inspect_approve(approval_case, capsys):
    paths, registry, review = approval_case
    registry.unlink()
    assert main(["init", "--approvals-path", str(registry)]) == 0
    before = {p: p.read_bytes() for p in [registry, *paths.values()]}
    capsys.readouterr()
    assert main(["list", "--approvals-path", str(registry)]) == 0
    assert json.loads(capsys.readouterr().out)["approved_hashes"] == []
    assert main(["inspect", *arguments(approval_case)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["review"]["review_state"] == "finalized" and not output["approval_issued"]
    assert all(p.read_bytes() == data for p, data in before.items())
    assert main(["approve", *arguments(approval_case), "--approvals-path", str(registry), *ACTION]) == 0
    assert load_approvals(registry).hashes == (record_sha256(asdict(review)),)
    assert all(p.read_bytes() == before[p] for p in paths.values())
    assert main(["approve", *arguments(approval_case), "--approvals-path", str(registry), *ACTION]) == 2


@pytest.mark.parametrize("flag", ["--confirm-independent-approval", "--operator-id", "--independent-justification"])
def test_required_explicit_action(approval_case, flag):
    action = list(ACTION)
    index = action.index(flag)
    del action[index:index + (1 if flag == "--confirm-independent-approval" else 2)]
    with pytest.raises(SystemExit) as error:
        main(["approve", *arguments(approval_case), "--approvals-path", str(approval_case[1]), *action])
    assert error.value.code == 2 and load_approvals(approval_case[1]).hashes == ()


@pytest.mark.parametrize("changes", [{"review_state": "draft"}, {"decision": "unresolved"}])
def test_inspect_blocked_review_without_issuing_approval(approval_case, changes):
    paths, registry, review = approval_case
    review = replace(review, **changes)
    write_reviews(paths, (review,))
    case = (paths, registry, review)
    assert main(["inspect", *arguments(case)]) == 0
    assert main(["approve", *arguments(case), "--approvals-path", str(registry), *ACTION]) == 2
    assert load_approvals(registry).hashes == ()


def test_missing_corrupt_registry_and_bad_hash(approval_case):
    _, registry, _ = approval_case
    options = arguments(approval_case)
    options[1] = "0" * 64
    assert main(["approve", *options, "--approvals-path", str(registry), *ACTION]) == 2
    registry.unlink()
    assert main(["list", "--approvals-path", str(registry)]) == 2
    assert main(["approve", *arguments(approval_case), "--approvals-path", str(registry), *ACTION]) == 2
    assert not registry.exists()
    registry.write_bytes(b"broken")
    assert main(["init", "--approvals-path", str(registry)]) == 2
    assert main(["approve", *arguments(approval_case), "--approvals-path", str(registry), *ACTION]) == 2
    assert registry.read_bytes() == b"broken"


@pytest.mark.parametrize("module", [True, False])
def test_entrypoint_help(module):
    command = ["-m", "scripts.approve_tactical_review"] if module else ["scripts/approve_tactical_review.py"]
    result = subprocess.run([sys.executable, "-B", *command, "approve", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--confirm-independent-approval" in result.stdout
