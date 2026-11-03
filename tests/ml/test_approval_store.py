"""Only fabricated hash fixtures in temporary registries."""

import json
import os
import subprocess
import sys

import pytest

from src.chessmind.ml.labeling import approval_store as store


def test_explicit_initialization_roundtrip_and_preservation(tmp_path):
    path = tmp_path / "approvals.json"
    with pytest.raises(FileNotFoundError):
        store.load_approvals(path)
    initial = store.save_approvals(path, (), expected_digest=None)
    assert path.read_bytes() == b"[]\n"
    with pytest.raises(ValueError):
        store.save_approvals(path, (), expected_digest=None)
    first = store.save_approvals(path, ["b" * 64], expected_digest=initial.digest)
    second = store.save_approvals(path, ["b" * 64, "a" * 64], expected_digest=first.digest)
    assert store.load_approvals(path) == second
    assert second.hashes == ("a" * 64, "b" * 64)
    before = path.read_bytes()
    store.save_approvals(path, ["a" * 64, "b" * 64], expected_digest=second.digest)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="remove"):
        store.save_approvals(path, ["a" * 64], expected_digest=second.digest)
    with pytest.raises(ValueError, match="stale"):
        store.save_approvals(path, ["b" * 64, "c" * 64], expected_digest=first.digest)
    assert path.read_bytes() == before


@pytest.mark.parametrize("raw", [b"", b"{", b"{}", b"null", b'[null]', b'[NaN]',
    b'[1]', b'["bad"]', json.dumps(["a" * 64] * 2).encode(), json.dumps(["A" * 64]).encode()])
def test_corrupt_registry_fails_closed(tmp_path, raw):
    path = tmp_path / "approvals.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError)):
        store.load_approvals(path)
    with pytest.raises((ValueError, TypeError)):
        store.save_approvals(path, (), expected_digest=None)
    assert path.read_bytes() == raw


def test_invalid_write_does_not_touch_existing_registry(tmp_path):
    path = tmp_path / "approvals.json"
    initial = store.save_approvals(path, (), expected_digest=None)
    for values in (["a" * 64] * 2, ["bad"], "a" * 64, {"a" * 64}):
        with pytest.raises(ValueError):
            store.save_approvals(path, values, expected_digest=initial.digest)
    assert path.read_bytes() == b"[]\n"
    with pytest.raises(ValueError):
        store.save_approvals(tmp_path / "new.json", ["a" * 64], expected_digest=None)
    assert not (tmp_path / "new.json").exists()


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace", "recheck"])
def test_failed_write_preserves_original_and_cleans_up(tmp_path, monkeypatch, failure):
    path = tmp_path / "approvals.json"
    initial = store.save_approvals(path, (), expected_digest=None)
    def fail(*args, **kwargs):
        raise OSError("injected failure")
    kwargs = {}
    if failure == "serialize":
        monkeypatch.setattr(store, "_serialize", fail)
    elif failure == "recheck":
        kwargs["before_replace"] = fail
    else:
        monkeypatch.setattr(store.os, failure, fail)
    with pytest.raises(OSError, match="injected"):
        store.save_approvals(path, ["a" * 64], expected_digest=initial.digest, **kwargs)
    assert path.read_bytes() == b"[]\n" and list(tmp_path.iterdir()) == [path]


def test_fsync_recheck_replace_order_and_atomic_destination(tmp_path, monkeypatch):
    path = tmp_path / "approvals.json"
    initial = store.save_approvals(path, (), expected_digest=None)
    sync, replace = os.fsync, os.replace
    events = []
    def fsync(fd):
        events.append("fsync")
        sync(fd)
    def recheck():
        assert events == ["fsync"]
        events.append("recheck")
    def atomic(source, destination):
        assert events == ["fsync", "recheck"]
        assert source.parent == destination.parent == tmp_path
        assert store.load_approvals(source).hashes == ("a" * 64,)
        assert path.read_bytes() == b"[]\n"
        events.append("replace")
        replace(source, destination)
    monkeypatch.setattr(store.os, "fsync", fsync)
    monkeypatch.setattr(store.os, "replace", atomic)
    store.save_approvals(path, ["a" * 64], expected_digest=initial.digest, before_replace=recheck)
    assert events == ["fsync", "recheck", "replace"]


def test_cross_process_lock_and_stale_lock_fail_closed(tmp_path):
    path = tmp_path / "approvals.json"
    initial = store.save_approvals(path, (), expected_digest=None)
    lock = path.with_name(path.name + ".lock")
    lock.write_bytes(b"owner")
    code = ("import sys; from src.chessmind.ml.labeling.approval_store import save_approvals; "
            "save_approvals(sys.argv[1], ['a'*64], expected_digest=sys.argv[2])")
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path), initial.digest],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "registry locked" in result.stderr
    assert lock.read_bytes() == b"owner" and path.read_bytes() == b"[]\n"


def test_corruption_after_snapshot_cannot_be_overwritten(tmp_path):
    path = tmp_path / "approvals.json"
    initial = store.save_approvals(path, (), expected_digest=None)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.save_approvals(path, ["a" * 64], expected_digest=initial.digest)
    assert path.read_bytes() == b"corrupt"
