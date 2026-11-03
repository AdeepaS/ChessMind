"""Synthetic reviews only; persistence never creates approvals."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from src.chessmind.ml.labeling import review_store as store
from tests.ml.test_review_workflow import artifacts, draft


def test_explicit_initialization_and_missing(tmp_path):
    path = tmp_path / "reviews.jsonl"
    with pytest.raises(FileNotFoundError):
        store.load_reviews(path)
    snapshot = store.save_reviews(path, (), expected_digest=None)
    assert path.read_bytes() == b"" and snapshot.reviews == ()
    assert store.load_reviews(path) == ()
    with pytest.raises(ValueError, match="already exists"):
        store.save_reviews(path, (), expected_digest=None)
    with pytest.raises(FileNotFoundError):
        store.save_reviews(tmp_path / "missing" / "reviews", (), expected_digest=None)


def test_roundtrip_order_metadata_and_determinism(tmp_path, artifacts):
    path = tmp_path / "reviews.jsonl"
    initial = store.save_reviews(path, (), expected_digest=None)
    first = draft(artifacts)
    records = (first, replace(first, reviewer_id="fixture:other", rationale="Unicode: é ♞"))
    saved = store.save_reviews(path, records, expected_digest=initial.digest)
    original = path.read_bytes()
    assert store.load_snapshot(path) == saved
    assert tuple(map(asdict, store.load_reviews(path))) == tuple(map(asdict, records))
    store.save_reviews(path, records, expected_digest=saved.digest)
    assert path.read_bytes() == original
    assert original.endswith(b"\n")


@pytest.mark.parametrize("raw", [b"{", b"null\n", b"[]\n", b"{}\n", b"\n",
    b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'\xff'])
def test_malformed_store_is_never_empty(tmp_path, raw):
    path = tmp_path / "reviews"
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError)):
        store.load_reviews(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("change", ["unknown", "missing", "schema", "state"])
def test_invalid_record_schema(tmp_path, artifacts, change):
    row = asdict(draft(artifacts))
    if change == "unknown":
        row["approval"] = True
    elif change == "missing":
        del row["review_state"]
    elif change == "schema":
        row["schema_version"] = "future"
    else:
        row["review_state"] = "approved"
    path = tmp_path / "reviews"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load_reviews(path)


def test_duplicates_and_finalized_conflicts(tmp_path, artifacts):
    path = tmp_path / "reviews"
    initial = store.save_reviews(path, (), expected_digest=None)
    review = draft(artifacts)
    final = replace(review, review_state="finalized")
    for rows in ((review, review), (final, replace(final, reviewer_id="fixture:other"))):
        with pytest.raises(ValueError):
            store.save_reviews(path, rows, expected_digest=initial.digest)
        assert path.read_bytes() == b""
        path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows), encoding="utf-8")
        with pytest.raises(ValueError):
            store.load_reviews(path)
        path.write_bytes(b"")


@pytest.mark.parametrize("failure", ["serialization", "fsync", "replace"])
def test_failure_preserves_store_and_cleans_temporary(tmp_path, artifacts, monkeypatch, failure):
    path = tmp_path / "reviews"
    initial = store.save_reviews(path, (), expected_digest=None)
    def fail(*args, **kwargs):
        raise OSError("injected failure")
    if failure == "serialization":
        monkeypatch.setattr(store, "_serialize", fail)
    else:
        monkeypatch.setattr(store.os, failure, fail)
    with pytest.raises(OSError, match="injected"):
        store.save_reviews(path, (draft(artifacts),), expected_digest=initial.digest)
    assert path.read_bytes() == b""
    assert list(tmp_path.iterdir()) == [path]


def test_replace_follows_fsync_and_uses_same_directory(tmp_path, artifacts, monkeypatch):
    path = tmp_path / "reviews"
    initial = store.save_reviews(path, (), expected_digest=None)
    fsync, replace_file = os.fsync, os.replace
    events = []
    def sync(fd):
        events.append("fsync")
        fsync(fd)
    def replace_checked(source, destination):
        assert events == ["fsync"]
        assert source.parent == destination.parent == tmp_path
        assert path.read_bytes() == b""
        assert store.load_reviews(source)
        events.append("replace")
        replace_file(source, destination)
    monkeypatch.setattr(store.os, "fsync", sync)
    monkeypatch.setattr(store.os, "replace", replace_checked)
    store.save_reviews(path, (draft(artifacts),), expected_digest=initial.digest)
    assert events == ["fsync", "replace"]


def test_stale_writer_and_corrupt_destination_rejected(tmp_path, artifacts):
    path = tmp_path / "reviews"
    initial = store.save_reviews(path, (), expected_digest=None)
    store.save_reviews(path, (draft(artifacts),), expected_digest=initial.digest)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        store.save_reviews(path, (), expected_digest=initial.digest)
    assert path.read_bytes() == before
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.save_reviews(path, (), expected_digest=initial.digest)
    assert path.read_bytes() == b"corrupt"


def test_lock_blocks_another_process_and_is_not_removed(tmp_path):
    path = tmp_path / "reviews"
    initial = store.save_reviews(path, (), expected_digest=None)
    lock = tmp_path / "reviews.lock"
    lock.write_bytes(b"existing owner")
    program = ("import sys; from src.chessmind.ml.labeling.review_store import save_reviews; "
               "save_reviews(sys.argv[1], (), expected_digest=sys.argv[2])")
    result = subprocess.run([sys.executable, "-B", "-c", program, str(path), initial.digest],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "store locked" in result.stderr
    assert lock.read_bytes() == b"existing owner" and path.read_bytes() == b""


def test_no_nonempty_implicit_initialization(tmp_path, artifacts):
    with pytest.raises(ValueError, match="initialize"):
        store.save_reviews(tmp_path / "reviews", (draft(artifacts),), expected_digest=None)
    assert list(tmp_path.iterdir()) == []
