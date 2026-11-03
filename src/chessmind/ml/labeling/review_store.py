"""Ordered adjudication JSONL with explicit initialization and cooperative locking.

All writers must use save_reviews on the same canonical path. An exclusive sibling
lock serializes digest-check/replace; a stale lock fails closed (no auto recovery).
External editors, hard-link aliases and unreliable network filesystems are outside
this protocol. File fsync is provided, not a cross-platform power-loss guarantee.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..data.dataset_readiness import _json
from .adjudication import AdjudicationRecord, _digest
from .review_workflow import _collection


MAX_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 10000


@dataclass(frozen=True)
class ReviewSnapshot:
    reviews: tuple[AdjudicationRecord, ...]
    digest: str


def _decode(data):
    if len(data) > MAX_BYTES:
        raise ValueError("review store exceeds 32 MiB")
    lines = data.decode("utf-8").splitlines()
    if len(lines) > MAX_RECORDS:
        raise ValueError("review store exceeds 10000 records")
    records = []
    expected = {field.name for field in fields(AdjudicationRecord)}
    for line in lines:
        row = _json(line)  # Blank lines, duplicate keys and nonfinite values fail.
        if type(row) is not dict or set(row) != expected:
            raise ValueError("invalid adjudication record fields")
        records.append(AdjudicationRecord(**row))
    return _collection(records)


def load_snapshot(path):
    """Read one byte snapshot; missing/corrupt stores raise, never become empty."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    return ReviewSnapshot(_decode(data), hashlib.sha256(data).hexdigest())


def load_reviews(path):
    return load_snapshot(path).reviews


def _serialize(reviews):
    records = _collection(reviews)
    data = "".join(json.dumps(asdict(row), sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False) + "\n"
                   for row in records).encode("utf-8")
    _decode(data)
    return data


def save_reviews(path, reviews, *, expected_digest):
    """CAS write; None explicitly initializes an absent EMPTY store only.

    Existing stores require the digest returned by load_snapshot. Parent directories
    must already exist. Return the committed snapshot. A lock left by process death
    must be investigated and removed manually while all writers are stopped.
    """
    records = _collection(reviews)
    data = _serialize(records)  # Complete validation/serialization before any I/O.
    if expected_digest is None:
        if records:
            raise ValueError("initialize an empty store before submitting reviews")
    else:
        _digest(expected_digest)
    path = Path(path).resolve()
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("review store locked; another writer or stale lock exists") from exc
    temporary = None
    try:
        os.close(descriptor)
        if expected_digest is None:
            if path.exists():
                raise ValueError("store already exists; initialization cannot overwrite it")
        elif load_snapshot(path).digest != expected_digest:
            raise ValueError("stale review store; reload before writing")
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        finally:
            lock.unlink()
    return ReviewSnapshot(records, hashlib.sha256(data).hexdigest())
