"""Append-only trusted hash registry; same cooperative CAS/lock pattern as Stage B.

This storage module does not authorize a human: only trusted operators may write
the registry. Local cooperating writers only; no protection from external editors,
hard-link aliases, or unreliable network filesystem locking. No automatic lock theft.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..data.dataset_readiness import _json
from .adjudication import _digest


MAX_BYTES = 32 * 1024 * 1024
MAX_APPROVALS = 10000


@dataclass(frozen=True)
class ApprovalSnapshot:
    hashes: tuple[str, ...]
    digest: str


def _validate(values):
    if type(values) not in (list, tuple):
        raise ValueError("approvals must be an array of hashes")
    if len(values) > MAX_APPROVALS:
        raise ValueError("approval record budget exceeded")
    for value in values:
        _digest(value)
    if len(set(values)) != len(values):
        raise ValueError("duplicate approval hash")
    return tuple(sorted(values))


def _decode(data):
    if len(data) > MAX_BYTES:
        raise ValueError("approval store exceeds 32 MiB")
    values = _json(data)
    if type(values) is not list:
        raise ValueError("approval registry must be a JSON array")
    return _validate(values)


def load_approvals(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    return ApprovalSnapshot(_decode(data), hashlib.sha256(data).hexdigest())


def _serialize(values):
    data = (json.dumps(_validate(values), indent=2, allow_nan=False) + "\n").encode("utf-8")
    _decode(data)
    return data


def save_approvals(path, hashes, *, expected_digest, before_replace=None):
    """Initialize absent [] with None; otherwise preserve every existing hash.

    before_replace is a caller validation hook, run under the registry lock before
    replacement. It must not mutate inputs. It allows the workflow to recheck its
    immutable artifact snapshot immediately before committing. It is not an artifact
    writer lock. Parent directories must already exist.
    """
    values = _validate(hashes)
    data = _serialize(values)
    if expected_digest is None:
        if values:
            raise ValueError("initialize an empty approval registry first")
    else:
        _digest(expected_digest)
    path = Path(path).resolve()
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("approval registry locked; another writer or stale lock exists") from exc
    temporary = None
    try:
        os.close(descriptor)
        if expected_digest is None:
            if path.exists():
                raise ValueError("approval registry already exists")
        else:
            previous = load_approvals(path)
            if previous.digest != expected_digest:
                raise ValueError("stale approval registry; reload before writing")
            if not set(previous.hashes) <= set(values):
                raise ValueError("cannot remove or replace existing approvals")
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        temporary = None
    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        finally:
            lock.unlink()
    return ApprovalSnapshot(values, hashlib.sha256(data).hexdigest())
