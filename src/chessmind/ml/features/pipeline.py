"""Stream candidate JSONL into deterministic feature records without an engine."""

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess

from .extractor import FeatureValidationError, extract_candidate_features
from .schema import FEATURE_SCHEMA_VERSION, feature_json


def _digest(path):
    # Kept local to avoid importing engine orchestration just to hash a file.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def generate_features(input_path, output_path, *, strict=True, source_manifest_path=None,
                      max_line_chars=2_000_000):
    """Publish new files; manifest is the completion marker for the output pair.

    Existing files are never replaced. Hard-link publication is atomic per file;
    ordinary failures roll back this run's output. Consumers must verify the
    manifest hash: a forced process kill between links can leave an orphan JSONL.
    Memory is bounded by one line/record plus scanned IDs (for deduplication).
    """
    if type(strict) is not bool or type(max_line_chars) is not int or max_line_chars < 1:
        raise ValueError("strict must be bool; max_line_chars must be a positive integer")
    started = time.perf_counter()
    input_path, output_path = Path(input_path), Path(output_path)
    manifest_path = output_path.with_name("feature_manifest.json")
    if output_path.resolve() in (input_path.resolve(), manifest_path.resolve()):
        raise ValueError("input/output/manifest paths must be distinct")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("feature output or manifest exists; choose a new output directory")
    input_hash = _digest(input_path)
    if source_manifest_path is None:
        adjacent = input_path.with_name("candidate_manifest.json")
        source_manifest_path = adjacent if adjacent.is_file() else None
    provenance = None
    source_count = None
    if source_manifest_path is not None:
        source_manifest_path = Path(source_manifest_path)
        if source_manifest_path.resolve() in (output_path.resolve(), manifest_path.resolve()):
            raise ValueError("source manifest must not be an output path")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict) or source_manifest.get("output_sha256") != input_hash:
            raise ValueError("candidate manifest input hash mismatch")
        source_count = source_manifest.get("candidate_count")
        if source_count is not None and (type(source_count) is not int or source_count < 0):
            raise ValueError("invalid source manifest candidate count")
        provenance = {"path": str(source_manifest_path), "sha256": _digest(source_manifest_path),
                      "schema_version": source_manifest.get("schema_version"),
                      "source_manifest_sha256": source_manifest.get("source_manifest_sha256"),
                      "input_sha256": source_manifest.get("input_sha256")}
    seen, reasons, examples = set(), Counter(), []
    processed = accepted = physical_line = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output_path.parent) as temporary:
        stage = Path(temporary)
        staged_output = stage / "features.jsonl"
        with input_path.open(encoding="utf-8") as source, staged_output.open(
                "w", encoding="utf-8", newline="\n") as target:
            while True:
                line = source.readline(max_line_chars + 1)
                if not line:
                    break
                physical_line += 1
                oversized = len(line) > max_line_chars
                if oversized:
                    while line and not line.endswith("\n"):
                        line = source.readline(max_line_chars + 1)
                elif not line.strip():
                    continue
                processed += 1
                try:
                    if oversized:
                        raise FeatureValidationError("line_too_large")
                    try:
                        candidate = json.loads(line, object_pairs_hook=_unique_object,
                                               parse_constant=_invalid_constant)
                    except (ValueError, RecursionError) as exc:
                        raise FeatureValidationError("invalid_json") from exc
                    if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str):
                        identity = candidate["candidate_id"]
                        if identity in seen:
                            raise FeatureValidationError("duplicate_candidate_id")
                        seen.add(identity)
                    record = extract_candidate_features(candidate)
                except FeatureValidationError as exc:
                    reasons[exc.reason] += 1
                    if len(examples) < 5:
                        examples.append({"line": physical_line, "reason": exc.reason})
                    if strict:
                        raise FeatureValidationError(f"line {physical_line}: {exc.reason}") from exc
                    continue
                target.write(feature_json(record) + "\n")
                accepted += 1
        if source_count is not None and source_count != processed:
            raise ValueError("candidate manifest count mismatch")
        if _digest(input_path) != input_hash:
            raise ValueError("input changed during feature generation")
        manifest = {
            "schema_version": FEATURE_SCHEMA_VERSION, "chess_version": chess.__version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_path), "input_sha256": input_hash,
            "output_file": output_path.name, "output_sha256": _digest(staged_output),
            "source_provenance": provenance, "total_processed": processed, "accepted": accepted,
            "rejected": processed - accepted, "unique_candidate_ids": accepted,
            "rejection_counts": dict(reasons), "error_examples": examples,
            "configuration": {"strict": strict, "max_line_chars": max_line_chars,
                "piece_values": {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0},
                "attack_definition": "python-chess geometric attacks, including pinned attackers",
                "ordinary_piece_safety_excludes_kings": True,
                "doubled_pawns": "excess pawns per file",
                "pawn_shield": "own pawns one rank forward on king file and adjacent files",
                "king_zone": "king square plus adjacent squares",
                "delta_convention": "after minus before; explicit colors; bools as 0/1",
                "mobility": "legal moves for actual side to move only"},
            "feature_value_count": 163, "position_value_count": 52,
            "numeric_delta_count": 49, "move_context_value_count": 10,
            "elapsed_seconds": time.perf_counter() - started,
        }
        staged_manifest = stage / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                   encoding="utf-8")
        published = False
        try:
            os.link(staged_output, output_path)
            published = True
            os.link(staged_manifest, manifest_path)
        except BaseException:
            if published:
                output_path.unlink()
            raise
    return manifest
