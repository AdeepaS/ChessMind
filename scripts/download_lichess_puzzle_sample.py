"""Acquire a bounded prefix of valid puzzles; run with python -m scripts...."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import zstandard

from src.chessmind.ml.data.puzzle_loader import DATASET_URL, PuzzleCSVReader


DEFAULT_OUTPUT = Path("data/raw/ml/puzzles")
SAMPLE_FILENAME = "lichess_puzzle_sample.csv"
MANIFEST_FILENAME = "source_manifest.json"


class BoundedDownload:
    """Limit bytes fetched and elapsed time, in addition to socket timeouts."""

    def __init__(self, response, max_bytes: int, deadline: float):
        self.response = response
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.bytes_read = 0

    def read(self, size: int) -> bytes:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("sample acquisition time limit reached")
        remaining = self.max_bytes - self.bytes_read
        if remaining <= 0:
            raise ValueError("compressed download byte limit reached before sample was complete")
        data = self.response.read(min(size, remaining))
        self.bytes_read += len(data)
        return data


def acquire_sample(output_dir: str | Path = DEFAULT_OUTPUT, *, count: int = 1000,
                   max_rows: int = 10000, max_bytes: int = 16 * 1024 * 1024,
                   timeout: float = 20, total_timeout: float = 120) -> dict:
    """Publish a CSV and manifest together by renaming a completed staging dir.

    The output directory must not exist. Exceptions/interruptions remove staging
    files; an abrupt process kill may leave a hidden staging directory, never a
    partial sample at the final path. This is a prefix sample, not a random one.
    """
    for name, value in (("count", count), ("max_rows", max_rows), ("max_bytes", max_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_rows < count:
        raise ValueError("max_rows must be at least count")
    if not all(math.isfinite(t) and t > 0 for t in (timeout, total_timeout)):
        raise ValueError("timeouts must be finite positive numbers")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; choose a new --output-dir")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + total_timeout
    request = Request(DATASET_URL, headers={
        "User-Agent": "ChessMind/Day8-puzzle-sample",
        "Accept-Encoding": "identity",
    })
    with tempfile.TemporaryDirectory(prefix=".puzzles-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        sample = stage / SAMPLE_FILENAME
        with urlopen(request, timeout=min(timeout, total_timeout)) as response:
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status {response.status} from {DATASET_URL}")
            download = BoundedDownload(response, max_bytes, deadline)
            with zstandard.ZstdDecompressor().stream_reader(download, read_size=16384) as decoded:
                with io.TextIOWrapper(decoded, encoding="utf-8-sig", newline="") as text:
                    reader = PuzzleCSVReader(text, strict=False, max_records=count, max_rows=max_rows)
                    with sample.open("w", encoding="utf-8", newline="") as target:
                        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for puzzle in reader:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("sample acquisition time limit reached")
                            writer.writerow(dict(puzzle.source_fields))
                    if reader.stats.accepted != count:
                        raise ValueError(
                            f"Incomplete sample: accepted {reader.stats.accepted}/{count}, "
                            f"processed {reader.stats.processed}, rejected {reader.stats.rejected}; "
                            "source ended or row limit reached"
                        )
        with sample.open("rb") as saved:
            sha256 = hashlib.file_digest(saved, "sha256").hexdigest()
        manifest = {
            "dataset_url": DATASET_URL,
            "source_page": "https://database.lichess.org/#puzzles",
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_file": SAMPLE_FILENAME,
            "sample_count": reader.stats.accepted,
            "sha256": sha256,
            "sampling_method": "first valid unique records in export order; not random",
            "processed_rows": reader.stats.processed,
            "rejected_rows": reader.stats.rejected,
            "rejection_counts": dict(reader.stats.rejection_counts),
            "error_examples": reader.stats.error_examples,
            "compressed_bytes_read": download.bytes_read,
        }
        (stage / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Both files are closed and complete before the directory becomes visible.
        os.rename(stage, output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--total-timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        manifest = acquire_sample(**vars(args))
    except (OSError, ValueError, zstandard.ZstdError) as exc:
        parser.exit(1, f"Acquisition failed for {DATASET_URL}: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Acquisition interrupted; no sample was published.\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
