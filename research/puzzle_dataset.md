# Lichess puzzle sample

This pipeline acquires and validates source puzzles only. It performs no feature
engineering, model training, predictions, or mistake labeling.

## Dependencies and commands

Run commands from the repository root with the project virtual environment.
The existing `python-chess`/`chess` dependencies validate positions and moves;
`zstandard==0.25.0` adds streaming `.csv.zst` support. Tests also require pytest,
as does the existing suite.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m scripts.download_lichess_puzzle_sample
.venv/Scripts/python.exe -m scripts.inspect_lichess_puzzles data/raw/ml/puzzles/lichess_puzzle_sample.csv --strict
.venv/Scripts/python.exe -m pytest tests/ml/test_puzzle_loader.py -q
.venv/Scripts/python.exe -m pytest tests/ml/test_taxonomy.py -q
.venv/Scripts/python.exe -m pytest -q
```

Imports and tests never initiate a download. Automated tests use synthetic,
offline data, including a mocked HTTP response for acquisition tests.

## Acquisition and provenance

The source is the [official Lichess puzzle database](https://database.lichess.org/#puzzles),
distributed under CC0. The script streams the compressed export at
`https://database.lichess.org/lichess_db_puzzle.csv.zst`, stops after 1,000 valid,
unique records, and closes the connection. It does not save the full archive.
This is a prefix sample in export order, not a random or representative sample.

Defaults also limit acquisition to 10,000 scanned rows, 16 MiB of compressed
input, a 20-second socket timeout and a 120-second elapsed-time budget. The
elapsed budget is checked between reads/records; an in-progress network read
can last until its socket timeout. Override with `--count`, `--max-rows`,
`--max-bytes`, `--timeout`, and `--total-timeout`.

The default output directory is `data/raw/ml/puzzles/`. It must not already
exist; use `--output-dir data/raw/ml/puzzles_another_sample` for another run.
Existing samples are never overwritten. A staging directory receives both:

- `lichess_puzzle_sample.csv`: original header and accepted source row values,
  including additional columns such as `DailyDate`; CSV quoting may normalize.
- `source_manifest.json`: dataset URL, UTC acquisition timestamp, accepted
  count, SHA-256 of the saved CSV, rejection statistics, bytes fetched and
  sampling method.

The completed directory is renamed into place only after both files close.
Network errors, a short source, exhausted budgets, or Ctrl+C fail without
publishing an incomplete sample. Abrupt process termination can leave a hidden
`.puzzles-*` staging directory, which is not a completed acquisition. No records
are fabricated when the official source is unavailable.

## Positions, solutions and annotations

`original_fen` is the source FEN **before** the opponent's setup move. The first
UCI token is `setup_move_uci`; applying it produces `puzzle_fen`, the position
presented to the solver. `solution_moves_uci` contains every remaining ply,
including the opponent's replies. `original_moves_uci` reconstructs the complete
sequence. Every move must be legal on the corresponding board, and at least
one solution move is required. Legality validation does not prove that a
solution is optimal or that the source's tactical theme is correct.

Records are frozen dataclasses with immutable tuples. `source_fields` preserves
all raw CSV values, including the original FEN, move string, themes, opening
tags and additional metadata. Theme order and spelling are preserved in
`themes`; `opening_tags` can be empty. `source` is `lichess_puzzles`.

Puzzle themes describe tactical opportunities in a puzzle-solving position.
They do not establish which mistake the previous human player made. For
example, a `fork` theme is not automatically ChessMind's `missed_fork` mistake.
No mapping to the Task 1 taxonomy or to severity classifications is performed.

## Loader and validation policy

```python
from src.chessmind.ml.data.puzzle_loader import open_puzzles

with open_puzzles("data/raw/ml/puzzles/lichess_puzzle_sample.csv",
                  strict=False, max_records=1000) as puzzles:
    for puzzle in puzzles:
        print(puzzle.puzzle_id, puzzle.puzzle_fen)
    print(puzzles.stats)
```

The context manager supports `.csv` and `.csv.zst`, uses UTF-8 (with optional
BOM), and closes handles on early exit. `PuzzleCSVReader` supports caller-owned
text streams. Both APIs stream records; they do not retain puzzle records.
Duplicate detection retains IDs, using O(unique IDs scanned) memory. Error
examples are capped at five; counters retain totals. Consumers should iterate
instead of calling `list()` on a full database.

- Required columns: `PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags`.
  Extra named columns are preserved. Missing/duplicate/empty header names are
  fatal schema errors. Zero-byte and header-only files yield zero records.
- FEN must have all six fields and describe a valid standard-chess board.
  Null moves, malformed UCI and illegal setup/solution moves are rejected.
- Numeric metadata must be integers; rating, rating deviation and play count
  must be nonnegative; popularity must lie between -100 and 100 inclusive.
- Empty ID, FEN, numeric fields or game URL, and mismatched row widths are
  rejected. Empty themes and opening tags are allowed.
- An ID's first occurrence reserves it even if its row is invalid; subsequent
  occurrences are rejected as duplicates. Unique puzzle counts refer to
  accepted records, not distinct rejected IDs.
- Strict mode (the loader default) counts and raises the first row rejection
  with its CSV line, ID and reason. Permissive mode skips rejected rows and
  records reason counts. Header, CSV syntax, I/O, encoding and decompression
  errors are fatal in both modes.
- `max_records` caps accepted records; rejections do not consume that limit.
  `max_rows` caps all processed records. Both allow zero; neither counts the
  header or blank physical lines. Limits validate only the consumed prefix.

Inspection defaults to permissive mode to report processed, accepted, rejected
and unique counts, five IDs, the ten most common themes (puzzles per theme),
and minimum/maximum/mean rating. Add `--strict` to fail on rejected records or
`--max-records` to inspect a prefix. No ratings are invented for an empty file;
the rating summary uses nulls.
