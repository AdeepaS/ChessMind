"""Stream and validate standard-chess Lichess puzzles without inferring mistakes."""

import csv
import io
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TextIO

import chess
import zstandard


DATASET_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
SOURCE = "lichess_puzzles"
REQUIRED_COLUMNS = (
    "PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity",
    "NbPlays", "Themes", "GameUrl", "OpeningTags",
)


class PuzzleValidationError(ValueError):
    """A rejected row, including a stable reason code and source location."""

    def __init__(self, reason: str, message: str, *, line: int, puzzle_id: str = ""):
        self.reason = reason
        self.line = line
        self.puzzle_id = puzzle_id
        super().__init__(f"CSV line {line}, PuzzleId {puzzle_id!r}: {reason}: {message}")


@dataclass(frozen=True)
class PuzzleRecord:
    puzzle_id: str
    original_fen: str
    puzzle_fen: str
    setup_move_uci: str
    solution_moves_uci: tuple[str, ...]
    rating: int
    rating_deviation: int
    popularity: int
    number_of_plays: int
    themes: tuple[str, ...]
    game_url: str
    opening_tags: tuple[str, ...]
    source: str
    # Preserve every source column and raw value, including future extra columns.
    source_fields: tuple[tuple[str, str], ...]

    @property
    def original_moves_uci(self) -> tuple[str, ...]:
        return (self.setup_move_uci,) + self.solution_moves_uci


@dataclass
class LoadStats:
    processed: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_counts: Counter = field(default_factory=Counter)
    error_examples: list[str] = field(default_factory=list)

    @property
    def unique_puzzles(self) -> int:
        """Accepted puzzles are unique; rejected IDs do not inflate this count."""
        return self.accepted


def _validate_row(row: dict[str, str], line: int) -> PuzzleRecord:
    puzzle_id = row.get("PuzzleId") or ""

    def reject(reason, message):
        raise PuzzleValidationError(reason, message, line=line, puzzle_id=puzzle_id)

    if None in row or any(value is None for value in row.values()):
        reject("missing_fields", "row width does not match the CSV header")
    for name in ("PuzzleId", "FEN", "Rating", "RatingDeviation", "Popularity", "NbPlays", "GameUrl"):
        if not row[name].strip():
            reject("missing_fields", f"{name} must not be empty")
    if any(char.isspace() for char in puzzle_id):
        reject("invalid_id", "PuzzleId must not contain whitespace")

    numeric = {}
    for name in ("Rating", "RatingDeviation", "Popularity", "NbPlays"):
        value = row[name]
        if re.fullmatch(r"[+-]?[0-9]+", value) is None:
            reject("invalid_metadata", f"{name} must be an integer, got {value!r}")
        try:
            number = int(value)
        except ValueError:
            reject("invalid_metadata", f"{name} exceeds the supported integer size")
        if (name == "Popularity" and not -100 <= number <= 100) or (name != "Popularity" and number < 0):
            reject("invalid_metadata", f"{name} is out of range: {number}")
        numeric[name] = number

    try:
        if len(row["FEN"].split()) != 6:
            raise ValueError("expected all six FEN fields")
        board = chess.Board(row["FEN"], chess960=False)
    except ValueError as exc:
        reject("invalid_fen", str(exc))
    if not board.is_valid():
        reject("invalid_board", f"invalid standard-chess position (status={board.status()})")

    moves = tuple(row["Moves"].split())
    if len(moves) < 2:
        reject("empty_solution", "need a setup move and at least one solution move")
    puzzle_fen = ""
    for index, uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            reject("invalid_moves", f"move {index + 1} is not UCI: {uci!r}")
        if not board.is_legal(move):
            reject("invalid_moves", f"move {index + 1} is illegal: {uci!r}")
        board.push(move)
        if index == 0:
            puzzle_fen = board.fen()

    return PuzzleRecord(
        puzzle_id=puzzle_id, original_fen=row["FEN"], puzzle_fen=puzzle_fen,
        setup_move_uci=moves[0], solution_moves_uci=moves[1:],
        rating=numeric["Rating"], rating_deviation=numeric["RatingDeviation"],
        popularity=numeric["Popularity"], number_of_plays=numeric["NbPlays"],
        themes=tuple(row["Themes"].split()), game_url=row["GameUrl"],
        opening_tags=tuple(row["OpeningTags"].split()), source=SOURCE,
        source_fields=tuple(row.items()),
    )


class PuzzleCSVReader(Iterator[PuzzleRecord]):
    """Validate a caller-owned text stream one row at a time.

    Strict mode raises at the first rejected row; permissive mode counts and
    skips it. Header/CSV syntax/encoding/decompression errors are always fatal:
    safely resynchronizing a broken CSV stream is not guaranteed.
    max_records counts accepted rows, max_rows counts all data rows. Neither
    includes the header. Zero is allowed. Blank physical lines are CSV blanks,
    not records. Duplicate IDs are rejected after their first occurrence,
    even if that occurrence was invalid. Only IDs and five error examples are
    retained, so memory is O(unique IDs scanned), not O(dataset contents).
    """

    def __init__(self, stream: TextIO, *, strict: bool = True,
                 max_records: int | None = None, max_rows: int | None = None):
        for name, limit in (("max_records", max_records), ("max_rows", max_rows)):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if type(strict) is not bool:
            raise TypeError("strict must be a bool")
        self.strict = strict
        self.max_records = max_records
        self.max_rows = max_rows
        self.stats = LoadStats()
        self._seen: set[str] = set()
        self._reader = csv.DictReader(stream, strict=True)
        try:
            self.fieldnames = tuple(self._reader.fieldnames or ())
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV header: {exc}") from exc
        if self.fieldnames:
            missing = set(REQUIRED_COLUMNS) - set(self.fieldnames)
            if missing or len(set(self.fieldnames)) != len(self.fieldnames) or "" in self.fieldnames:
                raise ValueError(f"Invalid CSV header: missing columns {sorted(missing)} or duplicate/empty names")

    def __iter__(self) -> "PuzzleCSVReader":
        return self

    def __next__(self) -> PuzzleRecord:
        while True:
            if ((self.max_records is not None and self.stats.accepted >= self.max_records)
                    or (self.max_rows is not None and self.stats.processed >= self.max_rows)):
                raise StopIteration
            try:
                row = next(self._reader)
            except csv.Error as exc:
                raise ValueError(f"Invalid CSV near line {self._reader.line_num}: {exc}") from exc
            self.stats.processed += 1
            puzzle_id = row.get("PuzzleId") or ""
            try:
                if puzzle_id and puzzle_id in self._seen:
                    raise PuzzleValidationError("duplicate_id", "ID already encountered",
                                                line=self._reader.line_num, puzzle_id=puzzle_id)
                if puzzle_id:
                    self._seen.add(puzzle_id)
                record = _validate_row(row, self._reader.line_num)
            except PuzzleValidationError as exc:
                self.stats.rejected += 1
                self.stats.rejection_counts[exc.reason] += 1
                if len(self.stats.error_examples) < 5:
                    self.stats.error_examples.append(str(exc))
                if self.strict:
                    raise
                continue
            self.stats.accepted += 1
            return record


@contextmanager
def open_puzzles(path: str | Path, *, strict: bool = True,
                 max_records: int | None = None, max_rows: int | None = None
                 ) -> Iterator[PuzzleCSVReader]:
    """Open a .csv or .csv.zst; close all handles even after early termination."""
    path = Path(path)
    options = dict(strict=strict, max_records=max_records, max_rows=max_rows)
    if path.name.endswith(".csv.zst"):
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
                with io.TextIOWrapper(decoded, encoding="utf-8-sig", newline="") as text:
                    yield PuzzleCSVReader(text, **options)
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as text:
            yield PuzzleCSVReader(text, **options)
    else:
        raise ValueError("Expected a .csv or .csv.zst puzzle file")
