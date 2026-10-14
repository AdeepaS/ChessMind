"""Synthetic offline fixtures for puzzle validation and bounded acquisition."""

import csv
import hashlib
import io
import json
from dataclasses import FrozenInstanceError
from datetime import datetime
from urllib.error import URLError

import chess
import pytest
import zstandard

from scripts import download_lichess_puzzle_sample as acquisition
from scripts.inspect_lichess_puzzles import inspect_puzzles
from src.chessmind.ml.data.puzzle_loader import (
    DATASET_URL, REQUIRED_COLUMNS, SOURCE, PuzzleCSVReader,
    PuzzleValidationError, open_puzzles,
)


def row(puzzle_id="test1", **changes):
    result = dict(zip(REQUIRED_COLUMNS, (
        puzzle_id, chess.STARTING_FEN, "e2e4 e7e5 g1f3 b8c6", "1500", "80",
        "95", "100", "fork opening unknownFutureTheme", "https://lichess.org/test#0", "",
    )))
    result.update(changes)
    return result


def csv_text(rows=(), columns=REQUIRED_COLUMNS):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def reader_for(rows, **options):
    return PuzzleCSVReader(io.StringIO(csv_text(rows)), **options)


@pytest.mark.parametrize("compressed", [False, True])
def test_load_valid_file_and_reconstruct_all_moves(tmp_path, compressed):
    path = tmp_path / ("puzzles.csv.zst" if compressed else "puzzles.csv")
    raw = csv_text([row()]).encode("utf-8")
    path.write_bytes(zstandard.ZstdCompressor().compress(raw) if compressed else raw)
    with open_puzzles(path) as puzzles:
        records = list(puzzles)
        assert puzzles.stats.processed == puzzles.stats.accepted == puzzles.stats.unique_puzzles == 1
        assert puzzles.stats.rejected == 0
    puzzle = records[0]
    assert puzzle.original_fen == chess.STARTING_FEN
    assert puzzle.setup_move_uci == "e2e4"
    assert puzzle.solution_moves_uci == ("e7e5", "g1f3", "b8c6")
    assert puzzle.original_moves_uci == tuple(row()["Moves"].split())
    board = chess.Board(puzzle.original_fen)
    board.push_uci(puzzle.setup_move_uci)
    assert board.fen() == puzzle.puzzle_fen
    assert board.turn == chess.BLACK
    solution_board = chess.Board(puzzle.puzzle_fen)
    for move in puzzle.solution_moves_uci:
        board.push_uci(move)
        solution_board.push_uci(move)
    assert board.fen() == solution_board.fen()
    assert (puzzle.rating, puzzle.rating_deviation, puzzle.popularity, puzzle.number_of_plays) == (1500, 80, 95, 100)
    assert puzzle.source == SOURCE
    assert puzzle.themes == ("fork", "opening", "unknownFutureTheme")
    assert puzzle.opening_tags == ()
    assert dict(puzzle.source_fields) == row()
    with pytest.raises(FrozenInstanceError):
        puzzle.rating = 1


@pytest.mark.parametrize("changes,reason", [
    ({"FEN": "not a fen"}, "invalid_fen"),
    ({"FEN": "8/8/8/8/8/8/8/8 w - - 0 1"}, "invalid_board"),
    ({"FEN": "8/8/8/8/8/8/4k3/4K3 w - - 0 1"}, "invalid_board"),
    ({"Moves": "e2e5 e7e5"}, "invalid_moves"),
    ({"Moves": "e2e4 e7e4"}, "invalid_moves"),
    ({"Moves": "e2e4 e7e5 g1g3"}, "invalid_moves"),
    ({"Moves": "not-uci e7e5"}, "invalid_moves"),
    ({"Moves": "e2e4 not-uci"}, "invalid_moves"),
    ({"Moves": "0000 e7e5"}, "invalid_moves"),
    ({"Moves": ""}, "empty_solution"),
    ({"Moves": "e2e4"}, "empty_solution"),
    ({"PuzzleId": ""}, "missing_fields"),
    ({"GameUrl": ""}, "missing_fields"),
    ({"Rating": ""}, "missing_fields"),
    ({"Rating": "NaN"}, "invalid_metadata"),
    ({"Rating": "1.5"}, "invalid_metadata"),
    ({"Rating": "1_500"}, "invalid_metadata"),
    ({"Rating": "9" * 5000}, "invalid_metadata"),
    ({"Rating": "-1"}, "invalid_metadata"),
    ({"RatingDeviation": "-1"}, "invalid_metadata"),
    ({"Popularity": "101"}, "invalid_metadata"),
    ({"Popularity": "-101"}, "invalid_metadata"),
    ({"NbPlays": "-1"}, "invalid_metadata"),
    ({"PuzzleId": "bad id"}, "invalid_id"),
])
def test_rejection_in_strict_and_permissive_modes(changes, reason):
    rows = [row(**changes), row("valid")]
    strict = reader_for(rows)
    with pytest.raises(PuzzleValidationError) as error:
        list(strict)
    assert error.value.reason == reason
    assert error.value.line == 2
    assert "CSV line 2" in str(error.value)
    assert strict.stats.processed == strict.stats.rejected == 1
    permissive = reader_for(rows, strict=False)
    assert [p.puzzle_id for p in permissive] == ["valid"]
    assert permissive.stats.processed == 2
    assert permissive.stats.accepted == permissive.stats.rejected == 1
    assert permissive.stats.rejection_counts == {reason: 1}


@pytest.mark.parametrize("strict", [False, True])
def test_missing_or_duplicate_header_fields_are_always_fatal(strict):
    for header in ("PuzzleId,FEN\n", ",".join(REQUIRED_COLUMNS + ("FEN",)) + "\n"):
        with pytest.raises(ValueError, match="Invalid CSV header"):
            PuzzleCSVReader(io.StringIO(header), strict=strict)


@pytest.mark.parametrize("extra", [False, True])
def test_row_width_mismatch(extra):
    fields = list(row().values())
    fields = fields + ["unexpected"] if extra else fields[:-1]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(REQUIRED_COLUMNS)
    writer.writerow(fields)
    stream.seek(0)
    reader = PuzzleCSVReader(stream, strict=False)
    assert list(reader) == []
    assert reader.stats.rejection_counts == {"missing_fields": 1}


def test_duplicates_and_accepted_limit():
    rows = [row("bad", FEN="bad"), row("one"), row("one"), row("two"), row("three")]
    reader = reader_for(rows, strict=False, max_records=2)
    assert [p.puzzle_id for p in reader] == ["one", "two"]
    assert reader.stats.processed == 4
    assert reader.stats.rejected == 2
    assert reader.stats.rejection_counts == {"invalid_fen": 1, "duplicate_id": 1}
    strict = reader_for([row(), row()])
    with pytest.raises(PuzzleValidationError, match="duplicate_id"):
        list(strict)
    assert strict.stats.accepted == 1


def test_repeated_invalid_id_is_still_duplicate_and_errors_are_bounded():
    reader = reader_for([row(FEN="bad")] * 10, strict=False)
    assert list(reader) == []
    assert reader.stats.rejection_counts == {"invalid_fen": 1, "duplicate_id": 9}
    assert len(reader.stats.error_examples) == 5


@pytest.mark.parametrize("limit,expected", [(0, 0), (1, 1), (5, 2)])
def test_max_records(limit, expected):
    reader = reader_for([row("one"), row("two")], max_records=limit)
    assert len(list(reader)) == expected
    assert reader.stats.processed == expected


@pytest.mark.parametrize("name", ["max_records", "max_rows"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_limits(name, value):
    with pytest.raises(ValueError, match=name):
        reader_for([], **{name: value})


def test_scan_limit_counts_rejections():
    reader = reader_for([row("bad", FEN="bad"), row("good")], strict=False, max_rows=1)
    assert list(reader) == []
    assert reader.stats.processed == reader.stats.rejected == 1


@pytest.mark.parametrize("text", ["", csv_text()])
def test_empty_datasets(text):
    reader = PuzzleCSVReader(io.StringIO(text))
    assert list(reader) == []
    assert reader.stats.processed == reader.stats.accepted == reader.stats.rejected == 0


def test_preserves_extra_columns_and_raw_annotations():
    source = row(Themes="fork  opening fork", OpeningTags="Italian_Game Example", DailyDate="12345")
    reader = PuzzleCSVReader(io.StringIO(csv_text([source], REQUIRED_COLUMNS + ("DailyDate",))))
    puzzle = next(reader)
    assert puzzle.themes == ("fork", "opening", "fork")
    assert puzzle.opening_tags == ("Italian_Game", "Example")
    assert dict(puzzle.source_fields) == source


@pytest.mark.parametrize("fen,moves", [
    ("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", "d7d5 e5d6"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1 e8c8"),
    ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q h8h7"),
])
def test_special_moves_survive_setup_reconstruction(fen, moves):
    puzzle = next(reader_for([row(FEN=fen, Moves=moves)]))
    board = chess.Board(fen)
    board.push_uci(puzzle.setup_move_uci)
    assert puzzle.puzzle_fen == board.fen()
    solution = chess.Board(puzzle.puzzle_fen)
    for move in puzzle.solution_moves_uci:
        board.push_uci(move)
        solution.push_uci(move)
    assert solution.fen() == board.fen()


def test_invalid_compressed_file_is_fatal(tmp_path):
    path = tmp_path / "broken.csv.zst"
    path.write_bytes(b"not a zstandard file")
    with pytest.raises(zstandard.ZstdError):
        with open_puzzles(path, strict=False) as puzzles:
            list(puzzles)


def test_record_limit_does_not_read_next_row():
    class GuardedLines:
        def __iter__(self):
            yield csv_text().strip() + "\n"
            yield csv_text([row()]).splitlines()[1] + "\n"
            raise AssertionError("reader consumed beyond accepted limit")
    assert len(list(PuzzleCSVReader(GuardedLines(), max_records=1))) == 1


@pytest.mark.parametrize("strict", [False, True])
def test_broken_csv_is_fatal(strict):
    reader = PuzzleCSVReader(io.StringIO(csv_text() + '"unterminated'), strict=strict)
    with pytest.raises(ValueError, match="Invalid CSV"):
        list(reader)


def test_inspection(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(csv_text([row("one"), row("two", Rating="2000"), row("bad", FEN="bad")]), encoding="utf-8")
    report = inspect_puzzles(path)
    assert report["processed_rows"] == 3
    assert report["accepted_rows"] == report["unique_puzzles"] == 2
    assert report["rejected_rows"] == 1
    assert report["sample_ids"] == ["one", "two"]
    assert report["ratings"] == {"min": 1500, "max": 2000, "mean": 1750}
    assert report["common_themes"] == {"fork": 2, "opening": 2, "unknownFutureTheme": 2}
    path.write_text(csv_text(), encoding="utf-8")
    assert inspect_puzzles(path)["ratings"] == {"min": None, "max": None, "mean": None}


class FakeResponse(io.BytesIO):
    status = 200


def fake_download(monkeypatch, data):
    response = FakeResponse(data)
    def open_response(request, timeout):
        assert request.full_url == DATASET_URL
        assert timeout > 0
        return response
    monkeypatch.setattr(acquisition, "urlopen", open_response)
    return response


def test_acquisition_manifest_and_preserved_header(tmp_path, monkeypatch):
    columns = REQUIRED_COLUMNS + ("DailyDate",)
    rows = [row("bad", FEN="bad", DailyDate=""), row("one", DailyDate=""), row("two", DailyDate="123")]
    data = zstandard.ZstdCompressor().compress(csv_text(rows, columns).encode())
    response = fake_download(monkeypatch, data)
    output = tmp_path / "puzzles"
    manifest = acquisition.acquire_sample(output, count=2)
    sample = output / acquisition.SAMPLE_FILENAME
    assert response.closed
    assert manifest == json.loads((output / acquisition.MANIFEST_FILENAME).read_text())
    assert manifest["dataset_url"] == DATASET_URL
    assert manifest["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert datetime.fromisoformat(manifest["acquired_at_utc"]).utcoffset().total_seconds() == 0
    assert manifest["sample_count"] == 2
    assert manifest["processed_rows"] == 3
    assert manifest["rejected_rows"] == 1
    with sample.open(newline="") as saved:
        reader = csv.DictReader(saved)
        assert tuple(reader.fieldnames) == columns
        assert list(reader) == rows[1:]
    assert inspect_puzzles(sample)["accepted_rows"] == 2


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError("timeout"), KeyboardInterrupt()])
def test_interrupted_acquisition_never_publishes(tmp_path, monkeypatch, failure):
    # Deliver a header and valid row before failing, so partial writes are exercised.
    compressor = zstandard.ZstdCompressor().compressobj()
    prefix = compressor.compress(csv_text([row()]).encode()) + compressor.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK)
    class InterruptedResponse(FakeResponse):
        def read(self, size):
            data = super().read(size)
            if data:
                return data
            raise failure
    response = InterruptedResponse(prefix)
    monkeypatch.setattr(acquisition, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(type(failure)):
        acquisition.acquire_sample(tmp_path / "puzzles", count=2)
    assert response.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("options", [{}, {"max_rows": 2}, {"max_bytes": 1}])
def test_incomplete_acquisition_never_publishes(tmp_path, monkeypatch, options):
    data = zstandard.ZstdCompressor().compress(csv_text([row(), row()]).encode())
    fake_download(monkeypatch, data)
    with pytest.raises(ValueError, match="Incomplete sample|byte limit"):
        acquisition.acquire_sample(tmp_path / "puzzles", count=2, **options)
    assert list(tmp_path.iterdir()) == []


def test_acquisition_never_overwrites_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "puzzles"
    output.mkdir()
    original = output / "keep.txt"
    original.write_text("original")
    def unexpected_network(*args, **kwargs):
        pytest.fail("network must not be accessed")
    monkeypatch.setattr(acquisition, "urlopen", unexpected_network)
    with pytest.raises(FileExistsError):
        acquisition.acquire_sample(output)
    assert original.read_text() == "original"


def test_download_byte_and_time_budgets():
    response = FakeResponse(b"abcdefgh")
    download = acquisition.BoundedDownload(response, 4, float("inf"))
    assert download.read(100) == b"abcd"
    assert download.bytes_read == 4
    with pytest.raises(ValueError, match="byte limit"):
        download.read(1)
    expired = acquisition.BoundedDownload(response, 4, 0)
    with pytest.raises(TimeoutError, match="time limit"):
        expired.read(1)
