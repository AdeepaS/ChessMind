"""Offline checks of statistics, multi-label semantics and reproducible reports."""

import csv
import hashlib
import io
import json
from dataclasses import replace

import pytest

from scripts.analyze_puzzle_dataset import main, write_research_report
from src.chessmind.ml.data.puzzle_exploration import (
    analyze_dataset, explore_puzzles, numeric_summary, rating_distribution, write_reports,
)
from src.chessmind.ml.data.puzzle_loader import PuzzleCSVReader, REQUIRED_COLUMNS


def csv_fixture(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
    writer.writeheader()
    for changes in rows:
        row = dict(zip(REQUIRED_COLUMNS, (
            "p1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4 e7e5", "1000", "80", "90", "10", "fork pin", "https://lichess.org/example#0", "",
        )))
        row.update(changes)
        writer.writerow(row)
    return stream.getvalue()


def puzzle(**changes):
    record = next(PuzzleCSVReader(io.StringIO(csv_fixture([{}]))))
    return replace(record, **changes)


def test_general_and_numeric_statistics():
    rows = [puzzle(puzzle_id=str(i), rating=value) for i, value in enumerate((100, 200, 300, 400))]
    result = explore_puzzles(iter(rows))
    general = result["general"]
    assert general["total_puzzles"] == general["unique_puzzles"] == 4
    assert general["ratings"] == {
        "count": 4, "min": 100, "max": 400, "mean": 250, "median": 250,
        "population_stddev": pytest.approx(111.80339887498948),
    }
    assert numeric_summary([3, 1, 2])["median"] == 2
    assert numeric_summary([1])["population_stddev"] == 0
    assert result["metadata"]["popularity"]["mean"] == 90
    assert result["metadata"]["number_of_plays"]["median"] == 10


def test_rating_boundaries_and_percentages():
    rows = rating_distribution([0, 499, 500, 999, 1000, 5000], (0, 500, 1000))
    assert [row["count"] for row in rows] == [2, 2, 2]
    assert [row["upper_exclusive"] for row in rows] == [500, 1000, None]
    assert [row["percentage"] for row in rows] == [33.333333] * 3
    assert rating_distribution([0, 5000], [0]) == [
        {"lower_inclusive": 0, "upper_exclusive": None, "count": 2, "percentage": 100.0}]


@pytest.mark.parametrize("edges", [[], [1, 100], [0, 0], [0, -1], [0, 2, 1], [False, 10], [0, 1.5]])
def test_invalid_buckets(edges):
    with pytest.raises(ValueError, match="bucket"):
        rating_distribution([], edges)


@pytest.mark.parametrize("rating", [-1, 1.5, True, float("nan"), "1000"])
def test_invalid_rating(rating):
    with pytest.raises(ValueError, match="Ratings"):
        rating_distribution([rating])
    with pytest.raises(ValueError, match="rating"):
        explore_puzzles([puzzle(rating=rating)])


def test_multi_label_counts_duplicate_tokens_and_pairs():
    result = explore_puzzles([
        puzzle(puzzle_id="a", themes=("pin", "fork", "fork")),
        puzzle(puzzle_id="b", themes=("mate", "fork", "pin")),
        puzzle(puzzle_id="c", themes=()),
    ], rare_threshold=1)
    assert result["themes"] == [
        {"theme": "fork", "count": 2, "percentage": 66.666667},
        {"theme": "pin", "count": 2, "percentage": 66.666667},
        {"theme": "mate", "count": 1, "percentage": 33.333333},
    ]
    assert result["total_theme_assignments"] == 5
    assert result["duplicate_theme_tokens"] == 1
    assert result["themes_per_puzzle"]["mean"] == pytest.approx(5 / 3)
    assert [(row["theme_a"], row["theme_b"], row["count"]) for row in result["theme_cooccurrence"]] == [
        ("fork", "pin", 2), ("fork", "mate", 1), ("mate", "pin", 1)]
    assert result["rare_themes"] == [{"theme": "mate", "count": 1, "percentage": 33.333333}]
    assert [row["theme_count"] for row in result["theme_count_distribution"]] == [0, 2, 3]


def test_duplicate_record_ids_are_visible_to_direct_callers():
    result = explore_puzzles([puzzle(), puzzle()])
    assert result["general"]["total_puzzles"] == 2
    assert result["general"]["unique_puzzles"] == result["general"]["duplicate_record_ids"] == 1


def test_order_does_not_affect_statistics_or_report_bytes(tmp_path):
    records = [puzzle(puzzle_id="a", themes=("pin", "fork")), puzzle(puzzle_id="b", themes=("mate", "fork"))]
    left = explore_puzzles(records)
    right = explore_puzzles([replace(p, themes=tuple(reversed(p.themes))) for p in reversed(records)])
    assert left == right
    write_reports(left, tmp_path / "first")
    write_reports(right, tmp_path / "second")
    for file in (tmp_path / "first").iterdir():
        assert file.read_bytes() == (tmp_path / "second" / file.name).read_bytes()


def test_missing_optional_metadata():
    records = [puzzle(themes=()), puzzle(opening_tags=("Example",), source_fields=(("DailyDate", ""),)),
               puzzle(source_fields=(("DailyDate", "123"),))]
    missing = explore_puzzles(records)["metadata"]["missing_optional"]
    assert missing == {"themes_empty": 1, "opening_tags_empty": 2,
                       "daily_date_empty": 1, "daily_date_column_absent": 1}


def test_empty_dataset_reports(tmp_path):
    input_file = tmp_path / "empty.csv"
    input_file.write_text(csv_fixture([]), encoding="utf-8")
    result = analyze_dataset(input_file)
    assert result["general"]["total_puzzles"] == result["general"]["unique_puzzles"] == 0
    assert result["general"]["ratings"]["mean"] is None
    assert result["themes"] == result["theme_cooccurrence"] == result["rare_themes"] == []
    assert all(row["count"] == row["percentage"] == 0 for row in result["rating_distribution"])
    write_reports(result, tmp_path / "reports")
    write_research_report(result, tmp_path / "research.md")
    assert json.loads((tmp_path / "reports/dataset_quality.json").read_text()) == result
    assert (tmp_path / "reports/theme_counts.csv").read_text() == "theme,count,percentage\n"
    assert "not available" in (tmp_path / "research.md").read_text()


def test_csv_json_and_validation_counts(tmp_path):
    path = tmp_path / "puzzles.csv"
    path.write_text(csv_fixture([{}, {"PuzzleId": "bad", "FEN": "bad"}, {}]), encoding="utf-8")
    result = analyze_dataset(path)
    assert result["validation"]["processed"] == 3
    assert result["validation"]["accepted"] == 1
    assert result["validation"]["rejected"] == 2
    assert result["validation"]["rejection_counts"] == {"invalid_fen": 1, "duplicate_id": 1}
    write_reports(result, tmp_path / "reports")
    quality = json.loads((tmp_path / "reports/dataset_quality.json").read_text())
    assert quality == result
    for filename, key in (("theme_counts.csv", "themes"), ("rating_distribution.csv", "rating_distribution"),
                          ("theme_cooccurrence.csv", "theme_cooccurrence")):
        with (tmp_path / "reports" / filename).open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == len(result[key])
        assert [int(row["count"]) for row in rows] == [row["count"] for row in result[key]]
    with pytest.raises(ValueError, match="invalid_fen"):
        analyze_dataset(path, strict=True)


def test_provenance_verification(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(csv_fixture([{}]), encoding="utf-8")
    manifest_path = tmp_path / "source_manifest.json"
    manifest = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sample_count": 1}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = analyze_dataset(path, manifest_path=manifest_path)["source"]
    assert source["manifest_hash_verified"] and source["manifest_count_matches"]
    manifest["sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_dataset(path, manifest_path=manifest_path)


@pytest.mark.parametrize("changes", [{"themes": "fork"}, {"themes": ("",)}, {"themes": ("two words",)},
                                     {"opening_tags": None}, {"popularity": 101}, {"number_of_plays": -1},
                                     {"puzzle_id": ""}])
def test_invalid_record_metadata(changes):
    with pytest.raises(ValueError):
        explore_puzzles([puzzle(**changes)])


def test_invalid_record_type_and_threshold():
    with pytest.raises(TypeError, match="PuzzleRecord"):
        explore_puzzles([{}])
    for threshold in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="rare_threshold"):
            explore_puzzles([], rare_threshold=threshold)
    with pytest.raises(ValueError, match="integers"):
        numeric_summary([float("inf")])


def test_cli_paths_configuration_and_repeatability(tmp_path, monkeypatch, capsys):
    path = tmp_path / "sample.csv"
    path.write_text(csv_fixture([{}]), encoding="utf-8")
    reports, research = tmp_path / "reports", tmp_path / "study.md"
    monkeypatch.setattr("sys.argv", ["analyze_puzzle_dataset", "--input", str(path),
                                     "--output-dir", str(reports), "--research-output", str(research),
                                     "--rating-buckets", "0", "1000", "--rare-threshold", "2"])
    main()
    assert "accepted 1" in capsys.readouterr().out
    first = {p.name: p.read_bytes() for p in reports.iterdir()}
    research_bytes = research.read_bytes()
    main()
    assert first == {p.name: p.read_bytes() for p in reports.iterdir()}
    assert research.read_bytes() == research_bytes
    assert json.loads((reports / "dataset_quality.json").read_text())["rating_distribution"][1]["count"] == 1
