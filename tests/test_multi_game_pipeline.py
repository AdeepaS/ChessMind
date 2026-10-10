
import json
from importlib import import_module

import pytest

from src.chessmind.pgn.multi_game_parser import load_multiple_pgn
from src.chessmind.players.player_identifier import find_player_games
from src.chessmind.players.player_statistics import (
    aggregate_player_statistics,
)
from src.chessmind.players.recurring_weakness_analyzer import (
    detect_recurring_weaknesses,
)
from src.chessmind.reports.player_report import (
    generate_player_report,
    save_player_report,
)


PHASES = ("opening", "middlegame", "endgame")


def make_stats(phase_data):
    """
    Construct statistics using the existing analyzer's format.

    phase_data example:
        {"opening": (10, 200)}
    means 10 moves and 200 total centipawns lost.
    """

    phases = {}

    for phase in PHASES:
        moves, cpl = phase_data.get(phase, (0, 0))

        phases[phase] = {
            "moves": moves,
            "total_centipawn_loss": cpl,
            "acpl": cpl / moves if moves else 0,
        }

    total_moves = sum(p["moves"] for p in phases.values())
    total_cpl = sum(
        p["total_centipawn_loss"] for p in phases.values()
    )

    return {
        "moves": total_moves,
        "total_centipawn_loss": total_cpl,
        "acpl": total_cpl / total_moves if total_moves else 0,
        "excellent": total_moves,
        "good": 0,
        "inaccuracy": 0,
        "mistake": 0,
        "blunder": 0,
        "phases": phases,
    }


def make_record(number, color, result, phase_data):
    return {
        "game_number": number,
        "color": color,
        "opponent": "Opponent",
        "result": result,
        "analysis": [],
        "stats": {
            color: make_stats(phase_data),
        },
    }


@pytest.fixture
def sample_player_games():
    games = load_multiple_pgn("data/raw/sample_games.pgn")
    return find_player_games(games, "Adeepa")


@pytest.fixture
def analyzed_records():
    return [
        make_record(
            1, "white", "1-0",
            {"opening": (10, 200), "middlegame": (20, 1200)},
        ),
        make_record(
            2, "black", "0-1",
            {"opening": (10, 300), "middlegame": (10, 800)},
        ),
        make_record(
            3, "white", "1/2-1/2",
            {"opening": (10, 100), "middlegame": (20, 1400)},
        ),
    ]


# ---------------------------------------
# PGN loading tests
# ---------------------------------------

def test_load_multiple_games(sample_player_games):
    assert len(sample_player_games) == 3


def test_empty_pgn_file(tmp_path):
    file_path = tmp_path / "empty.pgn"
    file_path.write_text("", encoding="utf-8")

    assert load_multiple_pgn(file_path) == []


def test_invalid_pgn_file(tmp_path):
    file_path = tmp_path / "invalid.pgn"

    file_path.write_text(
        '[Event "Invalid Game"]\n\n'
        '1. e4 e5 2.Ke3 1-0\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_multiple_pgn(file_path)


# ---------------------------------------
# Player identification tests
# ---------------------------------------

def test_identify_white_and_black(sample_player_games):
    assert [g["color"] for g in sample_player_games] == [
        "white", "black", "white"
    ]


def test_player_name_normalization():
    games = load_multiple_pgn("data/raw/sample_games.pgn")

    assert len(find_player_games(games, " ADEEPA ")) == 3
    assert len(find_player_games(games, "adeepa")) == 3


def test_unknown_player():
    games = load_multiple_pgn("data/raw/sample_games.pgn")

    assert find_player_games(games, "Unknown Player") == []


def test_empty_player_name():
    with pytest.raises(ValueError):
        find_player_games([], "   ")


# ---------------------------------------
# Statistics aggregation tests
# ---------------------------------------

def test_weighted_acpl(analyzed_records):
    stats = aggregate_player_statistics(analyzed_records)

    # Total: 1400 + 1100 + 1500 = 4000 cp
    # Moves: 30 + 20 + 30 = 80
    assert stats["total_moves"] == 80
    assert stats["total_centipawn_loss"] == 4000
    assert stats["overall_acpl"] == 50.0

    # Opening: (200 + 300 + 100) / 30
    assert stats["phase_statistics"]["opening"]["acpl"] == 20.0

    # Middlegame: (1200 + 800 + 1400) / 50
    assert stats["phase_statistics"]["middlegame"]["acpl"] == 68.0


def test_results_from_player_perspective(analyzed_records):
    stats = aggregate_player_statistics(analyzed_records)

    assert stats["total_games"] == 3
    assert stats["wins"] == 2
    assert stats["draws"] == 1
    assert stats["losses"] == 0


def test_empty_statistics():
    stats = aggregate_player_statistics([])

    assert stats["total_games"] == 0
    assert stats["total_moves"] == 0
    assert stats["overall_acpl"] is None


# ---------------------------------------
# Recurring weakness tests
# ---------------------------------------

def test_recurring_middlegame_weakness(analyzed_records):
    summary = aggregate_player_statistics(analyzed_records)

    report = {
        "games": analyzed_records,
        "summary": summary,
    }

    weaknesses = detect_recurring_weaknesses(report)

    assert weaknesses["middlegame"]["eligible_games"] == 3
    assert weaknesses["middlegame"]["weak_games"] == 3
    assert weaknesses["middlegame"]["is_recurring_weakness"]

    assert not weaknesses["opening"]["is_recurring_weakness"]


def test_insufficient_games(analyzed_records):
    games = analyzed_records[:2]

    report = {
        "games": games,
        "summary": aggregate_player_statistics(games),
    }

    weaknesses = detect_recurring_weaknesses(report)

    assert not weaknesses["middlegame"]["is_recurring_weakness"]


def test_invalid_weakness_threshold(analyzed_records):
    report = {
        "games": analyzed_records,
        "summary": aggregate_player_statistics(analyzed_records),
    }

    with pytest.raises(ValueError):
        detect_recurring_weaknesses(
            report,
            frequency_threshold=1.5,
        )


# ---------------------------------------
# Report generation tests
# ---------------------------------------

def test_generate_player_report(analyzed_records):
    data = {
        "games": analyzed_records,
        "summary": aggregate_player_statistics(analyzed_records),
    }

    report = generate_player_report("Adeepa", data)

    assert report["player"]["name"] == "Adeepa"
    assert report["overall_performance"]["total_games"] == 3
    assert len(report["games"]) == 3

    assert (
        "middlegame"
        in report["recurring_weaknesses"]["detected_phases"]
    )


def test_save_player_report(tmp_path, analyzed_records):
    data = {
        "games": analyzed_records,
        "summary": aggregate_player_statistics(analyzed_records),
    }

    report = generate_player_report("Adeepa", data)

    path = tmp_path / "reports" / "player_report.json"

    save_player_report(report, path)

    assert path.exists()

    with path.open(encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report


# ---------------------------------------
# Complete pipeline integration
# ---------------------------------------

def test_pipeline_with_mock_engine(
    sample_player_games,
    monkeypatch,
):
    """
    Exercise the complete Day 5 pipeline.

    Replace only the expensive Stockfish analysis,
    keeping actual game parsing and aggregation.
    """

    module = import_module(
        "src.chessmind.players.multi_game_analyzer"
    )

    def fake_analyze_game(game, depth=12):
        return [
            {
                "color": "white",
                "phase": "opening",
                "centipawn_loss": 10,
                "classification": "Excellent",
            },
            {
                "color": "black",
                "phase": "opening",
                "centipawn_loss": 20,
                "classification": "Excellent",
            },
        ]

    monkeypatch.setattr(
        module,
        "analyze_game",
        fake_analyze_game,
    )

    result = module.analyze_player_games(
        sample_player_games,
        depth=8,
    )

    assert result["summary"]["total_games"] == 3
    assert result["summary"]["total_moves"] == 3
    assert result["summary"]["total_centipawn_loss"] == 40
    assert result["summary"]["overall_acpl"] == 13.33

    report = generate_player_report("Adeepa", result)

    assert len(report["games"]) == 3
