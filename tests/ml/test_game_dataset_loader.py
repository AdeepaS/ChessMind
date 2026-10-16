"""Offline PGN validation, sampling, provenance and failure-path tests."""

import io
from dataclasses import FrozenInstanceError

import pytest
import zstandard

from scripts import download_lichess_game_sample as acquisition
from scripts.inspect_lichess_games import inspect_games
from src.chessmind.ml.data.game_dataset_loader import (
    GameFilters, GamePGNReader, GameResourceLimit, GameValidationError, open_games,
)


def pgn(game_id="abcdefgh", moves="1. e4 { [%eval 0.2] [%clk 0:10:00] } e5 2. Nf3 Nc6 1-0", **changes):
    headers = {"Event": "Rated Rapid game", "Site": f"https://lichess.org/{game_id}",
               "White": f"white{game_id}", "Black": f"black{game_id}",
               "Result": "1-0", "WhiteElo": "1000", "BlackElo": "1100",
               "UTCDate": "2025.01.01", "UTCTime": "00:00:00", "TimeControl": "600+0"}
    headers.update(changes)
    return "\n".join(f'[{k} "{v}"]' for k, v in headers.items() if v is not None) + "\n\n" + moves + "\n\n"


def reader(raw, **options):
    return GamePGNReader(io.StringIO(raw), **options)


@pytest.mark.parametrize("compressed", [False, True])
def test_files_and_immutable_provenance(tmp_path, compressed):
    raw = pgn()
    path = tmp_path / ("sample.pgn.zst" if compressed else "sample.pgn")
    path.write_bytes(zstandard.ZstdCompressor().compress(raw.encode()) if compressed else raw.encode())
    with open_games(path) as games:
        record, = games
    assert record.uci_moves == ("e2e4", "e7e5", "g1f3", "b8c6")
    assert record.original_pgn == raw
    assert record.source_archive == str(path)
    assert record.source_index == 1
    assert record.average_rating == 1050
    assert record.opening is record.eco is None
    assert record.player_keys[0] == "lichess:whiteabcdefgh"
    with pytest.raises(FrozenInstanceError):
        record.game_id = "changed"


@pytest.mark.parametrize("changes,moves,reason", [
    ({"White": None}, None, "missing_headers"),
    ({"Variant": "Chess960"}, None, "unsupported_variant"),
    ({"WhiteTitle": "BOT"}, None, "bot_player"),
    ({"BlackTitle": "BOT"}, None, "bot_player"),
    ({"WhiteElo": "?"}, None, "invalid_rating"),
    ({"WhiteElo": "0"}, None, "invalid_rating"),
    ({"TimeControl": "60+0"}, None, "time_control_filter"),
    ({"TimeControl": "-"}, None, "time_control_filter"),
    ({"Event": "Casual Rapid game"}, None, "unrated_game"),
    ({"Result": "*"}, "1. e4 *", "incomplete_game"),
    ({}, "1. e4 e5", "incomplete_game"),
    ({}, "1. e4 e5 0-1", "incomplete_game"),
    ({}, "1. e4 e5 2. Bh6 1-0", "invalid_moves"),
    ({}, "1. e4 garbage e5 1-0", "malformed_movetext"),
    ({}, "1. e4 {never closes 1-0", "malformed_movetext"),
    ({}, "1. e4 1-0 e5", "trailing_movetext"),
    ({"FEN": "invalid", "SetUp": "1"}, None, "invalid_moves"),
    ({"Site": "https://elsewhere/abcdefgh"}, None, "invalid_id"),
])
def test_rejections(changes, moves, reason):
    raw = pgn(**changes, **({"moves": moves} if moves is not None else {}))
    strict = reader(raw)
    with pytest.raises(GameValidationError) as error:
        list(strict)
    assert error.value.reason == reason
    permissive = reader(raw, strict=False)
    assert list(permissive) == []
    assert permissive.stats.rejection_counts == {reason: 1}


@pytest.mark.parametrize("result", ["1-0", "0-1", "1/2-1/2"])
def test_completed_nonterminal_games(result):
    game, = reader(pgn(Result=result, moves=f"1. e4 e5 {result}"))
    assert game.result == result


def test_multiple_duplicate_limits_and_determinism():
    raw = pgn() + pgn() + pgn("ijklmnop")
    first = reader(raw, strict=False)
    second = reader(raw, strict=False)
    assert list(first) == list(second)
    assert first.stats == second.stats
    assert first.stats.accepted == 2
    assert first.stats.rejection_counts == {"duplicate_id": 1}
    limited = reader(raw, max_records=1)
    assert len(list(limited)) == limited.stats.processed == 1
    assert list(reader(raw, max_records=0)) == []
    assert len(list(reader(raw, strict=False, max_games=2))) == 1


@pytest.mark.parametrize("raw", ["", "\n\n  \n"])
def test_empty(raw):
    games = reader(raw)
    assert list(games) == []
    assert games.stats.processed == 0


def test_filters_missing_metadata_and_multiline_comment():
    raw = pgn(UTCDate=None, UTCTime=None,
              moves="1. e4 {a\n\n[not a header]\nb} e5 1-0")
    game, = reader(raw)
    assert game.utc_date is game.utc_time is None
    for filters, reason in ((GameFilters(min_rating=1200), "rating_filter"),
                            (GameFilters(min_plies=10), "length_filter")):
        with pytest.raises(GameValidationError, match=reason):
            list(reader(raw, filters=filters))


def test_oversize_is_fatal_and_unknown_extension(tmp_path):
    with pytest.raises(GameResourceLimit):
        list(reader(pgn(), strict=False, max_game_chars=30))
    with pytest.raises(ValueError):
        with open_games(tmp_path / "sample.txt"):
            pass


def test_compression_window_larger_than_tiny_fixture(tmp_path):
    path = tmp_path / "window.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress((pgn() * 1000).encode()))
    with open_games(path, max_records=1) as games:
        assert len(list(games)) == 1


def test_invalid_first_occurrence_still_reserves_id():
    games = reader(pgn(WhiteElo="?") + pgn(), strict=False)
    assert list(games) == []
    assert games.stats.rejection_counts == {"invalid_rating": 1, "duplicate_id": 1}


def test_invalid_position_and_terminal_result():
    with pytest.raises(GameValidationError, match="invalid_position"):
        list(reader(pgn(FEN="8/8/8/8/8/8/8/8 w - - 0 1", SetUp="1",
                        moves="1-0")))
    with pytest.raises(GameValidationError, match="result_mismatch"):
        list(reader(pgn(moves="1. f3 e5 2. g4 Qh4# 1-0")))
    assert len(list(reader(pgn(Result="0-1", moves="1. f3 e5 2. g4 Qh4# 0-1")))) == 1


def test_classical_and_time_class_boundaries():
    from src.chessmind.ml.data.game_dataset_loader import time_class
    assert [time_class(t) for t in ("179+0", "180+0", "479+0", "480+0", "1499+0", "1500+0")] == [
        "bullet", "blitz", "blitz", "rapid", "rapid", "classical"]
    record, = reader(pgn(TimeControl="1800+0", Event="Rated Classical game"))
    assert record.time_class == "classical"


class Response(io.BytesIO):
    status = 200

    def geturl(self):
        return acquisition.DEFAULT_URL


def mock_download(monkeypatch, raw):
    data = zstandard.ZstdCompressor().compress(raw.encode())
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: Response(data))
    return data


def test_acquisition_atomic_validation_and_inspection(tmp_path, monkeypatch):
    raw = pgn() + pgn("ijklmnop", White="whiteabcdefgh") + pgn("qrstuvwx", WhiteElo="1500", BlackElo="1600")
    data = mock_download(monkeypatch, raw)
    output = tmp_path / "games"
    manifest = acquisition.acquire_sample(output, groups=((800, 1200, 1), (1200, 1800, 1)),
                                          filters=GameFilters())
    assert manifest["complete"]
    assert manifest["compressed_bytes_read"] == len(data)
    assert manifest["selection_filter_counts"] == {"rating_group_full": 1}
    report = inspect_games(output / "lichess_game_sample.pgn", strict=True,
                           manifest_path=output / "source_manifest.json")
    assert report["manifest_verified"] and report["unique_players"] == 4
    assert report["games_with_eval_annotations"] == 2
    assert report["mean_plies"] == 4
    with pytest.raises(FileExistsError):
        acquisition.acquire_sample(output)
    (output / "lichess_game_sample.pgn").write_text(pgn(), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        inspect_games(output / "lichess_game_sample.pgn", manifest_path=output / "source_manifest.json")


def test_shortage_and_repeat_player(tmp_path, monkeypatch):
    mock_download(monkeypatch, pgn() + pgn("ijklmnop", White="WHITEabcdefgh"))
    manifest = acquisition.acquire_sample(tmp_path / "games", groups=((800, 1200, 2),), filters=GameFilters())
    assert not manifest["complete"]
    assert manifest["rating_groups"][0]["shortage"] == 1
    assert manifest["selection_filter_counts"] == {"repeat_player": 1}


def test_budget_never_publishes_truncated_game(tmp_path, monkeypatch):
    mock_download(monkeypatch, pgn())
    manifest = acquisition.acquire_sample(tmp_path / "games", max_bytes=10, filters=GameFilters())
    assert manifest["accepted_count"] == 0
    assert manifest["stop_reason"] == "compressed_byte_budget"
    assert (tmp_path / "games/lichess_game_sample.pgn").read_bytes() == b""


@pytest.mark.parametrize("error", [OSError("connection lost"), KeyboardInterrupt()])
def test_interruption_cleanup(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(acquisition, "urlopen", fail)
    with pytest.raises(type(error)):
        acquisition.acquire_sample(tmp_path / "games")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("options", [
    {"max_bytes": 0}, {"timeout": float("nan")}, {"groups": ((800, 1200, 0),)},
    {"groups": ((800, 1300, 1), (1200, 1800, 1))}, {"source_url": "https://example.com/file"},
])
def test_invalid_acquisition_options(tmp_path, options):
    with pytest.raises(ValueError):
        acquisition.acquire_sample(tmp_path / "games", **options)
