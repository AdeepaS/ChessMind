"""Key identity tests using metadata and tiny fixture files, never Stockfish."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import chess
import chess.engine
import chess.variant
import pytest

from src.chessmind.engine import evaluation_cache_keys as keys


def metadata(**changes):
    values = {"Threads": ("spin", 1), "Hash": ("spin", 16),
              "Ponder": ("check", False), "MultiPV": ("spin", 1),
              "UCI_AnalyseMode": ("check", False), "UCI_Chess960": ("check", False)}
    options = {name: chess.engine.Option(name, kind, value, 1 if kind == "spin" else None,
                                        1024 if kind == "spin" else None, None)
               for name, (kind, value) in values.items()}
    config = {name: value for name, (_, value) in values.items()}
    target = {"Threads": 1, "Hash": 16}
    config.update(changes)
    target.update({k: v for k, v in changes.items() if k not in ("Ponder", "MultiPV", "UCI_AnalyseMode", "UCI_Chess960")})
    return keys.metadata_from_values({"name": "Stockfish test", "author": "fixture"},
                                     options, config, target)


@pytest.fixture
def identity(tmp_path):
    binary = tmp_path / "engine.exe"
    binary.write_bytes(b"deterministic engine fixture")
    return keys.build_engine_identity(binary, metadata())


def key(identity, board=None, limit=None, meta=None, **kwargs):
    return keys.build_cache_key(identity, metadata() if meta is None else meta,
                                chess.Board() if board is None else board,
                                chess.engine.Limit(depth=8) if limit is None else limit, **kwargs)


def replay():
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(move)
    return board


def test_identical_requests_and_independent_replays(identity):
    assert key(identity) == key(identity)
    assert key(identity, replay()) == key(identity, replay())


def test_identical_current_fen_different_history(identity):
    historical = replay()
    reconstructed = chess.Board(historical.fen())
    assert historical.fen() == reconstructed.fen()
    assert key(identity, historical) != key(identity, reconstructed)


@pytest.mark.parametrize("field,value", [("depth", 9), ("nodes", 1000), ("time", 0.5),
    ("mate", 2), ("white_clock", 10.0), ("black_clock", 11.0), ("white_inc", 1.0),
    ("black_inc", 2.0), ("remaining_moves", 5), ("clock_id", "clock")])
def test_each_limit_field_separates_requests(identity, field, value):
    limit = chess.engine.Limit(depth=8)
    setattr(limit, field, value)
    assert key(identity, limit=limit) != key(identity)


def test_different_binary_contents_change_identity(tmp_path):
    path = tmp_path / "engine"
    path.write_bytes(b"first")
    first = keys.build_engine_identity(path, metadata())
    path.write_bytes(b"second")
    second = keys.build_engine_identity(path, metadata())
    assert first.executable_path == second.executable_path
    assert key(first) != key(second)


def test_options_and_reported_identity_separate_keys(identity):
    assert key(identity, meta=metadata(Threads=2)) != key(identity)
    assert key(identity, options={"hAsH": 32}) != key(identity)
    different = replace(metadata(), reported_id=(("name", "another build"),))
    other = replace(identity, reported_id=different.reported_id)
    assert key(other, meta=different) != key(identity)


def test_first_search_normalization_and_case_insensitive_options(identity):
    before = metadata()
    after = metadata(UCI_AnalyseMode=True)
    assert key(identity, meta=before) == key(identity, meta=after)
    assert key(identity, options={"THREADS": 2}) == key(identity, options={"threads": 2})
    assert dict(key(identity).configuration)["uci_analysemode"] is True
    assert dict(key(identity).configuration)["ponder"] is False
    assert dict(key(identity, options={"UCI_AnalyseMode": False}).configuration)["uci_analysemode"] is False


@pytest.mark.parametrize("field,value", [("halfmove_clock", 7), ("fullmove_number", 12),
                                         ("castling_rights", 0), ("ep_square", chess.E3)])
def test_fen_state_separates_keys(identity, field, value):
    board = chess.Board()
    setattr(board, field, value)
    assert key(identity, board) != key(identity)


def test_chess960_separation_and_variant_bypass(identity):
    assert key(identity, chess.Board(chess960=True)) != key(identity)
    with pytest.raises(keys.CacheIdentityUnavailable):
        key(identity, chess.variant.AtomicBoard())


def test_null_history_and_custom_board_bypass(identity):
    board = chess.Board()
    board.push(chess.Move.null())
    with pytest.raises(keys.CacheIdentityUnavailable, match="Null"):
        key(identity, board)
    class CustomBoard(chess.Board):
        pass
    with pytest.raises(keys.CacheIdentityUnavailable, match="Unsupported"):
        key(identity, CustomBoard())


def test_inconsistent_history_is_not_silently_dropped(identity):
    board = replay()
    board.halfmove_clock += 1
    with pytest.raises(keys.CacheIdentityUnavailable, match="History"):
        key(identity, board)


def test_missing_metadata_and_binary_bypass(identity, tmp_path):
    with pytest.raises(keys.CacheIdentityUnavailable):
        key(None)
    with pytest.raises(keys.CacheIdentityUnavailable):
        keys.build_engine_identity(tmp_path / "absent", metadata())
    with pytest.raises(keys.CacheIdentityUnavailable):
        keys.metadata_from_values({}, {}, {}, {})
    with pytest.raises(keys.CacheIdentityUnavailable):
        keys.metadata_from_values(None, None, None, None)
    with pytest.raises(keys.CacheIdentityUnavailable):
        keys.build_cache_key(identity, None, chess.Board(), chess.engine.Limit(depth=8))
    with pytest.raises(keys.CacheIdentityUnavailable):
        key(identity, meta=replace(metadata(), config=(("unknown", 1),)))


def resource_metadata(value):
    base = metadata()
    return replace(base, options=base.options + (keys.OptionSpec("evalfile", "string", value, None, None, ()),),
                   config=base.config + (("evalfile", value),),
                   target_config=base.target_config + (("evalfile", value),))


def test_resource_fingerprints_and_changed_configuration(tmp_path):
    binary, resource = tmp_path / "engine", tmp_path / "network"
    binary.write_bytes(b"engine")
    resource.write_bytes(b"network one")
    meta = resource_metadata("network")
    first = keys.build_engine_identity(binary, meta, resource_paths={"EvalFile": resource})
    resource.write_bytes(b"network two")
    second = keys.build_engine_identity(binary, meta, resource_paths={"evalfile": resource})
    assert key(first, meta=meta) != key(second, meta=meta)
    with pytest.raises(keys.CacheIdentityUnavailable, match="resources changed"):
        key(first, meta=meta, options={"EvalFile": "another network"})
    with pytest.raises(keys.CacheIdentityUnavailable, match="Unidentified"):
        keys.build_engine_identity(binary, meta)
    embedded = keys.build_engine_identity(binary, meta, embedded_resources=("EvalFile",))
    assert embedded.resources[0][2] == "embedded"


def test_key_immutability_and_hashability(identity):
    value = key(identity)
    assert {value: "stored"}[key(identity)] == "stored"
    with pytest.raises(FrozenInstanceError):
        value.schema_version = 2
    assert isinstance(value.configuration, tuple)
    assert isinstance(value.search.limit, tuple)


def test_fingerprint_once_per_identity_not_per_request(tmp_path, monkeypatch):
    path = tmp_path / "engine"
    path.write_bytes(b"fixture")
    spy = Mock(wraps=keys._fingerprint)
    monkeypatch.setattr(keys, "_fingerprint", spy)
    identity = keys.build_engine_identity(path, metadata())
    for _ in range(5):
        key(identity)
    spy.assert_called_once_with(path)


def test_search_modes_preserve_explicit_multipv_and_root_restrictions(identity):
    assert key(identity, multipv=1) != key(identity)  # analyse returns list vs dict.
    assert key(identity, multipv=2) != key(identity, multipv=1)
    assert key(identity, root_moves=()) != key(identity)
    assert key(identity, root_moves=(chess.Move.from_uci("e2e4"),)) != key(identity)
    assert key(identity, info=chess.engine.INFO_SCORE) != key(identity)
    with pytest.raises(keys.CacheIdentityUnavailable):
        key(identity, root_moves=iter([]))


def test_snapshot_reads_on_loop_without_engine_commands():
    loop = asyncio.new_event_loop()
    ready = Event()
    holder = {}
    def run():
        asyncio.set_event_loop(loop)
        async def initialize():
            protocol = chess.engine.UciProtocol()
            protocol.id = {"name": "Stockfish fixture"}
            option = chess.engine.Option("Threads", "spin", 1, 1, 1024, None)
            protocol.options["Threads"] = option
            protocol.config["Threads"] = 1
            protocol.target_config["Threads"] = 1
            protocol.send_line = Mock(side_effect=AssertionError("No commands allowed"))
            holder["protocol"] = protocol
            ready.set()
        loop.create_task(initialize())
        loop.run_forever()
    thread = Thread(target=run)
    thread.start()
    try:
        assert ready.wait(5)
        engine = SimpleNamespace(protocol=holder["protocol"], timeout=2,
                                 analyse=Mock(), configure=Mock(), communicate=Mock())
        snapshot = keys.snapshot_engine_metadata(engine)
        assert dict(snapshot.reported_id)["name"] == "Stockfish fixture"
        engine.analyse.assert_not_called()
        engine.configure.assert_not_called()
        engine.communicate.assert_not_called()
        engine.protocol.send_line.assert_not_called()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_version_guard(identity, monkeypatch):
    monkeypatch.setattr(chess, "__version__", "future")
    with pytest.raises(keys.CacheIdentityUnavailable):
        key(identity)


@pytest.mark.parametrize("options", [{"MultiPV": 2}, {"Mystery": 1}, {"Threads": -1}])
def test_unsupported_options_bypass(identity, options):
    with pytest.raises(keys.CacheIdentityUnavailable):
        key(identity, options=options)


def test_tablebases_and_mutable_limit_bypass(identity):
    meta = metadata()
    meta = replace(meta, options=meta.options + (keys.OptionSpec("syzygypath", "string", "", None, None, ()),),
                   config=meta.config + (("syzygypath", "/tables"),))
    with pytest.raises(keys.CacheIdentityUnavailable, match="tablebase"):
        key(identity, meta=meta)
    with pytest.raises(keys.CacheIdentityUnavailable, match="scalars"):
        key(identity, limit=chess.engine.Limit(depth=8, clock_id=object()))
