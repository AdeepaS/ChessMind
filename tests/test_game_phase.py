"""Position fixtures and pipeline regressions for the Day 6 phase heuristic."""
import json
from unittest.mock import Mock

import chess
import chess.engine
import chess.pgn
import pytest

from src.chessmind.analysis.game_phase import (
    calculate_non_pawn_material, detect_game_phase, PHASE_ORDER,
)
from src.chessmind.analysis.game_analyzer import analyze_game, calculate_game_statistics
from src.chessmind.analysis.weakness_analyzer import detect_weakest_phase
from src.chessmind.players.multi_game_analyzer import analyze_player_games
from src.chessmind.reports.weakness_report import generate_weakness_report
from src.chessmind.reports.player_report import generate_player_report
from src.chessmind.pgn.multi_game_parser import load_multiple_pgn


def play(sans):
    board = chess.Board()
    for san in sans.split():
        board.push_san(san)
    assert board.is_valid()
    return board


OPENING = play("e4 e5 Nf3 Nc6").fen()
DEVELOPED = play(
    "e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 O-O d6 Nc3 O-O Bg5 Be6"
).fen()
QUEEN_EXCHANGE = play("d4 d6 e4 e5 dxe5 dxe5 Qxd8+ Kxd8").fen()
ROOK_ENDGAME = "4k2r/pp3ppp/8/8/8/8/PP3PPP/R3K3 w Qk - 0 1"
MINOR_ENDGAME = "6k1/5ppp/2n5/8/8/2B5/5PPP/6K1 w - - 0 1"
PAWN_ENDGAME = "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
QUEEN_ENDGAME = "3q2k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"
ONE_QUEEN_ENDGAME = "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"
QUEEN_ROOK_ENDGAME = "3q2kr/5ppp/8/8/8/8/5PPP/R2Q2K1 w - - 0 30"
# Queen plus rook/two-minor opposition is not sufficiently simplified under
# the chosen support threshold, despite a relatively low total material value.
QUEEN_WITH_SUPPORT = "r2n2k1/5ppp/2b5/8/8/8/5PPP/3Q2K1 w - - 0 30"


@pytest.mark.parametrize("fen,move,expected", [
    (chess.STARTING_FEN, 1, "opening"),
    (OPENING, 3, "opening"),
    (OPENING, 11, "opening"),  # Still six home minors beyond move ten.
    (DEVELOPED, 8, "middlegame"),
    (QUEEN_EXCHANGE, 5, "opening"),  # All eight minors home; 44 units remain.
    (ROOK_ENDGAME, 1, "endgame"),  # Material overrides a misleading FEN clock.
    (MINOR_ENDGAME, 1, "endgame"),
    (PAWN_ENDGAME, 1, "endgame"),
    (QUEEN_ENDGAME, 1, "endgame"),
    (ONE_QUEEN_ENDGAME, 1, "endgame"),
    (QUEEN_ROOK_ENDGAME, 30, "endgame"),
    (QUEEN_WITH_SUPPORT, 30, "middlegame"),
    (chess.STARTING_FEN, 40, "middlegame"),  # Late undeveloped play is not opening.
])
def test_position_policy_and_color_symmetry(fen, move, expected):
    board = chess.Board(fen)
    assert board.is_valid(), board.status()
    original = board.fen()
    assert detect_game_phase(board, move) == expected
    assert detect_game_phase(board.mirror(), move) == expected
    assert board.fen() == original


@pytest.mark.parametrize("previous,fen,expected", [
    ("middlegame", OPENING, "middlegame"),
    ("endgame", DEVELOPED, "endgame"),
    ("endgame", OPENING, "endgame"),
    ("opening", PAWN_ENDGAME, "endgame"),
])
def test_forward_only_phase_policy(previous, fen, expected):
    board = chess.Board(fen)
    assert board.is_valid()
    assert detect_game_phase(board, 11, previous_phase=previous) == expected


def test_promotion_does_not_reverse_endgame():
    board = chess.Board("3q2k1/P4ppp/8/8/8/8/5PPP/R2Q2K1 w - - 0 40")
    assert board.is_valid()
    previous = detect_game_phase(board, 40)
    assert previous == "endgame"
    board.push_san("a8=Q")
    assert board.is_valid()
    assert detect_game_phase(board, 40) == "middlegame"  # Now 32 units.
    assert detect_game_phase(board, 40, previous) == "endgame"


def test_castling_rights_do_not_control_phase():
    board = chess.Board(OPENING)
    board.castling_rights = chess.BB_EMPTY
    assert board.is_valid()
    assert detect_game_phase(board, 3) == "opening"
    board = chess.Board(ROOK_ENDGAME)
    assert board.has_castling_rights(chess.WHITE)
    assert detect_game_phase(board, 1) == "endgame"


def test_invalid_previous_phase_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown previous phase"):
        detect_game_phase(chess.Board(), 1, "unknown")


def test_material_helper_compatibility():
    assert calculate_non_pawn_material(chess.Board()) == 62
    assert calculate_non_pawn_material(chess.Board(QUEEN_EXCHANGE)) == 44
    assert calculate_non_pawn_material(chess.Board(PAWN_ENDGAME)) == 0


def test_analyzer_state_is_local_and_uses_pre_move_board(monkeypatch):
    # Legal development then retreat. The raw classifier can return opening,
    # but the game must stay middlegame after both sides have developed.
    sans = "Nf3 Nf6 Nc3 Nc6 e3 e6 Bd3 Bd6 Be2 Be7 Nb1 Nb8 Ng1 Ng8 Bf1 Bf8 a3 a6"
    board = play(sans)
    assert detect_game_phase(board, board.fullmove_number) == "opening"
    game = chess.pgn.Game.from_board(board)
    engine = Mock()
    engine.analyse.return_value = {
        "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)}
    factory = Mock(return_value=engine)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", factory)
    result = analyze_game(game)
    assert [move["phase"] for move in result[:8]] == ["opening"] * 8
    assert all(move["phase"] == "middlegame" for move in result[8:])
    assert result[8]["san"] == "Be2"
    assert engine.analyse.call_count == len(result) + 1
    engine.quit.assert_called_once()
    fresh = chess.pgn.Game.from_board(play("e4 e5"))
    assert all(m["phase"] == "opening" for m in analyze_game(fresh))
    assert factory.call_count == 2


@pytest.mark.parametrize("game_index", [0, 1, 2])
def test_historical_phase_pipeline_and_reporting(game_index, monkeypatch):
    game = load_multiple_pgn("data/raw/validation_games.pgn")[game_index]
    engine = Mock()
    engine.analyse.return_value = {
        "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)}
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", Mock(return_value=engine))
    # Each historical player is kept independent; no cross-player aggregation.
    multi = analyze_player_games([{
        "game": game, "game_number": 1, "color": "white",
        "opponent": game.headers["Black"], "result": game.headers["Result"],
    }])
    analysis = multi["games"][0]["analysis"]
    stats = multi["games"][0]["stats"]
    ranks = [PHASE_ORDER.index(m["phase"]) for m in analysis]
    assert ranks == sorted(ranks)
    assert engine.analyse.call_count == len(analysis) + 1
    assert sum(stats[c]["moves"] for c in ("white", "black")) == len(analysis)
    for color in ("white", "black"):
        assert sum(p["moves"] for p in stats[color]["phases"].values()) == stats[color]["moves"]
        assert stats[color]["acpl"] == 0
    single = generate_weakness_report(analysis, stats)
    report = generate_player_report(game.headers["White"], multi)
    assert json.loads(json.dumps(report)) == report
    assert json.loads(json.dumps(single)) == single
    assert detect_weakest_phase(stats)["white"] is not None
