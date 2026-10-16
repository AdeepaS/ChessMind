"""Hand-constructed board tests, independent of engines and network."""

from dataclasses import FrozenInstanceError, asdict

import chess
import pytest

from src.chessmind.ml.features.extractor import extract_move_features, extract_position_features
from src.chessmind.ml.features.schema import numeric_position_values


def board_with(pieces, turn=chess.WHITE):
    board = chess.Board(None)
    for square, symbol in {"h1": "K", "h8": "k", **pieces}.items():
        board.set_piece_at(chess.parse_square(square), chess.Piece.from_symbol(symbol))
    board.turn = turn
    assert board.is_valid()
    return board


def test_starting_material_counts_and_schema():
    value = extract_position_features(chess.Board())
    for color in (value.white, value.black):
        assert (color.pawns, color.knights, color.bishops, color.rooks, color.queens, color.kings) == (8, 2, 2, 2, 1, 1)
        assert color.material == 39 and color.non_pawn_material == 31 and color.bishop_pair
        assert color.pawn_shield == 3 and color.kingside_castling_right and color.queenside_castling_right
        assert color.attacked_pieces == 0
        assert color.doubled_excess_pawns == color.isolated_pawns == color.passed_pawns == 0
    assert value.material_balance == 0
    assert value.side_to_move_legal_moves == 20 and not value.side_to_move_in_check
    assert len(numeric_position_values(value)) == 49
    assert sum(len(v) if isinstance(v, dict) else 1 for v in asdict(value).values()) == 52
    with pytest.raises(FrozenInstanceError):
        value.white.pawns = 0


def test_asymmetric_material_and_bishop_pair():
    board = board_with({"c1": "B", "f1": "B", "a2": "P", "b8": "r"})
    features = extract_position_features(board)
    assert features.white.bishop_pair and not features.black.bishop_pair
    assert features.white.material == 7 and features.black.material == 5
    assert features.material_balance == 2


def test_capture_material_delta():
    board = chess.Board()
    for san in ("e4", "d5"):
        board.push_san(san)
    value = extract_move_features(board, chess.Move.from_uci("e4d5"))
    deltas = {d.feature: d.change for d in value.deltas}
    assert value.context.is_capture
    assert deltas["black.pawns"] == deltas["black.material"] == -1
    assert deltas["material_balance"] == 1
    assert deltas["white.material"] == 0


@pytest.mark.parametrize("piece,material", [("q", 9), ("r", 5), ("b", 3), ("n", 3)])
def test_promotion(piece, material):
    board = board_with({"a7": "P"})
    value = extract_move_features(board, chess.Move.from_uci("a7a8" + piece))
    assert value.after.white.pawns == 0 and value.after.white.material == material
    assert value.after.white.passed_pawns == 0
    assert value.context.promotion_piece == chess.piece_name(chess.PIECE_SYMBOLS.index(piece))
    assert dict((d.feature, d.change) for d in value.deltas)["white.material"] == material - 1


def test_attacked_defended_undefended_and_king_exclusion():
    board = board_with({"b2": "N", "c3": "P", "b8": "r"})
    f = extract_position_features(board)
    assert f.white.attacked_pieces == 1  # b2 knight, rook b8 ray.
    assert f.white.defended_pieces == 0  # c3 pawn attacks b4/d4, not b2.
    assert f.white.undefended_pieces == 2
    assert f.white.attacked_undefended_pieces == 1
    board.set_piece_at(chess.A1, chess.Piece(chess.BISHOP, chess.WHITE))
    f = extract_position_features(board)
    assert f.white.defended_pieces == 1 and f.white.attacked_undefended_pieces == 0
    kings = extract_position_features(board_with({}))
    assert kings.white.undefended_pieces == kings.black.undefended_pieces == 0


def test_pinned_geometric_attacker_is_not_legal_capture():
    board = chess.Board("4k3/4n3/2B5/8/8/8/8/K3R3 b - - 0 1")
    assert board.is_pinned(chess.BLACK, chess.E7)
    assert board.is_attacked_by(chess.BLACK, chess.C6)
    assert not board.is_legal(chess.Move.from_uci("e7c6"))
    features = extract_position_features(board)
    assert features.white.attacked_pieces == 1
    assert features.white.attacked_undefended_pieces == 1
    assert features.side_to_move_in_check


def test_pawn_edges_doubled_isolated_passed():
    board = board_with({"a2": "P", "a3": "P", "h2": "P", "b6": "p"})
    f = extract_position_features(board)
    assert f.white.doubled_excess_pawns == 1
    assert f.white.isolated_pawns == 3
    assert f.white.passed_pawns == 1  # h2 only; enemy b6 ahead of both a pawns.
    assert f.black.isolated_pawns == 1 and f.black.passed_pawns == 0
    mirrored = extract_position_features(board.mirror())
    for name in ("doubled_excess_pawns", "isolated_pawns", "passed_pawns"):
        assert getattr(f.white, name) == getattr(mirrored.black, name)
        assert getattr(f.black, name) == getattr(mirrored.white, name)


def test_pawns_same_rank_behind_connected_and_no_pawns():
    board = board_with({"a4": "P", "b3": "P", "b4": "p", "h7": "p"})
    f = extract_position_features(board)
    assert f.white.isolated_pawns == 0
    assert f.white.passed_pawns == 1  # a4 ignores enemy b4 on same rank.
    empty = extract_position_features(board_with({}))
    assert empty.white.passed_pawns == empty.black.isolated_pawns == 0


def test_king_edge_shield_zone_and_missing_rights():
    board = chess.Board("k5r1/8/8/8/8/8/6PP/7K w - - 0 1")
    f = extract_position_features(board)
    assert (f.white.king_square, f.white.king_file, f.white.king_rank) == ("h1", 7, 0)
    assert f.white.pawn_shield == 2
    assert f.white.attacked_king_zone_squares == 1  # g2, one square regardless of attacker count.
    assert not f.white.kingside_castling_right and not f.white.queenside_castling_right
    mirror = extract_position_features(board.mirror())
    assert mirror.black.pawn_shield == 2 and mirror.black.attacked_king_zone_squares == 1


def test_central_king_shield():
    board = chess.Board("7k/8/8/2PPP3/3K4/8/8/8 w - - 0 1")
    f = extract_position_features(board)
    assert f.white.pawn_shield == 3 and f.white.king_square == "d4"


def test_castling_and_rights_deltas():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    f = extract_move_features(board, chess.Move.from_uci("e1g1"))
    assert f.context.is_castling and not f.context.is_capture
    assert f.after.white.king_square == "g1"
    assert not f.after.white.kingside_castling_right and not f.after.white.queenside_castling_right
    assert f.after.black.kingside_castling_right
    assert {d.feature: d.change for d in f.deltas}["white.kingside_castling_right"] == -1


def test_en_passant_and_board_immutability():
    board = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        board.push_san(san)
    original, stack = board.fen(), list(board.move_stack)
    f = extract_move_features(board, chess.Move.from_uci("e5d6"))
    assert f.context.is_capture and f.context.is_en_passant
    assert f.after.black.pawns == 7
    assert board.fen() == original and board.move_stack == stack
    assert f.before.side_to_move == "white" and f.after.side_to_move == "black"
    assert f.before.side_to_move_legal_moves == board.legal_moves.count()
    after = board.copy()
    after.push_uci("e5d6")
    assert f.after.side_to_move_legal_moves == after.legal_moves.count()


def test_check_move_and_illegal_rejection():
    board = board_with({"a2": "R"})
    f = extract_move_features(board, chess.Move.from_uci("a2h2"))
    assert f.context.gives_check and f.after.side_to_move_in_check
    with pytest.raises(ValueError, match="illegal_move"):
        extract_move_features(chess.Board(), chess.Move.from_uci("e2e5"))
    with pytest.raises(ValueError):
        extract_position_features(chess.Board(None))
    with pytest.raises(ValueError):
        extract_position_features(chess.Board(chess960=True))
