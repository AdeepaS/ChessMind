"""Synthetic complete games and existing-format analysis for candidate tests."""

import io

import chess
import chess.engine
import chess.pgn
import pytest

from src.chessmind.analysis.game_analyzer import evaluate_move_quality
from src.chessmind.engine.analyzer import serialize_evaluation
from src.chessmind.ml.data.game_dataset_loader import GamePGNReader


@pytest.fixture
def candidate_fixture():
    def build(scores=None, *, game_id="abcdefgh", count=12):
        sequence = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5".split()[:count]
        board = chess.Board()
        for move in sequence:
            board.push_uci(move)
        pgn = chess.pgn.Game.from_board(board)
        pgn.headers.update({"Event": "Rated Rapid game", "Site": f"https://lichess.org/{game_id}",
                            "White": "Alice", "Black": "Bob", "WhiteElo": "1500",
                            "BlackElo": "1600", "Result": "1-0", "TimeControl": "600+0"})
        record = next(GamePGNReader(io.StringIO(str(pgn) + "\n\n")))
        scores = scores if scores is not None else [chess.engine.Cp(0)] * (count + 1)
        assert len(scores) == count + 1
        board, rows = chess.Board(), []
        for ply, uci in enumerate(sequence, 1):
            move = chess.Move.from_uci(uci)
            color = board.turn
            before, after = map(serialize_evaluation, scores[ply - 1:ply + 1])
            row = {"ply": ply, "move_number": board.fullmove_number,
                   "color": "white" if color else "black", "uci": uci,
                   "san": board.san(move), "phase": "opening", "fen_before": board.fen(),
                   "best_move": uci, "evaluation_before_detail": before,
                   "evaluation_after_detail": after, **evaluate_move_quality(before, after, color)}
            board.push(move)
            row["fen_after"] = board.fen()
            rows.append(row)
        return record, rows
    return build
