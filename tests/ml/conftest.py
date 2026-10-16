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


@pytest.fixture
def labeling_case():
    """Build a legal isolated candidate with explicit synthetic verification."""
    from dataclasses import asdict
    from src.chessmind.ml.data.candidate_schema import candidate_id
    from src.chessmind.ml.features.extractor import extract_candidate_features
    from src.chessmind.ml.labeling.schema import VerificationResult
    from src.chessmind.ml.data.candidate_extraction import _score

    def build(fen="1r5k/8/8/8/8/8/2N5/7K w - - 0 1", move="c2b4", *,
              before=None, after=None, before_pv=(), after_pv=("b8b4", "h1g1", "b4b5", "g1f1")):
        board = chess.Board(fen)
        assert board.is_valid() and board.is_legal(chess.Move.from_uci(move))
        mover, ply = board.turn, board.ply() + 1
        before = serialize_evaluation(before if before is not None else chess.engine.Cp(0))
        after = serialize_evaluation(after if after is not None else chess.engine.Cp(-500))
        quality = evaluate_move_quality(before, after, mover)
        c = {"candidate_id": candidate_id("lichess", "abcdefgh", ply), "source": "lichess",
             "source_game_id": "abcdefgh", "player_identifier": "lichess:fixture",
             "player_color": "white" if mover else "black", "ply_index": ply,
             "full_move_number": board.fullmove_number, "fen_before": board.fen(),
             "played_move_uci": move, "schema_version": "candidate_moves_v1",
             "source_evidence": {}, "selection_type": "SIGNIFICANT_CP_ERROR",
             "centipawn_loss": quality["centipawn_loss"], "existing_classification": quality["classification"],
             "mate_quality": quality["mate_quality"], "score_before": before, "score_after": after,
             "extraction_configuration": {"cp_loss_threshold": 100}, "analysis_profile": {"configured_depth": 8}}
        board.push_uci(move)
        c["fen_after"] = board.fen()
        feature = asdict(extract_candidate_features(c))
        # JSON-like lists match artifact representation.
        import json
        feature = json.loads(json.dumps(feature))
        v = VerificationResult("verified", before=_score(before), after=_score(after),
            before_pv=tuple(before_pv), after_pv=tuple(after_pv), before_depth=12, after_depth=12,
            centipawn_loss=quality["centipawn_loss"], classification=quality["classification"],
            mate_quality=quality["mate_quality"],
            significant_error_supported=quality["centipawn_loss"] >= 100 if quality["centipawn_loss"] is not None else None)
        return c, feature, v
    return build
