"""Borrowed-engine verification through existing scoring and cache interfaces."""

import math

import chess
import chess.engine

from ...analysis.game_analyzer import evaluate_move_quality
from ...engine.analyzer import analyze_board
from ..data.candidate_extraction import _score
from ..features.extractor import extract_candidate_features
from .schema import VerificationResult


def replay_pv(board, pv):
    replay = board.copy()
    for uci in pv:
        move = chess.Move.from_uci(uci)
        if not replay.is_legal(move):
            raise ValueError("illegal verification PV")
        replay.push(move)
    return replay


def _result(board, result, depth):
    score = _score(result["evaluation"])
    achieved = result.get("depth")
    pv = tuple(move.uci() for move in result.get("pv", ()))
    replay_pv(board, pv)
    if not board.is_game_over() and (not pv or type(achieved) is not int or achieved < depth):
        raise ValueError("missing PV or insufficient achieved depth")
    if score.type == "mate" and score.mate == 0:
        winner = "black" if board.turn else "white"
        if not board.is_checkmate() or score.winner != winner:
            raise ValueError("Mate(0) inconsistent with board")
    return score, pv, achieved


def verify_candidate(candidate, engine, *, depth=12, profile=None, cache=None, cache_context=None):
    """Two position requests, one token; never launches/configures/closes engine.

    FEN reconstruction cannot restore earlier repetition history. Callers receive
    an explicit failure; no fallback to original low-depth evaluations occurs.
    """
    original = dict(original_cp_loss=candidate.get("centipawn_loss"),
                    original_classification=candidate.get("existing_classification"),
                    original_mate_quality=candidate.get("mate_quality"), profile=profile)
    try:
        if type(depth) is not int or depth < 1:
            raise ValueError("invalid verification depth")
        extract_candidate_features(candidate)
        board = chess.Board(candidate["fen_before"])
        threshold = candidate["extraction_configuration"]["cp_loss_threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid original selection threshold")
        kwargs = {"game": object()}
        if cache is not None:
            kwargs.update(cache=cache, cache_context=cache_context)
        before = analyze_board(engine, board, depth=depth, **kwargs)
        before_score, before_pv, before_depth = _result(board, before, depth)
        mover = board.turn
        board.push_uci(candidate["played_move_uci"])
        after = analyze_board(engine, board, depth=depth, **kwargs)
        after_score, after_pv, after_depth = _result(board, after, depth)
        quality = evaluate_move_quality(before["evaluation"], after["evaluation"], mover)
        loss = quality["centipawn_loss"]
        old_loss = candidate.get("centipawn_loss")
        supported = loss >= threshold if loss is not None else None
        old_supported = old_loss >= threshold if old_loss is not None else None
        loss_change = loss - old_loss if loss is not None and old_loss is not None else None
        changed = supported != old_supported
        class_changed = quality["classification"] != candidate.get("existing_classification")
        material_change = (changed or class_changed or (loss_change is not None and abs(loss_change) >= 100)
                           or quality["mate_quality"] != candidate.get("mate_quality"))
        return VerificationResult("verified", before=before_score, after=after_score,
            before_pv=before_pv, after_pv=after_pv, before_depth=before_depth, after_depth=after_depth,
            centipawn_loss=loss, classification=quality["classification"], mate_quality=quality["mate_quality"],
            significant_error_supported=supported, cp_loss_change=loss_change,
            classification_changed=class_changed, significant_status_changed=changed,
            materially_changed=material_change, **original)
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, chess.engine.EngineError) as exc:
        return VerificationResult("failed", reason=f"{type(exc).__name__}: {exc}", **original)
