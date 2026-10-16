"""Select evidence from existing analysis, without searches or error-type labels."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import chess

from .candidate_schema import (
    AnalysisProfile, CandidateRecord, ExtractionConfig, ScoreEvidence,
    SourceEvidence, candidate_id,
)
from .game_dataset_loader import GameRecord


HARMFUL = frozenset({"lost_forced_mate", "allowed_forced_mate", "reversed_to_losing_mate"})
BENEFICIAL = frozenset({"escaped_forced_mate", "achieved_forced_mate",
                        "reversed_to_winning_mate", "delivered_checkmate"})
NEUTRAL = frozenset({"maintained_forced_mate", "remains_forced_loss"})


def _score(value) -> ScoreEvidence:
    if not isinstance(value, Mapping) or value.get("perspective") != "white":
        raise ValueError("structured White-perspective score required")
    kind, cp, mate, winner = (value.get(k) for k in ("type", "cp", "mate", "winner"))
    if kind == "cp":
        valid = type(cp) is int and mate is None and winner is None
    elif kind == "mate":
        valid = (cp is None and type(mate) is int and winner in ("white", "black")
                 and (mate == 0 or (mate > 0) == (winner == "white")))
    else:
        valid = False
    if not valid:
        raise ValueError("invalid structured score")
    return ScoreEvidence(kind, cp, mate, winner)


def _validated_rows(analysis, game):
    """Validate associations and preserve evidence; never recalculate quality."""
    if not isinstance(analysis, Sequence) or isinstance(analysis, (str, bytes)):
        raise ValueError("analysis must be a sequence of move results")
    if len(analysis) != game.move_count or len(analysis) != len(game.uci_moves):
        raise ValueError("analysis length does not match source game")
    board = chess.Board()
    rows = []
    previous_score = None
    required = {"ply", "move_number", "color", "uci", "san", "phase", "fen_before",
                "fen_after", "evaluation_before_detail", "evaluation_after_detail",
                "centipawn_loss", "classification"}
    for ply, row in enumerate(analysis, 1):
        if not isinstance(row, Mapping) or not required <= row.keys():
            raise ValueError(f"malformed analysis row at ply {ply}")
        if (type(row["ply"]) is not int or row["ply"] != ply or
                row["uci"] != game.uci_moves[ply - 1] or
                row["move_number"] != board.fullmove_number or
                row["color"] != ("white" if board.turn else "black") or
                row["fen_before"] != board.fen()):
            raise ValueError(f"source/position association mismatch at ply {ply}")
        move = chess.Move.from_uci(row["uci"])
        if not board.is_legal(move) or row["san"] != board.san(move):
            raise ValueError(f"invalid played move at ply {ply}")
        best = row.get("best_move")
        if best is not None and (not isinstance(best, str) or not board.is_legal(chess.Move.from_uci(best))):
            raise ValueError(f"invalid best move at ply {ply}")
        board.push(move)
        if row["fen_after"] != board.fen():
            raise ValueError(f"after position mismatch at ply {ply}")
        before, after = (_score(row[k]) for k in ("evaluation_before_detail", "evaluation_after_detail"))
        if previous_score is not None and before != previous_score:
            raise ValueError("adjacent analysis scores disagree")
        previous_score = after
        loss = row["centipawn_loss"]
        if loss is not None and (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0):
            raise ValueError("invalid centipawn loss")
        if (before.type == "mate" or after.type == "mate") and loss is not None:
            raise ValueError("mate transition must have undefined CP loss")
        if row["classification"] not in {"Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"}:
            raise ValueError("unknown existing classification")
        if row["phase"] not in {"opening", "middlegame", "endgame"}:
            raise ValueError("unknown game phase")
        if row.get("mate_quality") is not None and not isinstance(row["mate_quality"], str):
            raise ValueError("invalid mate quality")
        rows.append((row, before, after))
    return rows


def extract_candidates(analysis, game: GameRecord, *, config: ExtractionConfig | None = None,
                       profile: AnalysisProfile | None = None,
                       source: SourceEvidence | None = None) -> tuple[CandidateRecord, ...]:
    """Return one record per selected ply, in source order.

    Significant errors take precedence if configured CP ranges overlap.
    Mate-quality reasons come from the existing analyzer. Distance changes and
    unknown mate reasons are ambiguous, not tactical mistake labels.
    """
    config = config or ExtractionConfig()
    profile = profile or AnalysisProfile()
    source = source or SourceEvidence(source_archive=game.source_archive,
                                      archive_game_index=game.source_index)
    rows = _validated_rows(analysis, game)
    selected, controls = {}, []
    for row, before, after in rows:
        ply, loss = row["ply"], row["centipawn_loss"]
        if before.type == "mate" or after.type == "mate":
            if not config.include_mate_candidates:
                continue
            quality = row.get("mate_quality")
            if quality in NEUTRAL:
                continue
            if quality in HARMFUL:
                selected[ply] = ("MATE_RELATED", quality, "harmful")
            elif quality in BENEFICIAL:
                selected[ply] = ("MATE_RELATED", quality, "beneficial")
            else:
                selected[ply] = ("AMBIGUOUS", quality or "missing_mate_quality", "ambiguous")
        elif loss is None:
            selected[ply] = ("AMBIGUOUS", "undefined_cp_loss", None)
        elif loss >= config.cp_loss_threshold:
            selected[ply] = ("SIGNIFICANT_CP_ERROR", "cp_loss_at_or_above_threshold", None)
        elif loss <= config.control_cp_loss_max:
            controls.append(ply)
    def rank(ply):
        key = [config.random_seed, game.source, game.game_id, ply]
        return hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest(), ply
    for ply in sorted(controls, key=rank)[:config.max_control_moves_per_game]:
        selected[ply] = ("LOW_ERROR_CONTROL", "seeded_low_loss_sample", None)
    result = []
    for row, before, after in rows:
        if row["ply"] not in selected:
            continue
        kind, reason, status = selected[row["ply"]]
        player = game.player_keys[0 if row["color"] == "white" else 1]
        result.append(CandidateRecord(
            candidate_id(game.source, game.game_id, row["ply"]), game.game_id,
            game.source, player, row["color"], row["ply"], row["move_number"],
            row["fen_before"], row["fen_after"], row["uci"], row["san"], row.get("best_move"),
            row["phase"], before, after, row["centipawn_loss"], row["classification"],
            row.get("mate_quality"), status, kind, reason, config, profile, source,
            game.white_rating, game.black_rating, game.time_control, game.utc_date, game.utc_time))
    return tuple(result)
