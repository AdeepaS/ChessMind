"""Narrow geometric/legal/PV evidence. No tactical truth or broad classifiers."""

from dataclasses import replace

import chess

from ..features.material import material_features
from .engine_verification import replay_pv
from .schema import TacticalEvidence


def _balance(board, color):
    return material_features(board, color)["material"] - material_features(board, not color)["material"]


def capture_evidence(candidate, feature, verification):
    kind = "allowed_immediate_tactical_capture"
    if verification.status != "verified":
        return TacticalEvidence(kind, False, "verification_unavailable")
    before = chess.Board(candidate["fen_before"])
    played = chess.Move.from_uci(candidate["played_move_uci"])
    after = before.copy()
    after.push(played)
    pv = verification.after_pv
    if not pv:
        return TacticalEvidence(kind, False, "no_opponent_continuation")
    capture = chess.Move.from_uci(pv[0])
    if not after.is_legal(capture) or not after.is_capture(capture):
        return TacticalEvidence(kind, False, "pv_does_not_start_with_legal_capture", pv)
    if after.is_en_passant(capture):
        return TacticalEvidence(kind, False, "en_passant_outside_initial_detector_scope", pv)
    victim = after.piece_at(capture.to_square)
    moved = capture.to_square == played.to_square and not before.is_castling(played)
    origin = played.from_square if moved else capture.to_square
    previous = before.is_attacked_by(not before.turn, origin)
    defended = after.is_attacked_by(before.turn, capture.to_square)
    color_name = "white" if before.turn else "black"
    delta = next(d["change"] for d in feature["features"]["deltas"]
                 if d["feature"] == color_name + ".attacked_undefended_pieces")
    evidence = TacticalEvidence(kind, False, "pending", pv, capture.uci(),
        chess.square_name(capture.to_square), moved, defended, previous,
        attacked_undefended_delta=delta)
    if victim is None or victim.piece_type in (chess.PAWN, chess.KING):
        return replace(evidence, reason="victim_outside_nonpawn_scope")
    if previous:
        return replace(evidence, reason="victim_already_geometrically_attacked")
    if verification.before.type != "cp" or verification.after.type != "cp":
        return replace(evidence, reason="mate_scores_do_not_isolate_material_explanation")
    if verification.significant_error_supported is not True:
        return replace(evidence, reason="evaluation_does_not_support_significant_error")
    if len(pv) < 4 and not replay_pv(after, pv).is_game_over():
        return replace(evidence, reason="continuation_too_short_for_material_check")
    opponent = after.turn
    baseline = _balance(after, opponent)
    after_capture = after.copy()
    after_capture.push(capture)
    immediate_gain = _balance(after_capture, opponent) - baseline
    minimum_gain = immediate_gain
    for recapture in after_capture.legal_moves:
        if recapture.to_square == capture.to_square and after_capture.is_capture(recapture):
            reply = after_capture.copy()
            reply.push(recapture)
            minimum_gain = min(minimum_gain, _balance(reply, opponent) - baseline)
    # Inspect a bounded continuation; short lines ending mid-exchange cannot prove gain.
    inspected = pv[:8]
    final = replay_pv(after, inspected)
    gain = _balance(final, opponent) - baseline
    evidence = replace(evidence, pv_material_gain=gain,
                       minimum_gain_after_immediate_recapture=minimum_gain)
    # Include material earned by the played move itself (e.g. capturing a queen
    # before surrendering a knight). Post-move capture gain alone is misleading.
    whole_transition_gain = _balance(final, opponent) - _balance(before, opponent)
    if minimum_gain <= 0 or gain <= 0 or whole_transition_gain <= 0:
        return replace(evidence, reason="material_recovered_or_compensated")
    if final.is_check() or final.is_checkmate():
        return replace(evidence, reason="forcing_continuation_requires_review")
    # Any immediate capture available at the horizon can invalidate material accounting.
    if any(final.is_capture(move) for move in final.legal_moves):
        return replace(evidence, reason="unresolved_capture_at_pv_horizon")
    return replace(evidence, supported=True, reason="legal_pv_capture_with_persistent_material_gain")


def missed_mate_evidence(candidate, verification):
    kind = "missed_forced_mate"
    if verification.status != "verified":
        return TacticalEvidence(kind, False, "verification_unavailable")
    mover = candidate["player_color"]
    before, after = verification.before, verification.after
    if before.type != "mate" or before.winner != mover:
        return TacticalEvidence(kind, False, "no_verified_mover_mate_before")
    if after.type == "mate" and after.winner == mover:
        return TacticalEvidence(kind, False, "forced_mate_retained_or_delayed", verification.before_pv)
    if verification.mate_quality not in ("lost_forced_mate", "reversed_to_losing_mate"):
        return TacticalEvidence(kind, False, "mate_transition_not_supported")
    pv = verification.before_pv
    if not pv or pv[0] == candidate["played_move_uci"]:
        return TacticalEvidence(kind, False, "no_distinct_mating_alternative", pv)
    final = replay_pv(chess.Board(candidate["fen_before"]), pv)
    winner = "black" if final.turn else "white"
    if not final.is_checkmate() or winner != mover:
        return TacticalEvidence(kind, False, "mating_pv_not_complete", pv)
    if after.type == "cp":
        # A finite search returning CP does not exclude a longer forced mate.
        return TacticalEvidence(kind, False, "post_move_forced_mate_unconfirmed", pv)
    return TacticalEvidence(kind, True, "verified_mover_mate_lost_with_concrete_mating_alternative", pv)
