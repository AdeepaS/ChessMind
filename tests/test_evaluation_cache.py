"""Deterministic cache tests; no engine processes or cache-key builders."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import chess
import pytest

from src.chessmind.engine.evaluation_cache import CacheInfo, EvaluationCache


@pytest.fixture
def result():
    move = chess.Move.from_uci("e2e4")
    return {
        "score_cp": 35,
        "evaluation": {"type": "cp", "cp": 35, "mate": None,
                       "winner": None, "perspective": "white"},
        "best_move": move,
        "depth": 8,
        "pv": [move, chess.Move.from_uci("e7e5")],
        "metadata": {"labels": ["original"]},
    }


def test_default_capacity_and_empty_miss():
    cache = EvaluationCache()
    assert cache.info() == CacheInfo(0, 0, 0, 0, 4096)
    assert cache.get(("absent", 8)) is None
    assert cache.info() == CacheInfo(0, 1, 0, 0, 4096)


def test_hit_and_miss_counters(result):
    cache = EvaluationCache(2)
    assert cache.get("a") is None
    cache.put("a", result)
    assert cache.get("a") == result
    assert cache.get("a") == result
    assert cache.get("b") is None
    assert cache.info() == CacheInfo(2, 2, 0, 1, 2)


def test_get_refreshes_lru_and_evicts_oldest(result):
    cache = EvaluationCache(2)
    cache.put("a", result)
    cache.put("b", result)
    cache.get("a")
    cache.put("c", result)
    assert cache.info() == CacheInfo(1, 0, 1, 2, 2)
    assert cache.get("b") is None
    assert cache.get("a") == cache.get("c") == result


def test_capacity_one_evicts_on_new_key(result):
    cache = EvaluationCache(1)
    cache.put("a", result)
    cache.put("b", result)
    assert cache.info().evictions == 1
    assert cache.get("a") is None
    assert cache.get("b") == result


def test_replacement_refreshes_recency_without_eviction(result):
    cache = EvaluationCache(2)
    cache.put("a", result)
    cache.put("b", result)
    replacement = deepcopy(result)
    replacement["depth"] = 10
    cache.put("a", replacement)
    assert cache.info() == CacheInfo(0, 0, 0, 2, 2)
    cache.put("c", result)
    assert cache.get("b") is None
    assert cache.get("a") == replacement
    assert cache.info().evictions == 1


def test_clear_resets_entries_and_all_counters(result):
    cache = EvaluationCache(1)
    cache.put("a", result)
    cache.get("a")
    cache.get("missing")
    cache.put("b", result)
    cache.clear()
    assert cache.info() == CacheInfo(0, 0, 0, 0, 1)
    assert cache.get("b") is None
    cache.put("c", result)
    assert cache.get("c") == result


def test_zero_capacity_is_complete_no_op(result):
    cache = EvaluationCache(0)
    cache.put("a", result)
    cache.put([], None)  # Disabled mode skips key/value validation too.
    assert cache.get("a") is None
    assert cache.get([]) is None
    cache.clear()
    assert cache.info() == CacheInfo(0, 0, 0, 0, 0)


@pytest.mark.parametrize("capacity", [True, False, 1.0, "2", None, [], {}])
def test_noninteger_capacities_rejected(capacity):
    with pytest.raises(TypeError):
        EvaluationCache(capacity)


@pytest.mark.parametrize("capacity", [-1, -4096])
def test_negative_capacities_rejected(capacity):
    with pytest.raises(ValueError):
        EvaluationCache(capacity)


def test_insertion_copies_nested_values_and_moves(result):
    original = deepcopy(result)
    cache = EvaluationCache()
    cache.put("a", result)
    result["score_cp"] = 900
    result["evaluation"]["cp"] = 900
    result["metadata"]["labels"].append("changed")
    result["pv"][0].to_square = chess.E3
    result["pv"].append(chess.Move.from_uci("g1f3"))
    assert cache.get("a") == original


def test_retrieval_copies_nested_values_and_moves(result):
    cache = EvaluationCache()
    cache.put("a", result)
    retrieved = cache.get("a")
    retrieved["evaluation"]["cp"] = -500
    retrieved["metadata"]["labels"].clear()
    retrieved["best_move"].to_square = chess.E3
    retrieved["pv"].clear()
    assert cache.get("a") == result


@pytest.mark.parametrize("winner,legacy", [("white", 100000), ("black", -100000)])
def test_mate_zero_and_winner_preserved(result, winner, legacy):
    result["score_cp"] = legacy
    result["evaluation"] = {"type": "mate", "cp": None, "mate": 0,
                            "winner": winner, "perspective": "white"}
    result.update(best_move=None, pv=[], depth=None)
    cache = EvaluationCache()
    cache.put("mate", result)
    assert cache.get("mate") == result


def test_move_types_and_pv_order_preserved(result):
    cache = EvaluationCache()
    cache.put("a", result)
    stored = cache.get("a")
    assert isinstance(stored["best_move"], chess.Move)
    assert all(isinstance(move, chess.Move) for move in stored["pv"])
    assert [move.uci() for move in stored["pv"]] == ["e2e4", "e7e5"]
    assert stored["best_move"] == result["best_move"]
    assert stored["best_move"] is not result["best_move"]


def test_info_is_immutable_independent_snapshot(result):
    cache = EvaluationCache(1)
    snapshot = cache.info()
    with pytest.raises(FrozenInstanceError):
        snapshot.hits = 10
    cache.put("a", result)
    cache.get("a")
    assert snapshot == CacheInfo(0, 0, 0, 0, 1)
    assert cache.info() == CacheInfo(1, 0, 0, 1, 1)


@pytest.mark.parametrize("invalid,exception", [
    (None, TypeError), (RuntimeError("failed search"), TypeError),
    ([], TypeError), ({}, ValueError),
])
def test_invalid_result_does_not_replace_or_evict(result, invalid, exception):
    cache = EvaluationCache(1)
    cache.put("a", result)
    before = cache.info()
    for key in ("a", "b"):
        with pytest.raises(exception):
            cache.put(key, invalid)
        assert cache.info() == before
    assert cache.get("a") == result


@pytest.mark.parametrize("field", ["score_cp", "evaluation", "best_move", "depth", "pv"])
def test_required_fields(result, field):
    del result[field]
    with pytest.raises(ValueError):
        EvaluationCache().put("a", result)


@pytest.mark.parametrize("field,value", [("evaluation", None), ("pv", ())])
def test_nested_container_contract(result, field, value):
    result[field] = value
    with pytest.raises(TypeError):
        EvaluationCache().put("a", result)


def test_copy_failure_does_not_evict_or_replace(result):
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    cache = EvaluationCache(1)
    cache.put("a", result)
    before = cache.info()
    invalid = {**result, "extra": Uncopyable()}
    for key in ("a", "b"):
        with pytest.raises(RuntimeError, match="cannot copy"):
            cache.put(key, invalid)
        assert cache.info() == before
    assert cache.get("a") == result


def test_unhashable_key_does_not_change_cache(result):
    cache = EvaluationCache()
    with pytest.raises(TypeError):
        cache.put([], result)
    with pytest.raises(TypeError):
        cache.get([])
    assert cache.info() == CacheInfo(0, 0, 0, 0, 4096)
