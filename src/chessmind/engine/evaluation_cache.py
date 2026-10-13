"""Bounded, process-local storage for completed evaluation dictionaries.

This module has no engine dependency and does not decide search cacheability.
"""

from collections import OrderedDict
from collections.abc import Hashable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheInfo:
    """An immutable point-in-time snapshot, independent of subsequent operations."""

    hits: int
    misses: int
    evictions: int
    size: int
    max_entries: int


class EvaluationCache:
    """An LRU cache with independent copies on insertion and retrieval.

    Keys must be hashable and remain stable while stored. Values must be
    deepcopy-compatible dictionaries containing score_cp, evaluation, best_move,
    depth, and pv. evaluation must be a dictionary and pv must be a list.
    Additional fields are permitted and preserved. None, exceptions, and
    structurally incomplete results are rejected; score values and move objects
    are opaque and are never normalized or interpreted here. The caller must
    establish search success and semantic validity before calling put().

    Invalid values raise TypeError (wrong container type) or ValueError (missing
    fields). Copy failures propagate without changing entries or counters.
    Capacity zero disables validation, storage, and accounting for get/put.
    Capacity counts entries, not bytes. Instances are not thread-safe.
    """

    _REQUIRED_FIELDS = frozenset({"score_cp", "evaluation", "best_move", "depth", "pv"})

    def __init__(self, max_entries: int = 4096) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be a nonnegative integer")
        if max_entries < 0:
            raise ValueError("max_entries must be nonnegative")
        self._max_entries = max_entries
        self._entries: OrderedDict[Hashable, dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> dict[str, Any] | None:
        """Return an independent copy and refresh recency, or None on a miss."""
        if self._max_entries == 0:
            return None
        if key not in self._entries:
            self._misses += 1
            return None
        result = deepcopy(self._entries[key])
        self._entries.move_to_end(key)
        self._hits += 1
        return result

    def put(self, key: Hashable, result: dict[str, Any]) -> None:
        """Validate and copy before replacing or evicting any existing entry."""
        if self._max_entries == 0:
            return
        hash(key)
        if not isinstance(result, dict):
            raise TypeError("result must be an evaluation dictionary")
        missing = self._REQUIRED_FIELDS.difference(result)
        if missing:
            raise ValueError(f"result is missing required fields: {', '.join(sorted(missing))}")
        if not isinstance(result["evaluation"], dict):
            raise TypeError("result['evaluation'] must be a dictionary")
        if not isinstance(result["pv"], list):
            raise TypeError("result['pv'] must be a list")
        stored = deepcopy(result)
        self._entries[key] = stored
        self._entries.move_to_end(key)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def info(self) -> CacheInfo:
        """Return statistics without affecting recency or accounting."""
        return CacheInfo(self._hits, self._misses, self._evictions,
                         len(self._entries), self._max_entries)

    def clear(self) -> None:
        """Remove all entries and reset all counters, retaining capacity."""
        self._entries.clear()
        self._hits = self._misses = self._evictions = 0
