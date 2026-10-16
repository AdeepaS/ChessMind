"""Bounded, auditable standard PGN ingestion; no engine analysis or labels."""

import io
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn
import zstandard

from ...pgn.parser import get_moves


class GameValidationError(ValueError):
    def __init__(self, reason, index):
        self.reason = reason
        self.index = index
        super().__init__(f"Game {index}: {reason}")


class GameResourceLimit(ValueError):
    """Framing cannot safely continue after a per-game resource limit."""


@dataclass(frozen=True)
class GameFilters:
    rated_only: bool = True
    time_classes: tuple[str, ...] = ("rapid", "classical")
    min_rating: int = 1
    max_rating: int = 4000
    min_plies: int = 1

    def __post_init__(self):
        if not 1 <= self.min_rating <= self.max_rating <= 4000:
            raise ValueError("rating bounds must lie within 1..4000")
        if self.min_plies < 0 or not set(self.time_classes) <= {
                "ultrabullet", "bullet", "blitz", "rapid", "classical"}:
            raise ValueError("invalid length or time classes")


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    white_player: str
    black_player: str
    white_rating: int
    black_rating: int
    result: str
    utc_date: str | None
    utc_time: str | None
    time_control: str
    time_class: str
    event: str
    opening: str | None
    eco: str | None
    move_count: int  # plies (individual half-moves)
    uci_moves: tuple[str, ...]
    original_pgn: str
    source_archive: str
    source_index: int
    source: str = "lichess"

    @property
    def average_rating(self):
        return (self.white_rating + self.black_rating) / 2

    @property
    def player_keys(self):
        return tuple(f"{self.source}:{p.casefold()}" for p in
                     (self.white_player, self.black_player))


@dataclass
class GameLoadStats:
    processed: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_counts: Counter = field(default_factory=Counter)
    error_examples: list[str] = field(default_factory=list)


def time_class(control):
    if not re.fullmatch(r"[0-9]{1,6}\+[0-9]{1,6}", control):
        return None
    base, increment = map(int, control.split("+"))
    seconds = base + 40 * increment
    if seconds <= 0:
        return None
    return next(name for ceiling, name in ((29, "ultrabullet"), (179, "bullet"),
                (479, "blitz"), (1499, "rapid"), (float("inf"), "classical"))
                if seconds <= ceiling)


HEADER = re.compile(r'^\[([A-Za-z0-9_]+) "((?:[^"\\]|\\.)*)"\]\s*$')
# Full lexical coverage closes python-chess's deliberate tolerance of junk.
TOKEN = re.compile(r"\s+|\{[^}]*\}|;[^\n]*|[0-9]+\.(?:\.\.)?|"
                   r"1/2-1/2|1-0|0-1|\*|\$[0-9]+|[()!?+#]|"
                   r"(?:O-O-O|O-O|0-0-0|0-0)|"
                   r"[NBKRQ]?[a-h]?[1-8]?[x-]?[a-h][1-8](?:=?[NBRQnbrq])?"
                   , re.DOTALL)


def _validate(raw, index, archive, filters):
    def reject(reason):
        raise GameValidationError(reason, index)

    headers = {}
    lines = raw.splitlines(keepends=True)
    start = 0
    for start, line in enumerate(lines):
        if not line.strip():
            continue
        if not line.startswith("["):
            break
        match = HEADER.fullmatch(line)
        if not match or match[1] in headers:
            reject("malformed_headers")
        headers[match[1]] = match[2]
    else:
        reject("incomplete_game")
    if any(headers.get(k, "?") in ("", "?") for k in
           ("Site", "White", "Black", "Result", "Event", "TimeControl")):
        reject("missing_headers")
    if headers.get("Variant", "Standard") != "Standard":
        reject("unsupported_variant")
    if any(headers.get(k, "").upper() == "BOT" for k in ("WhiteTitle", "BlackTitle")):
        reject("bot_player")
    match = re.fullmatch(r"https://lichess.org/([A-Za-z0-9]{8})", headers["Site"])
    if not match:
        reject("invalid_id")
    game_id = match[1]
    result = headers["Result"]
    if result not in ("1-0", "0-1", "1/2-1/2") or headers.get("Termination", "").lower() in (
            "abandoned", "unterminated"):
        reject("incomplete_game")
    body = "".join(lines[start:])
    pos = depth = 0
    terminal = None
    while pos < len(body):
        token = TOKEN.match(body, pos)
        if token is None:
            reject("malformed_movetext")
        value = token[0]
        pos = token.end()
        if value.isspace() or value.startswith(("{", ";")):
            continue
        if terminal is not None:
            reject("trailing_movetext")
        if value == "(":
            depth += 1
        elif value == ")":
            depth -= 1
            if depth < 0:
                reject("malformed_movetext")
        elif value in ("1-0", "0-1", "1/2-1/2", "*") and depth == 0:
            terminal = value
    if depth or terminal != result:
        reject("incomplete_game")
    try:
        game = chess.pgn.read_game(io.StringIO(raw))
        if game is None or game.errors:
            reject("invalid_moves")
        board = game.board()
        if type(board) is not chess.Board or board.chess960 or not board.is_valid():
            reject("invalid_position")
        # Database standard games must start from the standard initial position.
        if board.fen() != chess.STARTING_FEN:
            reject("nonstandard_start")
        moves = get_moves(game)
        for move in moves:
            if not board.is_legal(move):
                reject("invalid_moves")
            board.push(move)
            if not board.is_valid():
                reject("invalid_position")
        if board.is_game_over() and board.result() != result:
            reject("result_mismatch")
    except GameValidationError:
        raise
    except ValueError:
        reject("invalid_position")
    ratings = []
    for key in ("WhiteElo", "BlackElo"):
        value = headers.get(key, "")
        if not re.fullmatch(r"[0-9]{1,4}", value) or not 1 <= int(value) <= 4000:
            reject("invalid_rating")
        ratings.append(int(value))
    if filters.rated_only and not headers["Event"].startswith("Rated "):
        reject("unrated_game")
    speed = time_class(headers["TimeControl"])
    if speed is None or speed not in filters.time_classes:
        reject("time_control_filter")
    if not filters.min_rating <= sum(ratings) / 2 <= filters.max_rating:
        reject("rating_filter")
    if len(moves) < max(1, filters.min_plies):
        reject("length_filter")
    def optional(key):
        value = headers.get(key)
        return None if not value or "?" in value or value == "-" else value
    return GameRecord(game_id, headers["White"], headers["Black"], *ratings,
                      result, optional("UTCDate"), optional("UTCTime"),
                      headers["TimeControl"], speed, headers["Event"],
                      optional("Opening"), optional("ECO"), len(moves),
                      tuple(m.uci() for m in moves), raw, archive, index)


class GamePGNReader:
    """Stream complete PGNs. Strict raises on any rejection; permissive skips.

    Memory: O(scanned IDs + max_game_chars). max_games bounds ID storage.
    Syntax errors are rejected per framed game. Oversized records and stream
    errors are fatal, since safe resynchronization is not assured.
    """
    def __init__(self, stream, *, strict=True, max_records=None, max_games=None,
                 max_game_chars=1_000_000, source_archive="", filters=None):
        for limit in (max_records, max_games):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError("limits must be nonnegative integers")
        if type(max_game_chars) is not int or max_game_chars < 1:
            raise ValueError("max_game_chars must be positive")
        self.stream, self.strict = stream, strict
        self.max_records, self.max_games = max_records, max_games
        self.max_game_chars, self.source_archive = max_game_chars, source_archive
        self.filters = filters or GameFilters()
        self.stats = GameLoadStats()
        self._seen = set()
        self._pending = ""

    def __iter__(self):
        return self

    def _raw_game(self):
        parts, size, body, comment = [], 0, False, False
        while True:
            line = self._pending or self.stream.readline(self.max_game_chars + 1)
            self._pending = ""
            if not line:
                return "".join(parts)
            if body and not comment and line.startswith("["):
                self._pending = line
                return "".join(parts)
            size += len(line)
            if size > self.max_game_chars:
                raise GameResourceLimit("maximum game characters exceeded")
            if not parts and not line.strip():
                size = 0
                continue
            parts.append(line)
            if line.strip() and (body or not line.startswith("[")):
                body = True
                for char in line:
                    if char == ";" and not comment:
                        break
                    if char == "{":
                        comment = True
                    elif char == "}":
                        comment = False
            if body and not comment and not line.strip():
                return "".join(parts)

    def __next__(self):
        while True:
            if ((self.max_records is not None and self.stats.accepted >= self.max_records)
                    or (self.max_games is not None and self.stats.processed >= self.max_games)):
                raise StopIteration
            raw = self._raw_game()
            if not raw:
                raise StopIteration
            self.stats.processed += 1
            try:
                identity = re.search(r'^\[Site "https://lichess.org/([A-Za-z0-9]{8})"\]', raw, re.M)
                if identity:
                    if identity[1] in self._seen:
                        raise GameValidationError("duplicate_id", self.stats.processed)
                    self._seen.add(identity[1])
                record = _validate(raw, self.stats.processed, self.source_archive, self.filters)
            except GameValidationError as exc:
                self.stats.rejected += 1
                self.stats.rejection_counts[exc.reason] += 1
                if len(self.stats.error_examples) < 5:
                    self.stats.error_examples.append(str(exc))
                if self.strict:
                    raise
                continue
            self.stats.accepted += 1
            return record


@contextmanager
def open_games(path, **options):
    path = Path(path)
    options.setdefault("source_archive", str(path))
    if path.name.endswith(".pgn.zst"):
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor(max_window_size=128 * 1024 * 1024).stream_reader(source) as decoded:
                with io.TextIOWrapper(decoded, encoding="utf-8-sig") as stream:
                    yield GamePGNReader(stream, **options)
    elif path.suffix == ".pgn":
        with path.open(encoding="utf-8-sig") as stream:
            yield GamePGNReader(stream, **options)
    else:
        raise ValueError("Expected .pgn or .pgn.zst")
