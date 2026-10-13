"""Conservative request identities; no search, configuration, or cache operations.

The metadata adapter is specific to chess 1.11.2 UciProtocol.config/target_config.
It mirrors UciProtocol.analysis and _position, including managed options. Re-audit
before enabling another library version. snapshot_engine_metadata reads on the
protocol event loop without scheduling a protocol command (which could cancel an
active search). Call between searches; concurrent configuration is unsupported.

Construct EngineIdentity once after each engine launch and reuse it for requests.
Executable and resource files must remain unchanged for that engine's lifetime.
No global fingerprint memoization is used. All unsupported metadata raises the
dedicated CacheIdentityUnavailable exception, not an engine-analysis exception.
"""

from concurrent.futures import Future
from dataclasses import dataclass, fields
import hashlib
import math
import os
from pathlib import Path
from typing import Mapping

import chess
import chess.engine

from src.chessmind.engine.embedded_resource_attestations import verified_embedded_resource


Scalar = str | int | float | bool | None
Pairs = tuple[tuple[str, Scalar], ...]
SUPPORTED_CHESS_VERSION = "1.11.2"
CACHE_SCHEMA_VERSION = 1


class CacheIdentityUnavailable(ValueError):
    """Metadata cannot safely identify this request; callers should bypass caching."""


@dataclass(frozen=True)
class OptionSpec:
    name: str
    type: str
    default: Scalar
    minimum: int | None
    maximum: int | None
    variants: tuple[str, ...]


@dataclass(frozen=True)
class EngineMetadata:
    reported_id: tuple[tuple[str, str], ...]
    options: tuple[OptionSpec, ...]
    config: Pairs
    target_config: Pairs


@dataclass(frozen=True)
class EngineIdentity:
    executable_path: str
    executable_sha256: str
    reported_id: tuple[tuple[str, str], ...]
    # (option name, option value, resolved resource path or 'embedded', digest)
    resources: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class SearchIdentity:
    limit: Pairs
    multipv: int | None
    root_moves: tuple[str, ...] | None
    info: int
    board_type: str
    variant: str
    chess960: bool
    root_fen: str
    moves: tuple[str, ...]
    current_fen: str


@dataclass(frozen=True)
class EvaluationKey:
    schema_version: int
    chess_version: str
    engine: EngineIdentity
    configuration: Pairs
    search: SearchIdentity


def _version() -> None:
    if chess.__version__ != SUPPORTED_CHESS_VERSION:
        raise CacheIdentityUnavailable("Unsupported python-chess implementation version")


def _scalar(value: object) -> Scalar:
    if type(value) not in (str, int, float, bool, type(None)):
        raise CacheIdentityUnavailable("Metadata values must be immutable scalars")
    if isinstance(value, float) and not math.isfinite(value):
        raise CacheIdentityUnavailable("Non-finite metadata")
    return value


def _names(mapping: Mapping) -> dict:
    result = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or name.lower() in result:
            raise CacheIdentityUnavailable("Invalid or duplicate option name")
        result[name.lower()] = value
    return result


def metadata_from_values(reported_id: Mapping, options: Mapping,
                         config: Mapping, target_config: Mapping) -> EngineMetadata:
    """Freeze a UCI snapshot; also usable with deterministic test metadata."""
    _version()
    if not all(isinstance(value, Mapping) for value in (reported_id, options, config, target_config)):
        raise CacheIdentityUnavailable("Missing UCI metadata mappings")
    if not reported_id.get("name") or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in reported_id.items()):
        raise CacheIdentityUnavailable("Missing engine-reported identity")
    specs = []
    for name, option in sorted(_names(options).items()):
        if not isinstance(option, chess.engine.Option) or option.name.lower() != name:
            raise CacheIdentityUnavailable("Invalid option declaration")
        if option.type not in ("check", "spin", "combo", "string", "button"):
            raise CacheIdentityUnavailable("Unsupported option type")
        specs.append(OptionSpec(name, option.type, _scalar(option.default),
                                option.min, option.max, tuple(option.var or ())))
    if not specs:
        raise CacheIdentityUnavailable("Missing UCI option declarations")
    return EngineMetadata(tuple(sorted(reported_id.items())), tuple(specs),
                          tuple(sorted((k, _scalar(v)) for k, v in _names(config).items())),
                          tuple(sorted((k, _scalar(v)) for k, v in _names(target_config).items())))


def snapshot_engine_metadata(engine: chess.engine.SimpleEngine) -> EngineMetadata:
    """Read on the engine loop, without analyse/configure/send_line/communicate.

    The caller must serialize this with analysis/configuration. A queued loop
    callback only reads metadata and does not interrupt any engine command.
    """
    _version()
    try:
        protocol = engine.protocol
        if type(protocol) is not chess.engine.UciProtocol:
            raise CacheIdentityUnavailable("Only installed UciProtocol is supported")
        future: Future = Future()

        def read() -> None:
            try:
                if protocol.returncode.done():
                    raise CacheIdentityUnavailable("Engine has terminated")
                future.set_result(metadata_from_values(
                    protocol.id, protocol.options, protocol.config, protocol.target_config))
            except Exception as exc:
                future.set_exception(exc)

        protocol.loop.call_soon_threadsafe(read)
        return future.result(timeout=engine.timeout if engine.timeout is not None else 10)
    except Exception as exc:
        raise CacheIdentityUnavailable(f"Engine metadata unavailable: {exc}") from exc


def _fingerprint(path: str | Path) -> tuple[str, str]:
    try:
        resolved = Path(path).resolve(strict=True)
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = resolved.stat()
        # Reading may update access time; that is not a content modification.
        def signature(stat):
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature(before) != signature(after):
            raise CacheIdentityUnavailable("File changed while fingerprinting")
        return str(resolved), digest
    except OSError as exc:
        raise CacheIdentityUnavailable(f"Cannot fingerprint {path}: {exc}") from exc


def _resource_values(configuration: Mapping) -> dict[str, str]:
    resources = {}
    for name, value in configuration.items():
        if name == "syzygypath" and value not in (None, "", "<empty>"):
            raise CacheIdentityUnavailable("External tablebase identity is unsupported")
        if name.startswith("evalfile"):
            if not isinstance(value, str) or not value or value == "<empty>":
                raise CacheIdentityUnavailable("Unknown evaluation resource")
            resources[name] = value
    return resources


def build_engine_identity(executable_path: str | Path, metadata: EngineMetadata, *,
                          resource_paths: Mapping[str, str | Path] | None = None,
                          embedded_resources: tuple[str, ...] = ()) -> EngineIdentity:
    """Hash once per invocation/engine lifetime, never in build_cache_key.

    resource_paths explicitly resolves EvalFile/EvalFileSmall to actual files.
    embedded_resources is a caller attestation that named options use networks
    embedded in this binary, not external files with the same name. Defaults are
    NOT assumed embedded: automatic attestation requires an audited binary hash
    and exact resource value. Unknown external resources/tablebases cause bypass.
    Rebuild identity if evaluation resources are reconfigured.
    """
    _version()
    if not isinstance(metadata, EngineMetadata) or not dict(metadata.reported_id).get("name"):
        raise CacheIdentityUnavailable("Missing engine metadata")
    path, digest = _fingerprint(executable_path)
    configuration = dict(metadata.config)
    configuration.update(metadata.target_config)
    values = _resource_values(configuration)
    explicit = _names(resource_paths or {})
    embedded = {name.lower() for name in embedded_resources}
    if (explicit.keys() | embedded) - values.keys() or explicit.keys() & embedded:
        raise CacheIdentityUnavailable("Ambiguous or unknown resource declaration")
    resources = []
    for name, value in sorted(values.items()):
        if name in explicit:
            resource_path, resource_digest = _fingerprint(explicit[name])
        elif name in embedded or verified_embedded_resource(digest, name, value):
            resource_path, resource_digest = "embedded", digest
        else:
            raise CacheIdentityUnavailable(f"Unidentified evaluation resource: {name}")
        resources.append((name, value, resource_path, resource_digest))
    return EngineIdentity(path, digest, metadata.reported_id, tuple(resources))


def build_search_identity(board: chess.Board, limit: chess.engine.Limit, *,
                          multipv: int | None = None,
                          root_moves: tuple[chess.Move, ...] | None = None,
                          info: chess.engine.Info = chess.engine.INFO_ALL) -> SearchIdentity:
    """Preserve all available history. FEN-only boards have an empty history.

    Supports standard Board including Chess960; variant/custom subclasses bypass.
    Never consume a caller's root-move generator or discard a null-move stack.
    """
    _version()
    if type(board) is not chess.Board or type(limit) is not chess.engine.Limit:
        raise CacheIdentityUnavailable("Unsupported board or limit implementation")
    if any(not move for move in board.move_stack):
        raise CacheIdentityUnavailable("Null-move history is unsupported")
    if multipv is not None and (type(multipv) is not int or multipv < 1):
        raise CacheIdentityUnavailable("Invalid MultiPV")
    if root_moves is not None and (type(root_moves) is not tuple or any(
            type(move) is not chess.Move or move not in board.legal_moves for move in root_moves)):
        raise CacheIdentityUnavailable("Root moves must be an immutable tuple of legal moves")
    root = board.root()
    current_fen = board.fen(shredder=board.chess960, en_passant="fen")
    replay = root.copy()
    for move in board.move_stack:
        if move not in replay.legal_moves:
            raise CacheIdentityUnavailable("Inconsistent move history")
        replay.push(move)
    if replay.fen(shredder=board.chess960, en_passant="fen") != current_fen:
        raise CacheIdentityUnavailable("History does not reproduce current board")
    return SearchIdentity(
        tuple((field.name, _scalar(getattr(limit, field.name))) for field in fields(limit)),
        multipv, None if root_moves is None else tuple(m.uci() for m in root_moves),
        int(info), f"{type(board).__module__}.{type(board).__qualname__}",
        type(board).uci_variant, board.chess960,
        root.fen(shredder=board.chess960, en_passant="fen"),
        tuple(move.uci() for move in board.move_stack), current_fen)


def effective_uci_configuration(metadata: EngineMetadata, search: SearchIdentity, *,
                                options: Mapping[str, Scalar] | None = None) -> Pairs:
    """Mirror 1.11.2 analysis option precedence without changing engine state."""
    _version()
    if not isinstance(metadata, EngineMetadata):
        raise CacheIdentityUnavailable("Missing engine metadata")
    specs = {option.name: option for option in metadata.options}
    config = dict(metadata.config)
    target = dict(metadata.target_config)
    overrides = _names(options or {})
    if (config.keys() | target.keys() | overrides.keys()) - specs.keys():
        raise CacheIdentityUnavailable("Undeclared UCI option")
    if (target.keys() | overrides.keys()) & set(chess.engine.MANAGED_OPTIONS):
        raise CacheIdentityUnavailable("Managed options cannot be explicitly configured")
    for name, value in (("ponder", False), ("multipv", search.multipv or 1)):
        if name in specs:
            config[name] = value
    if search.multipv not in (None, 1) and "multipv" not in specs:
        raise CacheIdentityUnavailable("MultiPV unsupported by engine")
    if "uci_analysemode" in specs and "uci_analysemode" not in target and "uci_analysemode" not in overrides:
        config["uci_analysemode"] = True
    config.update(target)
    config.update(overrides)
    for name, value, required in (("uci_variant", search.variant, search.variant != "chess"),
                                  ("uci_chess960", search.chess960, search.chess960)):
        if name in specs:
            config[name] = value
        elif required:
            raise CacheIdentityUnavailable(f"Engine does not support {name}")
    for name, spec in specs.items():
        if spec.type == "button":
            if name in target or name in overrides:
                raise CacheIdentityUnavailable("Button commands are not cacheable configuration")
            config.pop(name, None)
            continue
        if name not in config or config[name] is None:
            raise CacheIdentityUnavailable(f"Missing option value: {name}")
        option = chess.engine.Option(name, spec.type, spec.default, spec.minimum,
                                     spec.maximum, list(spec.variants))
        try:
            config[name] = _scalar(option.parse(config[name]))
        except (chess.engine.EngineError, TypeError, ValueError) as exc:
            raise CacheIdentityUnavailable(f"Invalid option value: {name}") from exc
    _resource_values(config)
    return tuple(sorted(config.items()))


def build_cache_key(identity: EngineIdentity | None, metadata: EngineMetadata,
                    board: chess.Board, limit: chess.engine.Limit, *,
                    multipv: int | None = None,
                    root_moves: tuple[chess.Move, ...] | None = None,
                    info: chess.engine.Info = chess.engine.INFO_ALL,
                    options: Mapping[str, Scalar] | None = None) -> EvaluationKey:
    """Build a key from an existing identity; never reads or hashes engine files."""
    if (not isinstance(identity, EngineIdentity) or not isinstance(metadata, EngineMetadata)
            or identity.reported_id != metadata.reported_id):
        raise CacheIdentityUnavailable("Missing or mismatched engine identity")
    search = build_search_identity(board, limit, multipv=multipv, root_moves=root_moves, info=info)
    configuration = effective_uci_configuration(metadata, search, options=options)
    if _resource_values(dict(configuration)) != {r[0]: r[1] for r in identity.resources}:
        raise CacheIdentityUnavailable("Evaluation resources changed; rebuild engine identity")
    return EvaluationKey(CACHE_SCHEMA_VERSION, chess.__version__, identity, configuration, search)
