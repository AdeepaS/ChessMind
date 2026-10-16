# Lightweight chess feature engineering

Task 6 adds deterministic, interpretable board characteristics alongside the
existing Stockfish evidence. It does not implement an evaluation function, train
models, detect tactical motifs, or assign mistake labels. Engine integration,
classification thresholds, caching, lifecycle behavior and taxonomy are unchanged.
No new dependencies or datasets are required.

## Interfaces and schema

Package: `src/chessmind/ml/features/`. Public functions:

```python
import chess
from src.chessmind.ml.features import (
    extract_position_features, extract_move_features, extract_candidate_features,
)

board = chess.Board()
position = extract_position_features(board)
transition = extract_move_features(board, chess.Move.from_uci("e2e4"))
# extract_candidate_features accepts an existing CandidateRecord or JSON mapping.
```

The feature schema is **`chess_features_v1`**. Dataclasses and nested records are
frozen; deltas use an immutable tuple. Standard, valid python-chess positions
with both kings are required; Chess960 and variant boards are rejected. Move
extraction requires a legal move and never mutates the caller's board, move
stack, turn, castling rights or en-passant state. No engine is launched.

Each candidate contains **163 feature values**:

- 52 position fields before the move and the same 52 after it.
- 49 numerical deltas, one for each numeric/boolean position field.
- 10 move-context fields.

This count includes categorical values and booleans, but excludes IDs, source
metadata, schema-version fields and delta field-name strings. Each position has
24 fields per color plus four shared fields. Feature names are stable dataclass
fields; JSONL uses UTF-8, LF, sorted object keys and fixed separators. Changed
definitions require a new feature schema version. Position-only objects use the
same definitions; their version is the exported `FEATURE_SCHEMA_VERSION`.

## Material (per color)

`pawns`, `knights`, `bishops`, `rooks`, `queens`, `kings` count pieces on the board.
`material` is a heuristic total in pawn units: pawn=1, knight=3, bishop=3, rook=5,
queen=9, king=0. Non-pawn values reuse the existing game-phase constants.
`non_pawn_material` excludes pawns and kings. `bishop_pair` means at least two
bishops, including promoted bishops; it does not require opposite square colors.
Shared `material_balance` is always **White material minus Black material**.
These quantities are not centipawns or Stockfish scores. Promotions and captures
change actual piece counts, including underpromotions and en-passant captures.

## Piece safety and geometric coverage (per color)

`attacked_pieces` counts own non-king pieces whose squares are attacked by the
opponent according to `Board.is_attacked_by`. `defended_pieces` counts own
non-king pieces attacked by another own piece. `undefended_pieces` counts all
own non-king pieces without such a defender, whether attacked or not.
`attacked_undefended_pieces` is the intersection of attacked and undefended.
All four ordinary piece counts exclude kings. Each piece counts once, regardless
of the number of attackers or defenders.

`geometrically_attacked_squares` counts the union of `Board.attacks` over all own
pieces, **including the king**, with each square counted once. Occupancy blocks
sliding rays; squares occupied by friendly pieces are included where attacked.
Pawn attacks are diagonals, not forward pushes. Pinned pieces still contribute
geometric attacks and defenses. En passant is not represented as a geometric
attack on the captured pawn's occupied square. The engine-independent capture
flag separately handles en passant.

Geometric attack is distinct from a legal capture and from an actual tactical
threat. A pinned knight can geometrically attack a piece while being legally
unable to capture it. A defended piece can still be lost tactically; an attacked
undefended piece can be harmless in context. No `hanging_piece` or other tactical
label is inferred from these counts.

## Pawn structure (per color)

- `doubled_excess_pawns`: sum `max(0, pawns_on_file - 1)` across files. Three
  pawns on one file contribute two, not three.
- `isolated_pawns`: count each pawn with no friendly pawn anywhere on either
  adjacent file. File-edge checks never wrap from a-file to h-file. Doubled
  isolated pawns each count as isolated.
- `passed_pawns`: count each pawn with no enemy pawn **strictly ahead**, on
  its own or an adjacent file. Ahead means higher rank for White, lower rank
  for Black. Enemy pawns on the same rank or behind do not prevent this flag.

The passed-pawn definition ignores non-pawn blockaders and friendly pawns ahead;
multiple pawns on a file can count as passed. These are static geometric
indicators, not judgments of strength or positional mistakes. Promotion removes
the promoted pawn from all pawn counts. No-pawn positions produce zero counts.

## King geometry (per color)

`king_square` is algebraic notation. `king_file` and `king_rank` are zero-based
indices: a1=(0,0), h8=(7,7). `kingside_castling_right` and
`queenside_castling_right` report retained rights, **not current castling
legality**; blocked paths or attacks can still prevent castling.

`pawn_shield` counts own pawns exactly one rank forward of the king, on the
king's file and its adjacent files (maximum three, clipped to board edges).
This definition applies equally to central, uncastled and castled kings; it
does not judge whether a king has castled or whether its shield is effective.

The king zone is the king square plus all adjacent squares. The field
`attacked_king_zone_squares` counts zone squares geometrically attacked by the
opponent, once per square, including pinned attackers. It does not count attack
multiplicity or assign a composite king-safety score. Missing kings, invalid
castling rights and other invalid standard-board states are rejected.

## Turn, mobility, move context and deltas

Shared position fields include `side_to_move`, `side_to_move_in_check` and
`side_to_move_legal_moves`. Legal mobility uses the actual board's legal-move
generator; the implementation never toggles turn to estimate the other player's
mobility. Before and after mobility generally describe **different players**.
Their difference is a raw transition characteristic, not a same-player mobility
gain or a measure of quality. History-dependent draw claims are not inferred
from isolated FENs.

Move context contains `moving_color`, `piece_type` (before moving),
`source_square`, `target_square`, `played_move_uci`, `is_capture`, `is_en_passant`,
`is_castling`, `promotion_piece` (null when absent), and `gives_check`.
All use python-chess's legal board/move definitions.

Every numerical delta uses **after minus before**, with explicit dotted field
paths such as `white.material` and `black.passed_pawns`. Boolean deltas first
convert false/true to 0/1. King file/rank deltas are coordinate displacement,
not safety scores. King-square strings and side-to-move strings have no numerical
delta. No mover-relative quantities are mixed with the White-minus-Black balance.

## Candidate integration and publication

The adapter preserves the original candidate ID, game ID, source-qualified player,
ply, candidate schema version and typed source evidence. It reuses the existing
stable candidate-ID function to check identity and the existing source-evidence
schema. It validates required fields, parses both FENs, validates the legal UCI
move and player/ply/full-move association, and requires the computed canonical
after-FEN to exactly equal the stored after-FEN, including counters. Candidate
schemas other than `candidate_moves_v1` are rejected. Unknown optional provenance
is explicit null. Source engine scores and selection labels are not copied into
board feature values; they remain joinable by candidate ID.

```powershell
.venv\Scripts\python.exe -m scripts.extract_chess_features --help
.venv\Scripts\python.exe -m scripts.extract_chess_features --input data/processed/ml/candidate_moves.jsonl --output data/processed/ml/candidate_features.jsonl
# For another run, choose fresh output paths:
.venv\Scripts\python.exe -m scripts.extract_chess_features --output data/processed/feature_repeat/candidate_features.jsonl --permissive
```

Strict mode is the default and aborts on the first rejected record. `--permissive`
counts and skips malformed records, including malformed JSON, duplicate JSON
keys, non-finite JSON constants, bad positions, missing fields and duplicate IDs.
The first textual occurrence reserves an ID even if validation subsequently
fails. Blank lines are ignored; empty input produces an empty JSONL and manifest.
Invalid UTF-8, I/O errors and provenance mismatches are fatal in either mode.
The line cap defaults to two million characters; oversized lines are drained
without retaining them and rejected. Memory is one bounded line/record plus
scanned IDs for duplicate detection, not all feature records.

An adjacent `candidate_manifest.json` is discovered automatically, or may be
provided with `--source-manifest`. Its output hash and candidate count must agree
with the input. The feature manifest records input/output SHA-256, source manifest
identity, definitions/configuration, versions, processed/accepted/rejected counts,
rejection reasons and elapsed time. Input content is rehashed before publication
to detect concurrent changes.

Both outputs are staged on the destination filesystem. Existing files are never
overwritten. Each file is atomically published using an exclusive hard link,
with `feature_manifest.json` published last as the completion marker. Ordinary
publication failures remove this run's output. A forced process kill between
links can leave an orphan JSONL; consumers must require a matching manifest and
verify its hash. This supports the existing candidate directory without replacing
or editing its candidate files. Filesystems must support hard links; otherwise
publication fails cleanly. This is not a claim of atomic two-file visibility or
power-loss durability.

## Real data validation

Generated on 2026-10-16 at 08:09:47 UTC (13:39:47 Asia/Colombo) using the existing
candidate dataset. Results:

| Measurement | Result |
| --- | ---: |
| Input candidates | 213 |
| Accepted feature records | 213 |
| Rejected records | 0 |
| Unique candidate IDs | 213 |
| Feature values per record | 163 |
| Processing time | 2.258 seconds |

Processing time includes hashing, validation, extraction and staged JSONL writing;
it excludes final manifest writing/publication. No engine ran. The audit confirmed
input ordering, matching IDs, uniform schemas, all provenance hashes and
**byte-identical regeneration of all 213 feature records**. Candidate source files
remain unchanged.

Input SHA-256:
`8ff3ac6b61895291ae6e8bcf7c340eaf4befb3ca97b88da10c2fd143313f5aa6`.

Output SHA-256:
`24deb2bba795e432c127cff30cbc7c41f717f3c92bca712ab50147270b423059`.

Artifacts are `data/processed/ml/candidate_features.jsonl` and
`data/processed/ml/feature_manifest.json`.

Inspected examples from game `rFLdfbbL`:

- Ply 21, White `c1g5`: quiet bishop move; both material totals stay at 35.
- Ply 67, White `a2b3`: pawn capture; Black material decreases 23 to 22,
  White stays at 15, and White-minus-Black balance delta is +1.
- Ply 94, Black `e3e2`: non-capturing rook check; material totals stay 1 and 17,
  and balance delta is zero.

## Limitations and future use

These features can supply interpretable covariates alongside engine evaluations
in later experiments. Categorical encoding, scaling and feature selection remain
future modeling choices. Do not infer an error label directly from material loss,
an isolated pawn, an attacked square or any other individual feature. Sacrifices,
forcing moves and positional compensation require further evidence.

The original candidates were selected using depth-8 scores from ten games, with
threshold, archive-order and control-sampling biases. This feature layer does
not remove those biases or make the candidates model-ready. Post-move features
are available only after a move is known; future experiments must distinguish
retrospective classification from pre-move prediction and avoid target leakage.
Recurring-player prediction still requires chronological player histories and
player-separated validation. Evidence-based tactical/positional detectors,
deeper engine verification and reviewed labels remain separate future tasks.

## Tests

All feature tests are offline and independent of Stockfish. They cover material,
promotions, capture deltas, pinned geometric attackers, pawn edges/direction and
mirroring, local king geometry, rights, checks, en passant, mobility, immutability,
candidate association, FEN mismatch, malformed JSONL, duplicate IDs, strict and
permissive modes, empty input, deterministic output, hashes and publication rollback.

```powershell
.venv\Scripts\python.exe -m pytest tests/ml/test_chess_features.py tests/ml/test_feature_pipeline.py -q
.venv\Scripts\python.exe -m pytest tests/ml -q
.venv\Scripts\python.exe -m pytest -q
```

Final verification: **39 focused feature tests, 276 ML tests and 518 full-suite
tests passed**. The full suite includes existing real-engine tests; feature tests
and feature generation do not use Stockfish. Windows permissions required
elevated access to existing datasets and pytest temporary directories. No test
failures or real-data rejections remain.
