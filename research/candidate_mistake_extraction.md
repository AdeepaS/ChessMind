# Candidate mistake extraction

Task 5 selects provisional move evidence from the existing game-analysis results.
It does not assign tactical or positional concept labels, confidence scores, or
training targets. No engine scoring, classification thresholds, cache code,
engine lifecycle code, taxonomy, or existing reports were changed.

## Existing interfaces and adapter limitations

Inspection found that `GameRecord` supplies validated complete PGN text, source
identity, ratings, UTC metadata and UCI moves. `analyze_game(game, depth, engine=,
cache=, cache_context=)` supplies a list of move dictionaries and already uses
N+1 position requests. Its borrowed-engine contract leaves shutdown to the owner
and creates a fresh game token for every game. `prepare_cache_arguments` creates
the existing conservative cache identity, falling back to uncached searches when
identity cannot be established.

The move results include both FENs, SAN/UCI, phase, optional best move, structured
White-perspective scores, genuine CP loss or null, classification and mate quality.
Legacy numeric evaluations can contain synthetic mate projections: extraction
ignores these and uses only the structured score details. `get_moves` and the
existing human-game loader remain the source of game validation.

The analysis interface does not return achieved depth, search bounds, timing,
principal variations, or engine identity per move. No adapter invents these:
achieved depth is null, configured depth is explicit, and observed engine metadata
is captured between searches at batch setup. Standalone extraction without an
analysis profile leaves engine identity and configured depth null. Per-game and
batch elapsed times are measured around the orchestration, not supplied by the
engine. The extractor validates source/ply/FEN/SAN associations and adjacent score
continuity without recalculating losses or classifications.

## Selection methodology

Frozen `ExtractionConfig` defaults:

| Parameter | Default |
| --- | ---: |
| Significant CP-loss threshold | 100 |
| Maximum control CP loss | 20 |
| Maximum controls per game | 5 |
| Seed | 42 |
| Include mate candidates | true |

These are experimental selection settings, not new severity thresholds.
`SIGNIFICANT_CP_ERROR` includes defined losses >=100. Existing classifications
are copied exactly: a loss of exactly 100 is selected while retaining the
existing `Inaccuracy` classification. CP loss measures evaluation deterioration
from the mover's perspective; it cannot identify whether a move involves a fork,
pin, strategic weakness or any other error concept.

`LOW_ERROR_CONTROL` considers defined losses <=20 after significant errors have
been removed. If configured ranges overlap, significant errors take precedence.
Eligible controls are ranked by SHA-256 of seed, platform, game ID and ply, and
the first five are selected. This is deterministic pseudo-random sampling without
dependence on global RNG state or Python's randomized `hash()`. All selected
records are then emitted in input-game and ascending-ply order. It is a sample
of the low-loss pool, not a guarantee of phase, player or rating balance.

Mate transitions never receive synthetic CP loss. The existing `mate_quality`
reason is retained and interpreted as follows:

| Existing reason | Selection | Transition status |
| --- | --- | --- |
| lost/allowed forced mate, reversed to losing mate | MATE_RELATED | harmful |
| escaped/achieved forced mate, reversed to winning mate, delivered checkmate | MATE_RELATED | beneficial |
| maintained forced mate, remains forced loss | omitted | neutral continuation |
| slower mate, faster defeat, checkmated, unknown/missing mate reason | AMBIGUOUS | ambiguous |

Mate-distance differences are search heuristics, so they are preserved as
ambiguous even when the existing classifier calls them inaccuracies. Beneficial
transitions are evidence records, not mistakes. Missing CP loss in a CP/CP result
is explicitly `AMBIGUOUS` with reason `undefined_cp_loss`. Disabling mate
candidates omits all transitions with a mate endpoint. No unknown case is forced
into a tactical concept.

## Schema, provenance and reproducibility

`candidate_schema.py` defines immutable records with immutable nested score,
configuration, engine-profile and source-evidence objects. Records carry both
ratings, source-qualified player identity, game ID, ply and full move number,
positions, moves, phase, existing classification and score evidence. Optional
best moves, UTC fields and unavailable metadata remain null.

Candidate IDs are SHA-256 of schema version (`candidate_moves_v1`), platform,
game ID and ply. Each ply appears at most once. IDs identify the source position,
not an engine run: compare configuration and provenance before merging runs.
Usernames are qualified by platform and case-normalized by the existing loader.
Cross-platform identity is never inferred.

Input PGN SHA-256 is checked against the adjacent `source_manifest.json` (or an
explicit manifest) before engine startup. The source manifest's own SHA-256,
archive URL and original game index are carried into the evidence where available.
The candidate manifest records input and output hashes, settings, engine profile,
loader rejections, failed games, selected counts, timings and cache statistics.

JSONL uses UTF-8, LF, sorted keys and fixed separators, with no timestamps or
timings in candidate records. Identical input identity, configuration, metadata
and engine results produce identical JSONL. Engine searches themselves are not
promised byte-identical across runs: version, binary/resources, options, history,
cache state and hardware can affect results. Timestamps and timings legitimately
vary in manifests. Existing caches are process-local and not persisted here.

## Pipeline and commands

The pipeline owns one lazily launched engine for a nonempty batch, sets Threads=1
and Hash=64 MiB, prepares the existing bounded 4,096-entry cache and passes the
same engine to `analyze_game` sequentially. It never launches per move or alters
the N+1 evaluation approach. Borrowed engines are never configured or closed;
their owner must supply the matching cache context to enable cache use. A
borrowed engine's binary hash remains null unless its path or identity is given.

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m scripts.extract_candidate_moves --help
.venv\Scripts\python.exe -m scripts.extract_candidate_moves --input data/raw/ml/games/lichess_game_sample.pgn --output data/processed/candidate_pilot_repeat/candidate_moves.jsonl --max-games 10 --depth 8 --cp-threshold 100 --control-threshold 20 --max-controls-per-game 5 --seed 42
```

The default output is `data/processed/ml/candidate_moves.jsonl` with sibling
`candidate_manifest.json`. **The output parent must be a new directory** so a
single atomic directory rename publishes both files together. Existing output
directories are never overwritten. JSONL is written incrementally to staging;
each game's records are validated and serialized before any are written.
Detected per-game analysis/validation failures are recorded and skipped without
retrying. Engine termination stops the batch. Startup, shutdown, stream I/O and
interrupt failures remove staging and publish no bundle. A forced process kill
can leave a hidden staging directory. Partial batches publish explicit failure
counts/shortages and exit 2; fatal failures exit 1, interruption 130.

`max_games` limits accepted games attempted, not successes: failures do not pull
replacement games into the pilot. Invalid/duplicate source games are counted by
the existing permissive loader. Empty inputs and zero-game batches launch no
engine. The default batch is only ten games, not the complete 500-game dataset.

## Real pilot benchmark

Run on 2026-10-16, manifest timestamp 05:46:18 UTC (11:16:18 Asia/Colombo), using
the first ten eligible games from the existing sample. No additional data was
downloaded. The engine reported **Stockfish 19**, chess library 1.11.2, configured
depth 8, Threads=1, Hash=64 MiB, MultiPV=1, full strength, no Syzygy path. The
complete observed options and embedded NNUE resource identity are in the manifest.

Engine binary SHA-256:
`45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0`.

Input PGN SHA-256:
`48d2798c43cff3eb92c63a3fb20d8ebc0fca6e6a5714f1f7fd49232d5586ca94`.

Source manifest SHA-256:
`4ec2061bd306bff67f211141d0ad944279dd44855d10c0fef4dce2a912afd0f6`.

Output JSONL SHA-256:
`8ff3ac6b61895291ae6e8bcf7c340eaf4befb3ca97b88da10c2fd143313f5aa6`.

| Measurement | Result |
| --- | ---: |
| Games analyzed | 10 |
| Analyzed plies | 720 |
| Significant CP-error candidates | 147 |
| Low-error controls | 50 |
| Mate-related records | 13 (9 harmful, 4 beneficial) |
| Ambiguous records | 3 (mate-distance cases) |
| Total records | 213 |
| Analysis failures / loader rejections | 0 / 0 |
| Total processing time | 6.681 s |
| Average processing time per game | 0.668 s |
| Cache hits / misses / evictions | 19 / 711 / 0 |

The cache accounts for all 730 position requests (720 plies plus ten initial
positions); 711 required engine searches. Batch time includes input hashing,
startup, metadata capture, analysis, extraction and shutdown, but excludes final
manifest serialization and publication. Per-game timings exclude engine startup.
These are one local pilot's measurements, not a performance guarantee.

Artifacts: `data/processed/ml/candidate_moves.jsonl` and
`data/processed/ml/candidate_manifest.json`.

## Limitations and future labeling

Depth 8 is a modest benchmark setting, not reliable ground truth. Candidates may
change at greater depth; controls can hide mistakes missed by the engine. Mate
winner/distance estimates can also change with search. The original PGN's engine
comments are not used as evaluations in this pipeline.

The source sample already has archive-prefix, quota and time-control selection
bias. Taking the first ten games adds ordering bias. CP thresholding favors sharp
evaluation drops, controls are capped per game, mate continuations are filtered,
and long games can supply more error candidates. Class proportions in the export
are selection artifacts, not population error rates. The source's player-disjoint
design leaves only one game per player, insufficient for recurring-weakness
learning. No model-ready annotations or verified mistake claims are made.

Future labels require separate evidence-based detectors, deeper verification of
candidate and alternative lines, tactical/positional tests, human review and
explicit uncertainty handling. Future player-history work also requires repeated
chronological observations and player-separated evaluation splits. None of that
labeling or model training is part of this task.

## Verification and discovered issues

Focused offline tests use synthetic PGNs and mocked existing-format analysis.
They cover boundaries, classifications, seeds, overlap, both colors, associations,
FENs, missing best moves, mate handling, undefined losses, malformed results,
stable IDs, deterministic serialization, provenance, duplicate games, failure
cleanup and borrowed/owned engine contracts. A conditional real-Stockfish smoke
test follows existing repository conventions and checks N+1 requests and shutdown.

During testing, converting a dataclass containing `Counter` via `asdict` produced
tuple keys when rejections were nonempty. The pipeline now explicitly serializes
rejection counts as a plain mapping, covered by the duplicate-game test. Review
also removed an unjustified default depth from standalone extraction metadata;
unknown analysis profiles now remain null. No existing engine correctness change
was required.

```powershell
.venv\Scripts\python.exe -m pytest tests/ml/test_candidate_extraction.py -q
.venv\Scripts\python.exe -m pytest tests/ml/test_candidate_pipeline.py -q
.venv\Scripts\python.exe -m pytest tests/ml -q
.venv\Scripts\python.exe -m pytest -q
```

Final results: **52 extraction tests passed; 18 pipeline tests passed (including
the real-engine smoke test); 237 ML tests passed; 479 full-suite tests passed**.
An independent audit of the saved pilot verified 213 unique IDs, selection
counts, 720 plies, matching engine profiles and all three input/source-manifest/
output SHA-256 values. Windows permissions required elevated execution for local
dataset access and pytest temporary directories. No unresolved test failures.
