# Evidence-based mistake label proposals

Task 7 produces **provisional, unreviewed proposals**, not ground-truth labels.
The only supported concepts are the existing taxonomy identifiers `hanging_piece`
and `missed_mate`. `allowed_immediate_tactical_capture` is an internal evidence
type, not a new concept. No positional labels, confidence probabilities, training,
taxonomy changes, scoring changes, cache changes or engine-lifecycle changes are
included. Candidate and feature source artifacts remain unchanged.

## Architecture and existing interfaces

New modules under `src/chessmind/ml/labeling/`:

- `schema.py`: frozen configuration, verification, evidence, proposal and record types.
- `engine_verification.py`: borrowed-engine verification using existing APIs.
- `tactical_evidence.py`: legal capture/material and mating-continuation evidence.
- `rules.py`: deterministic rules and explicit abstention.
- `pipeline.py`: input validation, bounded selection, sequential verification and export.

The implementation reuses candidate score validation and metadata conventions,
the feature extractor and serializer, material features, engine metadata capture,
`analyze_board`, `prepare_cache_arguments` and `evaluate_move_quality`. It does not
create a second chess evaluator or recompute classification thresholds.

The existing position analyzer supplies native structured scores, depth and PVs.
Unlike Task 5's game-level results, this interface lets verification preserve
achieved depth separately for each endpoint. Its legacy synthetic numeric score
is ignored. The API does not expose all raw UCI search-bound flags; these outputs
cannot certify exact evaluations. Reconstructing a pre-move FEN also cannot restore
earlier repetition history, although FEN counters and the played move are retained.

## Verification and selection stability

Each selected candidate is reconstructed and checked against its stored after-FEN.
The verifier borrows one engine and requests the before and after positions using
one fresh game token per candidate. It never starts or closes an engine. The batch
owner starts one process, configures it once, prepares the existing cache context,
passes it through unchanged and closes the process once. A caller-supplied engine
is neither configured nor closed; its cache context must belong to that engine.

Default settings are depth 12, one thread and 64 MiB engine hash, with a 4,096-entry
existing evaluation cache. The CLI permits depth 1..30 and threads 1..4; low-depth
settings are useful for smoke tests, not stronger verification. The pilot uses 12.
Both nonterminal endpoint searches must return a legal PV and achieved depth at
least the requested depth. Terminal positions may have empty PVs and depth zero.
Search/startup errors become explicit failed verifications with no label fallback.
`--skip-verification` produces abstentions only.

Structured evaluations remain White-perspective. CP loss and mate quality come
from `evaluate_move_quality`, including its mover-relative White/Black handling.
Any mate endpoint makes CP loss null. Signed mate distance and winner are retained;
Mate(0) is checked against the terminal board and winner, never treated as missing
or converted to arbitrary centipawns.

Every verification records original and verified CP loss, classification and mate
quality; CP-loss difference when both are defined; classification changes; and
significant-error support using the candidate's original extraction threshold.
The support value is three-state: true, false or null for mate transitions. Thus
`significant_status_changed` includes changes to/from null, not just crossings of
the CP threshold. `materially_changed` means a classification change, support-state
change, mate-quality change, or absolute CP-loss difference >=100. This reporting
definition does not replace historical classifications or claim objective truth.

## Capture evidence and hanging-piece rule

The first move of the verified after-PV must be a legal opponent capture. The
initial detector is deliberately limited to non-pawn, non-king victims; en passant
is outside its scope. For a hanging-piece proposal, the victim must be the piece
just moved to the played move's destination (excluding castling).

The detector then requires all of the following:

1. The victim was not already geometrically attacked before the played move.
   This conservative exclusion can abstain even when a prior attack was pinned.
2. Both verified scores are CP and significant deterioration is supported.
   Mate transitions do not isolate a material-based explanation here.
3. The verified PV has at least four plies, unless it reaches a terminal board.
4. The capture retains positive material gain after every immediate legal
   recapture onto the capturing piece's square.
5. Positive material gain persists over the first up to eight plies of the PV.
   The whole transition, including material earned by the original played move,
   must also leave positive opponent gain. Capturing a queen before surrendering
   a knight must not be mistaken for a net material loss.
6. The inspected horizon is not in check/checkmate and has no immediate legal
   capture available. Unresolved forcing or capture sequences require review.

Material uses the existing heuristic pawn-unit values, not engine CP. Evidence
stores the capture, victim, moved-piece association, defense and prior-attack
flags, full continuation, PV material gain and worst immediate-recapture gain.
The feature delta for the mover's attacked-and-undefended count is included as
context, never as sufficient proof. A defended piece can still be lost for net
material; conversely, geometric attack alone does not imply a legal capture.

Supported captures of other pieces remain internal evidence and require review,
without inventing another taxonomy label. The same capture never produces a
second synonymous concept. These rules are intentionally conservative and may
miss many real material mistakes.

## Missed-mate rule

A proposal requires a verified **mover-winning mate before** the played move and
an existing mate-quality result of `lost_forced_mate` or `reversed_to_losing_mate`.
The verified before-PV must begin with a different move, replay legally and finish
in checkmate with the mover as winner. Scores and legal continuations after the
played move are retained for inspection.

Allowing an opponent's mate from a CP position is not a missed-mate proposal.
Keeping a mover-winning mate, including slower mate, is not labeled `missed_mate`.
Distance changes can receive `requires_review`; missing/incomplete mating lines
cause abstention. A checkmating PV is a concrete illustrative line, not an
exhaustive proof over all defenses. A post-move CP score can mean the engine has
not found a longer mate, rather than that forced mate objectively disappeared.
That is a central false-positive risk requiring expert review.

## Proposal schema and decisions

Schema: `mistake_proposals_v1`; rule version: `conservative_tactics_v1`.
All record structures are immutable and preserve candidate/game/player/ply
identity, source schema versions, input hashes, canonical feature-record hash,
structured verification, evidence, zero-or-more proposals and reasons.

Automatic status is separate from expert review:

| Automatic status | Meaning |
| --- | --- |
| `proposed` | At least one narrow rule has provisional support. |
| `insufficient_evidence` | No supported concept, or verification unavailable. |
| `conflicting_evidence` | A significant-CP candidate no longer meets its threshold. |
| `requires_review` | Retained-mate distance uncertainty or unresolved capture evidence. |

`review_status` is always `unreviewed` in generated artifacts. Proposal lists can
be empty; no move is forced into one concept. The schema/rule collector supports
multiple concepts, although the current capture CP gate and mate rule normally
make the two supported concepts mutually exclusive. No probabilistic confidence
field or automatic expert-approval state exists.

## Input validation, selection and output

Candidate IDs and versions are validated through existing feature reconstruction.
Stored feature records must exactly match recomputed features, including numeric
and boolean types, schemas and source linkage. Input manifests are checked by
hash, including the feature manifest's link to its candidate dataset. Missing
features, orphan feature IDs, duplicate IDs and malformed JSON are explicit
errors. In permissive mode they generate diagnostics and are excluded; neither
copy of a duplicated ID is eligible. Provenance-hash mismatches and resource-limit
violations remain fatal. Inputs are rehashed before publication to detect changes.

The pilot reader caps each input at 32 MiB, 10,000 physical records and two million
characters per line. This is a bounded small-artifact join held in memory, not a
large-dataset streaming join. Defaults select at most 24 records: round-robin over
the four original selection types, taking the first available records per type,
then restoring original candidate order. This intentionally mixes errors,
controls, mates and ambiguities; it is not a representative sample.

```powershell
.venv\Scripts\python.exe -m scripts.generate_mistake_label_proposals --help
.venv\Scripts\python.exe -m scripts.generate_mistake_label_proposals --candidates data/processed/ml/candidate_moves.jsonl --features data/processed/ml/candidate_features.jsonl --output-dir data/labeled/labeling_repeat --max-candidates 24 --depth 12 --threads 1 --hash-mb 64 --strict
```

The output directory must be new. Default output is `data/labeled/ml/`, containing:

- `mistake_label_proposals.jsonl`
- `labeling_manifest.json`
- `label_quality_report.json`

The quality report is colocated with the other artifacts so one atomic staging
directory rename publishes all three together. It includes per-candidate stability
comparisons, proposal/status distributions, abstention reasons and verification
failures. Reasons can have multiple entries per abstention, so their counts need
not sum to the number of abstaining records. JSONL serialization is deterministic
for identical inputs and verified engine outputs. Timestamps/timings legitimately
vary in manifests. Existing output and source files are never overwritten.

Verification failures are published as failed, abstaining records and cause CLI
exit 2. Permissive input diagnostics also produce exit 2. Fatal input/setup-of-
artifacts/I/O/shutdown errors abort publication (exit 1); interruption exits 130.
Engine startup failure records explicit failures for the selected batch. Empty
inputs and zero-candidate limits launch no engine. Depth is the search bound;
there is no promised wall-clock deadline for a batch.

## Real pilot results and inspection

Run: 2026-10-16 at 08:47:52 UTC (14:17:52 Asia/Colombo). Stockfish 19,
chess library 1.11.2, depth 12, Threads=1, Hash=64 MiB, MultiPV=1, full strength,
no Syzygy path. Full observed options and embedded-resource identity are retained.

| Measurement | Result |
| --- | ---: |
| Valid candidate/feature joins | 213 |
| Verified candidates | 24 |
| Selected CP errors / controls / mates / ambiguous | 7 / 7 / 7 / 3 |
| Records with proposals / total proposals | 4 / 4 |
| `missed_mate` proposals | 4 |
| `hanging_piece` proposals | 0 |
| Abstentions | 20 |
| Insufficient / conflicting / requires-review statuses | 19 / 1 / 0 |
| Verification failures | 0 |
| Changed classifications | 8 |
| Changed three-state significant support | 2 |
| Materially changed comparisons | 10 |
| Processing time | 2.497 seconds |
| Cache hits / misses / evictions | 0 / 48 / 0 |

Timing includes input validation, engine setup/search/shutdown and staged output,
but excludes final manifest serialization/publication. There were two position
requests per candidate. No original classifications or features were modified.

The four proposals were inspected by replaying and rendering their alternative
lines. All four reach checkmate legally:

| Game / ply | Played move | Concrete alternative from verified PV |
| --- | --- | --- |
| rFLdfbbL / 94 | e3e2 | 47...Bh4+ 48.Kg1 Re1+ 49.Kh2 Bg3+ 50.Kg2 Be4# |
| rFLdfbbL / 96 | d3c4 | 48...f3 49.Kg1 Re1+ 50.Kh2 Bf4# |
| BXeKWqsb / 91 | b7a7 | 46.Rxf7+ Kh8 47.Rxf8+ Kg7 48.h8=Q# |
| BXeKWqsb / 93 | a7a4 | 47.Rxf7+ Kh8 48.Rxf8+ Kg7 49.h8=Q# |

This inspection is **not independent expert adjudication**. All four retain
winning CP evaluations after the played move (roughly 11.5 to 14.6 pawns from
the mover's perspective). Longer forced mates may still exist. A reviewer must
decide whether these are genuinely missed mating opportunities, delayed mates,
or search-horizon artifacts. In particular, ply 94 was a depth-8 low-error control
and now has a reported lost mate; it deserves explicit review, not silent relabeling.

Abstention examples:

- rFLdfbbL ply 23: original loss 106 (`Mistake`), verified loss 95 (`Inaccuracy`);
  below the original 100 threshold, recorded as conflicting evidence.
- rFLdfbbL plies 25 and 29: PV capture victims were already geometrically attacked;
  no new hanging-piece claim is made.
- rFLdfbbL ply 98, 0JJ4ygL3 ply 55 and BXeKWqsb ply 89: mover-winning mate retained;
  no missed-mate proposal.
- rFLdfbbL ply 102: terminal mate, no opponent continuation; no erroneous
  missed-mate label from Mate(0).

Independent artifact audit verified input/output/report hashes, 24 unique IDs,
all 48 PVs' legality, four mating alternatives, and identical rule decisions and
evidence regenerated from saved verification results. Source artifacts match
their original hashes. No additional high-depth searches were needed for this audit.

Output SHA-256:
`5ff294b0f5d78a776517dd86c363f6dfc364e90cc4bd51a9a2b3bf36dc6edd8f`.

Engine binary SHA-256:
`45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0`.

## Review workflow, limitations and future evaluation

A reviewer should inspect the original candidate, both positions, verified scores
and PVs, and detector reasons; accept, reject or revise concepts in a separate
review artifact linked by candidate ID and rule version. Do not overwrite these
automatic proposals or mark them expert reviewed merely because a PV is legal.
No accuracy, precision or confidence estimate can be inferred from this pilot.

Finite search, a single PV, unknown historical repetition, short material horizons,
unexamined defensive alternatives and long-term compensation can cause false
positives. Strict geometric prior-attack exclusions and quiet-horizon requirements
can cause false negatives. Zero hanging-piece proposals do not mean none of the
games contain such mistakes. Feature changes by themselves remain insufficient.

The pilot spans only three games; its four proposals cluster in two games with
nearby positions. They are not independent evidence. Archive-prefix bias,
candidate thresholding and stratified pilot selection compound the original
dataset limitations. Future evaluation needs independent expert annotations,
blind review with disagreement/adjudication tracking, per-concept precision and
recall, abstention coverage, and game/player-separated splits. Include both
proposed and abstaining examples to avoid evaluating only easy positives.

Future positional labeling needs explicit operational definitions, position/line
evidence, counterexamples and expert validation. Fork/pin/skewer detection,
advanced exchange analysis, broad positional rules and model training remain
outside this task.

## Verification

Focused tests use synthetic legal positions and mocked engine responses, with one
conditional real-engine integration smoke test matching repository conventions.
They cover legal versus pinned captures, defended/undefended pieces, recapture
and played-move compensation, mate retention and concrete missed-mate alternatives,
both colors, Mate(0), CP signs, depth disagreements, missing/duplicate/mismatched
features, empty proposals, abstention, failures, cache-context forwarding,
deterministic JSONL, provenance, strict/permissive behavior and engine ownership.

Testing caught an invalid synthetic pinned-piece setup, which was corrected to
start from a legal position. Review added whole-transition material accounting
and strict boolean/numeric feature-type comparison. No engine changes were needed.

```powershell
.venv\Scripts\python.exe -m pytest tests/ml/test_tactical_evidence.py tests/ml/test_engine_verification.py tests/ml/test_labeling_rules.py tests/ml/test_labeling_pipeline.py -q
.venv\Scripts\python.exe -m pytest tests/ml -q
.venv\Scripts\python.exe -m pytest -q
```

Final results: **43 focused labeling tests, 319 ML tests and 561 full-suite
tests passed**, including the real-engine smoke test. CLI help and final whitespace
checks also passed. Windows permissions required elevated execution for existing
artifact access and pytest temporary directories. No unresolved test failures.
