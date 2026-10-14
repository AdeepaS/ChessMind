# Lichess puzzle dataset exploration

## Dataset overview

Analyzed **1000 accepted records**, with **1000 unique IDs**.
Validation processed 1000 rows, accepted 1000, and rejected
0 in permissive mode. Rejection reasons: {}.
There are 63 distinct source themes and 4525
theme assignments. These are multi-label source annotations, not human mistake labels.

## Source information

- Input: `data/raw/ml/puzzles/lichess_puzzle_sample.csv`.
- SHA-256: `09709b9bf8bff1e82e32c3625c519b7cf25da7d0a95334be58bb772e3f507dc5`.
- Manifest hash verified: True; accepted count matches manifest: True.
- Source URL from manifest: https://database.lichess.org/lichess_db_puzzle.csv.zst.
- Acquisition timestamp (UTC): 2026-10-14T05:47:35.594151+00:00.
- Sampling method: first valid unique records in export order; not random.

The [official database documentation](https://database.lichess.org/#puzzles)
explains that the first move sets up the solving position. This analysis reuses
the existing loader's validation and keeps its original source annotations.

## Rating statistics

Minimum: 399.000; maximum: 3,050.000; mean:
1,441.965; median: 1,399.500; population standard
deviation: 544.823. Population standard deviation
describes this finite sample (denominator N), not an estimate of export-wide variance.

| Rating interval | Puzzles | % |
| --- | --- | --- |
| [0, 500) | 16 | 1.6 |
| [500, 1000) | 230 | 23.0 |
| [1000, 1500) | 327 | 32.7 |
| [1500, 2000) | 246 | 24.6 |
| [2000, 2500) | 148 | 14.8 |
| [2500, 3000) | 32 | 3.2 |
| [3000, infinity) | 1 | 0.1 |

Buckets include their lower boundary and exclude their upper boundary. The last
bucket is unbounded. Percentages use all accepted records and round to six
decimal places; rounding can make displayed totals differ slightly from 100%.
The largest bucket starts at 1000 and contains 327
puzzles (32.7%). This establishes within-sample imbalance only.

## Top 15 source themes

| Theme | Puzzles | % of accepted puzzles |
| --- | --- | --- |
| endgame | 511 | 51.1 |
| short | 504 | 50.4 |
| middlegame | 437 | 43.7 |
| crushing | 386 | 38.6 |
| mate | 336 | 33.6 |
| advantage | 270 | 27.0 |
| long | 257 | 25.7 |
| mateIn1 | 170 | 17.0 |
| oneMove | 170 | 17.0 |
| master | 150 | 15.0 |
| mateIn2 | 141 | 14.1 |
| fork | 128 | 12.8 |
| kingsideAttack | 79 | 7.9 |
| sacrifice | 78 | 7.8 |
| veryLong | 65 | 6.5 |

Each theme is counted once per puzzle. Theme counts and percentages must not
sum to the puzzle count or 100%, because one puzzle has multiple annotations.
All observed themes, including rare ones, appear in `theme_counts.csv`.

## Theme co-occurrence

| Theme A | Theme B | Puzzles | % |
| --- | --- | --- | --- |
| endgame | short | 269 | 26.9 |
| crushing | endgame | 249 | 24.9 |
| crushing | short | 215 | 21.5 |
| middlegame | short | 212 | 21.2 |
| mate | mateIn1 | 170 | 17.0 |
| mate | oneMove | 170 | 17.0 |
| mateIn1 | oneMove | 170 | 17.0 |
| mate | middlegame | 163 | 16.3 |
| endgame | mate | 155 | 15.5 |
| advantage | middlegame | 146 | 14.6 |
| advantage | short | 144 | 14.4 |
| mate | mateIn2 | 141 | 14.1 |
| mate | short | 141 | 14.1 |
| mateIn2 | short | 141 | 14.1 |
| endgame | long | 140 | 14.0 |

There are 683 distinct observed unordered pairs.
Pairs contain two different themes, count once per puzzle, and sort by descending
count then alphabetical pair. These are raw joint counts, not evidence of
causality or association beyond base rates. Phase, length and outcome tags can
dominate the leading pairs; nested mate tags can overlap by definition. Do not
interpret those pairs as two independent tactical skills or split overlapping
puzzles across training and evaluation sets.

## Data quality findings

The most frequent annotation is endgame (511 puzzles, 51.1%). Frequent general annotations can dominate a naive multi-label
objective. Rare is defined here as at most 5 puzzles per theme;
18 observed themes meet that threshold: doubleCheck (5), smotheredMate (5), equality (4), interference (4), capturingDefender (3), hookMate (3), mateIn4 (3), morphysMate (3), bodenMate (2), cornerMate (2), enPassant (2), epauletteMate (2), swallowstailMate (2), arabianMate (1), doubleBishopMate (1), superGM (1), vukovicMate (1), xRayAttack (1).
Themes absent from this sample cannot be enumerated from sample counts alone.

Distinct themes per puzzle have mean 4.525
and median 4.000.
Duplicate theme tokens removed from counting: 0.

| Distinct themes per puzzle | Puzzles | % |
| --- | --- | --- |
| 1 | 3 | 0.3 |
| 2 | 1 | 0.1 |
| 3 | 155 | 15.5 |
| 4 | 392 | 39.2 |
| 5 | 276 | 27.6 |
| 6 | 124 | 12.4 |
| 7 | 36 | 3.6 |
| 8 | 9 | 0.9 |
| 9 | 1 | 0.1 |
| 10 | 3 | 0.3 |

| Metadata | Minimum | Maximum | Mean | Median | Population stddev |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.000 | 100.000 | 86.898 | 90.000 | 12.929 |
| number_of_plays | 4.000 | 37,506.000 | 1,573.703 | 419.000 | 3,691.780 |

Missing optional annotations: 0 puzzles have no themes;
793 have no opening tags; 1000
have an empty DailyDate cell; 0 lack that column.
An empty opening tag is not automatically corruption: Lichess documents limited
opening-tag coverage. DailyDate only records a daily-feature event, so blank
values do not establish a data defect or a missing game date. Neither field is
an appropriate blanket row-exclusion rule. Popularity and play counts reflect
engagement and exposure, not objective annotation correctness.

Legal positions and move sequences do not prove that themes are complete,
solutions are optimal, or previous human moves satisfy a specific mistake
definition. Only ID uniqueness was validated here; repeated games, equivalent
positions and near-duplicate tactical patterns still need a separate audit.

## Taxonomy comparison: proposal only

The source's [theme definitions](https://github.com/lichess-org/lila/blob/master/translation/source/puzzleTheme.xml)
mix several annotation dimensions. A proposed separation for future annotation
design is below; these are examples, not assertions that each is present here.

| Source theme group | Examples | Proposed use |
| --- | --- | --- |
| Tactical motifs | fork, pin, skewer, discoveredAttack, hangingPiece, deflection | Opportunity annotations on the solving position |
| Game phases | opening, middlegame, endgame | Separate source phase context; no equivalence assumed with ChessMind's detector |
| Tactical outcomes and mating patterns | mate, mateIn1, mateIn2, backRankMate | Outcome/pattern annotations, separate from mistake severity |
| Strategic or endgame context | pawnEndgame, rookEndgame, advancedPawn, zugzwang, exposedKing | Candidate context for endgame and positional investigations |
| Objective, length and provenance | advantage, equality, crushing, oneMove, short, long, master, superGM | Separate objective/length/provenance metadata; rating is a separate difficulty signal |

The following editorial grouping accounts for the observed vocabulary. It is
only a documentation proposal; ambiguous concepts such as quiet moves, king
attacks and promotions may warrant multiple dimensions in a future schema.
Unknown future themes remain explicitly unreviewed, never silently mapped.

| Proposed annotation dimension | Observed themes (puzzle counts) |
| --- | --- |
| Tactical motifs and techniques | attraction (30), capturingDefender (3), clearance (20), defensiveMove (49), deflection (42), discoveredAttack (41), discoveredCheck (14), doubleCheck (5), enPassant (2), fork (128), hangingPiece (38), interference (4), intermezzo (12), pin (54), quietMove (41), sacrifice (78), skewer (27), trappedPiece (8), xRayAttack (1) |
| Game phases | endgame (511), middlegame (437), opening (52) |
| Tactical outcomes and mating patterns | arabianMate (1), backRankMate (37), bodenMate (2), cornerMate (2), doubleBishopMate (1), epauletteMate (2), hookMate (3), mate (336), mateIn1 (170), mateIn2 (141), mateIn3 (22), mateIn4 (3), morphysMate (3), operaMate (15), pillsburysMate (7), smotheredMate (5), swallowstailMate (2), vukovicMate (1) |
| Strategic or endgame context | advancedPawn (59), attackingF2F7 (9), bishopEndgame (13), exposedKing (40), kingsideAttack (79), knightEndgame (13), pawnEndgame (27), promotion (19), queenEndgame (8), queenRookEndgame (7), queensideAttack (14), rookEndgame (59), zugzwang (7) |
| Objective, length and provenance | advantage (270), crushing (386), equality (4), long (257), master (150), masterVsMaster (14), oneMove (170), short (504), superGM (1), veryLong (65) |
| Unreviewed source themes | none |

Possible relationships to the existing taxonomy are **research hypotheses**:

| Source evidence | Existing concept/category to investigate | Additional evidence required |
| --- | --- | --- |
| fork, pin, skewer | missed_fork, missed_pin, missed_skewer / tactical | A verified opportunity before the focal human move that the player failed to exploit |
| hangingPiece | hanging_piece / tactical | Ownership, defenders, and the focal move that left a piece capturable |
| mate and mateInN | missed_mate / tactical | A winning mate available to that player before their actual move, then lost |
| exposedKing, kingsideAttack, queensideAttack | king_safety / positional | Human move context and independent assessment of a king-safety error |
| pawnEndgame, rookEndgame | pawn_endgame, rook_endgame / endgame_technique | A technique error, not merely a matching material configuration |
| opening | opening_knowledge category | Evidence about development or center_control errors; phase alone is insufficient |

A `fork` puzzle does not establish `missed_fork`; a `pin` puzzle does not
establish `missed_pin`. The solver's opportunity and the previous player's
mistake have different actors and board states. The setup move alone cannot
prove causation: the opportunity may have existed earlier. No executable mapping
or mistake labels are produced, and `taxonomy.py` remains unchanged.

Current gaps include motifs such as discovered attack/check, deflection,
attraction, removal of a defender, interference, promotion and trapped pieces;
named mating patterns; and endgames beyond pawn and rook endings. The source
does not directly establish errors in pawn_structure, piece_activity,
weak_squares, development, center_control, or king_activity. Those need human
move context and positional evidence, not a guessed theme correspondence.

Future annotation design must distinguish:

1. **Missing an available tactic:** the player had a verified opportunity and
   chose a move that failed to exploit it.
2. **Allowing an opponent's tactic:** the player's move created or permitted an
   opposing opportunity; this is not the player's own missed tactic.
3. **Making a positional mistake:** structural or activity damage may have no
   immediate forcing sequence or puzzle motif.
4. **Evaluation loss without a clear motif:** retain severity separately and
   allow unassigned concepts/uncertain explanation. Never force every loss into
   a tactical label. Existing thresholds and mate policies remain authoritative.

## Sampling limitations

When the source manifest identifies a bounded prefix of the official export,
it is not a probability sample. Export order can produce selection bias, even when
IDs look arbitrary. This report measures only the supplied input; it does not
estimate full-database theme prevalence. Consult this input's source manifest;
missing provenance cannot establish that a sample was selected randomly.

Puzzle curation selects positions with actionable solutions, omitting ordinary
play, correct quiet moves, many positional mistakes, and negative examples.
Puzzle rating is not the source player's rating. Source theme absence is not
proof that the motif is absent. Rare support, hierarchical themes and overlapping
labels make this sample insufficient to establish training readiness or stable
per-label evaluation. No held-out performance or full-export comparison was
measured, so representativeness remains unproven.

## Recommendations for future collection and ML research

- For tactical recognition, plan a reproducible reservoir sample across the
  full export, then rating- and motif-aware multi-label stratification. Define
  rare-label quotas from evaluation needs, preserve inclusion probabilities,
  and maintain a separate natural-prevalence holdout. Do not count phase,
  length and nested outcomes as independent tactical skills.
- For mistake-classification support, separately collect consented or suitably
  licensed full games with focal moves, pre/post positions, legal alternatives,
  player perspective, and reliable evaluation evidence. Include missed tactics,
  allowed tactics, positional errors, correct moves, and uncertain explanations.
  Use reviewed annotations; a puzzle theme is only candidate supporting evidence.
- For personalized recommendations, seek player-level attempt outcomes,
  difficulty and skill coverage, while keeping source-player ratings distinct
  from puzzle ratings. Evaluate on held-out users and later attempts to reduce
  history leakage; popularity alone should not select training material.
- Audit normalized positions, original game IDs/URLs and near duplicates before
  splitting. Keep related games/positions in one partition, respect temporal
  boundaries when available, version the export and sampling seed, and document
  residual selection bias. Collection and training are future work only.

## Reproduction

Run from the repository root with the existing virtual environment:

```powershell
.venv/Scripts/python.exe -m scripts.analyze_puzzle_dataset
.venv/Scripts/python.exe -m pytest tests/ml/test_puzzle_exploration.py -q
```

Options: `--input`, `--output-dir`, `--research-output`, `--manifest`, `--strict`,
`--rating-buckets 0 500 1000 1500 2000 2500 3000`, and `--rare-threshold 5`.
The adjacent `source_manifest.json` is used if present; an explicit manifest
must exist. Its hash is checked before report files are written. No downloads
occur. Repeated runs with identical inputs/options produce identical report
bytes. Empty datasets have null numeric summaries and zero percentages.

Outputs: `theme_counts.csv`, `rating_distribution.csv`, `theme_cooccurrence.csv`,
and `dataset_quality.json`; CSVs share the same counts as the JSON. Exact medians
retain numeric columns in memory, but the pipeline never retains all board/move
records. No new dependencies are required.

## Conclusions

This dataset supports source-label exploration and future tactical retrieval
prototypes. Validation and descriptive counts do not establish suitability for
human mistake classification. Broader sampling, negative examples, actor-aware
annotations, and leakage-resistant evaluation are prerequisites for that claim.
