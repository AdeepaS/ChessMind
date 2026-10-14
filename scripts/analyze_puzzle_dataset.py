"""Explore an existing puzzle dataset and generate deterministic local reports."""

import argparse
from pathlib import Path

import zstandard

from src.chessmind.ml.data.puzzle_exploration import (
    DEFAULT_RATING_BUCKETS, analyze_dataset, write_reports,
)


def _table(headers, rows):
    def escaped(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(escaped(value) for value in row) + " |" for row in rows),
    ])


def _display(number):
    return "not available" if number is None else f"{number:,.3f}"


def _source_theme_groups(analysis):
    """Editorial grouping for the research table, not a mistake-label mapping."""
    groups = {
        "Tactical motifs and techniques": "attraction capturingDefender clearance defensiveMove deflection discoveredAttack discoveredCheck doubleCheck enPassant fork hangingPiece interference intermezzo pin quietMove sacrifice skewer trappedPiece xRayAttack castling",
        "Game phases": "opening middlegame endgame",
        "Tactical outcomes and mating patterns": "mate mateIn1 mateIn2 mateIn3 mateIn4 mateIn5 anastasiaMate arabianMate backRankMate balestraMate blindSwineMate bodenMate cornerMate doubleBishopMate dovetailMate epauletteMate hookMate killBoxMate morphysMate operaMate pillsburysMate smotheredMate swallowstailMate triangleMate vukovicMate",
        "Strategic or endgame context": "advancedPawn attackingF2F7 bishopEndgame exposedKing kingsideAttack knightEndgame pawnEndgame promotion queenEndgame queenRookEndgame queensideAttack rookEndgame underPromotion zugzwang",
        "Objective, length and provenance": "advantage crushing equality oneMove short long veryLong master masterVsMaster superGM",
    }
    counts = {row["theme"]: row["count"] for row in analysis["themes"]}
    remaining = set(counts)
    rows = []
    for group, vocabulary in groups.items():
        observed = sorted(remaining.intersection(vocabulary.split()))
        remaining.difference_update(observed)
        rows.append((group, ", ".join(f"{theme} ({counts[theme]})" for theme in observed) or "none"))
    rows.append(("Unreviewed source themes", ", ".join(sorted(remaining)) or "none"))
    return _table(("Proposed annotation dimension", "Observed themes (puzzle counts)"), rows)


def write_research_report(analysis: dict, path: str | Path) -> None:
    """Computed findings and a prose-only proposal; no runtime taxonomy mapping."""
    general = analysis["general"]
    ratings = general["ratings"]
    validation = analysis["validation"]
    source = analysis["source"]
    manifest = source["manifest"] or {}
    metadata = analysis["metadata"]
    missing = metadata["missing_optional"]
    top_table = _table(("Theme", "Puzzles", "% of accepted puzzles"), (
        (row["theme"], row["count"], row["percentage"]) for row in analysis["top_themes"]
    ))
    buckets = analysis["rating_distribution"]
    rating_table = _table(("Rating interval", "Puzzles", "%"), (
        (f"[{row['lower_inclusive']}, {row['upper_exclusive'] if row['upper_exclusive'] is not None else 'infinity'})",
         row["count"], row["percentage"]) for row in buckets
    ))
    pairs_table = _table(("Theme A", "Theme B", "Puzzles", "%"), (
        (row["theme_a"], row["theme_b"], row["count"], row["percentage"])
        for row in analysis["theme_cooccurrence"][:15]
    ))
    size_table = _table(("Distinct themes per puzzle", "Puzzles", "%"), (
        (row["theme_count"], row["count"], row["percentage"])
        for row in analysis["theme_count_distribution"]
    ))
    metadata_table = _table(("Metadata", "Minimum", "Maximum", "Mean", "Median", "Population stddev"), (
        (key, *(_display(metadata[key][stat]) for stat in ("min", "max", "mean", "median", "population_stddev")))
        for key in ("popularity", "number_of_plays")
    ))
    rare = ", ".join(f"{row['theme']} ({row['count']})" for row in analysis["rare_themes"]) or "none"
    largest = max(buckets, key=lambda row: row["count"])
    leading = analysis["top_themes"][0] if analysis["top_themes"] else None
    dominance = (f"The most frequent annotation is {leading['theme']} ({leading['count']} puzzles, "
                 f"{leading['percentage']}%)." if leading else "No themes were observed.")
    groups_table = _source_theme_groups(analysis)
    text = f"""# Lichess puzzle dataset exploration

## Dataset overview

Analyzed **{general['total_puzzles']} accepted records**, with **{general['unique_puzzles']} unique IDs**.
Validation processed {validation['processed']} rows, accepted {validation['accepted']}, and rejected
{validation['rejected']} in {validation['mode']} mode. Rejection reasons: {dict(validation['rejection_counts'])}.
There are {analysis['distinct_themes']} distinct source themes and {analysis['total_theme_assignments']}
theme assignments. These are multi-label source annotations, not human mistake labels.

## Source information

- Input: `{source['input_file']}`.
- SHA-256: `{source['sha256']}`.
- Manifest hash verified: {source['manifest_hash_verified']}; accepted count matches manifest: {source['manifest_count_matches']}.
- Source URL from manifest: {manifest.get('dataset_url', 'not provided')}.
- Acquisition timestamp (UTC): {manifest.get('acquired_at_utc', 'not provided')}.
- Sampling method: {manifest.get('sampling_method', 'not documented; do not assume random sampling')}.

The [official database documentation](https://database.lichess.org/#puzzles)
explains that the first move sets up the solving position. This analysis reuses
the existing loader's validation and keeps its original source annotations.

## Rating statistics

Minimum: {_display(ratings['min'])}; maximum: {_display(ratings['max'])}; mean:
{_display(ratings['mean'])}; median: {_display(ratings['median'])}; population standard
deviation: {_display(ratings['population_stddev'])}. Population standard deviation
describes this finite sample (denominator N), not an estimate of export-wide variance.

{rating_table}

Buckets include their lower boundary and exclude their upper boundary. The last
bucket is unbounded. Percentages use all accepted records and round to six
decimal places; rounding can make displayed totals differ slightly from 100%.
The largest bucket starts at {largest['lower_inclusive']} and contains {largest['count']}
puzzles ({largest['percentage']}%). This establishes within-sample imbalance only.

## Top 15 source themes

{top_table}

Each theme is counted once per puzzle. Theme counts and percentages must not
sum to the puzzle count or 100%, because one puzzle has multiple annotations.
All observed themes, including rare ones, appear in `theme_counts.csv`.

## Theme co-occurrence

{pairs_table}

There are {len(analysis['theme_cooccurrence'])} distinct observed unordered pairs.
Pairs contain two different themes, count once per puzzle, and sort by descending
count then alphabetical pair. These are raw joint counts, not evidence of
causality or association beyond base rates. Phase, length and outcome tags can
dominate the leading pairs; nested mate tags can overlap by definition. Do not
interpret those pairs as two independent tactical skills or split overlapping
puzzles across training and evaluation sets.

## Data quality findings

{dominance} Frequent general annotations can dominate a naive multi-label
objective. Rare is defined here as at most {analysis['rare_threshold']} puzzles per theme;
{len(analysis['rare_themes'])} observed themes meet that threshold: {rare}.
Themes absent from this sample cannot be enumerated from sample counts alone.

Distinct themes per puzzle have mean {_display(analysis['themes_per_puzzle']['mean'])}
and median {_display(analysis['themes_per_puzzle']['median'])}.
Duplicate theme tokens removed from counting: {analysis['duplicate_theme_tokens']}.

{size_table}

{metadata_table}

Missing optional annotations: {missing['themes_empty']} puzzles have no themes;
{missing['opening_tags_empty']} have no opening tags; {missing['daily_date_empty']}
have an empty DailyDate cell; {missing['daily_date_column_absent']} lack that column.
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

{groups_table}

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
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        target.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/ml/puzzles/lichess_puzzle_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/ml"))
    parser.add_argument("--research-output", type=Path, default=Path("research/puzzle_dataset_exploration.md"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--rating-buckets", type=int, nargs="+", default=DEFAULT_RATING_BUCKETS)
    parser.add_argument("--rare-threshold", type=int, default=5)
    args = parser.parse_args()
    try:
        manifest = args.manifest
        if manifest is None:
            adjacent = args.input.with_name("source_manifest.json")
            if adjacent.is_file():
                manifest = adjacent
        analysis = analyze_dataset(args.input, manifest_path=manifest, strict=args.strict,
                                   bucket_edges=args.rating_buckets, rare_threshold=args.rare_threshold)
        write_reports(analysis, args.output_dir)
        write_research_report(analysis, args.research_output)
    except (OSError, ValueError, zstandard.ZstdError) as exc:
        parser.exit(1, f"Dataset exploration failed: {exc}\n")
    stats = analysis["validation"]
    print(f"Processed {stats['processed']}; accepted {stats['accepted']}; rejected {stats['rejected']}; "
          f"unique {analysis['general']['unique_puzzles']}; themes {analysis['distinct_themes']}.")
    ratings = analysis["general"]["ratings"]
    print(f"Ratings: {ratings['min']}..{ratings['max']}; mean {_display(ratings['mean'])}; "
          f"median {_display(ratings['median'])}; population stddev {_display(ratings['population_stddev'])}.")
    print(f"Reports: {args.output_dir}; research: {args.research_output}")


if __name__ == "__main__":
    main()
