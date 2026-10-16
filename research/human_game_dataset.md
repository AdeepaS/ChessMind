# Human game dataset acquisition and validation

Task 4 implements ingestion and inspection only. No training, mistake labeling,
Stockfish analysis, caching changes or player-report changes are included.

## Source and provenance

The [official Lichess database](https://database.lichess.org/) publishes monthly
standard rated PGN archives under CC0. This sample uses
<https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst>.
Acquired on 2026-10-16 at 05:16:47 UTC (10:46:47 Asia/Colombo).

Artifacts under `data/raw/ml/games/`:

- `lichess_game_sample.pgn`: complete original PGN game text, including comments.
- `source_manifest.json`: source, timestamp, budgets, filters, requested/actual
  quotas, rejection counts, SHA-256, and source archive game indices for every ID.
- `inspection_report.json`: distributions, missing fields, annotation counts,
  source-qualified player frequencies and manifest verification.

Sample SHA-256:
`48d2798c43cff3eb92c63a3fb20d8ebc0fca6e6a5714f1f7fd49232d5586ca94`.
This hashes the saved PGN, not the monthly archive. Both count and digest were
verified after reloading the saved file; all 500 games passed strict validation.

## Selection and observed data

Selection is deterministic: first eligible games in archive order per rating
stratum. The arithmetic average of both ratings determines the stratum; each
individual rating is retained. Fractional averages use half-open ranges, e.g.
1199.5 belongs to `[800,1200)`. Ratings must be integers within 1..4000.
Games must be rated, standard, completed, at least 20 plies long, and rapid or
classical. BOT-tagged players are excluded. Selected players cannot recur by
default (case-insensitive, `lichess:`-qualified identity).

| Average rating interval | Requested | Accepted | Shortage |
| --- | ---: | ---: | ---: |
| 800 <= rating < 1200 | 150 | 150 | 0 |
| 1200 <= rating < 1800 | 200 | 200 | 0 |
| 1800 <= rating < 2400 | 150 | 150 | 0 |
| Total | 500 | 500 | 0 |

The recorded acquisition also applied an inclusive average-rating cap of 2399.
The CLI now defaults to 2400 so group selection includes half-integer averages
up to 2399.5. Use `--max-rating 2399` to reproduce this run's exact filters.

There are 500 unique game IDs and 1,000 unique players, each appearing once.
Average game ratings range from 802.5 to 2285, with mean 1504.318.
There are 482 rapid and 18 classical games. The most frequent controls are
600+0 (368), 600+5 (63), 900+10 (16), and 1800+0 (11); the inspection JSON
contains the complete distribution. Time class follows the
[Lichess FAQ](https://lichess.org/faq#time-controls): initial seconds plus
40 times increment; rapid is 480..1499 seconds and classical is >=1500.

Mean length is 67.058 plies; median is 61. White won 253 games, Black won 217,
and 30 were draws. UTC date/time, opening and ECO are present for all games.
97 games retain existing evaluation annotations; all 500 retain clock comments.
These annotations are source data, not newly generated labels or evaluations.

Acquisition read **2,654,208 compressed bytes** (approximately 2.53 MiB) and
processed 8,157 games. Of these, 982 passed loader filters and 7,175 were rejected:
7,036 time-control exclusions, 24 incomplete games, 61 short games, 23 unrated
games, 12 BOT games and 19 rating exclusions. Selection then skipped 470 games
in full strata and 12 repeat-player games. No complete monthly archive was saved.

## Commands and API

Run from the repository root with dependencies from `requirements.txt`:

```powershell
.venv\Scripts\python.exe -m scripts.download_lichess_game_sample --source-url https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst --max-rating 2399 --output-dir data/raw/ml/games_reproduction
.venv\Scripts\python.exe -m scripts.inspect_lichess_games data/raw/ml/games/lichess_game_sample.pgn --strict --manifest-path data/raw/ml/games/source_manifest.json
```

The output directory must not already exist. Default quotas total 500 games.
Use repeated `--group LOW:HIGH_EXCLUSIVE:COUNT` arguments to replace them, or
`--count 100 --min-rating 1000 --max-rating 1800` for one inclusive rating range.
Other controls include `--time-classes rapid classical`, `--min-plies`,
`--allow-repeat-players`, `--max-games`, `--max-bytes`, `--timeout` and
`--total-timeout`. Explicit rating bounds also filter custom groups.

```python
from src.chessmind.ml.data.game_dataset_loader import open_games, GameFilters

with open_games("data/raw/ml/games/lichess_game_sample.pgn",
                strict=False, max_records=500, max_games=10000,
                filters=GameFilters(min_plies=20)) as games:
    for game in games:
        print(game.game_id, game.average_rating, game.uci_moves)
    print(games.stats)
```

The same API supports `.pgn.zst`. `max_records` counts accepted games and
`max_games` counts all framed games. Zero limits and empty files are supported.
The frozen record preserves raw text, original archive/path and 1-based source
index. Missing optional fields are `None`. Usernames are Lichess identifiers,
never assumed to identify the same person on other platforms.

## Validation and resource behavior

The loader reuses the existing PGN move helper and python-chess parsing.
It checks explicit required headers (rather than synthesized parser defaults),
lexical movetext, matching terminal result, parser errors, standard initial
position and mainline legality. Unsupported variants, malformed or incomplete
games, invalid ratings and duplicates are rejected with stable reason codes.
A completed resignation, timeout or agreed draw does not require checkmate.
If the board is terminal, its result must agree with the header. Abandoned and
unterminated games are rejected. Only the first occurrence of an ID is eligible,
even if that occurrence fails validation.

Strict mode raises on the first rejection, including filter exclusions.
Permissive mode counts and skips rejected records. PGNs are framed using header
and blank-line boundaries outside brace comments. This targets the official
export format; malformed unclosed comments can consume subsequent text until
EOF or the resource cap. Ambiguous recovery is not claimed. Oversized games,
decoding/decompression errors and I/O errors stop processing. Original comments
are preserved; text-mode newline normalization is possible.

Defaults cap acquisition at 100,000 scanned games, 64 MiB compressed bytes,
300 seconds overall and 20 seconds per socket operation. The time budget is
checked cooperatively; a blocking socket read can overrun it by its timeout.
Each game is capped at one million characters and the decoder window at 128 MiB.
The actual archive requires an 8 MiB window. Loader memory is proportional to
scanned unique IDs plus one bounded game, rather than the archive size. The
acquisition game budget bounds ID tracking. Inspection retains per-player
counts and per-game lengths/average ratings and is intended for small samples.

Only complete, validated games are written to a staging file. The closed sample
is reparsed, hashed and checked against the manifest before the directory is
atomically renamed, publishing PGN, manifest and inspection together. A byte,
game or time budget shortage publishes a validated partial sample with explicit
shortages and CLI exit code 2. Network/decompression errors and KeyboardInterrupt
clean staging and publish nothing (exit 1/130). A forced process kill may leave
a hidden staging directory. Compression checksums for the unread remainder of
the monthly archive cannot be verified from a prefix.

## Limitations and future history acquisition

This is an ordered-prefix, quota-selected sample, **not representative of the
Lichess population**. Archive order, month, rating quotas, time-control filters,
minimum length, and repeat-player exclusions all introduce selection bias.
BOT exclusion detects declared tags only; it cannot prove that players are human.
Rating plausibility validation is not a claim that ratings are stable or accurate.

Each player appears once, so these games cannot establish recurring weaknesses
or train a recurring-weakness prediction model. Future work needs sufficiently
long chronological player histories, dates and rating trajectories, explicit
history windows, and adequate repeated observations per player. Evaluation must
separate source-qualified players across train/validation/test partitions and
respect chronological cutoffs to avoid leakage. Source engine annotations also
need explicit leakage controls before they could be used in future experiments.

## Verification

Offline synthetic fixtures cover loading, compression, multiple games, parser
failures, invalid positions/moves, headers, variants, bots, filters, duplicates,
completion, resignations/draws, limits, comments, deterministic statistics,
quota shortages, repeated players, byte limits, interruption cleanup, atomic
publication and hash/count disagreement. No tests require network access.

```powershell
.venv\Scripts\python.exe -m pytest tests/ml/test_game_dataset_loader.py -q
.venv\Scripts\python.exe -m pytest tests/ml -q
.venv\Scripts\python.exe -m pytest -q
```

Results: **42 focused tests passed; 167 ML tests passed; 409 full-suite tests
passed**. Windows sandbox permissions required elevated test and acquisition
execution. No unrelated project code or existing dataset files were changed.
