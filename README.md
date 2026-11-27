# Market snapshots

One file per completed NYSE session: every US-listed common stock ranked by relative strength, with trend-template checks. **Features and rankings only.** Nothing here is a buy or sell signal, a price target or investment advice. A second set of files records what actually followed each snapshot ([Forward validation](#forward-validation)), so that the rankings can be judged on this repository's own record rather than assumed to work.

> **This repository is a generated mirror.** Everything outside `data/` is written by an export script from a private source repository; `EXPORT_MANIFEST.json` records the source commit and the SHA-256 of every exported file. Do not edit those files here: the next export overwrites them. `data/` is written only by the scheduled workflow in this repository.

## Data files

```
data/latest.json                               pointer to the newest snapshot, written last
data/snapshots/YYYY/YYYY-MM-DD.csv.gz          one row per symbol
data/snapshots/YYYY/YYYY-MM-DD.meta.json       how that file was built
data/outcomes/YYYY/YYYY-MM-DD_hNN.csv.gz       what followed that day's snapshot over NN sessions, per symbol
data/validation/daily.csv                      one row per snapshot day and horizon
data/validation/summary.json                   per horizon: sample size, averages, interval, verdict
data/backfill/daily.csv                        historical backfill (separate, survivorship-biased): one row per day, horizon and tier
data/backfill/summary.json                     historical backfill: tier -> horizon -> slice -> metric
```

- `latest.json`: `trading_day`, `generated_at` (UTC), `snapshot` and `meta` (paths relative to `data/`), `snapshot_bytes`, `snapshot_sha256`.
- `*.meta.json`: `schema_version`, trading day, run time, provider (with the yfinance version), `rules_version`, the benchmark's strength, counts per stage, symbols that returned no data or failed validation, `columns`, `excluded_columns`, the full configuration, and the file hash.
- Raw URLs need no credentials: `https://raw.githubusercontent.com/raederhans/market-snap-shot/main/data/latest.json`. GitHub caches raw files for a few minutes.

```python
import json, urllib.request
import pandas as pd

base = 'https://raw.githubusercontent.com/raederhans/market-snap-shot/main/data/'
pointer = json.load(urllib.request.urlopen(base + 'latest.json'))
table = pd.read_csv(base + pointer['snapshot'])        # gzip is detected from the file name
leaders = table[(table.status == 'ranked') & (table.rs_percentile >= 90)]
```

The workflow (`.github/workflows/daily-market-snapshot.yml`) runs on weekdays at 23:30 UTC, in this order: tests, build the snapshot, commit the snapshot, run the forward validation, commit its results. A scheduled run never replaces a trading day that is already published, so a market-holiday run leaves the snapshot alone; a manual run replaces it. When an integrity gate fails nothing is committed, and that day has no file. The validation runs only after the snapshot commit is pushed: if it fails, the run is marked failed and the snapshot is already safe.

## Columns (schema version 2)

An empty cell means unknown; `status` says why. Booleans are `true` / `false`. Windows are counted in sessions. Prices are split- and dividend-adjusted.

| Column | Definition |
| --- | --- |
| `symbol` | Nasdaq Trader directory symbol (`BRK.B`, not `BRK-B`). |
| `group` | `stock`, or `etf` for the configured watchlist. |
| `status` | `ranked`: stock in the percentile pool. `reference`: ETF scored against that pool but not part of it. Otherwise the reason the row is not ranked: `stale` (last bar is not the snapshot day), `history_gaps` (its latest bars are not exactly the benchmark's sessions), `insufficient_bars` (fewer than `min_bars`), `below_min_price`, `below_min_dollar_volume`. |
| `last_bar_date` | Date of the symbol's last bar. |
| `bars` | Number of bars inside the roughly 18-month download window (not the listing age). |
| `avg_dollar_volume_50d` | Mean of close x volume over 50 sessions, USD. |
| `atr14_pct` | Wilder ATR(14) as a percentage of the close. |
| `return_63d_pct`, `return_126d_pct`, `return_189d_pct`, `return_252d_pct` | Percentage return over the latest 63 / 126 / 189 / 252 sessions. |
| `rs_strength_pct` | `0.4*R63 + 0.2*R126 + 0.2*R189 + 0.2*R252`, in percent. |
| `rs_vs_benchmark` | `(1 + strength) / (1 + strength of SPY)`; above 1 means stronger than SPY. |
| `momentum_12_1_pct` | Return from 252 to 21 sessions ago (12-month momentum skipping the latest month). |
| `rs_percentile` | Integer 1-99: mid-rank position of `rs_strength_pct` among `ranked` stocks, `100 * (below + equal/2) / N`, rounded half up. Ties share a value. |
| `distance_from_52w_high_pct`, `distance_from_52w_low_pct` | Close versus the highest high / lowest low of the latest 252 sessions. |
| `trend_template_status` | `passes`, `fails`, `partial_rs_unverified` (no percentile) or `insufficient_history`. |
| `trend_template_passed` | Number of the eight checks below that hold. |
| `tt_above_150_and_200` | Close above the 150- and 200-session averages. |
| `tt_150_above_200` | 150-session average above the 200-session average. |
| `tt_200_rising_21_sessions` | 200-session average higher than 21 sessions ago. |
| `tt_50_above_150_and_200` | 50-session average above the 150- and 200-session averages. |
| `tt_above_50` | Close above the 50-session average. |
| `tt_at_least_30pct_above_low` | Close at least 30% above the 52-week low. |
| `tt_within_25pct_of_high` | Close within 25% of the 52-week high. |
| `tt_rs_percentile_at_least_70` | `rs_percentile` is 70 or higher. |

A `stale` or `history_gaps` row has no features at all. An `insufficient_bars` row keeps what its history covers and leaves the rest empty; nothing is computed on a shortened window.

**Price levels are not published.** The pipeline also computes `close`, `sma50`, `sma150` and `sma200`, and leaves them out of these files (`output.exclude_columns` in `pipeline/config.toml`, listed as `excluded_columns` in each metadata file). Ratios, percentages, ranks and checks derived from them are kept.

## Forward validation

A snapshot describes a day. These files record what happened afterwards, from the first published snapshot on. The record is forward-only: a row is written once the later day has closed, from the snapshot as it was published, so it involves no hindsight in choosing stocks, and symbols that later disappear are counted instead of dropped.

**It is a record, not a result.** A horizon gets a verdict only after `min_non_overlapping_periods` (12) non-overlapping periods: about 3 months for 5 sessions, 1 year for 20, 3 years for 60. Until then every verdict is `insufficient_sample`, and the averages published meanwhile must not be quoted as evidence that a ranking or a check works or fails.

On each evaluation day T, for each horizon h in `validation.horizons` (5, 20 and 60 sessions), the job takes the snapshot of D, the XNYS session h sessions before T. If no snapshot was published for D the horizon is skipped that day and `summary.json` says why (`before_record_start` or `snapshot_missing`); a neighbouring day is never used. Returns are computed from the adjusted closes downloaded on T, both ends from the same download.

### `outcomes/YYYY/D_hNN.csv.gz`

| Column | Definition |
| --- | --- |
| `symbol` | Every `ranked` and `reference` row of the snapshot of D, in its order. |
| `outcome_status` | `ok`, or why there is no outcome: `no_data_at_T` (no usable bars in the download of T: no longer in the symbol directory, nothing returned, or bars rejected), `missing_bar_at_T` (bars, but none on T), `missing_bar_at_D` (a bar on T, none on D in today's series). Nothing is filled or carried forward. |
| `forward_return_pct` | `100 * (close on T / close on D - 1)`, adjusted closes; empty unless `ok`. |
| `forward_excess_pct` | `forward_return_pct` minus the same return of SPY (a simple difference). |

### `validation/daily.csv`

One row per (D, h); running the same day again replaces its rows. Statistics use the `ranked` stocks of D only (ETFs get an outcome but are not in the pool). Every `*_excess_pct` is the equal-weight mean of `forward_excess_pct` over the stocks of that group that have an outcome, and `*_n` is their number. A group with fewer than `min_group_size` (20) outcomes is left empty, and so is a difference that needs it.

| Column | Definition |
| --- | --- |
| `snapshot_day`, `horizon`, `evaluated_on` | D, h and T. |
| `rules_version` | `rules_version` of the snapshot of D. |
| `benchmark_return_pct` | Return of SPY from D to T. |
| `pool`, `with_outcome`, `missing`, `missing_pct` | Ranked stocks on D; those with status `ok`; the rest; the rest as a percentage of the pool. |
| `pool_excess_pct` | Mean excess of the whole pool: equal-weight stocks against cap-weighted SPY, not a property of any signal. |
| `rank_ic` | Spearman rank correlation between `rs_percentile` on D and `forward_excess_pct` (average ranks for ties). |
| `rs_q1_excess_pct`, `rs_q2_excess_pct`, `rs_q3_excess_pct`, `rs_q4_excess_pct`, `rs_q5_excess_pct` | Mean excess by `rs_percentile`: below 20, 20-39, 40-59, 60-79, 80 and above (`rs_quintile_edges`). |
| `rs_q5_minus_q1_pct` | Top group minus bottom group. |
| `template_pass_n`, `template_pass_excess_pct`, `template_fail_n`, `template_fail_excess_pct`, `template_pass_minus_fail_pct` | Stocks whose `trend_template_status` was `passes` against those with `fails`, and the difference. |
| `near_high_n`, `near_high_excess_pct`, `not_near_high_n`, `not_near_high_excess_pct`, `near_high_minus_rest_pct` | Stocks within 10% of the 52-week high (`distance_from_52w_high_pct >= -10`, `near_high_within_pct`) against the rest, and the difference. |
| `combo_n`, `combo_excess_pct`, `combo_minus_pool_pct` | Stocks with `rs_percentile` of 80 or more (`combo_min_rs_percentile`) that also pass the template; their mean excess minus `pool_excess_pct`. |
| `null_reasons` | Why a value is empty, as `group=reason` joined by `;` (for example `combo=group_too_small(n=7<20)`, `rank_ic=constant_input`). |

### `validation/summary.json`

Computed from `daily.csv` alone, so it can be recomputed from that file. Top level: `schema_version`, `validation_version`, `benchmark`, `confidence_level` (0.95), `min_non_overlapping_periods`, `record_start` (first published snapshot day), `latest_evaluation_day`, `latest_evaluated_snapshot_day`, `snapshot_rules_versions`, `mean_missing_pct`, `settings` (the `[validation]` configuration), `last_run` (per horizon: the snapshot day asked for, `evaluated` or `skipped`, and the reason), `horizons` and `limitations`.

Per horizon: `evaluated_days`, `non_overlapping_periods`, `sample_status` (`insufficient_sample` or `sufficient`), `first_snapshot_day`, `last_snapshot_day`, `snapshot_days_without_evaluation` (sessions inside that span with no row), `mean_missing_pct`, `non_overlapping_days` and `metrics`. The metrics are `rank_ic`, `rs_q5_minus_q1_pct`, `template_pass_minus_fail_pct`, `near_high_minus_rest_pct`, `combo_minus_pool_pct` and `pool_excess_pct`, each with:

- `days`, `daily_mean`, `daily_median`, `daily_positive_share`: all daily values. Description only: the daily rows of an h-session horizon share h-1 sessions with their neighbours.
- `non_overlapping`: `periods`, `mean`, `standard_error`, `ci95_low`, `ci95_high`, `t_critical`, from the rows of `non_overlapping_days` only: starting at the first evaluated day, a day is kept when it is at least h sessions after the last kept one. The interval is a two-sided 95% Student-t interval on those values.
- `verdict`: `insufficient_sample` while `periods` is below the minimum, whatever the numbers look like; after that `positive` or `negative` when the interval lies entirely above or below zero, otherwise `inconclusive`.

### What the record cannot show

- **Overlap and dependence.** Only the non-overlapping rows carry inference, and each day counts once: thousands of stocks moving together on a day are one observation.
- **One market period.** The record covers the conditions since `record_start` and nothing else. Twelve periods are the floor for computing an interval, not evidence that a result holds in other regimes.
- **Missing outcomes.** Delisted, acquired and renamed companies have no outcome; a takeover usually removes a gain and a failure removes a loss. Read every average next to `missing_pct`. A session on which the job did not run leaves no row for the days it would have evaluated; nothing is backfilled.
- **Not obtainable returns.** They run from the close of D, which is before the snapshot is published, and include no cost, slippage or tax. Averages are equal-weighted; one wrong provider value moves them.
- **Many intervals, looked at every day.** Several metrics and three horizons each get a 95% interval, so some will exclude zero by chance, and each verdict is recomputed as the record grows. Nothing is adjusted for either; a verdict that appears and later disappears was noise.
- **Not advice.** A description of features and rankings, not a trading strategy or a forecast.

## Historical backfill

The forward record needs months to years before it says anything. `data/backfill/` holds a **preliminary reference** made once by hand (workflow `.github/workflows/backfill.yml`, manual start only): the same statistics reconstructed on about six years of past sessions. **It is not the forward record, it is biased, and the two must not be mixed.** `summary.json` carries `record_type: "backfill"`.

How it is built (`pipeline/backfill.py`, settings in `[backfill]` of `pipeline/config.toml`):

1. About six years of adjusted daily bars for **the stocks listed today** and SPY, in one download.
2. An evaluation day D every 5 sessions, from the first session with 253 bars of history.
3. For each D, from bars up to D only: the ranked pool (same price, dollar-volume and history floors as the daily snapshot), `rs_percentile` within that pool, the eight trend-template checks and the distance from the 52-week high. `pipeline/backfill_features.py` computes them for all stocks at once; a test holds every value to the daily job's ranking code run on bars cut off at D.
4. Excess return over SPY from the close of D to the close 5, 20 and 60 sessions later, same download. A stock without a bar on D+h is counted as missing, never filled.
5. Two tiers: `top_dollar_volume`, the 500 ranked stocks with the highest `avg_dollar_volume_50d` on D (the primary result: large, liquid stocks), and `eligible_pool`, every ranked stock. RS groups use the percentile within the whole pool in both.
6. Slices of evaluation days: `full_sample`; `benchmark_above_sma200` and `benchmark_below_sma200` (SPY's close on D above, or at or below, its 200-session average); `year_YYYY` (the year of D). Each slice takes its own non-overlapping days and gets its own interval and verdict, `insufficient_sample` below 12 periods.

The daily cross-section and the aggregation are the forward record's functions, so every metric means what it means there.

### `backfill/daily.csv`

One row per (D, h, tier). The columns of `validation/daily.csv` with the same definitions (`snapshot_day` is D, although no snapshot of D exists; `evaluated_on` is D+h; `rules_version` is that of the code that ran; `pool` is the size of the tier), plus:

| Column | Definition |
| --- | --- |
| `tier` | `top_dollar_volume` or `eligible_pool`; third column of the file. |
| `year` | Calendar year of D. |
| `benchmark_trend` | `above` or `below`: SPY's close on D against its 200-session simple average (`regime_sma_sessions`); empty when fewer closes than that exist. |

### `backfill/summary.json`

Top level: `schema_version`, `record_type` (`backfill`), `backfill_version`, `validation_version`, `benchmark`, `confidence_level`, `min_non_overlapping_periods`, `generated_at`, `provider`, `rules_version`, `universe` (directory URL and file time, stocks after the name filter, `sample_limit`), `data` (symbols requested and fetched, rejected, first and last session, download time), `parameters` (the `[backfill]`, `[eligibility]` and `[validation]` settings), `evaluation` (first and last evaluation day, their number, days by benchmark trend, days skipped because fewer than `min_ranked_stocks` stocks ranked), `primary_tier`, `tiers` and `limitations`. `tiers.<tier>.horizons.<h>.slices.<slice>` has the per-horizon fields of `validation/summary.json` (without `non_overlapping_days` and `snapshot_days_without_evaluation`: the grid skips sessions by design, and the kept days follow from `daily.csv` by the same rule).

No per-symbol prices or outcomes are published.

### What the backfill cannot show

- **Survivorship.** Only stocks listed today are in it. Companies that were large in the past and were later acquired, taken private or failed are missing, so results are too favourable: most of all absolute ones (`pool_excess_pct`, the excess of stocks passing the template), less so the spreads between groups, which are still not free of it.
- **Today's adjustment.** Prices are adjusted as of the download. Returns, average comparisons and distances from the high are unaffected, but the price and dollar-volume floors and the dollar-volume ranking use adjusted levels, not the prices of that day.
- **Not obtainable, no costs.** Entry is the close of D, the close the features use. No commission, spread, market impact or tax.
- **Not a strategy backtest.** Average excess returns of groups in a cross-section: no entry or exit rule, sizing, turnover or capacity.
- **Many tests on one history.** Two tiers, three horizons, about nine slices and six metrics, each with a 95% interval: some exclude zero by chance, and nothing is adjusted.
- **Equal-weight means** react to bad prints and extreme moves; `rank_ic` is the more robust reference.
- **Few independent episodes.** The sessions below the 200-session average are mostly one bear market. The interval treats non-overlapping periods as independent, which overstates what a slice knows; a year slice rarely reaches 12 periods.

## How a snapshot is built

1. **Universe**: Nasdaq Trader `nasdaqtraded.txt` rows with `ETF = N` and `Test Issue = N`, minus warrants, units, rights, preferreds, debt, structured products and funds identified by name rules.
2. **Prices**: about 18 months of daily bars from Yahoo Finance through yfinance, pulled in full on every run. No cache and no second source. A bar missing any field is dropped, never filled.
3. **Gates**: the run stops and publishes nothing when the benchmark (SPY) is missing, not current or too short; when too few requested symbols return usable bars; when too many symbols lack the latest session; or when too few stocks qualify for ranking.
4. **Ranking**: stocks above the price and liquidity floors form the percentile pool; the ETF watchlist is scored against that pool.

Thresholds, the ETF watchlist, the name rules and the validation settings are in `pipeline/config.toml`; each metadata file carries the configuration that produced it.

## Limitations

- **Unofficial data.** Bars come from Yahoo Finance through yfinance, an unofficial interface with no service guarantee that can change or throttle without notice. A snapshot is whatever it returned at run time. Check anything that matters against an official source.
- **Adjusted prices.** Bars are split- and dividend-adjusted as of the run date, so past values change after later dividends and splits. Two days' snapshots are not one consistent series, and `avg_dollar_volume_50d` is slightly understated for dividend payers.
- **Survivorship.** The universe is the symbols listed on the run date. Delisted, acquired and renamed companies are absent, so the snapshots alone cannot support an unbiased backtest. The forward validation counts symbols that disappear after a snapshot, but only from the first published snapshot on.
- **RS is a community approximation** (formula from Fred6725/relative-strength, Apache-2.0), not the IBD RS Rating. Percentiles depend on this universe and its price and liquidity floors.
- **Features and rankings, not signals.** The trend-template checks are heuristics from a Minervini-inspired implementation (30% above the low variant). They describe state; they are not entry or exit signals.
- **Name-based universe filter.** The symbol directory has no security-type field. Closed-end funds without "Fund" in their name and SPAC shares remain; Nasdaq's "Closed End Fund" label also removes some business development companies (for example ARCC).
- **Missing days are possible.** A failed gate or provider outage leaves no file for that session; nothing is backfilled or carried forward.

## Running it yourself

```
python -m pip install -r pipeline/requirements.txt pytest==9.2.0     # Python 3.12
python -m pytest -q                                                  # mocked, no network
python -m pipeline.run --out <dir> [--limit N] [--closes <file>]     # real download; run from the repository root
python -m pipeline.validate --data data --closes <file> --out <dir>  # needs the closes of a run and published snapshots
python -m pipeline.backfill --out <dir> [--limit N]                  # historical backfill: one large download, minutes
```

`--limit N` samples N stocks for a trial; `sample_limit` in the metadata is then not null and percentiles are relative to the sample only. The calculations are in `remote-mcp/stock_research/` (`universe_rank.py`, `technical_intelligence.py`, `core.py`), kept at the path they have in the source repository; `pipeline/` fetches, checks, writes and publishes; the validation statistics are in `pipeline/outcomes.py` and `pipeline/summary.py`.
