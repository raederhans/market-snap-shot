# Market snapshots

One file per completed NYSE session: every US-listed common stock ranked by relative strength, with trend-template checks. **Features and rankings only.** Nothing here is a buy or sell signal, a price target or investment advice.

> **This repository is a generated mirror.** Everything outside `data/` is written by an export script from a private source repository; `EXPORT_MANIFEST.json` records the source commit and the SHA-256 of every exported file. Do not edit those files here: the next export overwrites them. `data/` is written only by the scheduled workflow in this repository.

## Data files

```
data/latest.json                               pointer to the newest snapshot, written last
data/snapshots/YYYY/YYYY-MM-DD.csv.gz          one row per symbol
data/snapshots/YYYY/YYYY-MM-DD.meta.json       how that file was built
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

The workflow (`.github/workflows/daily-market-snapshot.yml`) runs on weekdays at 23:30 UTC. A scheduled run never replaces a trading day that is already published, so a market-holiday run changes nothing; a manual run replaces it. When an integrity gate fails nothing is committed, and that day has no file.

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

## How a snapshot is built

1. **Universe**: Nasdaq Trader `nasdaqtraded.txt` rows with `ETF = N` and `Test Issue = N`, minus warrants, units, rights, preferreds, debt, structured products and funds identified by name rules.
2. **Prices**: about 18 months of daily bars from Yahoo Finance through yfinance, pulled in full on every run. No cache and no second source. A bar missing any field is dropped, never filled.
3. **Gates**: the run stops and publishes nothing when the benchmark (SPY) is missing, not current or too short; when too few requested symbols return usable bars; when too many symbols lack the latest session; or when too few stocks qualify for ranking.
4. **Ranking**: stocks above the price and liquidity floors form the percentile pool; the ETF watchlist is scored against that pool.

Thresholds, the ETF watchlist and the name rules are in `pipeline/config.toml`; each metadata file carries the configuration that produced it.

## Limitations

- **Unofficial data.** Bars come from Yahoo Finance through yfinance, an unofficial interface with no service guarantee that can change or throttle without notice. A snapshot is whatever it returned at run time. Check anything that matters against an official source.
- **Adjusted prices.** Bars are split- and dividend-adjusted as of the run date, so past values change after later dividends and splits. Two days' snapshots are not one consistent series, and `avg_dollar_volume_50d` is slightly understated for dividend payers.
- **Survivorship.** The universe is the symbols listed on the run date. Delisted, acquired and renamed companies are absent, so these files cannot support an unbiased backtest.
- **RS is a community approximation** (formula from Fred6725/relative-strength, Apache-2.0), not the IBD RS Rating. Percentiles depend on this universe and its price and liquidity floors.
- **Features and rankings, not signals.** The trend-template checks are heuristics from a Minervini-inspired implementation (30% above the low variant). They describe state; they are not entry or exit signals.
- **Name-based universe filter.** The symbol directory has no security-type field. Closed-end funds without "Fund" in their name and SPAC shares remain; Nasdaq's "Closed End Fund" label also removes some business development companies (for example ARCC).
- **Missing days are possible.** A failed gate or provider outage leaves no file for that session; nothing is backfilled or carried forward.

## Running it yourself

```
python -m pip install -r pipeline/requirements.txt pytest==9.2.0     # Python 3.12
python -m pytest -q                                                  # mocked, no network
python -m pipeline.run --out <dir> [--limit N]                       # real download; run from the repository root
```

`--limit N` samples N stocks for a trial; `sample_limit` in the metadata is then not null and percentiles are relative to the sample only. The calculations are in `remote-mcp/stock_research/` (`universe_rank.py`, `technical_intelligence.py`, `core.py`), kept at the path they have in the source repository; `pipeline/` fetches, checks, writes and publishes.
