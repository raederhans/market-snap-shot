"""Historical backfill of the validation statistics: a one-off, manually started run.

    python -m pipeline.backfill --out <dir> [--limit N]

The forward record (pipeline.validate) needs months before it can say anything. This
reconstructs the same statistics on past sessions as a preliminary reference: several
years of adjusted bars for today's stock universe, an evaluation day D every
`step_sessions` sessions, the point-in-time ranked pool of D (backfill_features.py, held
to the production ranking by a test), and the excess return from D to D+h. The daily
cross-section and the aggregation are the forward record's own functions
(outcomes.cross_section, summary.horizon_summary). Written under <out>:

    backfill/daily.csv      one row per (D, horizon, tier): the forward table's columns plus tier, year, benchmark_trend
    backfill/summary.json   record_type "backfill": tier -> horizon -> slice -> metric, with parameters and limitations

Nothing per symbol is written, and nothing under snapshots/, outcomes/ or validation/.
The universe is today's listed stocks, so the result is survivorship-biased; see
backfill_summary.LIMITATIONS. Not part of the daily job.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import pathlib
import sys
import time
from importlib.metadata import version
import pandas as pd
from stock_research import universe_rank
from stock_research.core import DataQualityError
from . import backfill_features as features, backfill_summary, gates, outcomes, prices, universe, validate
from .backfill_summary import TIER_POOL, TIER_TOP
from .run import load_config, sample

LABELS = ('tier', 'year', 'benchmark_trend')
DAILY_COLUMNS = (*outcomes.DAILY_COLUMNS[:2], 'tier', *outcomes.DAILY_COLUMNS[2:], 'year', 'benchmark_trend')


def check_settings(settings):
    if any(type(settings[name]) is not int or settings[name] < 1
           for name in ('history_calendar_days', 'step_sessions', 'top_dollar_volume', 'regime_sma_sessions')):
        raise ValueError('every [backfill] setting must be a positive integer')


def load_bars(config, now, limit):
    """Today's stock universe and the benchmark -> validated bars. Returns (bars, completed sessions, provenance)."""
    sessions = gates.completed_sessions(now)
    settings, benchmark = config['universe'], config['benchmark']
    directory, created = universe.parse_directory(universe.download_directory(settings['directory_url'], settings['download_attempts']))
    stocks, _ = universe.select_common_stocks(directory, settings)
    filtered = len(stocks)
    if benchmark in stocks:
        raise ValueError(f'benchmark {benchmark} is in the stock universe')
    if limit is not None:
        stocks = sample(stocks, limit)
    symbols = [*stocks, benchmark]
    last = sessions.index[-1]
    start = last - pd.Timedelta(days=config['backfill']['history_calendar_days'])
    fetch_settings = {k: v for k, v in config['prices'].items() if k != 'history_calendar_days'}
    started = time.monotonic()
    raw = prices.fetch_yahoo(symbols, start.date(), (last + pd.Timedelta(days=1)).date(), **fetch_settings)
    seconds = time.monotonic() - started
    bars, invalid = {}, 0
    for symbol in symbols:
        if symbol in raw:
            try:
                bars[symbol] = prices.session_bars(raw[symbol], sessions)
            except DataQualityError:   # one rejected bar anywhere in the history removes the symbol, as in the daily job
                invalid += 1
    gates.check_benchmark(bars, benchmark, sessions.iloc[-1])
    coverage = gates.check_coverage(len(symbols), len(bars), config['gates']['min_fetch_success_ratio'])
    facts = {
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds'),
        'provider': f'{prices.PROVIDER}; yfinance {version("yfinance")}', 'rules_version': universe_rank.VERSION,
        'universe': {'source': settings['directory_url'], 'file_creation_time': created, 'stocks_after_filter': filtered,
                     'sample_limit': limit,   # not None: a trial on a subset
                     'note': 'Stocks listed on the run date only; delisted, acquired and renamed companies are absent.'},
        'data': {'requested': len(symbols), 'fetched': len(raw), 'no_data': len(symbols) - len(raw), 'invalid_bars': invalid,
                 'fetch_success_ratio': round(coverage, 4), 'download_seconds': round(seconds, 1)}}
    return bars, sessions, facts


def evaluate(panel, config):
    """Daily rows as text for every (evaluation day, horizon, tier), and the days skipped for too small a pool."""
    settings, floors, benchmark = config['backfill'], config['eligibility'], config['benchmark']
    closes = pd.DataFrame(panel.close, index=panel.sessions, columns=panel.symbols)
    closes = pd.concat([closes, pd.Series(panel.benchmark_close, index=panel.sessions, name=benchmark)], axis=1)
    rows, skipped, last = [], [], len(panel.sessions) - 1
    for at in range(floors['min_bars'] - 1, last + 1, settings['step_sessions']):
        day = panel.sessions[at]
        pool = features.point_in_time(panel, at, floors, config['prices']['history_calendar_days'])
        if len(pool) < floors['min_ranked_stocks']:   # the daily job publishes no snapshot for such a session
            skipped.append({'day': day, 'ranked_stocks': len(pool)})
            continue
        tiers = {TIER_TOP: set(features.top_dollar_volume(pool, settings['top_dollar_volume']).symbol), TIER_POOL: set(pool.symbol)}
        labels = {'year': day[:4], 'benchmark_trend': features.benchmark_trend(panel, at, settings['regime_sma_sessions']) or ''}
        for horizon in config['validation']['horizons']:
            if at + horizon > last:   # D+h has not happened yet
                continue
            today = panel.sessions[at + horizon]
            results, benchmark_return = outcomes.forward_outcomes(list(pool.symbol), closes, day, today, benchmark)
            scored = pool.merge(pd.DataFrame(results, columns=outcomes.OUTCOME_COLUMNS), on='symbol')
            for tier, members in tiers.items():
                stats, reasons = outcomes.cross_section(scored[scored.symbol.isin(members)], config['validation'])
                row = validate.daily_row(day, horizon, today, universe_rank.VERSION, benchmark_return, stats, reasons)
                rows.append({**row, 'tier': tier, **labels})
    return sorted(rows, key=lambda row: (row['snapshot_day'], int(row['horizon']), row['tier'])), skipped


def typed(row):
    """A daily row as written -> numbers; the label columns stay text ('' = unknown trend)."""
    return {**validate.typed({k: v for k, v in row.items() if k not in LABELS}), **{k: row[k] for k in LABELS}}


def run(out_dir, *, limit=None, now=None, config=None):
    """Download, evaluate and write the two files. Returns the summary."""
    config = config or load_config()
    check_settings(config['backfill'])
    bars, sessions, facts = load_bars(config, pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now), limit)
    panel = features.build_panel(bars, config['benchmark'], sessions)
    rows, skipped = evaluate(panel, config)
    if not rows:
        raise gates.IntegrityError('no evaluation day has a ranked pool and an outcome: history too short or sample too small')
    facts['data'].update(first_session=panel.sessions[0], last_session=panel.sessions[-1], sessions=len(panel.sessions),
                         stocks_with_bars=len(panel.symbols))
    table = io.StringIO()
    writer = csv.DictWriter(table, DAILY_COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    document = backfill_summary.build([typed(row) for row in rows], sessions=panel.sessions, config=config, facts=facts)
    document['evaluation']['days_skipped_for_small_pool'] = skipped
    out = pathlib.Path(out_dir) / 'backfill'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'daily.csv').write_text(table.getvalue(), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pipeline.backfill', description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, help='directory that receives backfill/daily.csv and backfill/summary.json')
    parser.add_argument('--limit', type=int, help='trial: sample N stocks instead of the full universe')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be a positive number of stocks')
    try:
        document = run(args.out, limit=args.limit)
    except gates.IntegrityError as exc:
        print(f'BACKFILL FAILED, nothing written: {exc}', file=sys.stderr)
        return 1
    data, evaluation = document['data'], document['evaluation']
    print(f"{data['first_session']}..{data['last_session']}: {data['stocks_with_bars']} stocks, {evaluation['days']} evaluation days "
          f"from {evaluation['first_day']}, download {data['download_seconds']} s")
    for horizon, entry in document['tiers'][TIER_TOP]['horizons'].items():
        full = entry['slices'][backfill_summary.FULL_SAMPLE]
        print(f"h={horizon}: {full['evaluated_days']} days, {full['non_overlapping_periods']} non-overlapping, {full['sample_status']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
