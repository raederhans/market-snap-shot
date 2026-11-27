"""Daily market snapshot: universe -> prices -> integrity gates -> ranking -> files.

    python -m pipeline.run --out <dir> [--limit N] [--closes <file>]

Nothing is written unless every gate passes; a failed gate exits non-zero.
--closes also saves this run's adjusted closes for pipeline.validate (a scratch file,
never published).
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import tomllib
from collections import Counter
from importlib.metadata import version
import pandas as pd
from stock_research import universe_rank
from stock_research.core import DataQualityError
from . import gates, outcomes, prices, snapshot, universe

CONFIG_PATH = pathlib.Path(__file__).with_name('config.toml')


def load_config(path=CONFIG_PATH):
    config = tomllib.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if config['benchmark'] not in config['etf_watchlist']:
        raise ValueError('benchmark must be listed in etf_watchlist')
    snapshot.published_columns(config['output']['exclude_columns'])  # a typo fails here, before any download
    outcomes.check_settings(config['validation'])
    return config


def sample(symbols, limit):
    """Every k-th symbol across the alphabet, for a local trial run."""
    step = max(1, len(symbols) // limit)
    return symbols[::step][:limit]


def run(out_dir, *, limit=None, now=None, config=None, closes_path=None):
    config = config or load_config()
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    sessions = gates.completed_sessions(now)
    trading_day, trading_day_close = sessions.index[-1], sessions.iloc[-1]

    # 1. Universe: common stocks from the directory plus the configured ETF watchlist.
    settings = config['universe']
    directory, directory_created = universe.parse_directory(
        universe.download_directory(settings['directory_url'], settings['download_attempts']))
    stocks, excluded = universe.select_common_stocks(directory, settings)
    filtered = len(stocks)
    if limit is not None:
        stocks = sample(stocks, limit)
    etfs = sorted(config['etf_watchlist'])
    if set(etfs) & set(stocks):
        raise ValueError('etf_watchlist overlaps the stock universe: ' + ', '.join(sorted(set(etfs) & set(stocks))))
    groups = {**dict.fromkeys(stocks, universe_rank.STOCK), **dict.fromkeys(etfs, universe_rank.ETF)}

    # 2. Prices: full pull ending on the trading day (end is exclusive), then validation.
    fetch_settings = {k: v for k, v in config['prices'].items() if k != 'history_calendar_days'}
    start = trading_day - pd.Timedelta(days=config['prices']['history_calendar_days'])
    raw = prices.fetch_yahoo(list(groups), start.date(), (trading_day + pd.Timedelta(days=1)).date(), **fetch_settings)
    bars, invalid = {}, {}
    for symbol in groups:
        if symbol in raw:
            try:
                bars[symbol] = prices.session_bars(raw[symbol], sessions)
            except DataQualityError as exc:
                invalid[symbol] = str(exc)
    no_data = sorted(set(groups) - set(raw))

    # 3. Gates before ranking, 4. ranking, 5. gates on the result.
    benchmark = config['benchmark']
    gates.check_benchmark(bars, benchmark, trading_day_close)
    coverage = gates.check_coverage(len(groups), len(bars), config['gates']['min_fetch_success_ratio'])
    floors = config['eligibility']
    rows = universe_rank.rank_universe(
        bars, bars[benchmark], groups, min_price=floors['min_price'],
        min_avg_dollar_volume=floors['min_avg_dollar_volume_50d'], min_bars=floors['min_bars'],
        min_population=floors['min_ranked_stocks'])
    stale_ratio = gates.check_stale(rows, config['gates']['max_stale_ratio'])
    gates.check_population(rows, floors['min_ranked_stocks'])

    by_symbol = {r['symbol']: r for r in rows}
    meta = {
        'schema_version': snapshot.SCHEMA_VERSION,
        'trading_day': trading_day.date().isoformat(),
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds'),
        'provider': f'{prices.PROVIDER}; yfinance {version("yfinance")}',
        'rules_version': universe_rank.VERSION,
        'benchmark': {'symbol': benchmark, 'rs_strength_pct': by_symbol[benchmark]['rs_strength_pct']},
        'sample_limit': limit,  # not None: a local trial on a subset, never a publishable snapshot
        'symbol_directory': {'url': settings['directory_url'], 'file_creation_time': directory_created},
        'counts': {
            'directory_rows': len(directory), 'directory_excluded': excluded, 'stocks_after_filter': filtered,
            'stocks_requested': len(stocks), 'etfs_requested': len(etfs), 'requested': len(groups),
            'fetched': len(raw), 'no_data': len(no_data), 'invalid_bars': len(invalid),
            'snapshot_rows': len(rows), 'fetch_success_ratio': round(coverage, 4), 'stale_ratio': round(stale_ratio, 4),
            'rows_by_group_and_status': {g: dict(sorted(Counter(r['status'] for r in rows if r['group'] == g).items()))
                                         for g in (universe_rank.STOCK, universe_rank.ETF)},
        },
        'failed_symbols': {'no_data': no_data, 'invalid_bars': dict(sorted(invalid.items()))},
        'config': config,
    }
    meta = snapshot.write_snapshot(out_dir, meta['trading_day'], rows, meta, config['output']['exclude_columns'])
    if closes_path is not None:
        # The snapshot above is complete and cannot be recomputed later. Saving the scratch input of
        # the validation step must not stop it from being published: a failure is reported here and
        # pipeline.validate then fails on the missing file, after the snapshot has been pushed.
        try:
            prices.write_closes(closes_path, bars)
        except Exception as exc:
            print(f'closes not saved ({type(exc).__name__}: {exc}); the validation step will fail', file=sys.stderr)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pipeline.run', description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, help='directory that receives snapshots/ and latest.json')
    parser.add_argument('--limit', type=int, help='local trial: sample N stocks instead of the full universe')
    parser.add_argument('--closes', help='also write the adjusted closes of this run here (input of pipeline.validate)')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be a positive number of stocks')
    try:
        meta = run(args.out, limit=args.limit, closes_path=args.closes)
    except gates.IntegrityError as exc:
        print(f'INTEGRITY GATE FAILED, nothing published: {exc}', file=sys.stderr)
        return 1
    counts = meta['counts']
    print(f"{meta['trading_day']}: {counts['snapshot_rows']} rows, {counts['rows_by_group_and_status']}, "
          f"{meta['files']['snapshot_bytes']} bytes -> {meta['files']['snapshot']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
