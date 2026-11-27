"""Integrity gates. Any failure aborts the run before a single file is written."""
from __future__ import annotations
import exchange_calendars as xcals
import pandas as pd
from stock_research.universe_rank import BARS_FOR_RS, RANKED


class IntegrityError(RuntimeError):
    """The data cannot support a trustworthy snapshot; publish nothing."""


def completed_sessions(now):
    """XNYS session date -> close time (UTC) for every session already closed at `now`."""
    closes = xcals.get_calendar('XNYS').closes
    return closes[closes <= pd.Timestamp(now)]


def check_benchmark(bars, symbol, trading_day_close):
    """The benchmark defines the snapshot date, so it must be present, current and long enough."""
    d = bars.get(symbol)
    if d is None:
        raise IntegrityError(f'benchmark {symbol} was not fetched or failed validation')
    if d.index[-1] != trading_day_close:
        raise IntegrityError(f'benchmark {symbol} last bar {d.index[-1].date()} is not the latest completed '
                             f'XNYS session {trading_day_close.date()}')
    if len(d) < BARS_FOR_RS:
        raise IntegrityError(f'benchmark {symbol} has {len(d)} bars, needs {BARS_FOR_RS}')


def check_coverage(requested, usable, min_ratio):
    ratio = usable / requested if requested else 0.0
    if ratio < min_ratio:
        raise IntegrityError(f'only {usable} of {requested} requested symbols returned usable bars '
                             f'({ratio:.1%} < {min_ratio:.1%})')
    return ratio


def check_stale(rows, max_ratio):
    """Many symbols lagging the benchmark means the provider had not finished the session."""
    stale = sum(r['status'] == 'stale' for r in rows)
    ratio = stale / len(rows) if rows else 1.0
    if ratio > max_ratio:
        raise IntegrityError(f'{stale} of {len(rows)} symbols lack the latest session ({ratio:.1%} > {max_ratio:.1%})')
    return ratio


def check_population(rows, min_ranked):
    ranked = sum(r['status'] == RANKED for r in rows)
    if ranked < min_ranked:
        raise IntegrityError(f'only {ranked} ranked stocks; a 1-99 percentile needs at least {min_ranked}')
    return ranked
