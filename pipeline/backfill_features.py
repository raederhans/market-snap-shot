"""Point-in-time features of the whole universe for one past session, vectorised.

The daily job ranks a session with stock_research.universe_rank.rank_universe. Calling it
for every evaluation day of a multi-year history takes hours (about 15 ms per symbol and
day), so the backfill computes the few values it needs for all stocks at once from price
matrices. The definitions are the production ones: tests/test_backfill_features.py holds
every value to rank_universe run on bars cut off at that session (same pool, same numbers,
same checks).

Only rows up to the evaluation session are read, so nothing after D can reach D's features.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from stock_research import universe_rank
from stock_research.universe_rank import BARS_FOR_RS, RS_WEIGHTS, TEMPLATE_CHECKS
from .gates import IntegrityError

FIELDS = ('close', 'high', 'low', 'volume')
ABOVE, BELOW = 'above', 'below'
PASSES, FAILS = 'passes', 'fails'   # technical_intelligence.template statuses of a stock that has a percentile


@dataclass(frozen=True)
class Panel:
    """Daily bars as matrices: one row per benchmark session, one column per stock, NaN where a stock has no bar."""
    sessions: list               # ISO dates, ascending
    symbols: list                # sorted, without the benchmark
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    benchmark_close: np.ndarray


def build_panel(bars, benchmark, session_closes):
    """bars: {symbol: validated OHLCV stamped with the session close (prices.session_bars)} -> Panel.

    The benchmark's bars are the rows, so it must have one on every XNYS session between its
    first and last bar; a stock is then compared with those sessions exactly as in the daily job.
    """
    index = bars[benchmark].index
    expected = session_closes[(session_closes >= index[0]) & (session_closes <= index[-1])]
    if len(index) != len(expected) or (index != pd.DatetimeIndex(expected)).any():
        raise IntegrityError(f'benchmark {benchmark} has {len(index)} bars for {len(expected)} XNYS sessions between its first and last bar')
    symbols = sorted(symbol for symbol in bars if symbol != benchmark)
    matrices = {field: np.full((len(index), len(symbols)), np.nan) for field in FIELDS}
    for column, symbol in enumerate(symbols):
        d = bars[symbol]
        rows = index.get_indexer(d.index)
        inside = rows >= 0          # a bar before the benchmark's first one is outside the panel
        for field in FIELDS:
            matrices[field][rows[inside], column] = d[field].to_numpy()[inside]
    return Panel(sessions=[stamp.date().isoformat() for stamp in expected.index], symbols=symbols,
                 benchmark_close=bars[benchmark].close.to_numpy(), **matrices)


def point_in_time(panel, at, floors, history_calendar_days):
    """The ranked pool of session number `at` as rank_universe would have produced it that evening.

    One row per ranked stock in symbol order: close, avg_dollar_volume_50d, rs_strength_pct,
    rs_percentile (within this pool), trend_template_status, distance_from_52w_high_pct and the
    eight tt_* checks. floors: [eligibility]; history_calendar_days: the daily job's download
    window, inside which it counts `bars`. The caller applies min_ranked_stocks (a smaller pool
    has no percentile in production, and the daily job publishes nothing that day).
    """
    if at < BARS_FOR_RS - 1:
        raise ValueError(f'session {at} has fewer than {BARS_FOR_RS} benchmark bars behind it')
    window = slice(at - BARS_FOR_RS + 1, at + 1)
    # Production statuses stale / history_gaps: the latest 253 bars must be exactly the benchmark's sessions.
    aligned = ~np.isnan(panel.close[window]).any(axis=0)
    dates = np.array(panel.sessions, dtype='datetime64[D]')
    first = np.searchsorted(dates, dates[at] - np.timedelta64(history_calendar_days, 'D'))
    bars = (~np.isnan(panel.close[first:at + 1])).sum(axis=0)
    columns = np.flatnonzero(aligned & (bars >= floors['min_bars']))
    # One stock per row with the sessions along the last axis: numpy then adds a window in the same order as
    # pandas does for one stock's series, so the averages below equal the production values to the last bit.
    close, high, low, volume = (np.ascontiguousarray(getattr(panel, field)[window][:, columns].T) for field in FIELDS)
    dollar_volume = (close[:, -50:] * volume[:, -50:]).mean(axis=1)
    ranked = (close[:, -1] >= floors['min_price']) & (dollar_volume >= floors['min_avg_dollar_volume_50d'])
    close, high, low, dollar_volume = close[ranked], high[ranked], low[ranked], dollar_volume[ranked]
    last = close[:, -1]

    # universe_rank.rs_strength adds the four weighted returns with Python's built-in sum, which is compensated
    # for plain floats; the same expression on plain floats per stock gives the same last bit.
    returns = [(last / close[:, -n - 1] - 1).tolist() for n, _ in RS_WEIGHTS]
    strength = np.array([100 * sum(weight * part for (_, weight), part in zip(RS_WEIGHTS, parts)) for parts in zip(*returns)], dtype=float)
    population = np.sort(strength)
    percentile = np.array([universe_rank.percentile(value, population) for value in strength], dtype=int)

    # technical_intelligence.template, check by check in its order (TEMPLATE_CHECKS).
    ma = {n: close[:, -n:].mean(axis=1) for n in (50, 150, 200)}
    earlier200 = close[:, -221:-21].mean(axis=1)
    top, bottom = high[:, -252:].max(axis=1), low[:, -252:].min(axis=1)
    checks = dict(zip(TEMPLATE_CHECKS, (
        (last > ma[150]) & (last > ma[200]), ma[150] > ma[200], ma[200] > earlier200, ma[50] > np.maximum(ma[150], ma[200]),
        last > ma[50], last >= 1.3 * bottom, last >= .75 * top, percentile >= 70)))
    passes = np.logical_and.reduce(list(checks.values())) if len(last) else np.zeros(0, dtype=bool)
    return pd.DataFrame({
        'symbol': [panel.symbols[column] for column in columns[ranked]], 'close': last,
        'avg_dollar_volume_50d': dollar_volume, 'rs_strength_pct': strength, 'rs_percentile': percentile,
        'trend_template_status': np.where(passes, PASSES, FAILS), 'distance_from_52w_high_pct': 100 * (last / top - 1),
        **{'tt_' + name: value for name, value in checks.items()}})


def top_dollar_volume(pool, count):
    """The `count` ranked stocks with the highest 50-session average dollar volume (ties: symbol order); the whole pool when it is smaller."""
    return pool.sort_values('avg_dollar_volume_50d', ascending=False, kind='stable').head(count)


def benchmark_trend(panel, at, sma_sessions):
    """'above' when the benchmark closed above its simple average of the latest sma_sessions closes, 'below'
    when at or under it, None when the panel holds fewer closes than that by session `at` (no shorter average)."""
    if at + 1 < sma_sessions:
        return None
    window = panel.benchmark_close[at - sma_sessions + 1:at + 1]
    return ABOVE if window[-1] > window.mean() else BELOW
