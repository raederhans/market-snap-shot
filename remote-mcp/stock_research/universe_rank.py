"""Cross-sectional relative-strength ranking for one completed session.

Pure functions: no network, no storage, no buy/sell signal. The caller supplies
validated daily OHLCV (see core.require_ohlcv) for every symbol and the benchmark.

RS strength is the community approximation used by Fred6725/relative-strength
(https://github.com/Fred6725/relative-strength, Apache-2.0):
    strength = 0.4*R63 + 0.2*R126 + 0.2*R189 + 0.2*R252
    rs_vs_benchmark = (1 + strength_symbol) / (1 + strength_benchmark)
where R_n is the return over the latest n sessions. It is NOT the proprietary
IBD RS Rating. Deliberate differences from that repository's code: R_n spans
exactly n sessions (the repository takes the last n prices, so n-1 sessions); a
symbol with too few bars gets no value (the repository shortens the window); and
percentiles are mid-rank instead of pd.qcut(duplicates='drop'), whose bin count
silently shrinks when values tie.
"""
from __future__ import annotations
import math
import numpy as np
from .core import DataQualityError, finite_or_none
from . import technical_intelligence as ti

VERSION = '1.0.0'
RS_WEIGHTS = ((63, .4), (126, .2), (189, .2), (252, .2))
BARS_FOR_RS = 253  # R252 needs the close 252 sessions before the latest one
STOCK, ETF = 'stock', 'etf'
# Row status. Only 'ranked' rows form the percentile population; 'reference' rows
# (ETFs) are scored against that population without joining it.
RANKED, REFERENCE = 'ranked', 'reference'
FEATURES = ('close', 'avg_dollar_volume_50d', 'atr14_pct', 'return_63d_pct', 'return_126d_pct',
            'return_189d_pct', 'return_252d_pct', 'rs_strength_pct', 'rs_vs_benchmark', 'momentum_12_1_pct')
TEMPLATE_FIELDS = ('sma50', 'sma150', 'sma200', 'distance_from_52w_high_pct', 'distance_from_52w_low_pct',
                   'trend_template_status', 'trend_template_passed')
# The eight checks of technical_intelligence.template, in its order (a test keeps them in sync).
TEMPLATE_CHECKS = ('above_150_and_200', '150_above_200', '200_rising_21_sessions', '50_above_150_and_200',
                   'above_50', 'at_least_30pct_above_low', 'within_25pct_of_high', 'rs_percentile_at_least_70')
COLUMNS = (('symbol', 'group', 'status', 'last_bar_date', 'bars') + FEATURES + ('rs_percentile',)
           + TEMPLATE_FIELDS + tuple('tt_' + name for name in TEMPLATE_CHECKS))


def trailing_return(close, n):
    """Return over the latest n sessions, or None: never a shorter window."""
    if len(close) <= n:
        return None
    return finite_or_none(close.iloc[-1] / close.iloc[-n - 1] - 1)


def rs_strength(close):
    parts = [trailing_return(close, n) for n, _ in RS_WEIGHTS]
    if any(p is None for p in parts):
        return None
    return sum(w * p for (_, w), p in zip(RS_WEIGHTS, parts))


def momentum_12_1(close):
    """Return from 252 to 21 sessions ago: 12-month momentum skipping the latest month."""
    if len(close) < BARS_FOR_RS:
        return None
    return finite_or_none(close.iloc[-22] / close.iloc[-BARS_FOR_RS] - 1)


def percentile(value, population):
    """Mid-rank position of value in an ascending population, as an integer 1..99.

    p = 100 * (count below + half the count equal) / N, rounded half up and
    clamped to 1..99. Equal values therefore always share one percentile and N
    never changes the scale. A member of the population counts itself as equal.
    """
    below = np.searchsorted(population, value, 'left')
    equal = np.searchsorted(population, value, 'right') - below
    p = 100 * (below + .5 * equal) / len(population)
    return int(min(99, max(1, math.floor(p + .5))))


def symbol_features(d, benchmark, benchmark_strength):
    """Features of one symbol as of the benchmark's latest session.

    Returns (features, problem). problem is 'stale' when the last bar is not the
    benchmark's, 'history_gaps' when the trailing bars are not exactly the
    benchmark's sessions (an n-bar return would then span more than n sessions);
    both leave every feature None. A short history only blanks what it cannot cover.
    """
    out = dict.fromkeys(FEATURES)
    if d.index[-1] != benchmark.index[-1]:
        return out, 'stale'
    k = min(len(d), BARS_FOR_RS)
    if not d.index[-k:].equals(benchmark.index[-k:]):
        return out, 'history_gaps'
    c = d.close
    out['close'] = float(c.iloc[-1])
    if len(d) >= 50:
        out['avg_dollar_volume_50d'] = float((c * d.volume).iloc[-50:].mean())
    out['atr14_pct'] = finite_or_none(ti.feature_series(d).atr_pct.iloc[-1])  # existing Wilder ATR
    for n, _ in RS_WEIGHTS:
        r = trailing_return(c, n)
        out[f'return_{n}d_pct'] = None if r is None else 100 * r
    strength = rs_strength(c)
    if strength is not None:
        out['rs_strength_pct'] = 100 * strength
        out['rs_vs_benchmark'] = (1 + strength) / (1 + benchmark_strength)
    m = momentum_12_1(c)
    out['momentum_12_1_pct'] = None if m is None else 100 * m
    return out, None


def _status(group, problem, f, bars, min_price, min_avg_dollar_volume, min_bars):
    if problem:
        return problem
    if bars < min_bars:
        return 'insufficient_bars'
    if group == ETF:
        return REFERENCE  # liquidity floors define the stock pool only
    if f['close'] < min_price:
        return 'below_min_price'
    if f['avg_dollar_volume_50d'] < min_avg_dollar_volume:
        return 'below_min_dollar_volume'
    return RANKED


def _template_fields(d, rs_percentile):
    """Existing Minervini-inspired template, now fed a real cross-sectional percentile."""
    t = ti.template(d, rs_percentile=rs_percentile)  # 'insufficient_history' below 252 bars
    out = {f'sma{n}': t.get('ma', {}).get(str(n)) for n in (50, 150, 200)}
    for name in ('distance_from_52w_high_pct', 'distance_from_52w_low_pct'):
        out[name] = t.get(name)
    out.update(trend_template_status=t['status'], trend_template_passed=t.get('passed'))
    out.update({'tt_' + name: t['checks'].get(name) for name in TEMPLATE_CHECKS})
    return out


def rank_universe(bars, benchmark, groups, *, min_price, min_avg_dollar_volume, min_bars, min_population):
    """Rank {symbol: OHLCV} against the benchmark; one row (COLUMNS) per symbol, sorted by symbol.

    groups maps each symbol to 'stock' or 'etf'. Stocks passing the floors are
    'ranked' and receive a percentile within that pool; ETFs are 'reference' and
    are placed on the same stock distribution. Every other status has no percentile.
    """
    if min_bars < BARS_FOR_RS:
        raise ValueError(f'min_bars must be at least {BARS_FOR_RS} so every ranked symbol has all four RS windows')
    benchmark_strength = rs_strength(benchmark.close)
    if benchmark_strength is None:
        raise DataQualityError(f'benchmark needs at least {BARS_FOR_RS} bars')
    rows = []
    for symbol in sorted(bars):
        d = bars[symbol]
        f, problem = symbol_features(d, benchmark, benchmark_strength)
        status = _status(groups[symbol], problem, f, len(d), min_price, min_avg_dollar_volume, min_bars)
        rows.append({'symbol': symbol, 'group': groups[symbol], 'status': status,
                     'last_bar_date': d.index[-1].tz_convert('America/New_York').date().isoformat(),
                     'bars': len(d), **f, 'rs_percentile': None})
    population = np.sort([r['rs_strength_pct'] for r in rows if r['status'] == RANKED])
    # Fewer names than the scale has steps: leave every percentile unknown rather than stretch it.
    scored = len(population) >= max(1, min_population)
    for row in rows:
        if scored and row['status'] in (RANKED, REFERENCE):
            row['rs_percentile'] = percentile(row['rs_strength_pct'], population)
        if row['close'] is None:  # stale or gapped: nothing is known as of this session
            row.update(dict.fromkeys(COLUMNS[COLUMNS.index('sma50'):]))
        else:
            row.update(_template_fields(bars[row['symbol']], row['rs_percentile']))
    return rows
