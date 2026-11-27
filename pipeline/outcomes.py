"""Forward outcomes of one published snapshot and their cross-sectional statistics.

Pure functions: no files, no network. pipeline.validate reads and writes; summary.py
aggregates the daily rows produced here.

A snapshot of day D is matched with what happened by day T, `horizon` XNYS sessions
later, using the adjusted closes downloaded on T. Nothing is filled: a symbol without
both closes gets a status and no number, and stays out of every average.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

VERSION = '1.0.0'   # of the definitions in this module and in summary.py
# Outcome status of one symbol.
OK = 'ok'
NO_DATA = 'no_data_at_T'          # no usable bars in today's download: gone from the directory, nothing returned, or rejected
MISSING_T = 'missing_bar_at_T'    # downloaded, but no bar on the evaluation day
MISSING_D = 'missing_bar_at_D'    # a bar today, none on the snapshot day in today's series
OUTCOME_COLUMNS = ('symbol', 'outcome_status', 'forward_return_pct', 'forward_excess_pct')
QUINTILES = 5
# One row of data/validation/daily.csv per (snapshot day, horizon). `*_excess_pct` is an equal-weight mean of
# forward_excess_pct over the ranked stocks of the group that have an outcome; `*_n` is how many those are.
STAT_COLUMNS = (
    'pool', 'with_outcome', 'missing', 'missing_pct', 'pool_excess_pct', 'rank_ic',
    *(f'rs_q{q}_excess_pct' for q in range(1, QUINTILES + 1)), 'rs_q5_minus_q1_pct',
    'template_pass_n', 'template_pass_excess_pct', 'template_fail_n', 'template_fail_excess_pct', 'template_pass_minus_fail_pct',
    'near_high_n', 'near_high_excess_pct', 'not_near_high_n', 'not_near_high_excess_pct', 'near_high_minus_rest_pct',
    'combo_n', 'combo_excess_pct', 'combo_minus_pool_pct')
DAILY_COLUMNS = ('snapshot_day', 'horizon', 'evaluated_on', 'rules_version', 'benchmark_return_pct', *STAT_COLUMNS, 'null_reasons')
# The daily values that summary.py aggregates and gives a verdict for.
METRICS = ('rank_ic', 'rs_q5_minus_q1_pct', 'template_pass_minus_fail_pct', 'near_high_minus_rest_pct',
           'combo_minus_pool_pct', 'pool_excess_pct')


class ValidationError(RuntimeError):
    """The inputs cannot support a trustworthy reconciliation; nothing is written."""


def check_settings(settings):
    """Reject a [validation] configuration these statistics cannot honour (called when the configuration is loaded)."""
    horizons, edges = settings['horizons'], settings['rs_quintile_edges']
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError('validation.horizons must be distinct positive session counts')
    if len(edges) != QUINTILES - 1 or list(edges) != sorted(set(edges)):
        raise ValueError(f'validation.rs_quintile_edges must be {QUINTILES - 1} ascending percentiles')
    if settings['min_group_size'] < 2 or settings['min_non_overlapping_periods'] < 2:
        raise ValueError('validation.min_group_size and min_non_overlapping_periods must be at least 2')


def session_before(today, horizon, sessions):
    """The session `horizon` sessions before `today`. sessions: ascending ISO dates of the exchange calendar."""
    if today not in sessions:
        raise ValidationError(f'{today} is not an exchange session')
    at = sessions.index(today)
    if at < horizon:
        raise ValidationError(f'the calendar does not reach {horizon} sessions before {today}')
    return sessions[at - horizon]


def forward_outcomes(symbols, closes, day, today, benchmark):
    """Close-to-close return from `day` to `today` for each symbol, and its excess over the benchmark.

    closes: today's adjusted closes, indexed by ISO session date, one column per symbol with usable
    bars, NaN where a symbol has no bar. Both series come from the same download, so the return
    includes dividends. Returns (rows with OUTCOME_COLUMNS, in the order of `symbols`; benchmark
    return in percent). Excess is the simple difference of the two percentages.
    """
    for date in (day, today):
        if date not in closes.index:
            raise ValidationError(f'today\'s prices have no row for session {date}')
    start, end = closes.loc[day], closes.loc[today]
    if benchmark not in closes.columns or not (np.isfinite(start[benchmark]) and np.isfinite(end[benchmark])):
        raise ValidationError(f'benchmark {benchmark} lacks a close on {day} or {today}')
    base = 100 * (float(end[benchmark]) / float(start[benchmark]) - 1)
    rows = []
    for symbol in symbols:
        if symbol not in closes.columns:
            status = NO_DATA
        elif not np.isfinite(end[symbol]):
            status = MISSING_T
        elif not np.isfinite(start[symbol]):
            status = MISSING_D
        else:
            status = OK
        value = 100 * (float(end[symbol]) / float(start[symbol]) - 1) if status == OK else None
        rows.append({'symbol': symbol, 'outcome_status': status, 'forward_return_pct': value,
                     'forward_excess_pct': None if value is None else value - base})
    return rows, base


def spearman(x, y):
    """Spearman rank correlation: Pearson correlation of average ranks, so ties share a rank.

    None when either side is constant (the correlation is undefined, not zero).
    """
    a, b = (pd.Series(v, dtype=float).rank(method='average').to_numpy() for v in (x, y))
    a, b = a - a.mean(), b - b.mean()
    scale = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / scale) if scale > 0 else None


def cross_section(pool, settings):
    """Statistics of one snapshot day's ranked stocks. Returns (values keyed by STAT_COLUMNS, null reasons).

    pool: one row per ranked stock with the snapshot's rs_percentile, trend_template_status and
    distance_from_52w_high_pct, and that stock's outcome_status and forward_excess_pct.
    A group, or the rank-IC sample, with fewer than min_group_size outcomes is None and its name is
    in the reasons; a spread is None when either of its two groups is. An unknown feature puts a
    stock in neither group of that comparison.
    """
    if not len(pool):
        raise ValidationError('the snapshot has no ranked stocks')
    known = pool[pool.outcome_status == OK]
    excess = known.forward_excess_pct.to_numpy(dtype=float)
    rs = known.rs_percentile.to_numpy(dtype=float)
    distance = known.distance_from_52w_high_pct.to_numpy(dtype=float)
    template = known.trend_template_status.to_numpy()
    minimum, reasons = settings['min_group_size'], {}
    out = {'pool': len(pool), 'with_outcome': len(known), 'missing': len(pool) - len(known),
           'missing_pct': 100 * (len(pool) - len(known)) / len(pool)}

    def group(name, mask):
        """(mean excess or None, count) of the stocks selected by mask."""
        n = int(mask.sum())
        if n < minimum:
            reasons[name] = f'group_too_small(n={n}<{minimum})'   # no comma: the daily table stays plain CSV
            return None, n
        return float(excess[mask].mean()), n

    def spread(high, low):
        return None if high is None or low is None else high - low

    everything = np.ones(len(known), dtype=bool)
    out['pool_excess_pct'], _ = group('pool', everything)
    if len(known) < minimum:
        out['rank_ic'], reasons['rank_ic'] = None, reasons['pool']
    else:
        out['rank_ic'] = spearman(rs, excess)
        if out['rank_ic'] is None:
            reasons['rank_ic'] = 'constant_input'

    # Quintile k holds edges[k-2] <= rs_percentile < edges[k-1]; the top one is at or above the last edge.
    quintile = np.searchsorted(settings['rs_quintile_edges'], rs, side='right') + 1
    for q in range(1, QUINTILES + 1):
        out[f'rs_q{q}_excess_pct'], _ = group(f'rs_q{q}', quintile == q)
    out['rs_q5_minus_q1_pct'] = spread(out['rs_q5_excess_pct'], out['rs_q1_excess_pct'])

    passes = template == 'passes'
    out['template_pass_excess_pct'], out['template_pass_n'] = group('template_pass', passes)
    out['template_fail_excess_pct'], out['template_fail_n'] = group('template_fail', template == 'fails')
    out['template_pass_minus_fail_pct'] = spread(out['template_pass_excess_pct'], out['template_fail_excess_pct'])

    near = distance >= -settings['near_high_within_pct']   # distances are negative below the high; NaN is in neither group
    out['near_high_excess_pct'], out['near_high_n'] = group('near_high', near)
    out['not_near_high_excess_pct'], out['not_near_high_n'] = group('not_near_high', distance < -settings['near_high_within_pct'])
    out['near_high_minus_rest_pct'] = spread(out['near_high_excess_pct'], out['not_near_high_excess_pct'])

    out['combo_excess_pct'], out['combo_n'] = group('combo', passes & (rs >= settings['combo_min_rs_percentile']))
    out['combo_minus_pool_pct'] = spread(out['combo_excess_pct'], out['pool_excess_pct'])
    return out, reasons
