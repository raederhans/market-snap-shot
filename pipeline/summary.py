"""Aggregate the daily validation rows into the published summary. Pure functions.

A day's row for horizon h describes returns over the h sessions after it, so consecutive
rows share h-1 of those sessions and are far from independent. All daily values are
described (mean, median, share above zero), but the interval and the verdict use only
rows at least h sessions apart, and never before there are enough of them.
"""
from __future__ import annotations
import math
import statistics
from .outcomes import METRICS, VERSION

SCHEMA_VERSION = 1
CONFIDENCE = 0.95
INSUFFICIENT, SUFFICIENT = 'insufficient_sample', 'sufficient'
POSITIVE, NEGATIVE, INCONCLUSIVE = 'positive', 'negative', 'inconclusive'
# 0.975 quantile of Student's t by degrees of freedom (two-sided 95%). Source: NIST/SEMATECH e-Handbook of
# Statistical Methods, section 1.3.6.7.2, "Critical Values of the Student's t Distribution", column 0.975.
# A test integrates the t density numerically and checks every entry.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
         40: 2.021, 60: 2.000, 120: 1.980}
LIMITATIONS = [
    'Overlapping windows: the daily rows of an h-session horizon share h-1 sessions with their neighbours and are not '
    'independent. daily_mean, daily_median and daily_positive_share only describe them. The interval and the verdict use '
    'rows at least h sessions apart, starting from the first evaluated day.',
    'Stocks move together within a day, so each day is reduced to one cross-sectional number before any inference: the '
    'thousands of stocks in a day are not thousands of independent observations.',
    'The record covers only the market conditions since record_start. A verdict describes that period and says nothing about '
    'other regimes; min_non_overlapping_periods is the floor for computing an interval, not evidence of stability.',
    'A symbol without an outcome (no usable data on the evaluation day, or no bar on one of the two days) is counted in '
    'missing_pct and left out of every average, never filled. Delisted, acquired and renamed companies end up there: a '
    'takeover usually removes a gain and a failure removes a loss, so averages can be biased in either direction when '
    'missing_pct is not small.',
    'Returns run from the close of the snapshot day to the close h sessions later, in adjusted prices from Yahoo Finance '
    'through yfinance (unofficial) as downloaded on the evaluation day; excess is the simple difference from the benchmark. '
    'The snapshot is published after that close, so the entry price was not obtainable, and no cost, slippage or tax is '
    'deducted. Averages are equal-weighted; one bad provider value moves them.',
    'Each horizon reports several metrics, each with its own 95% interval; with this many intervals some exclude zero by '
    'chance alone, and every verdict is recomputed as the record grows, which gives chance more opportunities. No '
    'multiple-comparison or repeated-look adjustment is made: a verdict that appears and later disappears was noise.',
    'A descriptive record of features and rankings. Not a trading strategy, not a forecast and not investment advice.',
]


def t_critical(df):
    """0.975 quantile of Student's t. Between and beyond the tabulated values the largest tabulated
    degrees of freedom not above `df` is used: the interval is then slightly wide, never narrow."""
    if df < 1:
        raise ValueError('a t interval needs at least two values')
    return T_975[max(k for k in T_975 if k <= df)]


def non_overlapping(days, horizon, position):
    """Greedy subsample from the first day: keep a day when it is at least `horizon` sessions after the last kept one.

    position maps an ISO session date to its index in the exchange calendar, so a gap in
    the record (a day without a snapshot) cannot make two kept windows overlap.
    """
    kept = []
    for day in sorted(days):
        if not kept or position[day] - position[kept[-1]] >= horizon:
            kept.append(day)
    return kept


def interval(values):
    """Mean of independent values with its standard error and two-sided 95% t interval (None where undefined)."""
    n = len(values)
    out = {'periods': n, 'mean': None, 'standard_error': None, 'ci95_low': None, 'ci95_high': None, 't_critical': None}
    if n:
        out['mean'] = math.fsum(values) / n
    if n >= 2:
        error = statistics.stdev(values) / math.sqrt(n)   # sample standard deviation (n - 1)
        critical = t_critical(n - 1)
        out.update(standard_error=error, t_critical=critical, ci95_low=out['mean'] - critical * error,
                   ci95_high=out['mean'] + critical * error)
    return out


def verdict(estimate, min_periods):
    """insufficient_sample below min_periods, whatever the numbers say; otherwise by whether the interval excludes zero."""
    if estimate['periods'] < min_periods:
        return INSUFFICIENT
    if estimate['ci95_low'] > 0:
        return POSITIVE
    if estimate['ci95_high'] < 0:
        return NEGATIVE
    return INCONCLUSIVE


def _rounded(value):
    return None if value is None else round(value, 4)


def metric_summary(by_day, kept, min_periods):
    """One metric of one horizon. by_day: {snapshot day: value or None}; kept: the non-overlapping days."""
    values = [v for v in by_day.values() if v is not None]
    estimate = interval([by_day[day] for day in kept if by_day[day] is not None])
    return {'days': len(values),
            'daily_mean': _rounded(math.fsum(values) / len(values)) if values else None,
            'daily_median': _rounded(statistics.median(values)) if values else None,
            'daily_positive_share': _rounded(sum(v > 0 for v in values) / len(values)) if values else None,
            'non_overlapping': {k: _rounded(v) if isinstance(v, float) else v for k, v in estimate.items()},
            'verdict': verdict(estimate, min_periods)}


def horizon_summary(rows, horizon, sessions, min_periods):
    """All metrics of one horizon. rows: that horizon's typed daily rows; sessions: ascending ISO session dates."""
    by_day = {row['snapshot_day']: row for row in rows}
    days = sorted(by_day)
    position = {day: at for at, day in enumerate(sessions)}
    kept = non_overlapping(days, horizon, position)
    span = position[days[-1]] - position[days[0]] + 1 if days else 0
    return {'evaluated_days': len(days), 'non_overlapping_periods': len(kept),
            'sample_status': SUFFICIENT if len(kept) >= min_periods else INSUFFICIENT,
            'first_snapshot_day': days[0] if days else None, 'last_snapshot_day': days[-1] if days else None,
            # Sessions inside that span with no row: no snapshot was published, or validation did not run h sessions later.
            'snapshot_days_without_evaluation': span - len(days),
            'mean_missing_pct': _rounded(math.fsum(by_day[d]['missing_pct'] for d in days) / len(days)) if days else None,
            'non_overlapping_days': kept,
            'metrics': {name: metric_summary({day: by_day[day][name] for day in days}, kept, min_periods) for name in METRICS}}


def build(rows, *, sessions, settings, benchmark, record_start, evaluation_day, last_run):
    """The summary document. rows: every typed row of the daily table; last_run: what this run did per horizon."""
    return {
        'schema_version': SCHEMA_VERSION, 'validation_version': VERSION,
        'description': 'Forward record of the daily market snapshots: each published snapshot is matched with the returns '
                       'realised a fixed number of sessions later, computed once that day has closed.',
        'benchmark': benchmark, 'confidence_level': CONFIDENCE,
        'min_non_overlapping_periods': settings['min_non_overlapping_periods'],
        'record_start': record_start,                 # first published snapshot day
        'latest_evaluation_day': evaluation_day,      # the day whose closes this run used
        'latest_evaluated_snapshot_day': max((row['snapshot_day'] for row in rows), default=None),
        'snapshot_rules_versions': sorted({row['rules_version'] for row in rows}),
        'mean_missing_pct': _rounded(math.fsum(row['missing_pct'] for row in rows) / len(rows)) if rows else None,
        'settings': settings, 'last_run': last_run,
        'horizons': {str(h): horizon_summary([row for row in rows if row['horizon'] == h], h, sessions,
                                             settings['min_non_overlapping_periods']) for h in settings['horizons']},
        'limitations': LIMITATIONS,
    }
