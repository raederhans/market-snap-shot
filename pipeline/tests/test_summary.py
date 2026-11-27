"""Aggregation of the daily rows: non-overlapping sampling, t interval, verdicts, summary structure. No files, no network."""
import json
import math
import statistics
import exchange_calendars as xcals
import pytest
from pipeline import outcomes, summary
from pipeline.run import load_config

SESSIONS = [stamp.date().isoformat() for stamp in xcals.get_calendar('XNYS').sessions]
POSITION = {day: at for at, day in enumerate(SESSIONS)}
START = SESSIONS.index('2026-11-25')
SETTINGS = load_config()['validation']


def days(count, start=START):
    return SESSIONS[start:start + count]


def t_density(x, df):
    return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi) * (1 + x * x / df) ** (-(df + 1) / 2)


def central_probability(t, df, steps=20_000):
    """P(-t <= T <= t) for Student's t by Simpson's rule on the density: independent of the table."""
    width = t / steps
    inner = sum((4 if k % 2 else 2) * t_density(k * width, df) for k in range(1, steps))
    return 2 * width / 3 * (t_density(0, df) + inner + t_density(t, df))


def test_every_tabulated_t_value_covers_95_percent():
    assert sorted(summary.T_975) == [*range(1, 31), 40, 60, 120]
    for df, value in summary.T_975.items():
        assert central_probability(value, df) == pytest.approx(summary.CONFIDENCE, abs=1e-4), df
    assert central_probability(summary.T_975[11], 12) > 0.95 > central_probability(summary.T_975[12], 11)   # the test can tell rows apart


def test_t_lookup_is_exact_inside_the_table_and_conservative_outside():
    assert summary.t_critical(1) == 12.706 and summary.t_critical(11) == 2.201 and summary.t_critical(30) == 2.042
    assert summary.t_critical(35) == 2.042 and summary.t_critical(40) == 2.021 and summary.t_critical(59) == 2.021
    assert summary.t_critical(119) == 2.000 and summary.t_critical(5000) == 1.980   # never below the normal 1.960
    values = [summary.t_critical(df) for df in range(1, 200)]
    assert values == sorted(values, reverse=True)
    for df in (35, 59, 119, 5000):                                # a smaller df than the true one: wider, never narrower
        assert central_probability(summary.t_critical(df), df) >= 0.95
    with pytest.raises(ValueError):
        summary.t_critical(0)


def test_non_overlapping_days_are_at_least_one_horizon_apart():
    twelve = days(12)
    assert summary.non_overlapping(twelve, 5, POSITION) == [twelve[0], twelve[5], twelve[10]]
    assert summary.non_overlapping(twelve, 1, POSITION) == twelve
    assert summary.non_overlapping(twelve, 20, POSITION) == [twelve[0]] and summary.non_overlapping([], 5, POSITION) == []
    assert summary.non_overlapping(list(reversed(twelve)), 5, POSITION) == [twelve[0], twelve[5], twelve[10]]   # order of input is irrelevant
    # A gap in the record: the next kept day is the first one at least 5 sessions later, not "five rows" later.
    gapped = [twelve[i] for i in (0, 1, 2, 6, 7, 11)]
    assert summary.non_overlapping(gapped, 5, POSITION) == [twelve[0], twelve[6], twelve[11]]
    # Distance is counted in sessions: 25 November to 3 December spans a holiday and a weekend but is exactly 5 sessions.
    assert summary.non_overlapping(['2026-11-25', '2026-12-02', '2026-12-03'], 5, POSITION) == ['2026-11-25', '2026-12-03']


def test_interval_by_hand():
    out = summary.interval([1.0, 2.0, 3.0, 4.0])
    # mean 2.5; squared deviations 2.25 + .25 + .25 + 2.25 = 5 -> sd = sqrt(5/3); se = sd / 2; t(3) = 3.182
    error = math.sqrt(5 / 3) / 2
    assert out['periods'] == 4 and out['mean'] == 2.5 and out['standard_error'] == pytest.approx(error) and out['t_critical'] == 3.182
    assert out['ci95_low'] == pytest.approx(2.5 - 3.182 * error) and out['ci95_high'] == pytest.approx(2.5 + 3.182 * error)
    assert out['ci95_low'] == pytest.approx(0.44603, abs=1e-5) and out['ci95_high'] == pytest.approx(4.55397, abs=1e-5)
    assert summary.interval([3.0]) == {'periods': 1, 'mean': 3.0, 'standard_error': None, 'ci95_low': None, 'ci95_high': None, 't_critical': None}
    assert summary.interval([]) == {'periods': 0, 'mean': None, 'standard_error': None, 'ci95_low': None, 'ci95_high': None, 't_critical': None}


@pytest.mark.parametrize('values,minimum,expected', [
    ([1.0, 2.0] * 6, 12, 'positive'),              # exactly the minimum: mean 1.5, interval about 1.17 .. 1.83
    ([-1.0, -2.0] * 6, 12, 'negative'),
    ([-1.0, 2.0] * 6, 12, 'inconclusive'),         # mean .5, interval about -.50 .. 1.50
    ([-1.0, 1.0] * 6, 12, 'inconclusive'),         # mean exactly 0
    ([1.0, 2.0] * 5 + [1.0], 12, 'insufficient_sample'),   # eleven periods, however clear they look
    ([1.0, 2.0] * 6, 13, 'insufficient_sample'),
    ([5.0], 12, 'insufficient_sample'), ([], 12, 'insufficient_sample'),
    ([1.0, 2.0], 2, 'inconclusive'),               # t(1) = 12.706 makes two periods very wide
    ([1.0, 1.1], 2, 'positive'),
])
def test_verdict_branches(values, minimum, expected):
    assert summary.verdict(summary.interval(values), minimum) == expected


def test_metric_summary_describes_all_days_but_infers_from_the_kept_ones():
    seven = days(7)
    by_day = dict(zip(seven, [0.5, -0.1, None, 0.3, 0.2, 0.1, -0.4]))
    out = summary.metric_summary(by_day, [seven[0], seven[2], seven[5]], 12)   # the kept day without a value drops out
    assert out['days'] == 6 and out['daily_mean'] == pytest.approx(0.1) and out['daily_median'] == pytest.approx(0.15)
    assert out['daily_positive_share'] == pytest.approx(0.6667, abs=1e-4)     # four of six above zero
    estimate = out['non_overlapping']
    assert estimate['periods'] == 2 and estimate['mean'] == pytest.approx(0.3) and estimate['t_critical'] == 12.706
    assert estimate['standard_error'] == pytest.approx(0.2) and estimate['ci95_low'] == pytest.approx(0.3 - 12.706 * 0.2, abs=1e-4)
    assert out['verdict'] == 'insufficient_sample'
    empty = summary.metric_summary(dict.fromkeys(seven), seven[:1], 12)
    assert empty == {'days': 0, 'daily_mean': None, 'daily_median': None, 'daily_positive_share': None, 'verdict': 'insufficient_sample',
                     'non_overlapping': {'periods': 0, 'mean': None, 'standard_error': None, 'ci95_low': None, 'ci95_high': None, 't_critical': None}}


def row(day, horizon, k, **changes):
    """A typed daily row whose metrics alternate around a positive level."""
    base = {name: 0.05 + 0.01 * (k % 2) for name in outcomes.METRICS}
    return {'snapshot_day': day, 'horizon': horizon, 'rules_version': '1.0.0', 'missing_pct': 0.5 + (k % 2), **base, **changes}


@pytest.mark.parametrize('count,periods,status', [(55, 11, 'insufficient_sample'), (56, 12, 'sufficient'), (61, 13, 'sufficient')])
def test_horizon_reaches_a_verdict_only_at_the_minimum_number_of_periods(count, periods, status):
    rows = [row(day, 5, k) for k, day in enumerate(days(count))]
    out = summary.horizon_summary(rows, 5, SESSIONS, 12)
    assert out['evaluated_days'] == count and out['non_overlapping_periods'] == periods and out['sample_status'] == status
    assert out['non_overlapping_days'] == days(count)[::5] and out['snapshot_days_without_evaluation'] == 0
    assert out['first_snapshot_day'] == '2026-11-25' and out['last_snapshot_day'] == days(count)[-1]
    assert set(out['metrics']) == set(outcomes.METRICS)
    metric = out['metrics']['rank_ic']
    kept = [0.05 + 0.01 * (k % 2) for k in range(0, count, 5)]                 # every fifth day, worked out independently
    error = statistics.stdev(kept) / math.sqrt(len(kept))
    assert metric['days'] == count and metric['non_overlapping']['periods'] == periods
    assert metric['non_overlapping']['mean'] == pytest.approx(statistics.fmean(kept), abs=1e-4)
    assert metric['non_overlapping']['ci95_low'] == pytest.approx(statistics.fmean(kept) - summary.T_975[periods - 1] * error, abs=1e-4)
    assert metric['verdict'] == ('positive' if periods >= 12 else 'insufficient_sample')
    assert metric['daily_positive_share'] == 1.0     # all 55 daily values are positive, and that still decides nothing


def test_horizon_summary_counts_gaps_and_ignores_null_metrics():
    ten = days(10)
    rows = [row(day, 5, k) for k, day in enumerate(ten) if k not in (3, 4, 5)]     # three sessions without a row
    rows[0]['combo_minus_pool_pct'] = None                                         # the first kept day has no combo value
    out = summary.horizon_summary(rows, 5, SESSIONS, 12)
    assert out['evaluated_days'] == 7 and out['snapshot_days_without_evaluation'] == 3
    assert out['non_overlapping_days'] == [ten[0], ten[6]] and out['mean_missing_pct'] == pytest.approx((4 * 0.5 + 3 * 1.5) / 7, abs=1e-4)
    assert out['metrics']['rank_ic']['non_overlapping']['periods'] == 2
    assert out['metrics']['combo_minus_pool_pct']['non_overlapping']['periods'] == 1 and out['metrics']['combo_minus_pool_pct']['days'] == 6


def test_summary_document_structure():
    rows = [row(day, 5, k) for k, day in enumerate(days(56))] + [row(day, 20, k, rules_version='1.1.0') for k, day in enumerate(days(30))]
    last_run = {'5': {'snapshot_day': days(56)[-1], 'status': 'evaluated', 'reason': None},
                '20': {'snapshot_day': days(30)[-1], 'status': 'evaluated', 'reason': None},
                '60': {'snapshot_day': '2026-09-01', 'status': 'skipped', 'reason': 'before_record_start'}}
    out = summary.build(rows, sessions=SESSIONS, settings=SETTINGS, benchmark='SPY', record_start='2026-11-25',
                        evaluation_day='2027-02-24', last_run=last_run)
    assert json.loads(json.dumps(out, allow_nan=False)) == out                  # plain JSON, no NaN
    assert list(out) == ['schema_version', 'validation_version', 'description', 'benchmark', 'confidence_level',
                         'min_non_overlapping_periods', 'record_start', 'latest_evaluation_day', 'latest_evaluated_snapshot_day',
                         'snapshot_rules_versions', 'mean_missing_pct', 'settings', 'last_run', 'horizons', 'limitations']
    assert out['schema_version'] == 1 and out['validation_version'] == outcomes.VERSION and out['confidence_level'] == 0.95
    assert out['min_non_overlapping_periods'] == 12 and out['record_start'] == '2026-11-25' and out['benchmark'] == 'SPY'
    assert out['latest_evaluation_day'] == '2027-02-24' and out['latest_evaluated_snapshot_day'] == days(56)[-1]
    assert out['snapshot_rules_versions'] == ['1.0.0', '1.1.0'] and out['settings'] == SETTINGS and out['last_run'] == last_run
    assert out['mean_missing_pct'] == pytest.approx(1.0) and list(out['horizons']) == ['5', '20', '60']
    five, twenty, sixty = (out['horizons'][h] for h in ('5', '20', '60'))
    assert (five['sample_status'], twenty['sample_status'], sixty['sample_status']) == ('sufficient', 'insufficient_sample', 'insufficient_sample')
    assert twenty['evaluated_days'] == 30 and twenty['non_overlapping_periods'] == 2      # 30 overlapping days are two periods
    assert {m['verdict'] for m in twenty['metrics'].values()} == {'insufficient_sample'}
    assert sixty['evaluated_days'] == 0 and sixty['first_snapshot_day'] is None and sixty['mean_missing_pct'] is None
    assert sixty['metrics']['rank_ic']['daily_mean'] is None and sixty['metrics']['rank_ic']['verdict'] == 'insufficient_sample'
    assert list(five['metrics']['rank_ic']) == ['days', 'daily_mean', 'daily_median', 'daily_positive_share', 'non_overlapping', 'verdict']
    text = ' '.join(out['limitations'])
    for phrase in ('Overlapping windows', 'move together within a day', 'since record_start', 'never filled', 'not obtainable',
                   'multiple-comparison', 'not investment advice'):
        assert phrase in text
    nothing = summary.build([], sessions=SESSIONS, settings=SETTINGS, benchmark='SPY', record_start=None, evaluation_day='2026-11-25', last_run={})
    assert nothing['latest_evaluated_snapshot_day'] is None and nothing['mean_missing_pct'] is None and nothing['snapshot_rules_versions'] == []
    assert all(h['sample_status'] == 'insufficient_sample' for h in nothing['horizons'].values())
