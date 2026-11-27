"""pipeline.backfill end to end on a mocked provider, and the slice summary on a hand-made table. No network."""
import copy
import csv
import json
import pandas as pd
import pytest
from pipeline import backfill, backfill_summary, outcomes, prices, run as pipeline_run, summary, universe
from .test_run import DIRECTORY, NOW, synthetic

STOCKS = 27                     # kept by the name rules from the fixture directory


@pytest.fixture
def config():
    config = copy.deepcopy(pipeline_run.load_config())
    config['eligibility']['min_ranked_stocks'] = 20
    config['backfill'].update(history_calendar_days=730, top_dollar_volume=10)   # 502 sessions: grid 252, 257, ..., 497
    config['validation'].update(min_group_size=2)
    return config


@pytest.fixture
def provider(monkeypatch):
    state = {'fetch': synthetic, 'calls': []}
    def fetch_yahoo(symbols, start, end, **settings):
        state['calls'].append({'symbols': list(symbols), 'start': str(start), 'end': str(end), 'settings': settings})
        return state['fetch'](symbols, start, end, **settings)
    monkeypatch.setattr(universe, 'download_directory', lambda url, attempts: DIRECTORY)
    monkeypatch.setattr(prices, 'fetch_yahoo', fetch_yahoo)
    return state


def table(out):
    with (out / 'backfill/daily.csv').open(newline='') as handle:
        return list(csv.DictReader(handle))


def test_run_writes_the_daily_table_and_the_summary_and_nothing_else(tmp_path, config, provider):
    document = backfill.run(tmp_path, now=NOW, config=config)
    call = provider['calls'][0]
    assert (call['start'], call['end']) == ('2024-11-25', '2026-11-26') and len(call['symbols']) == STOCKS + 1
    assert call['symbols'][-1] == 'SPY' and 'QQQ' not in call['symbols'] and call['settings']['batch_size'] == 100
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*') if p.is_file()) == ['backfill/daily.csv', 'backfill/summary.json']
    assert json.loads((tmp_path / 'backfill/summary.json').read_text()) == document

    rows = table(tmp_path)
    assert tuple(rows[0]) == backfill.DAILY_COLUMNS and set(backfill.DAILY_COLUMNS) - set(outcomes.DAILY_COLUMNS) == {'tier', 'year', 'benchmark_trend'}
    sessions = document['data']['sessions']
    assert sessions == 502 and document['data']['first_session'] == '2024-11-25' and document['data']['last_session'] == '2026-11-25'
    # The grid: the first session with 253 bars, then every fifth; a horizon has a row while D+h exists.
    days = sorted({row['snapshot_day'] for row in rows})
    calendar = pd.DatetimeIndex(synthetic(['X'], '2024-11-25', '2026-11-26')['X'].index).strftime('%Y-%m-%d').tolist()
    assert days == calendar[252:497:5] and document['evaluation'] == {
        'first_day': calendar[252], 'last_day': calendar[492], 'days': 49, 'step_sessions': 5,
        'days_by_benchmark_trend': {'above': 49, 'below': 0}, 'days_skipped_for_small_pool': []}
    per_horizon = {h: [row for row in rows if row['horizon'] == h and row['tier'] == 'eligible_pool'] for h in ('5', '20', '60')}
    assert {h: len(found) for h, found in per_horizon.items()} == {'5': 49, '20': 46, '60': 38}   # D+h must exist: the last session is number 501
    assert all(row['evaluated_on'] == calendar[calendar.index(row['snapshot_day']) + int(row['horizon'])] for row in rows)
    assert len(rows) == 2 * (49 + 46 + 38) and {row['tier'] for row in rows} == {'top_dollar_volume', 'eligible_pool'}
    first = per_horizon['5'][0]
    assert first['pool'] == str(STOCKS) and first['missing'] == '0' and first['year'] == '2025' and first['benchmark_trend'] == 'above'
    assert first['rules_version'] == '1.0.0' and float(first['rank_ic']) == 1.0      # the k-th stock drifts k% a year
    top = next(row for row in rows if row['tier'] == 'top_dollar_volume')
    assert top['pool'] == '10' and float(top['rs_q5_excess_pct']) > float(top['pool_excess_pct'])   # the ten fastest risers trade the most dollars

    assert document['record_type'] == 'backfill' and document['schema_version'] == 1 and document['primary_tier'] == 'top_dollar_volume'
    assert document['limitations'] == backfill_summary.LIMITATIONS and 'Survivorship' in document['limitations'][0]
    assert document['rules_version'] == '1.0.0' and 'Yahoo' in document['provider'] and document['universe']['sample_limit'] is None
    assert document['universe']['source'] == config['universe']['directory_url'] and document['universe']['stocks_after_filter'] == STOCKS
    assert document['data']['requested'] == 28 and document['data']['fetch_success_ratio'] == 1.0 and document['data']['stocks_with_bars'] == STOCKS
    assert document['parameters'] == {'backfill': config['backfill'], 'eligibility': config['eligibility'], 'validation': config['validation']}
    assert list(document['tiers']) == ['top_dollar_volume', 'eligible_pool']
    for tier in document['tiers'].values():
        assert list(tier['horizons']) == ['5', '20', '60']
        five = tier['horizons']['5']['slices']
        assert list(five) == ['full_sample', 'benchmark_above_sma200', 'benchmark_below_sma200', 'year_2025', 'year_2026']
        assert five['full_sample']['evaluated_days'] == five['full_sample']['non_overlapping_periods'] == 49
        assert five['benchmark_below_sma200']['evaluated_days'] == 0 and five['benchmark_below_sma200']['sample_status'] == 'insufficient_sample'
        assert tier['horizons']['60']['slices']['full_sample']['non_overlapping_periods'] == 4          # days 0, 12, 24, 36 of the grid
        assert tier['horizons']['60']['slices']['full_sample']['metrics']['rank_ic']['verdict'] == 'insufficient_sample'
        assert set(five['full_sample']['metrics']) == set(outcomes.METRICS) and 'non_overlapping_days' not in five['full_sample']


def test_missing_outcomes_are_counted_and_small_pools_are_skipped(tmp_path, config, provider):
    def partial(symbols, start, end, **settings):
        data = synthetic(symbols, start, end)
        data['MU'] = data['MU'].iloc[:-7]                # last bar on session 494
        return data
    provider['fetch'] = partial
    backfill.run(tmp_path / 'a', now=NOW, config=config)
    rows = {(r['snapshot_day'], r['horizon'], r['tier']): r for r in table(tmp_path / 'a')}
    days = sorted({key[0] for key in rows})
    last = rows[(days[-1], '5', 'eligible_pool')]         # D = session 492, D+5 = 497: MU ranked on D, no bar on D+5
    assert (last['pool'], last['with_outcome'], last['missing']) == ('27', '26', '1') and last['missing_pct'] == '3.7037'
    earlier = rows[(days[-2], '5', 'eligible_pool')]      # D+5 = 492: MU still has its bar
    assert (earlier['with_outcome'], earlier['missing']) == ('27', '0')

    config['eligibility']['min_ranked_stocks'] = 27       # one stale stock then leaves 26 ranked
    def stale(symbols, start, end, **settings):
        data = synthetic(symbols, start, end)
        data['MU'] = data['MU'].iloc[:-12]                # last bar on session 489: stale on D = 492
        return data
    provider['fetch'] = stale
    document = backfill.run(tmp_path / 'b', now=NOW, config=config)
    skipped = document['evaluation']['days_skipped_for_small_pool']   # sessions 492 and 497, the latter without any outcome yet
    assert skipped[0] == {'day': days[-1], 'ranked_stocks': 26} and [entry['ranked_stocks'] for entry in skipped] == [26, 26]
    assert document['evaluation']['days'] == 48 and days[-1] not in {r['snapshot_day'] for r in table(tmp_path / 'b')}


def test_gates_and_settings(tmp_path, config, provider, capsys, monkeypatch):
    def holed(symbols, start, end, **settings):
        data = synthetic(symbols, start, end)
        data['SPY'] = data['SPY'].drop(data['SPY'].index[100])
        return data
    provider['fetch'] = holed
    with pytest.raises(backfill.gates.IntegrityError, match='501 bars for 502 XNYS sessions'):
        backfill.run(tmp_path, now=NOW, config=config)
    provider['fetch'] = lambda symbols, start, end, **settings: {s: d for s, d in synthetic(symbols, start, end).items() if s == 'SPY' or s < 'D'}
    with pytest.raises(backfill.gates.IntegrityError, match='requested symbols returned usable bars'):
        backfill.run(tmp_path, now=NOW, config=config)
    provider['fetch'] = synthetic
    short = copy.deepcopy(config)
    short['backfill']['history_calendar_days'] = 365      # 252 sessions: the benchmark gate of the daily job
    with pytest.raises(backfill.gates.IntegrityError, match='needs 253'):
        backfill.run(tmp_path, now=NOW, config=short)
    assert not any(tmp_path.iterdir())
    for name, value in (('step_sessions', 0), ('top_dollar_volume', 2.5), ('regime_sma_sessions', -1)):
        bad = copy.deepcopy(config)
        bad['backfill'][name] = value
        with pytest.raises(ValueError, match='positive integer'):
            backfill.run(tmp_path, now=NOW, config=bad)
    monkeypatch.setattr(backfill, 'run', lambda out, limit: (_ for _ in ()).throw(backfill.gates.IntegrityError('benchmark SPY missing')))
    assert backfill.main(['--out', str(tmp_path)]) == 1 and 'BACKFILL FAILED, nothing written' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        backfill.main(['--out', str(tmp_path), '--limit', '0'])


def test_limit_and_unknown_trend(tmp_path, config, provider, capsys, monkeypatch):
    config['backfill']['regime_sma_sessions'] = 260       # the first two grid days (253 and 258 closes) have no 260-session average
    monkeypatch.setattr(backfill, 'load_config', lambda: config)
    monkeypatch.setattr(pd.Timestamp, 'now', classmethod(lambda cls, tz=None: pd.Timestamp(NOW)))
    assert backfill.main(['--out', str(tmp_path), '--limit', '24']) == 0
    printed = capsys.readouterr().out
    assert '24 stocks, 49 evaluation days' in printed and 'h=60: 38 days, 4 non-overlapping, insufficient_sample' in printed
    document = json.loads((tmp_path / 'backfill/summary.json').read_text())
    assert document['universe']['sample_limit'] == 24 and document['data']['requested'] == 25
    rows = [row for row in table(tmp_path) if row['horizon'] == '5' and row['tier'] == 'eligible_pool']
    assert [row['benchmark_trend'] for row in rows[:3]] == ['', '', 'above'] and document['evaluation']['days_by_benchmark_trend'] == {'above': 47, 'below': 0}
    five = document['tiers']['eligible_pool']['horizons']['5']['slices']
    assert five['full_sample']['evaluated_days'] == 49 and five['benchmark_above_sma260']['evaluated_days'] == 47
    assert rows[0]['pool'] == '24' and next(r for r in table(tmp_path) if r['tier'] == 'top_dollar_volume')['pool'] == '10'


def test_slice_summary_by_hand():
    """Horizon 10 on a grid of every fifth session: each slice takes its own rows at least 10 sessions apart."""
    sessions = [f'2022-{1 + k // 28:02d}-{1 + k % 28:02d}' for k in range(80)]
    # (session number, trend, rank_ic); every other metric is constant.
    grid = [(0, 'above', .10), (5, 'below', .50), (10, 'below', .90), (15, 'above', .20), (20, 'above', .30), (25, 'below', .70),
            (30, 'below', -.10), (35, '', .40), (40, 'above', .60)]
    rows = [{'snapshot_day': sessions[at], 'horizon': 10, 'tier': 'eligible_pool', 'missing_pct': 1.0, 'year': '2022' if at < 28 else '2023',
             'benchmark_trend': trend, **dict.fromkeys(outcomes.METRICS, 1.0), 'rank_ic': ic} for at, trend, ic in grid]
    config = {'validation': {'horizons': [10], 'min_non_overlapping_periods': 3}, 'backfill': {'regime_sma_sessions': 200}}
    found = backfill_summary.tier_summary(rows, sessions, config)['10']['slices']
    assert list(found) == ['full_sample', 'benchmark_above_sma200', 'benchmark_below_sma200', 'year_2022', 'year_2023']
    def ic(name):
        return found[name]['evaluated_days'], found[name]['metrics']['rank_ic']['non_overlapping'], found[name]['metrics']['rank_ic']['verdict']
    days, estimate, verdict = ic('full_sample')            # sessions 0, 10, 20, 30, 40: .10 .90 .30 -.10 .60
    assert (days, estimate['periods'], estimate['mean']) == (9, 5, 0.36) and verdict == 'inconclusive'
    assert estimate['standard_error'] == 0.1778 and estimate['t_critical'] == 2.776 and estimate['ci95_low'] == round(.36 - 2.776 * (.158 / 5) ** .5, 4)
    days, estimate, verdict = ic('benchmark_above_sma200')  # sessions 0, 15, 40 (20 is 5 after 15): .10 .20 .60
    assert (days, estimate['periods'], estimate['mean']) == (4, 3, 0.3) and verdict == 'inconclusive'
    days, estimate, verdict = ic('benchmark_below_sma200')  # sessions 5, 25 (10 and 30 overlap them): .50 .70
    assert (days, estimate['periods'], estimate['mean']) == (4, 2, 0.6) and verdict == 'insufficient_sample'
    assert found['benchmark_below_sma200']['sample_status'] == 'insufficient_sample'
    days, estimate, verdict = ic('year_2023')               # sessions 30, 40 (35 has no trend but is in its year)
    assert (days, estimate['periods'], estimate['mean']) == (3, 2, 0.25) and verdict == 'insufficient_sample'
    constant = found['benchmark_above_sma200']['metrics']['pool_excess_pct']
    assert constant['non_overlapping']['mean'] == 1.0 and constant['verdict'] == 'positive'   # zero spread: the interval is the point 1.0
    # The same rows through the forward function give the same numbers: nothing is defined twice.
    direct = summary.horizon_summary([r for r in rows if r['benchmark_trend'] == 'below'], 10, sessions, 3)
    assert {k: v for k, v in direct.items() if k not in backfill_summary.DROPPED} == found['benchmark_below_sma200']
