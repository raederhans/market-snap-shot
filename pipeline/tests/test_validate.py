"""pipeline.validate on small hand-made snapshots and closes: files, skipping, idempotence, failures. No network."""
import copy
import csv
import gzip
import io
import json
import shutil
import pandas as pd
import pytest
from pipeline import outcomes, prices, run as pipeline_run, snapshot, universe, validate
from pipeline.outcomes import ValidationError
from .test_run import DIRECTORY, synthetic

NAN = float('nan')
DAY, TODAY = '2026-11-19', '2026-11-27'     # five sessions apart (Thanksgiving in between)
# Twelve ranked stocks (two of them spelled like missing/boolean values), two reference ETFs and one unranked row.
# (symbol, group, status, rs_percentile, template, distance from high, close on DAY, close on TODAY or None)
STOCKS = [
    ('A', 95, 'passes', -2., 107.), ('B', 85, 'passes', -5., 103.), ('C', 80, 'fails', -10., 105.), ('D', 70, 'passes', -8., 102.),
    ('E', 60, 'fails', -12., 101.), ('F', 45, 'fails', -30., 100.), ('G', 40, 'fails', -20., 104.), ('H', 25, 'fails', -40., 99.),
    ('I', 10, 'fails', -50., 97.), ('J', 5, 'fails', -60., 95.), ('NA', 90, 'passes', -1., None), ('TRUE', 15, 'fails', -55., NAN)]


def snapshot_rows():
    blank = dict.fromkeys(snapshot.COLUMNS)
    rows = [{**blank, 'symbol': s, 'group': 'stock', 'status': 'ranked', 'rs_percentile': rs, 'trend_template_status': template,
             'distance_from_52w_high_pct': distance} for s, rs, template, distance, _ in STOCKS]
    rows += [{**blank, 'symbol': s, 'group': 'etf', 'status': 'reference', 'rs_percentile': 50, 'trend_template_status': 'fails',
              'distance_from_52w_high_pct': -3.} for s in ('QQQ', 'SPY')]
    return rows + [{**blank, 'symbol': 'STL', 'group': 'stock', 'status': 'stale'}]


def closes_frame(day=DAY, today=TODAY, **changes):
    """Every stock at 100 on `day`; SPY +1%, QQQ +2%. 'NA' has no column, 'TRUE' no bar today."""
    end = {**{s: close for s, *_, close in STOCKS if close is not None}, 'QQQ': 510., 'SPY': 404., 'STL': 9., **changes}
    start = {**dict.fromkeys(end, 100.), 'QQQ': 500., 'SPY': 400., 'STL': 9.}
    return pd.DataFrame([start, {**start, 'A': 250.}, end], index=[day, '2026-11-20' if day == DAY else '2026-11-23', today])   # the middle row is never used


def save_closes(path, frame):
    frame.to_csv(path, index_label='session', compression='gzip')
    return path


def add_snapshot(data, day, rows=None):
    snapshot.write_snapshot(data, day, snapshot_rows() if rows is None else rows,
                            {'generated_at': day + 'T23:40:00+00:00', 'rules_version': '1.0.0'}, ['close'])


@pytest.fixture
def config():
    config = copy.deepcopy(pipeline_run.load_config())
    config['validation'].update(horizons=[5, 20], min_group_size=2)
    return config


@pytest.fixture
def world(tmp_path, config):
    """A published data directory holding the snapshot of DAY, today's closes, and a runner for validate."""
    data = tmp_path / 'data'
    add_snapshot(data, DAY)
    closes = save_closes(tmp_path / 'closes.csv.gz', closes_frame())
    return {'data': data, 'closes': closes, 'out': tmp_path / 'out', 'config': config,
            'run': lambda **kw: validate.validate(kw.get('data', data), kw.get('closes', closes), tmp_path / 'out', config=config)}


def files_in(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob('*')) if p.is_file()}


def daily_rows(directory):
    return list(csv.DictReader(io.StringIO((directory / 'validation' / 'daily.csv').read_text())))


def test_evaluates_the_snapshot_five_sessions_back_and_writes_three_outputs(world):
    document = world['run']()
    assert sorted(files_in(world['out'])) == ['outcomes/2026/2026-11-19_h05.csv.gz', 'validation/daily.csv', 'validation/summary.json']
    lines = gzip.decompress((world['out'] / 'outcomes/2026/2026-11-19_h05.csv.gz').read_bytes()).decode().splitlines()
    assert lines[0] == 'symbol,outcome_status,forward_return_pct,forward_excess_pct' and len(lines) == 1 + 14   # ranked + reference, not STL
    assert lines[1] == 'A,ok,7.0000,6.0000' and lines[10] == 'J,ok,-5.0000,-6.0000'      # 107/100 - 1 = 7%, minus SPY's 1%
    assert lines[11] == 'NA,no_data_at_T,,' and lines[12] == 'TRUE,missing_bar_at_T,,'
    assert lines[13] == 'QQQ,ok,2.0000,1.0000' and lines[14] == 'SPY,ok,1.0000,0.0000'

    (row,) = daily_rows(world['out'])
    assert list(row) == list(outcomes.DAILY_COLUMNS)
    expected = {   # the hand-worked cross-section of test_outcomes.py: the ETFs are not part of the pool
        'snapshot_day': DAY, 'horizon': '5', 'evaluated_on': TODAY, 'rules_version': '1.0.0', 'benchmark_return_pct': '1.0000',
        'pool': '12', 'with_outcome': '10', 'missing': '2', 'missing_pct': '16.6667', 'pool_excess_pct': '0.3000', 'rank_ic': '0.8545',
        'rs_q1_excess_pct': '-5.0000', 'rs_q2_excess_pct': '', 'rs_q3_excess_pct': '1.0000', 'rs_q4_excess_pct': '0.5000',
        'rs_q5_excess_pct': '4.0000', 'rs_q5_minus_q1_pct': '9.0000', 'template_pass_n': '3', 'template_pass_excess_pct': '3.0000',
        'template_fail_n': '7', 'template_fail_excess_pct': '-0.8571', 'template_pass_minus_fail_pct': '3.8571',
        'near_high_n': '4', 'near_high_excess_pct': '3.2500', 'not_near_high_n': '6', 'not_near_high_excess_pct': '-1.6667',
        'near_high_minus_rest_pct': '4.9167', 'combo_n': '2', 'combo_excess_pct': '4.0000', 'combo_minus_pool_pct': '3.7000',
        'null_reasons': 'rs_q2=group_too_small(n=1<2)'}
    assert row == expected

    assert json.loads((world['out'] / 'validation/summary.json').read_text()) == document
    assert document['record_start'] == DAY and document['latest_evaluation_day'] == TODAY and document['latest_evaluated_snapshot_day'] == DAY
    assert document['last_run'] == {'5': {'snapshot_day': DAY, 'status': 'evaluated', 'reason': None},
                                    '20': {'snapshot_day': '2026-10-29', 'status': 'skipped', 'reason': 'before_record_start'}}
    five = document['horizons']['5']
    assert five['evaluated_days'] == 1 and five['non_overlapping_periods'] == 1 and five['sample_status'] == 'insufficient_sample'
    assert five['mean_missing_pct'] == 16.6667 and document['mean_missing_pct'] == 16.6667
    ic = five['metrics']['rank_ic']
    assert ic['daily_mean'] == 0.8545 and ic['non_overlapping']['mean'] == 0.8545 and ic['non_overlapping']['ci95_low'] is None
    assert {m['verdict'] for h in document['horizons'].values() for m in h['metrics'].values()} == {'insufficient_sample'}
    assert five['metrics']['rs_q5_minus_q1_pct']['daily_mean'] == 9.0 and document['horizons']['20']['evaluated_days'] == 0


def test_a_horizon_without_its_snapshot_is_skipped_and_no_neighbouring_day_is_used(world):
    add_snapshot(world['data'], '2026-10-30')                   # one session after the day horizon 20 needs
    add_snapshot(world['data'], '2026-10-28')                   # and one before: the record now starts earlier
    document = world['run']()
    assert document['last_run']['20'] == {'snapshot_day': '2026-10-29', 'status': 'skipped', 'reason': 'snapshot_missing'}
    assert document['record_start'] == '2026-10-28' and document['horizons']['20']['evaluated_days'] == 0
    assert [name for name in files_in(world['out']) if name.startswith('outcomes/')] == ['outcomes/2026/2026-11-19_h05.csv.gz']
    assert [(r['snapshot_day'], r['horizon']) for r in daily_rows(world['out'])] == [(DAY, '5')]


def test_before_any_snapshot_exists_the_record_is_empty_not_an_error(tmp_path, world):
    document = world['run'](data=tmp_path / 'nothing-published')
    assert {entry['reason'] for entry in document['last_run'].values()} == {'before_record_start'}
    assert document['record_start'] is None and document['latest_evaluated_snapshot_day'] is None
    assert sorted(files_in(world['out'])) == ['validation/daily.csv', 'validation/summary.json'] and daily_rows(world['out']) == []
    assert (world['out'] / 'validation/daily.csv').read_text() == ','.join(outcomes.DAILY_COLUMNS) + '\n'


def test_a_rerun_writes_identical_bytes_and_replaces_its_row(tmp_path, world):
    world['run']()
    first = files_in(world['out'])
    world['run']()
    assert files_in(world['out']) == first
    shutil.copytree(world['out'], world['data'], dirs_exist_ok=True)       # published; the next run starts from it
    world['run']()
    assert files_in(world['out']) == first and len(daily_rows(world['out'])) == 1
    # The same (snapshot day, horizon) with other prices replaces the row instead of adding one; other rows stay.
    table = world['data'] / 'validation/daily.csv'
    other = {**daily_rows(world['data'])[0], 'snapshot_day': '2026-11-18', 'evaluated_on': '2026-11-25', 'rank_ic': '-0.2500'}
    table.write_text(table.read_text() + ','.join(other[c] for c in outcomes.DAILY_COLUMNS) + '\n')
    world['run'](closes=save_closes(tmp_path / 'revised.csv.gz', closes_frame(A=108.)))
    rows = daily_rows(world['out'])
    assert [(r['snapshot_day'], r['horizon']) for r in rows] == [('2026-11-18', '5'), (DAY, '5')]
    assert rows[0] == other and rows[1]['rs_q5_excess_pct'] == '4.3333' and rows[1]['pool_excess_pct'] == '0.4000'
    document = json.loads((world['out'] / 'validation/summary.json').read_text())
    assert document['horizons']['5']['evaluated_days'] == 2 and document['horizons']['5']['non_overlapping_periods'] == 1
    assert document['horizons']['5']['metrics']['rank_ic']['daily_mean'] == pytest.approx((0.8545 - 0.25) / 2, abs=1e-4)   # from the table


def test_rows_accumulate_day_after_day(tmp_path, world):
    world['run']()
    shutil.copytree(world['out'], world['data'], dirs_exist_ok=True)
    add_snapshot(world['data'], '2026-11-20')
    later = save_closes(tmp_path / 'monday.csv.gz', closes_frame(day='2026-11-20', today='2026-11-30', J=90.))
    document = world['run'](closes=later)
    rows = daily_rows(world['out'])
    assert [(r['snapshot_day'], r['evaluated_on']) for r in rows] == [(DAY, TODAY), ('2026-11-20', '2026-11-30')]
    assert rows[1]['rs_q1_excess_pct'] == '-7.5000'                         # I -3% and J -10%, minus SPY's 1%
    assert 'outcomes/2026/2026-11-20_h05.csv.gz' in files_in(world['out'])
    five = document['horizons']['5']
    assert five['evaluated_days'] == 2 and five['non_overlapping_periods'] == 1 and five['non_overlapping_days'] == [DAY]
    assert five['metrics']['rs_q5_minus_q1_pct']['daily_mean'] == pytest.approx((9.0 + 11.5) / 2)
    assert document['latest_evaluation_day'] == '2026-11-30' and document['record_start'] == DAY


def test_failures_write_nothing(tmp_path, world, capsys, monkeypatch):
    def fails(match, **kw):
        with pytest.raises(ValidationError, match=match):
            world['run'](**kw)
        assert not world['out'].exists()
    fails('closes file .* is missing', closes=tmp_path / 'absent.csv.gz')
    holiday = closes_frame().rename(index={TODAY: '2026-11-26'})            # the last row is not a session
    fails('2026-11-26 is not an exchange session', closes=save_closes(tmp_path / 'holiday.csv.gz', holiday))
    fails('benchmark SPY lacks a close', closes=save_closes(tmp_path / 'nospy.csv.gz', closes_frame(SPY=NAN)))
    fails('no row for session 2026-11-19', closes=save_closes(tmp_path / 'short.csv.gz', closes_frame().iloc[1:]))
    # Horizon 5 is fine but horizon 20 cannot be evaluated: nothing at all is written.
    add_snapshot(world['data'], '2026-10-29')
    fails('no row for session 2026-10-29')
    shutil.rmtree(world['data'] / 'snapshots/2026')
    add_snapshot(world['data'], DAY)
    meta = world['data'] / f'snapshots/2026/{DAY}.meta.json'
    meta.write_text(json.dumps({**json.loads(meta.read_text()), 'files': {'snapshot_sha256': '0' * 64}}))
    fails('does not match the hash')
    add_snapshot(world['data'], DAY)
    table = world['data'] / 'validation/daily.csv'
    table.parent.mkdir()
    table.write_text('snapshot_day,horizon,rank_ic\n2026-11-18,5,0.1\n')    # written by some other version
    fails('other columns than this version writes')
    table.write_text(','.join(outcomes.DAILY_COLUMNS) + '\n2026-11-18,5,2026-11-25\n')
    fails('malformed row')
    table.unlink()
    monkeypatch.setattr(validate, 'SNAPSHOT_COLUMNS', (*validate.SNAPSHOT_COLUMNS, 'sector'))
    fails('lacks columns: sector')
    monkeypatch.undo()
    monkeypatch.setattr(validate, 'load_config', lambda: world['config'])
    assert validate.main(['--data', str(world['data']), '--closes', str(tmp_path / 'absent.csv.gz'), '--out', str(world['out'])]) == 1
    assert 'FORWARD VALIDATION FAILED, nothing written' in capsys.readouterr().err and not world['out'].exists()
    assert validate.main(['--data', str(world['data']), '--closes', str(world['closes']), '--out', str(world['out'])]) == 0
    printed = capsys.readouterr().out
    assert 'h=5: snapshot 2026-11-19 evaluated; 1 days, 1 non-overlapping, insufficient_sample' in printed
    assert 'h=20: snapshot 2026-10-29 skipped (before_record_start); 0 days' in printed


def test_end_to_end_with_the_snapshot_job_over_six_sessions(tmp_path, monkeypatch, config):
    """Build -> publish snapshot -> validate -> publish validation, as the workflow does, with mocked prices."""
    monkeypatch.setattr(universe, 'download_directory', lambda url, attempts: DIRECTORY)
    monkeypatch.setattr(prices, 'fetch_yahoo', synthetic)
    config['eligibility']['min_ranked_stocks'] = 20
    config['validation'].update(horizons=[1, 5], min_group_size=3)
    data = tmp_path / 'data'
    for day in ('2026-11-19', '2026-11-20', '2026-11-23', '2026-11-24', '2026-11-25', '2026-11-27'):
        work = tmp_path / day
        pipeline_run.run(work / 'snapshot', now=day + 'T23:30:00Z', config=config, closes_path=work / 'closes.csv.gz')
        shutil.copytree(work / 'snapshot', data, dirs_exist_ok=True)
        document = validate.validate(data, work / 'closes.csv.gz', work / 'validation', config=config)
        shutil.copytree(work / 'validation', data, dirs_exist_ok=True)
    rows = daily_rows(data)
    assert [(r['snapshot_day'], r['horizon'], r['evaluated_on']) for r in rows] == [
        ('2026-11-19', '1', '2026-11-20'), ('2026-11-19', '5', '2026-11-27'), ('2026-11-20', '1', '2026-11-23'),
        ('2026-11-23', '1', '2026-11-24'), ('2026-11-24', '1', '2026-11-25'), ('2026-11-25', '1', '2026-11-27')]
    # The k-th symbol grows (1 + k/100) a year, so strength and forward return have the same order: rank IC is 1.
    assert all(r['rank_ic'] == '1.0000' and r['pool'] == '27' and r['missing'] == '0' for r in rows)
    symbols = sorted(pd.read_csv(data / 'snapshots/2026/2026-11-19.csv.gz', keep_default_na=False).symbol)
    spy, aapl = (100 * ((1 + symbols.index(s) / 100) ** (5 / 252) - 1) for s in ('SPY', 'AAPL'))
    outcome = pd.read_csv(data / 'outcomes/2026/2026-11-19_h05.csv.gz', keep_default_na=False).set_index('symbol')
    assert len(outcome) == 40 and (outcome.outcome_status == 'ok').all()
    assert float(outcome.loc['AAPL', 'forward_return_pct']) == pytest.approx(aapl, abs=1e-4)
    assert float(outcome.loc['AAPL', 'forward_excess_pct']) == pytest.approx(aapl - spy, abs=1e-4)
    assert float(rows[1]['benchmark_return_pct']) == pytest.approx(spy, abs=1e-4)
    one, five = document['horizons']['1'], document['horizons']['5']
    assert one['evaluated_days'] == 5 and one['non_overlapping_periods'] == 5 and five['evaluated_days'] == 1
    assert one['sample_status'] == five['sample_status'] == 'insufficient_sample'
    assert one['metrics']['rank_ic']['verdict'] == 'insufficient_sample' and one['metrics']['rank_ic']['daily_positive_share'] == 1.0
    assert document['record_start'] == '2026-11-19' and document['snapshot_rules_versions'] == ['1.0.0']
    assert sorted(p.name for p in (data / 'outcomes/2026').iterdir()) == [
        '2026-11-19_h01.csv.gz', '2026-11-19_h05.csv.gz', '2026-11-20_h01.csv.gz', '2026-11-23_h01.csv.gz',
        '2026-11-24_h01.csv.gz', '2026-11-25_h01.csv.gz']
