"""End-to-end run with a mocked symbol directory and price provider. No network."""
import copy
import gzip
import io
import json
import pathlib
import numpy as np
import pandas as pd
import pytest
from pipeline import gates, prices, run as pipeline_run, snapshot, universe
from pipeline.gates import IntegrityError
from stock_research import universe_rank

DIRECTORY = (pathlib.Path(__file__).parent / 'fixtures' / 'nasdaqtraded_sample.txt').read_text()
NOW = '2026-11-26T23:30:00Z'   # Thanksgiving evening: the snapshot day must be Wednesday the 25th
DAY = '2026-11-25'
PRICE_LEVELS = ['close', 'sma50', 'sma150', 'sma200']   # computed for the checks, not published


def synthetic(symbols, start, end, **settings):
    """Deterministic adjusted bars for every requested symbol: the k-th symbol drifts k% a year."""
    dates = gates.completed_sessions(pd.Timestamp(end, tz='UTC')).index
    dates = dates[dates >= pd.Timestamp(start)]
    out = {}
    for k, symbol in enumerate(sorted(symbols)):
        close = 50.0 * (1 + k / 100) ** (np.arange(len(dates)) / 252)
        out[symbol] = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * .99, 'close': close,
                                    'volume': 1_000_000.}, index=dates)
    return out


@pytest.fixture
def config():
    config = copy.deepcopy(pipeline_run.load_config())
    config['eligibility']['min_ranked_stocks'] = 20   # the fixture directory keeps 27 stocks
    return config


@pytest.fixture
def provider(monkeypatch):
    """Replace the directory download and the price provider. Tests swap provider['fetch']."""
    state = {'fetch': synthetic, 'calls': []}
    def fetch_yahoo(symbols, start, end, **settings):
        state['calls'].append({'symbols': list(symbols), 'start': start, 'end': end, 'settings': settings})
        return state['fetch'](symbols, start, end, **settings)
    monkeypatch.setattr(universe, 'download_directory', lambda url, attempts: DIRECTORY)
    monkeypatch.setattr(prices, 'fetch_yahoo', fetch_yahoo)
    return state


def read_snapshot(out):
    with gzip.open(next(out.glob('snapshots/*/*.csv.gz')), 'rt') as handle:
        return pd.read_csv(handle).set_index('symbol')


def test_successful_run_writes_snapshot_metadata_and_pointer(tmp_path, config, provider):
    meta = pipeline_run.run(tmp_path, now=NOW, config=config)
    call = provider['calls'][0]
    assert str(call['end']) == '2026-11-26' and str(call['start']) == '2025-05-26'   # 548 days back, end exclusive
    assert call['settings'] == {'batch_size': 100, 'max_attempts': 3, 'backoff_seconds': 20, 'pause_seconds': 1.0}
    assert 'BRK.B' in call['symbols'] and 'SPY' in call['symbols'] and 'AACIW' not in call['symbols']

    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*') if p.is_file())
    assert files == ['latest.json', f'snapshots/2026/{DAY}.csv.gz', f'snapshots/2026/{DAY}.meta.json']
    table = read_snapshot(tmp_path)
    published = [c for c in universe_rank.COLUMNS if c not in PRICE_LEVELS]
    assert list(table.reset_index().columns) == published and len(published) == len(universe_rank.COLUMNS) - 4
    assert len(table) == 27 + 13 and (table.last_bar_date == DAY).all()
    assert (table[table.group == 'stock'].status == 'ranked').all()
    assert (table[table.group == 'etf'].status == 'reference').all()
    stocks = table[table.group == 'stock'].sort_values('rs_strength_pct')
    assert stocks.rs_percentile.is_monotonic_increasing and stocks.rs_percentile.between(1, 99).all()
    assert stocks.rs_percentile.iloc[0] == 2 and stocks.rs_percentile.iloc[-1] == 98   # 100*.5/27, 100*26.5/27
    assert table.loc['SPY', 'rs_vs_benchmark'] == 1.0

    saved = json.loads((tmp_path / f'snapshots/2026/{DAY}.meta.json').read_text())
    assert saved == meta and saved['trading_day'] == DAY and saved['sample_limit'] is None
    assert saved['schema_version'] == 2 and saved['columns'] == published and saved['excluded_columns'] == PRICE_LEVELS
    counts = saved['counts']
    assert counts['directory_rows'] == 53 and counts['stocks_after_filter'] == 27 and counts['requested'] == 40
    assert counts['fetched'] == 40 and counts['no_data'] == 0 and counts['fetch_success_ratio'] == 1.0
    assert counts['directory_excluded']['warrant'] == 3 and counts['directory_excluded']['etf'] == 2
    assert counts['rows_by_group_and_status'] == {'stock': {'ranked': 27}, 'etf': {'reference': 13}}
    assert saved['config'] == config and saved['rules_version'] == universe_rank.VERSION
    assert saved['symbol_directory']['file_creation_time'] == '1125202621:34'
    pointer = json.loads((tmp_path / 'latest.json').read_text())
    assert pointer['trading_day'] == DAY and pointer['snapshot'] == f'snapshots/2026/{DAY}.csv.gz'
    assert pointer['meta'] == f'snapshots/2026/{DAY}.meta.json'   # both relative to the directory of latest.json
    assert pointer['snapshot_sha256'] == saved['files']['snapshot_sha256']


def test_excluded_columns_come_from_the_configuration(tmp_path, config, provider):
    config['output']['exclude_columns'] = []                       # nothing excluded: the full table
    meta = pipeline_run.run(tmp_path / 'all', now=NOW, config=config)
    table = read_snapshot(tmp_path / 'all')
    assert list(table.reset_index().columns) == list(universe_rank.COLUMNS) and meta['excluded_columns'] == []
    assert table.loc['AAPL', 'close'] > 0 and table.loc['AAPL', 'sma200'] > 0

    config['output']['exclude_columns'] = ['sma200', 'atr14_pct']  # any column, recorded in table order
    meta = pipeline_run.run(tmp_path / 'two', now=NOW, config=config)
    two = read_snapshot(tmp_path / 'two')
    assert meta['excluded_columns'] == ['atr14_pct', 'sma200'] and not {'sma200', 'atr14_pct'} & set(two.columns)
    assert 'sma200' not in meta['columns'] and two.rs_percentile.equals(table.rs_percentile)   # ranking is unaffected
    assert two.tt_above_150_and_200.equals(table.tt_above_150_and_200)   # checks still use the excluded averages

    config['output']['exclude_columns'] = ['close', 'closing_price']
    with pytest.raises(ValueError, match='unknown columns: closing_price'):
        pipeline_run.run(tmp_path / 'typo', now=NOW, config=config)
    assert not (tmp_path / 'typo').exists()


def test_stale_invalid_and_missing_symbols_are_reported_not_ranked(tmp_path, config, provider):
    def partial(symbols, start, end, **settings):
        data = synthetic(symbols, start, end)
        data['MU'] = data['MU'].iloc[:-1]                       # no bar for the snapshot day
        data['NOW'].iloc[5, data['NOW'].columns.get_loc('low')] *= 2   # low above high
        del data['UBER']
        return data
    provider['fetch'] = partial
    config['gates'].update(min_fetch_success_ratio=.9, max_stale_ratio=.05)
    meta = pipeline_run.run(tmp_path, now=NOW, config=config)
    table = read_snapshot(tmp_path)
    assert table.loc['MU', 'status'] == 'stale' and table.loc['MU', 'last_bar_date'] == '2026-11-24'
    assert pd.isna(table.loc['MU', 'rs_strength_pct']) and pd.isna(table.loc['MU', 'rs_percentile'])
    assert 'NOW' not in table.index and 'UBER' not in table.index
    assert meta['failed_symbols'] == {'no_data': ['UBER'], 'invalid_bars': {'NOW': 'OHLC price consistency violated'}}
    assert meta['counts']['fetched'] == 39 and meta['counts']['invalid_bars'] == 1 and meta['counts']['snapshot_rows'] == 38
    assert meta['counts']['rows_by_group_and_status']['stock'] == {'ranked': 24, 'stale': 1}


def drop(*symbols):
    def fetch(names, start, end, **settings):
        data = synthetic(names, start, end)
        return {s: d for s, d in data.items() if s not in symbols}
    return fetch


def lagging(*symbols):
    def fetch(names, start, end, **settings):
        data = synthetic(names, start, end)
        return {s: d.iloc[:-1] if s in symbols else d for s, d in data.items()}
    return fetch


def thin(names, start, end, **settings):
    return {s: d.assign(volume=10.) if s not in ('SPY', 'AAPL') else d for s, d in synthetic(names, start, end).items()}


@pytest.mark.parametrize('fetch,message', [
    (drop('SPY'), 'benchmark SPY was not fetched'),
    (lagging('SPY'), 'benchmark SPY last bar 2026-11-24'),
    (drop('AAPL', 'MSFT', 'NVDA'), 'only 37 of 40 requested symbols'),          # 92.5% < 95%
    (lagging('AAPL', 'MSFT', 'NVDA'), '3 of 40 symbols lack the latest session'),  # 7.5% > 5%
    (thin, 'only 1 ranked stocks'),
    (lambda names, start, end, **settings: {}, 'benchmark SPY was not fetched'),
])
def test_every_gate_failure_publishes_nothing(tmp_path, config, provider, fetch, message):
    provider['fetch'] = fetch
    with pytest.raises(IntegrityError, match=message):
        pipeline_run.run(tmp_path, now=NOW, config=config)
    assert list(tmp_path.iterdir()) == []


def test_cli_exit_codes(tmp_path, config, provider, monkeypatch, capsys):
    monkeypatch.setattr(pipeline_run, 'load_config', lambda: config)
    assert pipeline_run.main(['--out', str(tmp_path / 'ok')]) == 0
    assert (tmp_path / 'ok' / 'latest.json').is_file() and 'rows' in capsys.readouterr().out
    provider['fetch'] = drop('SPY')
    assert pipeline_run.main(['--out', str(tmp_path / 'bad')]) == 1
    assert 'INTEGRITY GATE FAILED' in capsys.readouterr().err and not (tmp_path / 'bad').exists()
    with pytest.raises(SystemExit):
        pipeline_run.main(['--out', str(tmp_path / 'bad'), '--limit', '0'])



def test_limit_samples_stocks_but_always_keeps_the_watchlist(tmp_path, config, provider):
    config['eligibility']['min_ranked_stocks'] = 5
    meta = pipeline_run.run(tmp_path, now=NOW, config=config, limit=9)
    requested = provider['calls'][0]['symbols']
    assert meta['sample_limit'] == 9 and meta['counts']['stocks_requested'] == 9 and meta['counts']['stocks_after_filter'] == 27
    assert len(requested) == 9 + 13 and set(config['etf_watchlist']) <= set(requested)
    assert pipeline_run.sample(list('abcdefghij'), 3) == ['a', 'd', 'g'] and pipeline_run.sample(['a'], 5) == ['a']


def test_configuration_errors_stop_the_run(tmp_path, config, provider):
    config['etf_watchlist'] = config['etf_watchlist'] + ['AAPL']       # a stock cannot also be a reference ETF
    with pytest.raises(ValueError, match='overlaps'):
        pipeline_run.run(tmp_path, now=NOW, config=config)
    bad = tmp_path / 'config.toml'
    bad.write_text(pipeline_run.CONFIG_PATH.read_text().replace('benchmark = "SPY"', 'benchmark = "VTI"'))
    with pytest.raises(ValueError, match='benchmark'):
        pipeline_run.load_config(bad)
    bad.write_text(pipeline_run.CONFIG_PATH.read_text().replace('"sma200"]', '"sma_200"]'))
    with pytest.raises(ValueError, match='unknown columns: sma_200'):
        pipeline_run.load_config(bad)
    assert pipeline_run.load_config()['output']['exclude_columns'] == PRICE_LEVELS


def test_csv_cells_and_reproducible_bytes():
    row = dict.fromkeys(universe_rank.COLUMNS)
    row.update(symbol='BRK.B', group='stock', status='ranked', bars=379, close=579.31999, rs_percentile=77,
               avg_dollar_volume_50d=1234567.89, rs_vs_benchmark=1.234567, atr14_pct=2.3449, tt_above_50=True,
               tt_150_above_200=False)
    data = snapshot.csv_bytes([row], universe_rank.COLUMNS)
    assert data == snapshot.csv_bytes([row], universe_rank.COLUMNS) and data[4:8] == b'\x00\x00\x00\x00'   # gzip mtime field is zero
    line = dict(zip(*[l.split(',') for l in gzip.GzipFile(fileobj=io.BytesIO(data)).read().decode().splitlines()]))
    assert line['close'] == '579.3200' and line['avg_dollar_volume_50d'] == '1234568' and line['rs_vs_benchmark'] == '1.2346'
    assert line['atr14_pct'] == '2.34' and line['rs_percentile'] == '77' and line['bars'] == '379'
    assert line['tt_above_50'] == 'true' and line['tt_150_above_200'] == 'false' and line['sma50'] == ''
    public = gzip.decompress(snapshot.csv_bytes([row], snapshot.published_columns(PRICE_LEVELS))).decode().splitlines()
    assert 'close' not in public[0].split(',') and '579.3200' not in public[1] and public[1].startswith('BRK.B,stock,ranked,,379,1234568,')
