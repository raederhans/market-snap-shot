"""backfill_features against the production ranking, and its point-in-time guarantees. Synthetic bars, no network."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from pipeline import backfill_features as features, gates, prices
from pipeline.gates import IntegrityError
from stock_research import universe_rank

SESSIONS = gates.completed_sessions(pd.Timestamp('2026-11-26T23:30:00Z')).iloc[-420:]
FLOORS = {'min_price': 5.0, 'min_avg_dollar_volume_50d': 5_000_000, 'min_bars': 253, 'min_ranked_stocks': 5}
WINDOW_DAYS = 548
CHECKED = ('close', 'avg_dollar_volume_50d', 'rs_strength_pct', 'rs_percentile', 'trend_template_status',
           'distance_from_52w_high_pct', *('tt_' + name for name in universe_rank.TEMPLATE_CHECKS))


def walk(rng, n, start=50., drift=0., volume=2_000_000.):
    close = start * np.exp(np.cumsum(rng.normal(drift, .02, n)))
    spread = np.abs(rng.normal(0, .01, n))
    return pd.DataFrame({'open': close, 'high': close * (1 + spread), 'low': close * (1 - spread), 'close': close,
                         'volume': volume * rng.uniform(.5, 1.5, n)}, index=SESSIONS.index[-n:])


def universe():
    """Forty ordinary stocks with different drifts plus one of every case the daily job treats specially."""
    rng, n = np.random.default_rng(7), len(SESSIONS)
    raw = {f'S{k:02d}': walk(rng, n, drift=(k - 20) / 8000) for k in range(40)}
    raw['SPY'] = walk(rng, n, 400., .0003, 5e7)
    raw['CHEAP'] = walk(rng, n, 2.)                           # below_min_price
    raw['THIN'] = walk(rng, n, volume=20_000.)                # below_min_dollar_volume
    raw['YOUNG'] = walk(rng, 300)                             # insufficient_bars on early days, ranked later
    raw['GAP'] = walk(rng, n).drop(SESSIONS.index[330])       # history_gaps while session 330 is in the window
    raw['GONE'] = walk(rng, n).iloc[:380]                     # stale after session 379
    raw['FLAT'] = walk(rng, n).assign(open=10.01, high=10.01, low=10.01, close=10.01)   # every comparison is a tie
    return {symbol: prices.session_bars(d, SESSIONS) for symbol, d in raw.items()}


@pytest.fixture(scope='module')
def bars():
    return universe()


@pytest.fixture(scope='module')
def panel(bars):
    return features.build_panel(bars, 'SPY', SESSIONS)


def production(bars, at):
    """rank_universe on bars cut off at session `at`, inside the daily job's download window."""
    day = SESSIONS.iloc[at]
    start = (SESSIONS.index[at] - pd.Timedelta(days=WINDOW_DAYS)).tz_localize('UTC')
    cut = {symbol: d[(d.index >= start) & (d.index <= day)] for symbol, d in bars.items()}
    cut = {symbol: d for symbol, d in cut.items() if len(d)}
    groups = {symbol: universe_rank.ETF if symbol == 'SPY' else universe_rank.STOCK for symbol in cut}
    rows = universe_rank.rank_universe(cut, cut['SPY'], groups, min_price=FLOORS['min_price'], min_bars=FLOORS['min_bars'],
                                       min_avg_dollar_volume=FLOORS['min_avg_dollar_volume_50d'], min_population=FLOORS['min_ranked_stocks'])
    return {row['symbol']: row for row in rows}


@pytest.mark.parametrize('at', [252, 300, 331, 379, 380, 419])
def test_every_value_equals_the_production_ranking_on_bars_cut_off_at_that_session(bars, panel, at):
    expected = production(bars, at)
    pool = features.point_in_time(panel, at, FLOORS, WINDOW_DAYS)
    ranked = sorted(symbol for symbol, row in expected.items() if row['status'] == universe_rank.RANKED)
    assert list(pool.symbol) == ranked and len(ranked) >= 40          # the same pool, in symbol order
    for row in pool.to_dict('records'):
        for name in CHECKED:                                           # exact equality: floats to the last bit
            assert row[name] == expected[row['symbol']][name], (row['symbol'], name)
    statuses = {symbol: expected[symbol]['status'] for symbol in ('CHEAP', 'THIN', 'YOUNG', 'GAP', 'GONE') if symbol in expected}
    assert statuses['CHEAP'] == 'below_min_price' and statuses['THIN'] == 'below_min_dollar_volume'
    assert statuses['YOUNG'] == ('insufficient_bars' if at < 372 else 'ranked')
    assert statuses['GAP'] == ('history_gaps' if 330 <= at <= 582 else 'ranked')
    assert statuses['GONE'] == ('ranked' if at <= 379 else 'stale')


def test_a_longer_min_bars_counts_bars_inside_the_production_window(bars, panel):
    floors = {**FLOORS, 'min_bars': 300}
    expected = universe_rank.rank_universe(
        {s: d[d.index >= (SESSIONS.index[419] - pd.Timedelta(days=WINDOW_DAYS)).tz_localize('UTC')] for s, d in bars.items()}, bars['SPY'],
        {s: 'stock' for s in bars}, min_price=5., min_avg_dollar_volume=5e6, min_bars=300, min_population=5)
    ranked = sorted(row['symbol'] for row in expected if row['status'] == 'ranked' and row['symbol'] != 'SPY')
    assert list(features.point_in_time(panel, 419, floors, WINDOW_DAYS).symbol) == ranked
    assert 'YOUNG' not in features.point_in_time(panel, 419, {**FLOORS, 'min_bars': 301}, WINDOW_DAYS).symbol.values   # it has 300 bars


def test_prices_after_the_evaluation_session_cannot_change_its_features_or_tiers(panel):
    at = 330
    changed = {field: getattr(panel, field).copy() for field in features.FIELDS}
    for matrix in changed.values():                       # every later bar rewritten, some removed
        matrix[at + 1:] *= 3.7
        matrix[at + 5:, ::2] = np.nan
    later = dataclasses.replace(panel, benchmark_close=np.r_[panel.benchmark_close[:at + 1], panel.benchmark_close[at + 1:] / 9], **changed)
    before, after = (features.point_in_time(p, at, FLOORS, WINDOW_DAYS) for p in (panel, later))
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert list(features.top_dollar_volume(before, 10).symbol) == list(features.top_dollar_volume(after, 10).symbol)
    assert features.benchmark_trend(panel, at, 200) == features.benchmark_trend(later, at, 200)
    assert not features.point_in_time(later, at + 6, FLOORS, WINDOW_DAYS).equals(features.point_in_time(panel, at + 6, FLOORS, WINDOW_DAYS))
    with pytest.raises(ValueError, match='fewer than 253'):
        features.point_in_time(panel, 251, FLOORS, WINDOW_DAYS)   # never a shorter window


def test_top_dollar_volume_tier():
    pool = pd.DataFrame({'symbol': list('ABCDE'), 'avg_dollar_volume_50d': [5., 9., 9., 1., 7.]})
    assert list(features.top_dollar_volume(pool, 3).symbol) == ['B', 'C', 'E']       # ties keep symbol order
    assert list(features.top_dollar_volume(pool, 2).symbol) == ['B', 'C']
    assert list(features.top_dollar_volume(pool, 50).symbol) == ['B', 'C', 'E', 'A', 'D']   # a smaller pool: all of it


def test_benchmark_trend_needs_the_full_average(panel):
    closes = np.r_[np.full(199, 100.), 101., 100.25, 90.]
    flat = dataclasses.replace(panel, benchmark_close=closes)
    assert features.benchmark_trend(flat, 198, 200) is None           # 199 closes: no shorter average is used
    assert features.benchmark_trend(flat, 199, 200) == 'above'        # 101 against 100.005
    assert features.benchmark_trend(flat, 200, 200) == 'above'        # 100.25 against 100.00625
    assert features.benchmark_trend(flat, 201, 200) == 'below'
    assert features.benchmark_trend(dataclasses.replace(panel, benchmark_close=np.full(250, 100.)), 249, 200) == 'below'   # at the average


def test_panel_rows_are_the_benchmark_sessions(bars, panel):
    assert panel.sessions == [stamp.date().isoformat() for stamp in SESSIONS.index] and 'SPY' not in panel.symbols
    assert panel.symbols == sorted(set(bars) - {'SPY'}) and panel.close.shape == (420, len(bars) - 1)
    gap = panel.symbols.index('GAP')
    assert np.isnan(panel.close[330, gap]) and panel.close[329, gap] == bars['GAP'].close.iloc[329]
    assert np.isnan(panel.close[:120, panel.symbols.index('YOUNG')]).all() and np.isnan(panel.volume[380:, panel.symbols.index('GONE')]).all()
    assert (panel.benchmark_close == bars['SPY'].close.to_numpy()).all()
    holed = {**bars, 'SPY': bars['SPY'].drop(bars['SPY'].index[100])}
    with pytest.raises(IntegrityError, match='419 bars for 420 XNYS sessions'):
        features.build_panel(holed, 'SPY', SESSIONS)
    short = features.build_panel({**bars, 'SPY': bars['SPY'].iloc[50:]}, 'SPY', SESSIONS)   # stock bars before the benchmark's first: outside
    assert len(short.sessions) == 370 and short.close[0, short.symbols.index('S00')] == bars['S00'].close.iloc[50]
