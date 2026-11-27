"""universe_rank: local arithmetic only, no network. Expected numbers are hand-computed in comments."""
import numpy as np
import pandas as pd
import pytest
from stock_research import technical_intelligence as ti, universe_rank as ur
from stock_research.core import DataQualityError
from test_technical_intelligence import frame

N = 300
FLOORS = dict(min_price=5.0, min_avg_dollar_volume=50_000, min_bars=253, min_population=1)


def stepped(last, back63, back126, back189, back252, back21=None, base=100.0, n=N):
    """Flat series with chosen closes at the exact look-back sessions the formula reads."""
    c = np.full(n, base)
    c[-1], c[-64], c[-127], c[-190], c[-253] = last, back63, back126, back189, back252
    if back21 is not None:
        c[-22] = back21
    return c


# R63 = 120/100-1 = .20, R126 = 120/80-1 = .50, R189 = 120/150-1 = -.20, R252 = 120/60-1 = 1.00
# strength = .4*.20 + .2*.50 + .2*(-.20) + .2*1.00 = .08 + .10 - .04 + .20 = .34
LEADER = stepped(120, 100, 80, 150, 60, back21=90)
# Every window is 110/100-1 = .10, so strength = .10
BENCHMARK = stepped(110, 100, 100, 100, 100)


def test_trailing_return_and_strength_by_hand():
    close = pd.Series(LEADER)
    assert ur.trailing_return(close, 63) == pytest.approx(.20)
    assert ur.trailing_return(close, 252) == pytest.approx(1.00)
    assert ur.rs_strength(close) == pytest.approx(.34)
    assert ur.rs_strength(pd.Series(BENCHMARK)) == pytest.approx(.10)
    # 12-1 momentum: close 21 sessions ago over close 252 sessions ago = 90/60-1
    assert ur.momentum_12_1(close) == pytest.approx(.50)


def test_short_history_gives_none_not_a_shorter_window():
    close = pd.Series(LEADER)
    assert ur.trailing_return(close.iloc[-64:], 63) == pytest.approx(.20)
    assert ur.trailing_return(close.iloc[-63:], 63) is None
    assert ur.rs_strength(close.iloc[-252:]) is None  # R252 needs 253 closes
    assert ur.momentum_12_1(close.iloc[-252:]) is None


def test_percentile_mid_rank_ties_and_clamp():
    population = np.array([-10., 0., 0., 30.])
    assert ur.percentile(-10., population) == 13  # 100*(0+.5)/4 = 12.5, rounds half up
    assert ur.percentile(0., population) == 50    # 100*(1+.5*2)/4: both tied names share it
    assert ur.percentile(30., population) == 88   # 100*(3+.5)/4 = 87.5
    assert ur.percentile(10., population) == 75   # outside value: three of four are below
    assert ur.percentile(999., population) == 99 and ur.percentile(-999., population) == 1
    assert ur.percentile(5., np.array([5.])) == 50
    # 200 distinct values use the whole 1..99 scale and never 0 or 100
    wide = np.arange(200.)
    assert [ur.percentile(v, wide) for v in (0., 99., 199.)] == [1, 50, 99]


def universe():
    full = {'LEAD': LEADER, 'FLAT': np.full(N, 100.), 'TWIN': np.full(N, 100.),
            'LAG': stepped(80, 100, 100, 100, 100),  # every window 80/100-1 = -.20
            'PENNY': np.full(N, 3.), 'ETFX': stepped(121, 100, 100, 100, 100)}  # ETF strength .21
    bars = {s: frame(c) for s, c in full.items()}
    bars['THIN'] = frame(np.full(N, 100.), volume=np.full(N, 100.))  # $10k a day
    bars['SPY'] = frame(BENCHMARK)
    bars['YOUNG'] = frame(np.full(N, 100.)).iloc[-200:]
    bars['STALE'] = frame(np.full(N, 100.)).iloc[:-1]
    bars['GAPPY'] = frame(np.full(N, 100.)).drop(index=bars['SPY'].index[-10])
    groups = {s: ur.ETF if s in ('SPY', 'ETFX') else ur.STOCK for s in bars}
    return bars, groups


def ranked(**overrides):
    bars, groups = universe()
    rows = ur.rank_universe(bars, bars['SPY'], groups, **{**FLOORS, **overrides})
    return {r['symbol']: r for r in rows}


def test_rank_universe_statuses_and_hand_percentiles():
    rows = ranked()
    assert list(rows) == sorted(rows) and all(tuple(r) == ur.COLUMNS for r in rows.values())
    assert {s: r['status'] for s, r in rows.items()} == {
        'LEAD': 'ranked', 'FLAT': 'ranked', 'TWIN': 'ranked', 'LAG': 'ranked', 'PENNY': 'below_min_price',
        'THIN': 'below_min_dollar_volume', 'YOUNG': 'insufficient_bars', 'STALE': 'stale', 'GAPPY': 'history_gaps',
        'SPY': 'reference', 'ETFX': 'reference'}
    lead = rows['LEAD']
    assert lead['rs_strength_pct'] == pytest.approx(34.0)
    assert lead['rs_vs_benchmark'] == pytest.approx(1.34 / 1.10)
    assert lead['return_126d_pct'] == pytest.approx(50.0) and lead['momentum_12_1_pct'] == pytest.approx(50.0)
    # Ranked pool strengths: LAG -20, FLAT 0, TWIN 0, LEAD 34 (N = 4)
    assert rows['LAG']['rs_percentile'] == 13             # 100*(0+.5)/4 = 12.5
    assert rows['FLAT']['rs_percentile'] == rows['TWIN']['rs_percentile'] == 50  # tie: 100*(1+1)/4
    assert lead['rs_percentile'] == 88                     # 100*(3+.5)/4 = 87.5
    # ETFs are placed on the stock distribution without joining it
    assert rows['SPY']['rs_percentile'] == 75              # strength 10: three of four stocks below
    assert rows['ETFX']['rs_percentile'] == 75             # strength 21: still between FLAT and LEAD
    assert rows['SPY']['rs_vs_benchmark'] == pytest.approx(1.0)
    for symbol in ('PENNY', 'THIN', 'YOUNG', 'STALE', 'GAPPY'):
        assert rows[symbol]['rs_percentile'] is None


def test_flat_stock_features_by_hand():
    flat = ranked()['FLAT']
    assert flat['close'] == 100.0 and flat['bars'] == N
    assert flat['avg_dollar_volume_50d'] == pytest.approx(100_000)  # 100 * 1000 shares
    assert flat['atr14_pct'] == pytest.approx(1.0)                  # range 100.5-99.5 = 1 on a 100 close
    assert flat['distance_from_52w_high_pct'] == pytest.approx(100 * (100 / 100.5 - 1))
    assert flat['distance_from_52w_low_pct'] == pytest.approx(100 * (100 / 99.5 - 1))
    assert flat['sma50'] == flat['sma200'] == pytest.approx(100.0)
    assert flat['trend_template_status'] == 'fails' and flat['tt_rs_percentile_at_least_70'] is False


def test_template_gets_the_real_percentile():
    rows = ranked()
    assert rows['LEAD']['tt_rs_percentile_at_least_70'] is True   # 88
    assert rows['SPY']['tt_rs_percentile_at_least_70'] is True    # 75, reference percentile
    assert rows['PENNY']['tt_rs_percentile_at_least_70'] is None  # not ranked: stays unknown
    bars, _ = universe()
    expected = ti.template(bars['LEAD'], rs_percentile=88)
    assert rows['LEAD']['trend_template_status'] == expected['status']
    assert rows['LEAD']['trend_template_passed'] == expected['passed']
    assert tuple(expected['checks']) == ur.TEMPLATE_CHECKS
    assert all(rows['LEAD']['tt_' + k] == v for k, v in expected['checks'].items())


def test_short_stale_and_gapped_rows_stay_null_with_a_reason():
    rows = ranked()
    young = rows['YOUNG']  # 200 bars: 63- and 126-session returns exist, the rest cannot
    assert young['return_63d_pct'] == pytest.approx(0.0) and young['return_126d_pct'] == pytest.approx(0.0)
    assert young['return_252d_pct'] is None and young['rs_strength_pct'] is None and young['momentum_12_1_pct'] is None
    assert young['trend_template_status'] == 'insufficient_history' and young['sma200'] is None
    assert young['tt_above_50'] is None
    bars, _ = universe()
    for symbol in ('STALE', 'GAPPY'):
        row = rows[symbol]
        assert all(row[c] is None for c in ur.COLUMNS[ur.COLUMNS.index('close'):])
    assert rows['STALE']['last_bar_date'] == bars['SPY'].index[-2].tz_convert('America/New_York').date().isoformat()
    assert rows['GAPPY']['last_bar_date'] == rows['SPY']['last_bar_date']


def test_small_population_leaves_every_percentile_unknown():
    rows = ranked(min_population=5)  # only four ranked stocks
    assert all(r['rs_percentile'] is None for r in rows.values())
    assert rows['LEAD']['status'] == 'ranked' and rows['LEAD']['tt_rs_percentile_at_least_70'] is None
    assert rows['LEAD']['trend_template_status'] in ('fails', 'partial_rs_unverified')


def test_rejects_unusable_settings_and_benchmark():
    bars, groups = universe()
    with pytest.raises(ValueError):
        ur.rank_universe(bars, bars['SPY'], groups, **{**FLOORS, 'min_bars': 252})
    with pytest.raises(DataQualityError):
        ur.rank_universe(bars, bars['SPY'].iloc[-252:], groups, **FLOORS)
