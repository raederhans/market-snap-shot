"""Forward outcomes and one day's cross-sectional statistics, against numbers worked out by hand. No files, no network."""
import copy
import exchange_calendars as xcals
import pandas as pd
import pytest
from pipeline import outcomes
from pipeline.outcomes import ValidationError
from pipeline.run import load_config

SESSIONS = [stamp.date().isoformat() for stamp in xcals.get_calendar('XNYS').sessions]
SETTINGS = {**load_config()['validation'], 'min_group_size': 2}
NAN = float('nan')


@pytest.mark.parametrize('today,horizon,expected', [
    ('2026-11-27', 1, '2026-11-25'),    # Thursday the 26th is Thanksgiving
    ('2026-11-27', 5, '2026-11-19'),    # 25, 24, 23, 20, 19
    ('2026-11-30', 5, '2026-11-20'),    # 27, 25, 24, 23, 20: the half session on the 27th counts
    ('2026-11-25', 20, '2026-10-28'),   # no holiday in between: four full weeks
    ('2027-01-04', 5, '2026-12-24'),    # 31, 30, 29, 28, 24: Christmas and New Year's Day are skipped
    ('2026-11-25', 60, '2026-09-01'),   # 17 earlier sessions in November, 22 in October, 21 in September (Labor Day out)
])
def test_snapshot_day_is_counted_in_exchange_sessions(today, horizon, expected):
    assert outcomes.session_before(today, horizon, SESSIONS) == expected


def test_snapshot_day_needs_a_session_inside_the_calendar():
    with pytest.raises(ValidationError, match='not an exchange session'):
        outcomes.session_before('2026-11-26', 5, SESSIONS)     # Thanksgiving
    with pytest.raises(ValidationError, match='does not reach'):
        outcomes.session_before(SESSIONS[3], 5, SESSIONS)


CLOSES = pd.DataFrame(
    {'AAA': [100., 104., 110.], 'BBB': [50., 49., 45.], 'CCC': [NAN, 31., 30.], 'DDD': [20., 21., NAN],
     'EEE': [NAN, 7., NAN], 'SPY': [400., 402., 404.]}, index=['2026-11-19', '2026-11-20', '2026-11-27'])


def test_forward_return_and_excess_by_hand():
    rows, benchmark = outcomes.forward_outcomes(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'GONE', 'SPY'], CLOSES,
                                                '2026-11-19', '2026-11-27', 'SPY')
    by = {row['symbol']: row for row in rows}
    assert [row['symbol'] for row in rows] == ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'GONE', 'SPY'] and list(rows[0]) == list(outcomes.OUTCOME_COLUMNS)
    assert benchmark == pytest.approx(1.0)                                 # 404 / 400 - 1
    assert by['AAA']['outcome_status'] == 'ok' and by['AAA']['forward_return_pct'] == pytest.approx(10.0)
    assert by['AAA']['forward_excess_pct'] == pytest.approx(9.0)            # 10 - 1, a simple difference
    assert by['BBB']['forward_return_pct'] == pytest.approx(-10.0) and by['BBB']['forward_excess_pct'] == pytest.approx(-11.0)
    assert by['SPY']['forward_excess_pct'] == pytest.approx(0.0)
    # The middle row (104, 49 ...) is never used: only the two closes count.
    for symbol, status in (('CCC', 'missing_bar_at_D'), ('DDD', 'missing_bar_at_T'), ('EEE', 'missing_bar_at_T'), ('GONE', 'no_data_at_T')):
        assert by[symbol] == {'symbol': symbol, 'outcome_status': status, 'forward_return_pct': None, 'forward_excess_pct': None}


def test_missing_benchmark_or_session_row_stops_the_horizon():
    with pytest.raises(ValidationError, match='benchmark DDD lacks a close'):
        outcomes.forward_outcomes(['AAA'], CLOSES, '2026-11-19', '2026-11-27', 'DDD')
    with pytest.raises(ValidationError, match='benchmark QQQ lacks a close'):
        outcomes.forward_outcomes(['AAA'], CLOSES, '2026-11-19', '2026-11-27', 'QQQ')
    with pytest.raises(ValidationError, match='no row for session 2026-11-18'):
        outcomes.forward_outcomes(['AAA'], CLOSES, '2026-11-18', '2026-11-27', 'SPY')


def test_spearman_by_hand_including_ties():
    assert outcomes.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert outcomes.spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)
    # x ranks 1, 2.5, 2.5, 4 (the two 2s share a rank); y ranks 1, 3, 2, 4. Deviations from 2.5:
    # x -1.5, 0, 0, 1.5 and y -1.5, .5, -.5, 1.5 -> 4.5 / sqrt(4.5 * 5).
    assert outcomes.spearman([1, 2, 2, 3], [10, 30, 20, 40]) == pytest.approx(4.5 / (4.5 * 5) ** .5)
    assert outcomes.spearman([1, 2, 2, 3], [10, 30, 20, 40]) == pytest.approx(0.948683, abs=1e-6)
    # Ties on both sides: x ranks 1.5, 1.5, 3.5, 3.5 (deviations -1, -1, 1, 1); y ranks 1, 2.5, 2.5, 4
    # (deviations -1.5, 0, 0, 1.5) -> (1.5 + 1.5) / sqrt(4 * 4.5).
    assert outcomes.spearman([5, 5, 9, 9], [1, 2, 2, 3]) == pytest.approx(3 / 18 ** .5)
    assert outcomes.spearman([7, 7, 7], [1, 2, 3]) is None and outcomes.spearman([1, 2, 3], [4, 4, 4]) is None


def pool(rows):
    columns = ['symbol', 'rs_percentile', 'trend_template_status', 'distance_from_52w_high_pct', 'outcome_status', 'forward_excess_pct']
    return pd.DataFrame(rows, columns=columns)


# Twelve ranked stocks, ten with an outcome. Excess sums to 3.
POOL = pool([
    ('A', 95, 'passes', -2., 'ok', 6.), ('B', 85, 'passes', -5., 'ok', 2.), ('C', 80, 'fails', -10., 'ok', 4.),
    ('D', 70, 'passes', -8., 'ok', 1.), ('E', 60, 'fails', -12., 'ok', 0.), ('F', 45, 'fails', -30., 'ok', -1.),
    ('G', 40, 'fails', -20., 'ok', 3.), ('H', 25, 'fails', -40., 'ok', -2.), ('I', 10, 'fails', -50., 'ok', -4.),
    ('J', 5, 'fails', -60., 'ok', -6.),
    ('K', 90, 'passes', -1., 'no_data_at_T', NAN), ('L', 15, 'fails', -55., 'missing_bar_at_T', NAN)])


def test_cross_section_by_hand():
    out, reasons = outcomes.cross_section(POOL, SETTINGS)
    assert set(out) == set(outcomes.STAT_COLUMNS)
    assert (out['pool'], out['with_outcome'], out['missing']) == (12, 10, 2) and out['missing_pct'] == pytest.approx(100 * 2 / 12)
    assert out['pool_excess_pct'] == pytest.approx(0.3)                       # 3 / 10
    # RS ranks 10..1 for A..J; excess ranks A10 B7 C9 D6 E5 F4 G8 H3 I2 J1; squared differences sum to 24.
    assert out['rank_ic'] == pytest.approx(1 - 6 * 24 / (10 * 99))
    # Quintiles by percentile: <20 I J; 20-39 H alone; 40-59 F G; 60-79 D E; >=80 A B C (80 is in the top one).
    assert out['rs_q1_excess_pct'] == pytest.approx(-5.0) and out['rs_q2_excess_pct'] is None
    assert out['rs_q3_excess_pct'] == pytest.approx(1.0) and out['rs_q4_excess_pct'] == pytest.approx(0.5)
    assert out['rs_q5_excess_pct'] == pytest.approx(4.0) and out['rs_q5_minus_q1_pct'] == pytest.approx(9.0)
    # Template: A B D pass (K passes but has no outcome), the other seven fail.
    assert (out['template_pass_n'], out['template_fail_n']) == (3, 7)
    assert out['template_pass_excess_pct'] == pytest.approx(3.0) and out['template_fail_excess_pct'] == pytest.approx(-6 / 7)
    assert out['template_pass_minus_fail_pct'] == pytest.approx(3 + 6 / 7)
    # Within 10% of the high: A B C D (-10 is inside); the rest: E..J.
    assert (out['near_high_n'], out['not_near_high_n']) == (4, 6)
    assert out['near_high_excess_pct'] == pytest.approx(3.25) and out['not_near_high_excess_pct'] == pytest.approx(-10 / 6)
    assert out['near_high_minus_rest_pct'] == pytest.approx(3.25 + 10 / 6)
    # RS >= 80 and passing: A B (C fails the template, K has no outcome).
    assert out['combo_n'] == 2 and out['combo_excess_pct'] == pytest.approx(4.0) and out['combo_minus_pool_pct'] == pytest.approx(3.7)
    assert reasons == {'rs_q2': 'group_too_small(n=1<2)'}


def test_small_groups_are_null_with_the_reason_and_so_are_their_spreads():
    out, reasons = outcomes.cross_section(POOL, {**SETTINGS, 'min_group_size': 4})
    assert out['combo_excess_pct'] is None and out['combo_minus_pool_pct'] is None and out['combo_n'] == 2
    assert out['template_pass_excess_pct'] is None and out['template_pass_minus_fail_pct'] is None
    assert out['template_fail_excess_pct'] == pytest.approx(-6 / 7)            # the other side is still reported
    assert out['near_high_minus_rest_pct'] == pytest.approx(3.25 + 10 / 6)    # 4 and 6: both large enough
    assert out['rs_q5_minus_q1_pct'] is None and out['rank_ic'] is not None and out['pool_excess_pct'] == pytest.approx(0.3)
    assert reasons == {name: f'group_too_small(n={n}<4)' for name, n in
                       {'combo': 2, 'template_pass': 3, 'rs_q1': 2, 'rs_q2': 1, 'rs_q3': 2, 'rs_q4': 2, 'rs_q5': 3}.items()}
    out, reasons = outcomes.cross_section(POOL, {**SETTINGS, 'min_group_size': 11})   # ten outcomes are not enough for anything
    assert all(out[name] is None for name in outcomes.METRICS) and out['with_outcome'] == 10
    assert reasons['pool'] == reasons['rank_ic'] == 'group_too_small(n=10<11)'


def test_configured_thresholds_define_the_groups():
    out, _ = outcomes.cross_section(POOL, {**SETTINGS, 'near_high_within_pct': 5, 'combo_min_rs_percentile': 90,
                                           'rs_quintile_edges': [10, 45, 60, 85], 'min_group_size': 1})
    assert (out['near_high_n'], out['not_near_high_n'], out['combo_n']) == (2, 8, 1)     # A B within 5%; only A at 90 or more
    assert out['rs_q1_excess_pct'] == pytest.approx(-6.0)                     # only J is below 10
    assert out['rs_q5_excess_pct'] == pytest.approx(4.0) and out['rs_q3_excess_pct'] == pytest.approx(-1.0)   # A B; F alone


def test_unknown_features_and_a_constant_rank_are_not_guessed():
    rows = POOL.copy()
    rows.loc[rows.symbol == 'A', 'distance_from_52w_high_pct'] = NAN
    rows.loc[rows.symbol == 'B', 'trend_template_status'] = 'partial_rs_unverified'
    out, _ = outcomes.cross_section(rows, SETTINGS)
    assert (out['near_high_n'], out['not_near_high_n']) == (3, 6)             # A is in neither group
    assert (out['template_pass_n'], out['template_fail_n']) == (2, 7)         # B is neither a pass nor a fail
    flat = POOL.assign(rs_percentile=50)
    out, reasons = outcomes.cross_section(flat, SETTINGS)
    assert out['rank_ic'] is None and reasons['rank_ic'] == 'constant_input'
    with pytest.raises(ValidationError, match='no ranked stocks'):
        outcomes.cross_section(POOL.iloc[:0], SETTINGS)
    nothing = POOL.assign(outcome_status='no_data_at_T', forward_excess_pct=NAN)   # every outcome missing
    out, reasons = outcomes.cross_section(nothing, SETTINGS)
    assert out['missing_pct'] == 100 and out['with_outcome'] == 0 and all(out[name] is None for name in outcomes.METRICS)


@pytest.mark.parametrize('change,message', [
    ({'horizons': []}, 'horizons'), ({'horizons': [5, 5]}, 'horizons'), ({'horizons': [0, 5]}, 'horizons'),
    ({'horizons': [5.0]}, 'horizons'), ({'rs_quintile_edges': [20, 40, 60]}, 'rs_quintile_edges'),
    ({'rs_quintile_edges': [20, 60, 40, 80]}, 'rs_quintile_edges'), ({'rs_quintile_edges': [20, 20, 40, 80]}, 'rs_quintile_edges'),
    ({'min_group_size': 1}, 'at least 2'), ({'min_non_overlapping_periods': 1}, 'at least 2'),
])
def test_unusable_settings_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        outcomes.check_settings({**copy.deepcopy(SETTINGS), **change})
    config = load_config()['validation']
    assert config == {'horizons': [5, 20, 60], 'rs_quintile_edges': [20, 40, 60, 80], 'near_high_within_pct': 10,
                      'combo_min_rs_percentile': 80, 'min_group_size': 20, 'min_non_overlapping_periods': 12}
