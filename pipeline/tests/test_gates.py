"""Every integrity gate, pass and fail. No network."""
import pandas as pd
import pytest
from pipeline import gates
from pipeline.gates import IntegrityError


@pytest.mark.parametrize('now,expected', [
    ('2026-11-25T20:59:00Z', '2026-11-24'),   # one minute before the close: the session is not complete
    ('2026-11-25T21:00:00Z', '2026-11-25'),
    ('2026-11-26T23:30:00Z', '2026-11-25'),   # Thanksgiving: still Wednesday's session
    ('2026-11-27T18:00:00Z', '2026-11-27'),   # early close at 13:00 New York
    ('2026-11-29T12:00:00Z', '2026-11-27'),   # weekend
    ('2026-07-06T23:30:00Z', '2026-07-06'),   # summer time: close at 20:00 UTC
])
def test_latest_completed_session(now, expected):
    sessions = gates.completed_sessions(now)
    assert sessions.index[-1] == pd.Timestamp(expected)
    assert sessions.iloc[-1] <= pd.Timestamp(now)


def bars(last='2026-11-25', n=300):
    closes = gates.completed_sessions(last + 'T23:00:00Z').iloc[-n:]
    return pd.DataFrame({'close': 1.0}, index=pd.DatetimeIndex(closes))


CLOSE = pd.Timestamp('2026-11-25T21:00:00Z')


def test_benchmark_gate_passes_on_a_current_full_history():
    gates.check_benchmark({'SPY': bars()}, 'SPY', CLOSE)


@pytest.mark.parametrize('universe,message', [
    ({}, 'not fetched'),
    ({'QQQ': bars()}, 'not fetched'),
    ({'SPY': bars(last='2026-11-24')}, 'not the latest completed'),
    ({'SPY': bars(n=252)}, 'needs 253'),
])
def test_benchmark_gate_failures(universe, message):
    with pytest.raises(IntegrityError, match=message):
        gates.check_benchmark(universe, 'SPY', CLOSE)


def test_coverage_gate():
    assert gates.check_coverage(1000, 950, .95) == pytest.approx(.95)   # exactly at the floor passes
    with pytest.raises(IntegrityError, match='949 of 1000'):
        gates.check_coverage(1000, 949, .95)
    with pytest.raises(IntegrityError):
        gates.check_coverage(0, 0, .95)                                 # nothing requested is not a success


def rows(**counts):
    return [{'status': status} for status, n in counts.items() for _ in range(n)]


def test_stale_gate():
    assert gates.check_stale(rows(ranked=95, stale=5), .05) == pytest.approx(.05)
    with pytest.raises(IntegrityError, match='6 of 100'):
        gates.check_stale(rows(ranked=94, stale=6), .05)
    with pytest.raises(IntegrityError):
        gates.check_stale([], .05)


def test_population_gate():
    assert gates.check_population(rows(ranked=100, reference=13, below_min_price=40), 100) == 100
    with pytest.raises(IntegrityError, match='only 99 ranked'):
        gates.check_population(rows(ranked=99, reference=13, below_min_price=400), 100)
