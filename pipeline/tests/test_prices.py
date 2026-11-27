"""Price fetch and bar validation with a mocked provider. No network."""
import numpy as np
import pandas as pd
import pytest
from pipeline import gates, prices
from stock_research.core import DataQualityError

DATES = pd.DatetimeIndex(['2025-11-25', '2025-11-26', '2025-11-28'])  # 27th is Thanksgiving
SETTINGS = dict(batch_size=2, max_attempts=3, backoff_seconds=20, pause_seconds=1.0)


def ohlcv(close=10.0, dates=DATES):
    c = np.full(len(dates), close)
    return pd.DataFrame({'Open': c, 'High': c * 1.01, 'Low': c * .99, 'Close': c, 'Volume': 1000.}, index=dates)


def reply(frames):
    """Shape of yf.download(group_by='ticker'): columns (Ticker, Price), rows padded to all dates."""
    return pd.concat(frames, axis=1, keys=list(frames), names=['Ticker', 'Price'])


class Provider:
    """Scripted stand-in for yf.download: one entry per call, an Exception entry is raised."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def __call__(self, tickers, start, end):
        self.calls.append((tuple(tickers), start, end))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def fetch(provider, symbols, **overrides):
    waits = []
    out = prices.fetch_yahoo(symbols, '2025-11-01', '2025-11-29', download=provider, sleep=waits.append,
                             **{**SETTINGS, **overrides})
    return out, waits


def test_batches_symbol_spelling_and_missing_symbols():
    empty = ohlcv().assign(Open=np.nan, High=np.nan, Low=np.nan, Close=np.nan, Volume=np.nan)
    provider = Provider(reply({'AAPL': ohlcv(), 'BRK-B': ohlcv(500.)}), reply({'MU': ohlcv(90.), 'NOPE': empty}),
                        reply({'SPY': ohlcv(700.)}))
    out, waits = fetch(provider, ['AAPL', 'BRK.B', 'MU', 'NOPE', 'SPY'])
    assert [c[0] for c in provider.calls] == [('AAPL', 'BRK-B'), ('MU', 'NOPE'), ('SPY',)]
    assert provider.calls[0][1:] == ('2025-11-01', '2025-11-29')
    assert set(out) == {'AAPL', 'BRK.B', 'MU', 'SPY'}          # directory spelling back, all-NaN symbol absent
    assert list(out['BRK.B'].columns) == prices.FIELDS and out['BRK.B'].close.iloc[-1] == 500.
    assert waits == [1.0, 1.0]                                 # pause between batches only, none after the last


def test_padded_rows_of_a_shorter_history_are_dropped():
    short = ohlcv(dates=DATES[1:]).reindex(DATES)              # no bar on the first date
    out, _ = fetch(Provider(reply({'AAPL': ohlcv(), 'IPO': short})), ['AAPL', 'IPO'])
    assert len(out['AAPL']) == 3 and list(out['IPO'].index) == list(DATES[1:])


def test_bar_with_a_null_close_is_dropped_not_filled():
    unsettled = ohlcv()
    unsettled.iloc[-1, unsettled.columns.get_loc('Close')] = np.nan   # seen live: volume present, close null
    out, _ = fetch(Provider(reply({'AAPL': ohlcv(), 'SNDK': unsettled})), ['AAPL', 'SNDK'])
    assert list(out['SNDK'].index) == list(DATES[:-1]) and not out['SNDK'].isna().any().any()   # stale downstream


def test_partial_batch_is_not_retried():
    provider = Provider(reply({'AAPL': ohlcv()}))
    out, waits = fetch(provider, ['AAPL', 'NOPE'])
    assert set(out) == {'AAPL'} and len(provider.calls) == 1 and waits == []


def test_empty_batch_is_retried_with_doubling_backoff():
    provider = Provider(ConnectionError('reset'), pd.DataFrame(), reply({'AAPL': ohlcv(), 'MU': ohlcv()}))
    out, waits = fetch(provider, ['AAPL', 'MU'])
    assert set(out) == {'AAPL', 'MU'} and len(provider.calls) == 3 and waits == [20, 40]


def test_retries_are_bounded_and_the_run_moves_on():
    provider = Provider(RuntimeError('429'), RuntimeError('429'), RuntimeError('429'), reply({'SPY': ohlcv()}))
    out, waits = fetch(provider, ['AAPL', 'MU', 'SPY'])
    assert set(out) == {'SPY'} and len(provider.calls) == 4
    assert waits == [20, 40, 1.0]                              # two backoffs, then the normal pause


def sessions():
    return gates.completed_sessions('2025-12-01T00:00:00Z')


def lower(frame):
    return frame.rename(columns=str.lower)


def test_session_bars_stamps_exchange_closes_including_the_early_close():
    d = prices.session_bars(lower(ohlcv()), sessions())
    assert [t.isoformat() for t in d.index] == ['2025-11-25T21:00:00+00:00', '2025-11-26T21:00:00+00:00',
                                               '2025-11-28T18:00:00+00:00']  # day after Thanksgiving closes 13:00 ET
    assert list(d.columns) == prices.FIELDS and d.close.iloc[-1] == 10.0
    assert len(prices.session_bars(lower(ohlcv()).iloc[-1:], sessions())) == 1  # one bar is valid, just short


@pytest.mark.parametrize('dates', [
    pd.DatetimeIndex(['2025-11-26', '2025-11-27']),   # holiday
    pd.DatetimeIndex(['2025-11-28', '2025-11-29']),   # Saturday
    pd.DatetimeIndex(['2025-11-28', '2025-12-01']),   # session not yet completed at the cutoff
])
def test_session_bars_rejects_bars_outside_completed_sessions(dates):
    with pytest.raises(DataQualityError):
        prices.session_bars(lower(ohlcv(dates=dates)), sessions())


@pytest.mark.parametrize('column,value', [('volume', np.nan), ('close', np.nan), ('close', -1.0), ('volume', -5.0)])
def test_session_bars_rejects_missing_or_impossible_values(column, value):
    raw = lower(ohlcv())
    raw.iloc[1, raw.columns.get_loc(column)] = value
    with pytest.raises(DataQualityError):
        prices.session_bars(raw, sessions())


def test_adjustment_float_noise_is_snapped_but_real_inversions_are_rejected():
    raw = lower(ohlcv())
    raw['low'] = raw['close']
    raw.iloc[1, raw.columns.get_loc('low')] = 10.0 * (1 + 2e-16) + 2e-15   # low a few ulps above the close
    raw['high'] = raw['close']
    raw.iloc[2, raw.columns.get_loc('high')] = 10.0 - 2e-15                # high a few ulps below the close
    assert raw.low.iloc[1] > raw.close.iloc[1] and raw.high.iloc[2] < raw.close.iloc[2]
    d = prices.session_bars(raw, sessions())
    assert d.low.iloc[1] == d.close.iloc[1] and d.high.iloc[2] == d.close.iloc[2]
    raw.iloc[1, raw.columns.get_loc('low')] = 10.01                        # a cent above the close is bad data
    with pytest.raises(DataQualityError):
        prices.session_bars(raw, sessions())
