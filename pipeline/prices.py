"""Daily OHLCV for the whole universe.

One provider (Yahoo Finance through yfinance, unofficial), pulled in full on every
run: no price cache, no second source, no substitute when a symbol is missing.
"""
from __future__ import annotations
import logging
import os
import pathlib
import sys
import time
import pandas as pd
from stock_research.core import require_ohlcv, DataQualityError
from .universe import yahoo_symbol

PROVIDER = 'Yahoo Finance via yfinance (unofficial; split- and dividend-adjusted, auto_adjust=True)'
FIELDS = ['open', 'high', 'low', 'close', 'volume']
FLOAT_NOISE = 1e-9  # relative; adjustment arithmetic, far below one cent on any listed price


def _yf_download(tickers, start, end):
    import yfinance as yf
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)  # per-symbol misses are counted, not logged
    return yf.download(tickers, start=start, end=end, interval='1d', auto_adjust=True, actions=False,
                       group_by='ticker', threads=True, progress=False)


def _batch(download, batch, start, end):
    """One request for up to batch_size symbols -> {symbol: frame}; an error counts as nothing."""
    try:
        frame = download([yahoo_symbol(s) for s in batch], start, end)
    except Exception as exc:  # network, rate limit or an all-empty reply: retried by the caller
        print(f'batch starting {batch[0]} failed: {type(exc).__name__}', file=sys.stderr)
        return {}
    out = {}
    if frame is None or frame.empty:
        return out
    present = set(frame.columns.get_level_values(0))
    for symbol in batch:
        y = yahoo_symbol(symbol)
        if y not in present:
            continue
        d = frame[y].rename(columns=str.lower)
        if not set(FIELDS) <= set(d.columns):
            continue
        # Rows padded to other symbols' dates are empty, and Yahoo can return a session with a
        # null close while it is unsettled. Neither is a bar: dropping them leaves the symbol
        # stale or gapped downstream. Nothing is filled in.
        d = d[FIELDS].dropna()
        if len(d):
            out[symbol] = d
    return out


def fetch_yahoo(symbols, start, end, *, batch_size, max_attempts, backoff_seconds, pause_seconds,
                download=_yf_download, sleep=time.sleep):
    """symbols, start, end (exclusive) -> {symbol: OHLCV frame indexed by session date}.

    Symbols use the directory spelling (BRK.B). A symbol Yahoo has nothing for is
    absent from the result. A batch that returns nothing at all is treated as a
    transient failure and retried, at most max_attempts times with doubling backoff.
    """
    out = {}
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    for number, batch in enumerate(batches, 1):
        for attempt in range(1, max_attempts + 1):
            got = _batch(download, batch, str(start), str(end))
            if got or attempt == max_attempts:
                break
            sleep(backoff_seconds * 2 ** (attempt - 1))
        out.update(got)
        if number % 10 == 0 or number == len(batches):
            print(f'prices: batch {number}/{len(batches)}, {len(out)} symbols with data', file=sys.stderr)
        if number < len(batches):
            sleep(pause_seconds)
    return out


def session_bars(raw, session_closes):
    """Stamp daily bars with their XNYS session close (UTC) and validate them.

    session_closes maps completed session dates to close times. A bar on any
    other date, or OHLCV that core.require_ohlcv rejects, raises DataQualityError.
    """
    dates = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    if not dates.isin(session_closes.index).all():
        raise DataQualityError('bar dated outside completed XNYS sessions')
    d = raw.astype(float)
    d.index = pd.DatetimeIndex(session_closes.loc[dates])
    return require_ohlcv(_snap_float_noise(d), min_rows=1)


def _snap_float_noise(d):
    """Undo one-ulp OHLC inversions created by price adjustment.

    auto_adjust scales each field separately, so a high (low) that equalled the
    open or close can land a float step below (above) it and fail validation.
    Only differences within FLOAT_NOISE are snapped; real inconsistencies still raise.
    """
    top, bottom = d[['open', 'close']].max(axis=1), d[['open', 'close']].min(axis=1)
    d['high'] = d.high.mask((d.high < top) & (top - d.high <= FLOAT_NOISE * top), top)
    d['low'] = d.low.mask((d.low > bottom) & (d.low - bottom <= FLOAT_NOISE * bottom), bottom)
    return d


def write_closes(path, bars):
    """Adjusted closes of this run as one gzip CSV: a row per session date, a column per symbol with usable bars.

    Scratch input for pipeline.validate, kept outside the published directory. An empty
    cell means the symbol has no bar that session. Written under a temporary name and
    renamed, so a reader never sees half a file.
    """
    frame = pd.concat({symbol: d.close for symbol, d in bars.items()}, axis=1).sort_index()
    frame.index = [stamp.tz_convert('America/New_York').date().isoformat() for stamp in frame.index]
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    frame.to_csv(partial, index_label='session', compression='gzip')   # full float repr: read back exactly
    os.replace(partial, path)


def read_closes(path):
    """The matrix written by write_closes: ISO session dates ascending, one column per symbol, NaN = no bar."""
    return pd.read_csv(path, index_col='session', compression='gzip', float_precision='round_trip')   # the default parser is not exact
