"""Data contracts and OHLCV validation. All historical inputs must be source-dated."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
import math
import pandas as pd
import numpy as np


class DataQualityError(ValueError):
    pass


@dataclass(frozen=True)
class Provenance:
    provider: str
    symbol: str
    venue: str
    currency: str
    interval: str
    fetched_at: str
    last_observation: str
    adjustment: str
    session: str = 'regular'
    url: Optional[str] = None

    def validate(self) -> None:
        if self.adjustment not in {'adjusted', 'unadjusted', 'unknown'}:
            raise DataQualityError('adjustment must be adjusted/unadjusted/unknown')
        if self.session not in {'regular', 'extended', 'mixed', 'unknown'}:
            raise DataQualityError('unknown trading session')
        if not self.provider or not self.symbol or not self.venue or not self.currency:
            raise DataQualityError('provider, symbol, venue, currency required')
        for t in (self.fetched_at, self.last_observation):
            v = datetime.fromisoformat(t.replace('Z', '+00:00'))
            if v.tzinfo is None:
                raise DataQualityError('timestamps require timezone')


def require_ohlcv(data: pd.DataFrame, min_rows: int = 2) -> pd.DataFrame:
    """Validate, never silently fabricate/forward-fill missing candles."""
    if data is None or len(data) < min_rows:
        raise DataQualityError(f'need at least {min_rows} observations')
    df = data.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    required = {'open', 'high', 'low', 'close', 'volume'}
    if missing := required - set(df.columns):
        raise DataQualityError(f'missing OHLCV fields: {sorted(missing)}')
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQualityError('DatetimeIndex required')
    if df.index.tz is None:
        raise DataQualityError('Timezone-aware timestamps required')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise DataQualityError('timestamps must be strictly increasing without duplicates')
    for name in df.columns:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise DataQualityError('OHLCV contains missing or infinite values')
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any() or (df.volume < 0).any():
        raise DataQualityError('negative volume or non-positive price')
    if (df.high < df[['open', 'close', 'low']].max(axis=1)).any() or (df.low > df[['open', 'close', 'high']].min(axis=1)).any():
        raise DataQualityError('OHLC price consistency violated')
    return df


def finite_or_none(x):
    if x is None:
        return None
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except (ValueError, TypeError):
        return None


def pct_change(last, previous):
    if previous is None or not np.isfinite(previous) or previous == 0:
        return None
    return finite_or_none(last / previous - 1)

