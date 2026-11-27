"""Common-stock universe from the Nasdaq Trader symbol directory.

nasdaqtraded.txt lists every US-listed security but has no "common stock" flag,
so warrants, units, rights, preferreds and debt are removed by the name rules in
config.toml. This module only applies those rules; it holds none of its own.
"""
from __future__ import annotations
import csv
import io
import re
import time
import urllib.request
from collections import Counter

TRAILER = 'File Creation Time'


def download_directory(url, attempts, *, opener=urllib.request.urlopen, sleep=time.sleep):
    """Fetch the directory text; a failed request is retried a bounded number of times."""
    for attempt in range(1, attempts + 1):
        try:
            with opener(url, timeout=60) as response:
                return response.read().decode('utf-8')
        except OSError:
            if attempt == attempts:
                raise
            sleep(5 * attempt)


def parse_directory(text):
    """Return (rows, file_creation_time). Rows are dicts keyed by the header names."""
    lines = text.splitlines()
    created = next((line.split('|')[0].split(':', 1)[1].strip() for line in lines if line.startswith(TRAILER)), None)
    body = [line for line in lines if line and not line.startswith(TRAILER)]
    rows = list(csv.DictReader(io.StringIO('\n'.join(body)), delimiter='|', quoting=csv.QUOTE_NONE))
    required = {'Symbol', 'Security Name', 'ETF', 'Test Issue'}
    if not rows or not required <= set(rows[0]):
        raise ValueError('symbol directory format changed: missing ' + ', '.join(sorted(required - set(rows[0] if rows else {}))))
    return rows, created


def exclusion_reason(row, settings):
    """Why a directory row is not a common stock, or None when it is kept."""
    if row['ETF'] != 'N':
        return 'etf'
    if row['Test Issue'] != 'N':
        return 'test_issue'
    if re.search(settings['exclude_symbol_pattern'], row['Symbol']):
        return 'symbol_pattern'
    name = row['Security Name']
    for rule in settings['name_rule']:
        if re.search(rule['pattern'], name, re.I) and not ('unless' in rule and re.search(rule['unless'], name, re.I)):
            return rule['reason']
    return None


def select_common_stocks(rows, settings):
    """Return (sorted kept symbols, {reason: excluded row count})."""
    kept, excluded = [], Counter()
    for row in rows:
        reason = exclusion_reason(row, settings)
        if reason is None:
            kept.append(row['Symbol'])
        else:
            excluded[reason] += 1
    if len(set(kept)) != len(kept):
        raise ValueError('symbol directory lists a kept symbol twice')
    return sorted(kept), dict(sorted(excluded.items()))


def yahoo_symbol(symbol):
    """Directory class shares use a dot (BRK.B); Yahoo writes them with a dash (BRK-B)."""
    return symbol.replace('.', '-')
