"""Universe filter against rows copied verbatim from nasdaqtraded.txt (2026-11-25 file). No network."""
import copy
import pathlib
import pytest
from pipeline import universe
from pipeline.run import load_config

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'nasdaqtraded_sample.txt'
SETTINGS = load_config()['universe']

# Common equity that naive rules would lose.
KEEP = {
    'AAPL', 'NVDA', 'MSFT', 'AVGO', 'TSLA',
    'UBER', 'PLTR', 'MU', 'NOW',          # tickers ending in R / U / W are not rights, units or warrants
    'TSM', 'ASML', 'ARM', 'BABA',         # ADRs; TSM's name has no share-class words at all
    'AMX',                                # ADR "representing the right to receive"
    'KOF',                                # ADR "representing 10 Units"
    'ACN', 'LIN', 'SPOT',                 # "Ordinary Shares"
    'GOOG', 'SHOP',                       # "Class C Capital Stock", "Subordinate Voting Shares"
    'BRK.A', 'BRK.B', 'BF.B',             # dotted share classes
    'ET', 'AB', 'IEP',                    # partnership units are the common equity
    'CEG',                                # stale "When-Issued" label on a seasoned stock
}
DROP = {
    'SPY': 'etf', 'QQQ': 'etf', 'ZZZTX': 'test_issue',
    'BAC$E': 'symbol_pattern', 'ATH$A': 'symbol_pattern', 'ADC$A': 'symbol_pattern',
    'AACIW': 'warrant', 'ALUB.W': 'warrant', 'PSNYW': 'warrant',
    'WAFDP': 'preferred', 'DDT': 'preferred', 'BPOPM': 'preferred',
    'SOJC': 'debt', 'AFGB': 'debt', 'VXX': 'debt',
    'GJT': 'structured_product', 'KTN': 'debt',  # KTN's coupon is matched first
    'AACIU': 'unit', 'AAC.U': 'unit', 'SONN': 'unit', 'NOVTU': 'unit',
    'AACPR': 'right', 'AIIA.R': 'right', 'CELG.R': 'right',
    'ARCC': 'fund', 'PDI': 'fund',
}


@pytest.fixture(scope='module')
def rows():
    return universe.parse_directory(FIXTURE.read_text())[0]


def test_parse_reads_rows_and_drops_the_trailer(rows):
    _, created = universe.parse_directory(FIXTURE.read_text())
    assert created == '1125202621:34'
    assert len(rows) == len(KEEP) + len(DROP)
    assert all(row['Symbol'] and not row['Symbol'].startswith('File') for row in rows)


def test_keeps_every_common_stock_and_nothing_else(rows):
    kept, excluded = universe.select_common_stocks(rows, SETTINGS)
    assert set(kept) == KEEP and kept == sorted(kept)
    assert sum(excluded.values()) == len(DROP)


@pytest.mark.parametrize('symbol,reason', sorted(DROP.items()))
def test_exclusion_reason(rows, symbol, reason):
    row = next(r for r in rows if r['Symbol'] == symbol)
    assert universe.exclusion_reason(row, SETTINGS) == reason


def test_preferred_depositary_shares_are_dropped_by_name_alone(rows):
    # "Depositary Shares each representing ... Preferred" must go even without the "$" symbol marker.
    settings = {**SETTINGS, 'exclude_symbol_pattern': '(?!)'}
    by_symbol = {r['Symbol']: r for r in rows}
    assert universe.exclusion_reason(by_symbol['ADC$A'], settings) == 'preferred'
    assert universe.exclusion_reason(by_symbol['ATH$A'], settings) == 'preferred'
    assert universe.exclusion_reason(by_symbol['BAC$E'], settings) == 'preferred'  # "Pfd"
    assert universe.exclusion_reason(by_symbol['BABA'], settings) is None          # ADS "each representing"


def test_rules_come_from_configuration(rows):
    settings = copy.deepcopy(SETTINGS)
    settings['name_rule'] = [r for r in settings['name_rule'] if r['reason'] != 'fund']
    kept, _ = universe.select_common_stocks(rows, settings)
    assert {'ARCC', 'PDI'} <= set(kept)
    settings['name_rule'] = [{'reason': 'adr', 'pattern': 'depositary'}]
    assert universe.exclusion_reason(next(r for r in rows if r['Symbol'] == 'BABA'), settings) == 'adr'


def test_yahoo_symbol_spelling():
    assert universe.yahoo_symbol('BRK.B') == 'BRK-B' and universe.yahoo_symbol('MU') == 'MU'


def test_format_change_is_an_error_not_an_empty_universe():
    with pytest.raises(ValueError):
        universe.parse_directory('Ticker|Name\nAAPL|Apple\n')
    with pytest.raises(ValueError):
        universe.parse_directory('')


def test_duplicate_kept_symbol_is_an_error(rows):
    with pytest.raises(ValueError):
        universe.select_common_stocks(rows + [rows[0]], SETTINGS)


class _Reply:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def read(self): return self.body


def test_download_retries_a_bounded_number_of_times():
    calls, waits = [], []
    def flaky(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise OSError('reset')
        return _Reply(b'ok')
    assert universe.download_directory('u', 3, opener=flaky, sleep=waits.append) == 'ok'
    assert len(calls) == 3 and waits == [5, 10]
    calls.clear()
    with pytest.raises(OSError):
        universe.download_directory('u', 2, opener=lambda url, timeout: flaky(url, timeout), sleep=waits.append)
    assert len(calls) == 2
