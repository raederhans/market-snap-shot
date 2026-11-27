"""Summary document of the historical backfill. Pure functions.

The statistics are summary.horizon_summary, the function behind the forward record: the
non-overlapping subsample, the t interval and the verdict are computed here once per tier,
horizon and slice of evaluation days. Nothing is defined a second time.
"""
from __future__ import annotations
from . import outcomes, summary
from .backfill_features import ABOVE, BELOW

SCHEMA_VERSION = 1
VERSION = '1.0.0'
RECORD_TYPE = 'backfill'
TIER_TOP, TIER_POOL = 'top_dollar_volume', 'eligible_pool'
FULL_SAMPLE = 'full_sample'
# horizon_summary fields that only describe a daily record: the backfill grid skips sessions by design, and
# the kept days follow from daily.csv by the greedy rule.
DROPPED = ('non_overlapping_days', 'snapshot_days_without_evaluation')
LIMITATIONS = [
    'Survivorship: the universe is the stocks listed today. Companies that were large and liquid in the past and were later '
    'acquired, taken private or failed are absent, so the results are too favourable. The bias is larger for absolute '
    'statements (pool_excess_pct, the excess of stocks passing the template) than for spreads between groups, but spreads are '
    'not free of it: among weak stocks only those that survived are counted.',
    'A reconstruction, not the forward record: no snapshot of these days was ever published. It is a preliminary reference '
    'kept apart from validation/summary.json and must not be merged with it or quoted as the project\'s forward result.',
    'Prices are split- and dividend-adjusted as of the download date. Returns, moving-average comparisons and distances from '
    'the high are unaffected by a later adjustment, but the price floor and the dollar-volume floor and ranking are applied to '
    'adjusted levels, which differ from the prices of that day.',
    'Returns run from the close of D, the same close the features use, so the entry was not obtainable. No commission, '
    'spread, market impact or tax is deducted.',
    'Cross-sectional average excess returns of groups, not a strategy backtest: no entry or exit rule, position sizing, '
    'turnover or capacity is modelled.',
    'Multiple comparisons: two tiers, several horizons, slices and metrics are tested on one stretch of history, each with its '
    'own 95% interval. Some intervals exclude zero by chance alone and nothing is adjusted for it.',
    'Averages are equal-weighted and sensitive to bad prints and extreme moves; rank_ic is reported beside them as the more '
    'robust reference.',
    'Only rows at least h sessions apart carry the interval, starting from the first evaluation day (one of several possible '
    'starting phases). The interval treats them as independent. A slice covers few distinct episodes (the sessions below the '
    'benchmark average are mostly one bear market), so its periods are less independent than that, and a year slice rarely '
    'reaches min_non_overlapping_periods.',
    'A stock without a bar on D+h is counted in missing_pct and left out of every average, never filled.',
    'Unofficial data (Yahoo Finance through yfinance). A descriptive study of features and rankings: not a trading strategy, '
    'not a forecast and not investment advice.',
]


def slices(rows, sma_sessions):
    """{slice name: rows}: the full sample, the two benchmark-trend states of D and each calendar year of D.

    A day whose benchmark trend is unknown is in the full sample and its year only.
    """
    out = {FULL_SAMPLE: rows}
    for state in (ABOVE, BELOW):
        out[f'benchmark_{state}_sma{sma_sessions}'] = [row for row in rows if row['benchmark_trend'] == state]
    for year in sorted({row['year'] for row in rows}):
        out[f'year_{year}'] = [row for row in rows if row['year'] == year]
    return out


def tier_summary(rows, sessions, config):
    """horizon -> slice -> summary.horizon_summary of that slice's rows (its own non-overlapping subsample)."""
    settings, out = config['validation'], {}
    for horizon in settings['horizons']:
        parts = slices([row for row in rows if row['horizon'] == horizon], config['backfill']['regime_sma_sessions'])
        out[str(horizon)] = {'slices': {
            name: {key: value for key, value in summary.horizon_summary(part, horizon, sessions, settings['min_non_overlapping_periods']).items()
                   if key not in DROPPED} for name, part in parts.items()}}
    return out


def build(rows, *, sessions, config, facts):
    """The backfill summary. rows: every typed row of the daily table; facts: provenance collected by the run."""
    settings = config['backfill']
    days = sorted({row['snapshot_day'] for row in rows})
    return {
        'schema_version': SCHEMA_VERSION, 'record_type': RECORD_TYPE, 'backfill_version': VERSION,
        'validation_version': outcomes.VERSION,
        'description': 'Historical backfill, not the forward record: the features of past sessions were reconstructed from '
                       'today\'s download for the stocks listed today and matched with the returns that followed. '
                       'Survivorship-biased preliminary reference; read the limitations.',
        'benchmark': config['benchmark'], 'confidence_level': summary.CONFIDENCE,
        'min_non_overlapping_periods': config['validation']['min_non_overlapping_periods'],
        **facts,
        'parameters': {'backfill': settings, 'eligibility': config['eligibility'], 'validation': config['validation']},
        'evaluation': {'first_day': days[0] if days else None, 'last_day': days[-1] if days else None, 'days': len(days),
                       'step_sessions': settings['step_sessions'],
                       'days_by_benchmark_trend': {state: len({row['snapshot_day'] for row in rows if row['benchmark_trend'] == state})
                                                   for state in (ABOVE, BELOW)}},
        'primary_tier': TIER_TOP,
        'tiers': {
            TIER_TOP: {'description': f"The {settings['top_dollar_volume']} ranked stocks with the highest avg_dollar_volume_50d on D "
                                      '(all of them when fewer are ranked). RS groups use the percentile within the whole ranked pool.',
                       'horizons': tier_summary([row for row in rows if row['tier'] == TIER_TOP], sessions, config)},
            TIER_POOL: {'description': 'Every ranked stock of D (price, dollar-volume and history floors): the comparison tier.',
                        'horizons': tier_summary([row for row in rows if row['tier'] == TIER_POOL], sessions, config)}},
        'limitations': LIMITATIONS,
    }
