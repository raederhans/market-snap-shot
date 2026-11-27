"""Forward reconciliation: match earlier snapshots with what happened by today.

    python -m pipeline.validate --data <published data dir> --closes <file from pipeline.run> --out <dir>

Runs after today's snapshot has been published, so a failure here cannot touch it. For
each configured horizon h it takes the snapshot published h XNYS sessions before the
evaluation day (the last session in the closes file) and writes, under <out>:

    outcomes/YYYY/DAY_hNN.csv.gz   per symbol: outcome status, forward return, excess over the benchmark
    validation/daily.csv           one row per (snapshot day, horizon); a rerun replaces its rows
    validation/summary.json        aggregate per horizon, with the sample status

A horizon whose snapshot day has no published snapshot is skipped and the reason is
recorded in the summary; a neighbouring day is never used instead. pipeline.publish
--validation commits <out> into the data directory. Calculations: outcomes.py, summary.py.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import pathlib
import sys
import exchange_calendars as xcals
import pandas as pd
from stock_research.universe_rank import RANKED, REFERENCE
from . import outcomes, prices, snapshot, summary
from .run import load_config

SNAPSHOT_COLUMNS = ('symbol', 'status', 'rs_percentile', 'trend_template_status', 'distance_from_52w_high_pct')
TEXT = ('snapshot_day', 'evaluated_on', 'rules_version', 'null_reasons')
COUNTS = ('horizon', 'pool', 'with_outcome', 'missing', 'template_pass_n', 'template_fail_n', 'near_high_n',
          'not_near_high_n', 'combo_n')
DAILY_DECIMALS = 4


def published_days(data):
    """Trading days that have a snapshot under data/snapshots, ascending."""
    return sorted(path.name.removesuffix('.csv.gz') for path in (data / 'snapshots').glob('*/*.csv.gz'))


def read_snapshot(data, day):
    """The columns validation needs from one published snapshot, verified against its metadata, and its rules version."""
    base = data / 'snapshots' / day[:4]
    meta = json.loads((base / f'{day}.meta.json').read_text(encoding='utf-8'))
    raw = (base / f'{day}.csv.gz').read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['files']['snapshot_sha256']:
        raise outcomes.ValidationError(f'snapshot {day} does not match the hash in its metadata')
    # Symbols are text whatever they spell (NA, TRUE); only an empty cell is unknown.
    table = pd.read_csv(io.BytesIO(raw), compression='gzip', usecols=lambda name: name in SNAPSHOT_COLUMNS,
                        dtype={'symbol': str, 'status': str, 'trend_template_status': str},
                        keep_default_na=False, na_values=[''])
    missing = sorted(set(SNAPSHOT_COLUMNS) - set(table.columns))
    if missing:
        raise outcomes.ValidationError(f'snapshot {day} lacks columns: {", ".join(missing)}')
    return table, meta['rules_version']


def read_daily(path):
    """Existing daily rows as {(snapshot day, horizon): {column: text}}; no file yet means an empty record."""
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != outcomes.DAILY_COLUMNS:
            raise outcomes.ValidationError(f'{path} has other columns than this version writes; migrate it deliberately')
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):   # a line with more or fewer cells than the header
        raise outcomes.ValidationError(f'{path} has a malformed row')
    return {(row['snapshot_day'], int(row['horizon'])): row for row in rows}


def _text(value):
    if value is None:
        return ''
    return f'{value:.{DAILY_DECIMALS}f}' if isinstance(value, float) else str(value)


def typed(row):
    """A daily row as read from the file -> numbers, with None for an empty cell. The summary is computed from
    these published values, so anyone can recompute it from daily.csv."""
    return {column: text if column in TEXT else None if text == '' else int(text) if column in COUNTS else float(text)
            for column, text in row.items()}


def evaluate(data, closes, day, today, horizon, config):
    """One (snapshot day, horizon): returns (outcome rows for the file, daily row as text)."""
    table, rules_version = read_snapshot(data, day)
    eligible = table[table.status.isin([RANKED, REFERENCE])]   # reference ETFs get an outcome but are not in the pool
    rows, benchmark_return = outcomes.forward_outcomes(list(eligible.symbol), closes, day, today, config['benchmark'])
    pool = eligible.merge(pd.DataFrame(rows, columns=outcomes.OUTCOME_COLUMNS), on='symbol')
    stats, reasons = outcomes.cross_section(pool[pool.status == RANKED], config['validation'])
    daily = {'snapshot_day': day, 'horizon': horizon, 'evaluated_on': today, 'rules_version': rules_version,
             'benchmark_return_pct': benchmark_return, **stats,
             'null_reasons': ';'.join(f'{name}={why}' for name, why in sorted(reasons.items()))}
    return rows, {column: _text(daily[column]) for column in outcomes.DAILY_COLUMNS}


def validate(data_dir, closes_path, out_dir, *, config=None):
    """Evaluate every configured horizon for the day of the closes file; write the three outputs. Returns the summary."""
    config = config or load_config()
    data, out = pathlib.Path(data_dir), pathlib.Path(out_dir)
    if not pathlib.Path(closes_path).is_file():
        raise outcomes.ValidationError(f'closes file {closes_path} is missing: the build step did not save it')
    closes = prices.read_closes(closes_path)
    today = closes.index[-1]
    sessions = [stamp.date().isoformat() for stamp in xcals.get_calendar('XNYS').sessions]
    published = published_days(data)
    daily = read_daily(data / 'validation' / 'daily.csv')
    files, last_run = {}, {}
    for horizon in config['validation']['horizons']:
        day = outcomes.session_before(today, horizon, sessions)
        if day not in published:
            # No substitute day: before the record began this is expected, later it is a gap in the record.
            reason = 'before_record_start' if not published or day < published[0] else 'snapshot_missing'
            last_run[str(horizon)] = {'snapshot_day': day, 'status': 'skipped', 'reason': reason}
            continue
        rows, daily[(day, horizon)] = evaluate(data, closes, day, today, horizon, config)
        files[f'outcomes/{day[:4]}/{day}_h{horizon:02d}.csv.gz'] = snapshot.csv_bytes(rows, outcomes.OUTCOME_COLUMNS)
        last_run[str(horizon)] = {'snapshot_day': day, 'status': 'evaluated', 'reason': None}

    table = io.StringIO()
    writer = csv.DictWriter(table, outcomes.DAILY_COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(daily[key] for key in sorted(daily))
    document = summary.build([typed(daily[key]) for key in sorted(daily)], sessions=sessions, settings=config['validation'],
                             benchmark=config['benchmark'], record_start=published[0] if published else None,
                             evaluation_day=today, last_run=last_run)
    files['validation/daily.csv'] = table.getvalue().encode('utf-8')
    files['validation/summary.json'] = (json.dumps(document, indent=2) + '\n').encode('utf-8')
    for name, content in files.items():   # nothing is written before every horizon has been computed
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(content)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pipeline.validate', description=__doc__.splitlines()[0])
    parser.add_argument('--data', required=True, help='published data directory holding snapshots/ (read only)')
    parser.add_argument('--closes', required=True, help='adjusted closes written by pipeline.run --closes')
    parser.add_argument('--out', required=True, help='directory that receives outcomes/ and validation/')
    args = parser.parse_args(argv)
    try:
        document = validate(args.data, args.closes, args.out)
    except outcomes.ValidationError as exc:
        print(f'FORWARD VALIDATION FAILED, nothing written: {exc}', file=sys.stderr)
        return 1
    for horizon, entry in document['last_run'].items():
        total = document['horizons'][horizon]
        print(f"h={horizon}: snapshot {entry['snapshot_day']} {entry['status']}{' (' + entry['reason'] + ')' if entry['reason'] else ''}; "
              f"{total['evaluated_days']} days, {total['non_overlapping_periods']} non-overlapping, {total['sample_status']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
