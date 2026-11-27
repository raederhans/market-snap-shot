"""Write one trading day's snapshot: gzip CSV rows, a metadata JSON and the latest pointer.

CSV rather than JSON: the rows are one flat table, so CSV avoids repeating ~35 keys
per symbol (several times smaller), opens directly in pandas or a spreadsheet, and
an empty cell is an unambiguous "unknown".
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import pathlib
from stock_research.universe_rank import COLUMNS

SCHEMA_VERSION = 2  # 2: columns listed in output.exclude_columns are absent (metadata: excluded_columns)
# Output precision only; ranking uses the unrounded values.
DECIMALS = {'close': 4, 'sma50': 4, 'sma150': 4, 'sma200': 4, 'rs_vs_benchmark': 4, 'avg_dollar_volume_50d': 0}
DEFAULT_DECIMALS = 2


def _cell(column, value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        digits = DECIMALS.get(column, DEFAULT_DECIMALS)
        return str(int(round(value))) if digits == 0 else f'{value:.{digits}f}'
    return str(value)


def published_columns(exclude):
    """COLUMNS without the configured exclusions, in the original order."""
    unknown = sorted(set(exclude) - set(COLUMNS))
    if unknown:
        raise ValueError('output.exclude_columns names unknown columns: ' + ', '.join(unknown))
    return [c for c in COLUMNS if c not in exclude]


def csv_bytes(rows, columns):
    text = io.StringIO()
    writer = csv.writer(text, lineterminator='\n')
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(c, row[c]) for c in columns])
    raw = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as handle:  # mtime=0: same rows, same bytes
        handle.write(text.getvalue().encode('utf-8'))
    return raw.getvalue()


def write_snapshot(out_dir, trading_day, rows, meta, exclude_columns):
    """Create snapshots/YYYY/YYYY-MM-DD.csv.gz and .meta.json, then latest.json. Returns meta.

    Paths in the metadata and the pointer are relative to out_dir. exclude_columns
    (config output.exclude_columns) are dropped here, after ranking, and recorded.
    """
    out_dir = pathlib.Path(out_dir)
    base = f'snapshots/{trading_day[:4]}/{trading_day}'
    columns = published_columns(exclude_columns)
    data = csv_bytes(rows, columns)
    meta = {**meta, 'columns': columns, 'excluded_columns': [c for c in COLUMNS if c in exclude_columns],
            'files': {'snapshot': base + '.csv.gz', 'meta': base + '.meta.json',
                      'snapshot_bytes': len(data), 'snapshot_sha256': hashlib.sha256(data).hexdigest()}}
    (out_dir / base).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / (base + '.csv.gz')).write_bytes(data)
    (out_dir / (base + '.meta.json')).write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    # Written last: the pointer only ever names a complete snapshot.
    pointer = {'trading_day': trading_day, 'generated_at': meta['generated_at'], **meta['files']}
    (out_dir / 'latest.json').write_text(json.dumps(pointer, indent=2) + '\n', encoding='utf-8')
    return meta
