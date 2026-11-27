"""Daily market snapshot batch job, run by a scheduled GitHub Actions workflow.

The calculations live in the shared library under remote-mcp/, so make it
importable when the job is started as `python -m pipeline.run` from the repository root.
"""
import pathlib
import sys

_CANONICAL = str(pathlib.Path(__file__).resolve().parents[1] / 'remote-mcp')
if _CANONICAL not in sys.path:
    sys.path.insert(0, _CANONICAL)
