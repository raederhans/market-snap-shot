"""Commit one built snapshot to the data directory of the repository that runs the job.

    python -m pipeline.publish --snapshot <dir written by pipeline.run> --branch <branch> [--overwrite]

Run from the root of a clean checkout of <branch>. Without --overwrite (the
scheduled job) a trading day that is already published is left alone: a run on a
market holiday rebuilds the previous session. Never force-pushes: if the branch
moved during the run the commit is rebased onto it, and any failure exits non-zero.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from .run import load_config


def git(repo, *args, check=True):
    return subprocess.run(['git', '-C', str(repo), *args], check=check).returncode


def publish(snapshot_dir, repo_dir, branch, *, overwrite, settings):
    """Returns True when a commit was pushed, False when the day was already published."""
    snapshot_dir, repo_dir = pathlib.Path(snapshot_dir), pathlib.Path(repo_dir)
    pointer = json.loads((snapshot_dir / 'latest.json').read_text(encoding='utf-8'))
    day, data = pointer['trading_day'], repo_dir / settings['directory']
    git(repo_dir, 'pull', '--rebase', 'origin', branch)   # decide against what is published right now
    if not overwrite and (data / pointer['snapshot']).exists():
        print(f'Snapshot for {day} is already published; nothing to do.')
        return False
    shutil.copytree(snapshot_dir, data, dirs_exist_ok=True)
    git(repo_dir, 'add', '--', settings['directory'])     # the data directory only, never the rest of the tree
    git(repo_dir, 'commit', '--quiet', '-m', f'Market snapshot {day}')
    for _ in range(settings['push_attempts']):
        git(repo_dir, 'pull', '--rebase', 'origin', branch)   # a no-op unless the branch moved; a conflict raises
        if git(repo_dir, 'push', 'origin', f'HEAD:refs/heads/{branch}', check=False) == 0:
            return True
    raise RuntimeError(f"push to {branch} was rejected {settings['push_attempts']} times; nothing was published")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pipeline.publish', description=__doc__.splitlines()[0])
    parser.add_argument('--snapshot', required=True, help='directory holding latest.json and snapshots/')
    parser.add_argument('--branch', required=True, help='branch to commit to and push')
    parser.add_argument('--overwrite', action='store_true', help='replace a trading day that is already published')
    args = parser.parse_args(argv)
    publish(args.snapshot, pathlib.Path.cwd(), args.branch, overwrite=args.overwrite, settings=load_config()['publish'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
