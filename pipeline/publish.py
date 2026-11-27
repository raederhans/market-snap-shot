"""Commit the job's outputs to the data directory of the repository that runs it.

    python -m pipeline.publish --snapshot <dir written by pipeline.run> --branch <branch> [--overwrite]
    python -m pipeline.publish --validation <dir written by pipeline.validate> --branch <branch>

Two separate commits, the snapshot first: it is a point-in-time record that cannot be
recomputed, so nothing that follows may stand between it and the push. Run from the
root of a clean checkout of <branch>. Without --overwrite (the scheduled job) a trading
day that is already published is left alone: a run on a market holiday rebuilds the
previous session. Validation outputs always replace what is there (their rows are keyed
by snapshot day and horizon). Never force-pushes: if the branch moved during the run the
commit is rebased onto it, and any failure exits non-zero.
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


def _commit_and_push(source_dir, repo_dir, branch, message, settings):
    """Copy source_dir into the data directory, commit that directory only and push. False when nothing changed."""
    shutil.copytree(source_dir, repo_dir / settings['directory'], dirs_exist_ok=True)
    git(repo_dir, 'add', '--', settings['directory'])     # the data directory only, never the rest of the tree
    if git(repo_dir, 'diff', '--cached', '--quiet', check=False) == 0:
        print(f'{message}: identical to what is published; nothing to commit.')
        return False
    git(repo_dir, 'commit', '--quiet', '-m', message)
    for _ in range(settings['push_attempts']):
        git(repo_dir, 'pull', '--rebase', 'origin', branch)   # a no-op unless the branch moved; a conflict raises
        if git(repo_dir, 'push', 'origin', f'HEAD:refs/heads/{branch}', check=False) == 0:
            return True
    raise RuntimeError(f"push to {branch} was rejected {settings['push_attempts']} times; nothing was published")


def publish(snapshot_dir, repo_dir, branch, *, overwrite, settings):
    """Returns True when a commit was pushed, False when the day was already published."""
    snapshot_dir, repo_dir = pathlib.Path(snapshot_dir), pathlib.Path(repo_dir)
    pointer = json.loads((snapshot_dir / 'latest.json').read_text(encoding='utf-8'))
    day, data = pointer['trading_day'], repo_dir / settings['directory']
    git(repo_dir, 'pull', '--rebase', 'origin', branch)   # decide against what is published right now
    if not overwrite and (data / pointer['snapshot']).exists():
        print(f'Snapshot for {day} is already published; nothing to do.')
        return False
    return _commit_and_push(snapshot_dir, repo_dir, branch, f'Market snapshot {day}', settings)


def publish_validation(validation_dir, repo_dir, branch, *, settings):
    """Commit the outputs of pipeline.validate. Returns True when a commit was pushed, False when nothing changed."""
    validation_dir, repo_dir = pathlib.Path(validation_dir), pathlib.Path(repo_dir)
    document = json.loads((validation_dir / 'validation' / 'summary.json').read_text(encoding='utf-8'))
    git(repo_dir, 'pull', '--rebase', 'origin', branch)
    return _commit_and_push(validation_dir, repo_dir, branch, f"Forward validation {document['latest_evaluation_day']}", settings)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pipeline.publish', description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--snapshot', help='directory holding latest.json and snapshots/')
    source.add_argument('--validation', help='directory holding outcomes/ and validation/')
    parser.add_argument('--branch', required=True, help='branch to commit to and push')
    parser.add_argument('--overwrite', action='store_true', help='replace a trading day that is already published')
    args = parser.parse_args(argv)
    settings = load_config()['publish']
    if args.validation:
        publish_validation(args.validation, pathlib.Path.cwd(), args.branch, settings=settings)
    else:
        publish(args.snapshot, pathlib.Path.cwd(), args.branch, overwrite=args.overwrite, settings=settings)
    return 0


if __name__ == '__main__':
    sys.exit(main())
