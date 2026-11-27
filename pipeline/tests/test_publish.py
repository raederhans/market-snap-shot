"""Publish step against a local bare repository. Real git, no network."""
import json
import subprocess
import pytest
from pipeline import publish, snapshot
from pipeline.run import load_config

SETTINGS = load_config()['publish']
ROW = {'symbol': 'AAPL', 'status': 'ranked'}


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def build(directory, day, generated_at='2026-11-25T23:40:00+00:00'):
    """A snapshot directory laid out as pipeline.run leaves it, holding one row."""
    rows = [{**dict.fromkeys(snapshot.COLUMNS), **ROW}]
    snapshot.write_snapshot(directory, day, rows, {'generated_at': generated_at}, [])
    return directory


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """origin (bare, one commit on main) and the job's shallow checkout of it."""
    for key, value in {'GIT_AUTHOR_NAME': 'job', 'GIT_AUTHOR_EMAIL': 'job@localhost', 'GIT_COMMITTER_NAME': 'job',
                       'GIT_COMMITTER_EMAIL': 'job@localhost', 'GIT_CONFIG_GLOBAL': '/dev/null',
                       'GIT_CONFIG_NOSYSTEM': '1'}.items():
        monkeypatch.setenv(key, value)
    origin, seed = tmp_path / 'origin.git', tmp_path / 'seed'
    git(tmp_path, 'init', '--quiet', '--bare', '--initial-branch=main', str(origin))
    git(tmp_path, 'clone', '--quiet', str(origin), str(seed))
    (seed / 'README.md').write_text('mirror\n')
    git(seed, 'add', 'README.md'), git(seed, 'commit', '--quiet', '-m', 'Export'), git(seed, 'push', '--quiet', 'origin', 'HEAD:main')
    return {'origin': origin, 'seed': seed, 'job': checkout(tmp_path, origin, 'job')}


def checkout(tmp_path, origin, name):
    git(tmp_path, 'clone', '--quiet', '--depth', '1', '--branch', 'main', origin.as_uri(), str(tmp_path / name))
    return tmp_path / name


def run(repos, snapshot_dir, overwrite=False, job='job'):
    return publish.publish(snapshot_dir, repos[job], 'main', overwrite=overwrite, settings=SETTINGS)


def published(origin, path):
    return git(origin, 'show', f'main:{path}')


def test_first_and_next_day_publish_only_the_data_directory(tmp_path, repos):
    assert run(repos, build(tmp_path / 'day1', '2026-11-24')) is True
    assert run(repos, build(tmp_path / 'day2', '2026-11-25')) is True
    origin = repos['origin']
    assert git(origin, 'ls-tree', '-r', '--name-only', 'main').split('\n') == [
        'README.md', 'data/latest.json', 'data/snapshots/2026/2026-11-24.csv.gz', 'data/snapshots/2026/2026-11-24.meta.json',
        'data/snapshots/2026/2026-11-25.csv.gz', 'data/snapshots/2026/2026-11-25.meta.json']
    assert json.loads(published(origin, 'data/latest.json'))['trading_day'] == '2026-11-25'
    assert git(origin, 'log', '--format=%s', 'main').split('\n') == ['Market snapshot 2026-11-25', 'Market snapshot 2026-11-24', 'Export']
    assert git(origin, 'show', '--format=', '--name-only', 'main').split('\n') == [
        'data/latest.json', 'data/snapshots/2026/2026-11-25.csv.gz', 'data/snapshots/2026/2026-11-25.meta.json']


def test_scheduled_rerun_keeps_the_published_day_and_a_manual_run_replaces_it(tmp_path, repos, capsys):
    run(repos, build(tmp_path / 'first', '2026-11-25'))
    head = git(repos['origin'], 'rev-parse', 'main')
    rerun = build(tmp_path / 'rerun', '2026-11-25', generated_at='2026-11-26T23:40:00+00:00')   # holiday run
    assert run(repos, rerun) is False and 'already published' in capsys.readouterr().out
    assert git(repos['origin'], 'rev-parse', 'main') == head and git(repos['job'], 'status', '--porcelain') == ''
    assert run(repos, rerun, overwrite=True) is True
    assert git(repos['origin'], 'rev-list', '--count', 'main') == '3'
    assert json.loads(published(repos['origin'], 'data/latest.json'))['generated_at'] == '2026-11-26T23:40:00+00:00'


def second_export(seed):
    (seed / 'README.md').write_text('mirror, second export\n')
    git(seed, 'commit', '--quiet', '-am', 'Export 2')


def test_branch_that_moved_while_the_snapshot_was_built_is_picked_up_first(tmp_path, repos):
    second_export(repos['seed'])
    git(repos['seed'], 'push', '--quiet', 'origin', 'HEAD:main')
    assert run(repos, build(tmp_path / 'day', '2026-11-25')) is True
    assert git(repos['origin'], 'log', '--format=%s', 'main').split('\n') == ['Market snapshot 2026-11-25', 'Export 2', 'Export']
    assert published(repos['origin'], 'README.md') == 'mirror, second export'


def test_commit_is_rebased_when_the_branch_moves_between_commit_and_push(tmp_path, repos):
    second_export(repos['seed'])
    # Client-side hook: just before the job's first push is sent, the other commit reaches origin.
    hook = repos['job'] / '.git' / 'hooks' / 'pre-push'
    hook.write_text(f'''#!/bin/sh
[ -e "{tmp_path}/raced" ] && exit 0
touch "{tmp_path}/raced"
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE
git -C "{repos['seed']}" push --quiet origin HEAD:main
''')
    hook.chmod(0o755)
    assert run(repos, build(tmp_path / 'day', '2026-11-25')) is True and (tmp_path / 'raced').exists()
    assert git(repos['origin'], 'log', '--format=%s', 'main').split('\n') == ['Market snapshot 2026-11-25', 'Export 2', 'Export']
    assert published(repos['origin'], 'README.md') == 'mirror, second export'
    assert json.loads(published(repos['origin'], 'data/latest.json'))['trading_day'] == '2026-11-25'


def test_a_day_published_by_another_run_meanwhile_is_not_replaced_by_a_scheduled_run(tmp_path, repos):
    other = {**repos, 'other': checkout(tmp_path, repos['origin'], 'other')}
    run(other, build(tmp_path / 'theirs', '2026-11-25'), job='other')
    head = git(repos['origin'], 'rev-parse', 'main')
    assert run(repos, build(tmp_path / 'ours', '2026-11-25', generated_at='2026-11-25T23:59:00+00:00')) is False
    assert git(repos['origin'], 'rev-parse', 'main') == head


def test_a_rejected_push_fails_the_job_without_forcing(tmp_path, repos):
    hook = repos['origin'] / 'hooks' / 'pre-receive'
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o755)
    head = git(repos['origin'], 'rev-parse', 'main')
    with pytest.raises(RuntimeError, match='rejected 3 times'):
        run(repos, build(tmp_path / 'day', '2026-11-25'))
    assert git(repos['origin'], 'rev-parse', 'main') == head


def test_a_conflicting_change_on_the_branch_fails_the_job(tmp_path, repos):
    run(repos, build(tmp_path / 'first', '2026-11-25'))
    stale = checkout(tmp_path, repos['origin'], 'stale')           # checked out before the second publish below
    run(repos, build(tmp_path / 'second', '2026-11-25', generated_at='2026-11-25T23:50:00+00:00'), overwrite=True)
    head = git(repos['origin'], 'rev-parse', 'main')
    third = build(tmp_path / 'third', '2026-11-25', generated_at='2026-11-25T23:55:00+00:00')
    (stale / 'data' / 'latest.json').write_text('{}\n')            # local commit that cannot be rebased cleanly
    git(stale, 'commit', '--quiet', '-am', 'local edit')
    with pytest.raises(subprocess.CalledProcessError):
        publish.publish(third, stale, 'main', overwrite=True, settings=SETTINGS)
    assert git(repos['origin'], 'rev-parse', 'main') == head


def test_cli_publishes_from_the_current_directory(tmp_path, repos, monkeypatch):
    monkeypatch.chdir(repos['job'])
    assert publish.main(['--snapshot', str(build(tmp_path / 'day', '2026-11-25')), '--branch', 'main']) == 0
    assert published(repos['origin'], 'data/latest.json').count('2026-11-25') >= 1
