"""Publish steps against a local bare repository: the snapshot commit, then the validation commit. Real git, no network."""
import json
import subprocess
import pytest
from pipeline import publish, snapshot, validate
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


def built_validation(directory, day='2026-11-25', body='snapshot_day,horizon\n'):
    """A validation directory laid out as pipeline.validate leaves it."""
    for name, text in {'validation/summary.json': json.dumps({'latest_evaluation_day': day}) + '\n', 'validation/daily.csv': body,
                       'outcomes/2026/2026-11-18_h05.csv.gz': 'outcome bytes'}.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)
    return directory


def test_validation_is_a_second_commit_after_the_snapshot(tmp_path, repos, capsys):
    assert run(repos, build(tmp_path / 'day', '2026-11-25')) is True
    snapshot_commit = git(repos['origin'], 'rev-parse', 'main')
    results = built_validation(tmp_path / 'validation')
    assert publish.publish_validation(results, repos['job'], 'main', settings=SETTINGS) is True
    origin = repos['origin']
    assert git(origin, 'log', '--format=%s', 'main').split('\n') == ['Forward validation 2026-11-25', 'Market snapshot 2026-11-25', 'Export']
    assert git(origin, 'rev-parse', 'main~1') == snapshot_commit                       # the snapshot commit is untouched history
    assert git(origin, 'show', '--format=', '--name-only', 'main').split('\n') == [
        'data/outcomes/2026/2026-11-18_h05.csv.gz', 'data/validation/daily.csv', 'data/validation/summary.json']
    assert json.loads(published(origin, 'data/latest.json'))['trading_day'] == '2026-11-25'
    # A rerun with identical results (a market-holiday run) commits nothing; changed results replace the files.
    head = git(origin, 'rev-parse', 'main')
    assert publish.publish_validation(results, repos['job'], 'main', settings=SETTINGS) is False
    assert 'nothing to commit' in capsys.readouterr().out and git(origin, 'rev-parse', 'main') == head
    built_validation(results, body='snapshot_day,horizon\n2026-11-18,5\n')
    assert publish.publish_validation(results, repos['job'], 'main', settings=SETTINGS) is True
    assert published(origin, 'data/validation/daily.csv') == 'snapshot_day,horizon\n2026-11-18,5'
    assert git(origin, 'rev-list', '--count', 'main') == '4' and git(repos['job'], 'status', '--porcelain') == ''


def test_a_failed_validation_leaves_the_published_snapshot_alone(tmp_path, repos, monkeypatch, capsys):
    """The workflow order: snapshot pushed first; the validation steps then fail without touching it."""
    monkeypatch.chdir(repos['job'])
    assert publish.main(['--snapshot', str(build(tmp_path / 'day', '2026-11-25')), '--branch', 'main']) == 0
    head = git(repos['origin'], 'rev-parse', 'main')
    arguments = ['--data', 'data', '--closes', str(tmp_path / 'never-written.csv.gz'), '--out', str(tmp_path / 'validation')]
    assert validate.main(arguments) == 1 and 'FORWARD VALIDATION FAILED' in capsys.readouterr().err   # the step exits non-zero
    with pytest.raises(FileNotFoundError):                                           # and so would the publish step after it
        publish.main(['--validation', str(tmp_path / 'validation'), '--branch', 'main'])
    assert git(repos['origin'], 'rev-parse', 'main') == head and git(repos['job'], 'status', '--porcelain') == ''
    assert git(repos['origin'], 'ls-tree', '-r', '--name-only', 'main').split('\n') == [
        'README.md', 'data/latest.json', 'data/snapshots/2026/2026-11-25.csv.gz', 'data/snapshots/2026/2026-11-25.meta.json']
    # A validation push that is rejected fails as well, and the snapshot is still what is published.
    hook = repos['origin'] / 'hooks' / 'pre-receive'
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o755)
    with pytest.raises(RuntimeError, match='rejected 3 times'):
        publish.publish_validation(built_validation(tmp_path / 'results'), repos['job'], 'main', settings=SETTINGS)
    assert git(repos['origin'], 'rev-parse', 'main') == head


def test_cli_publishes_validation_and_requires_exactly_one_source(tmp_path, repos, monkeypatch):
    monkeypatch.chdir(repos['job'])
    assert publish.main(['--validation', str(built_validation(tmp_path / 'validation')), '--branch', 'main']) == 0
    assert git(repos['origin'], 'log', '-1', '--format=%s', 'main') == 'Forward validation 2026-11-25'
    for arguments in (['--branch', 'main'], ['--snapshot', 'a', '--validation', 'b', '--branch', 'main']):
        with pytest.raises(SystemExit):
            publish.main(arguments)


def built_backfill(directory, last='2026-11-25', body='snapshot_day,horizon,tier\n'):
    """A backfill directory laid out as pipeline.backfill leaves it."""
    (directory / 'backfill').mkdir(parents=True, exist_ok=True)
    (directory / 'backfill/summary.json').write_text(json.dumps({'record_type': 'backfill', 'data': {'last_session': last}}) + '\n')
    (directory / 'backfill/daily.csv').write_text(body)
    return directory


def test_backfill_is_its_own_commit_and_touches_only_its_directory(tmp_path, repos, monkeypatch, capsys):
    run(repos, build(tmp_path / 'day', '2026-11-25'))
    publish.publish_validation(built_validation(tmp_path / 'validation'), repos['job'], 'main', settings=SETTINGS)
    origin = repos['origin']
    before = {path: published(origin, path) for path in ('data/latest.json', 'data/validation/daily.csv', 'data/validation/summary.json')}
    results = built_backfill(tmp_path / 'backfill')
    assert publish.publish_backfill(results, repos['job'], 'main', settings=SETTINGS) is True
    assert git(origin, 'log', '-1', '--format=%s', 'main') == 'Historical backfill through 2026-11-25'
    assert git(origin, 'show', '--format=', '--name-only', 'main').split('\n') == ['data/backfill/daily.csv', 'data/backfill/summary.json']
    assert {path: published(origin, path) for path in before} == before          # the forward record and the snapshot are untouched
    head = git(origin, 'rev-parse', 'main')
    assert publish.publish_backfill(results, repos['job'], 'main', settings=SETTINGS) is False and git(origin, 'rev-parse', 'main') == head
    capsys.readouterr()
    # A source directory holding anything else is refused before git is touched: it could overwrite the forward record.
    for extra in ('validation/summary.json', 'snapshots/2026/x.csv.gz', 'outcomes/2026/x.csv.gz', 'latest.json'):
        mixed = built_backfill(tmp_path / ('mixed-' + extra.split('/')[0]))
        (mixed / extra).parent.mkdir(parents=True, exist_ok=True)
        (mixed / extra).write_text('{}')
        with pytest.raises(RuntimeError, match='must hold backfill/ only'):
            publish.publish_backfill(mixed, repos['job'], 'main', settings=SETTINGS)
    with pytest.raises(FileNotFoundError):
        publish.publish_backfill(tmp_path / 'never-written', repos['job'], 'main', settings=SETTINGS)
    assert git(origin, 'rev-parse', 'main') == head and git(repos['job'], 'status', '--porcelain') == ''
    # The command line, and a moved branch: rebased, not forced.
    second_export(repos['seed'])
    git(repos['seed'], 'pull', '--quiet', '--rebase', 'origin', 'main'), git(repos['seed'], 'push', '--quiet', 'origin', 'HEAD:main')
    monkeypatch.chdir(repos['job'])
    assert publish.main(['--backfill', str(built_backfill(results, body='snapshot_day,horizon,tier\n2021-11-24,5,eligible_pool\n')), '--branch', 'main']) == 0
    assert git(origin, 'log', '--format=%s', '-2', 'main').split('\n') == ['Historical backfill through 2026-11-25', 'Export 2']
    with pytest.raises(SystemExit):
        publish.main(['--backfill', 'a', '--validation', 'b', '--branch', 'main'])
    hook = origin / 'hooks' / 'pre-receive'
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o755)
    with pytest.raises(RuntimeError, match='rejected 3 times'):
        publish.publish_backfill(built_backfill(results, last='2026-11-27'), repos['job'], 'main', settings=SETTINGS)
