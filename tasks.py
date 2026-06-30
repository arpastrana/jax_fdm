# -*- coding: utf-8 -*-
from __future__ import print_function

import contextlib
import glob
import os
import sys
from shutil import rmtree

from invoke import Exit
from invoke import task

try:
    input = raw_input
except NameError:
    pass


BASE_FOLDER = os.path.dirname(__file__)


class Log(object):
    def __init__(self, out=sys.stdout, err=sys.stderr):
        self.out = out
        self.err = err

    def flush(self):
        self.out.flush()
        self.err.flush()

    def write(self, message):
        self.flush()
        self.out.write(message + '\n')
        self.out.flush()

    def info(self, message):
        self.write('[INFO] %s' % message)

    def warn(self, message):
        self.write('[WARN] %s' % message)


log = Log()


def confirm(question):
    while True:
        response = input(question).lower().strip()

        if not response or response in ('n', 'no'):
            return False

        if response in ('y', 'yes'):
            return True

        print('Focus, kid! It is either (y)es or (n)o', file=sys.stderr)


@task(default=True)
def help(ctx):
    """Lists available tasks and usage."""
    ctx.run('invoke --list')
    log.write('Use "invoke -h <taskname>" to get detailed help for a task.')


@task(help={
    'bytecode': 'True to clean up compiled python files, otherwise False.',
    'builds': 'True to clean up build/packaging artifacts, otherwise False.'})
def clean(ctx, bytecode=True, builds=True):
    """Cleans the local copy from compiled artifacts."""

    with chdir(BASE_FOLDER):
        if bytecode:
            for root, dirs, files in os.walk(BASE_FOLDER):
                for f in files:
                    if f.endswith('.pyc'):
                        os.remove(os.path.join(root, f))
                if '.git' in dirs:
                    dirs.remove('.git')

        folders = ['dist/']

        if bytecode:
            for t in ('src', 'tests'):
                folders.extend(glob.glob('{}/**/__pycache__'.format(t), recursive=True))

        if builds:
            folders.append('build/')
            folders.append('src/jax_fdm.egg-info/')

        for folder in folders:
            rmtree(os.path.join(BASE_FOLDER, folder), ignore_errors=True)


@task()
def lint(ctx):
    """Check the consistency of coding style."""
    log.write('Running ruff linter...')
    ctx.run('ruff check src')


@task(help={
      'checks': 'True to run all checks before testing, otherwise False.'})
def test(ctx, checks=False, doctest=False):
    """Run all tests."""
    if checks:
        lint(ctx)

    with chdir(BASE_FOLDER):
        cmd = ['pytest']
        if doctest:
            cmd.append('--doctest-modules')

        ctx.run(' '.join(cmd))


@task
def prepare_changelog(ctx):
    """Prepare changelog for next release."""
    UNRELEASED_CHANGELOG_TEMPLATE = '## Unreleased\n\n### Added\n\n### Changed\n\n### Removed\n\n\n## '

    with chdir(BASE_FOLDER):
        # Preparing changelog for next release
        with open('CHANGELOG.md', 'r+') as changelog:
            content = changelog.read()
            changelog.seek(0)
            changelog.write(content.replace(
                '## ', UNRELEASED_CHANGELOG_TEMPLATE, 1))

        ctx.run('git add CHANGELOG.md && git commit -m "Prepare changelog for next release"')


@task(help={
      'release_type': 'Type of release follows semver rules. Must be one of: major, minor, patch.'})
def release(ctx, release_type):
    """Releases the project in one swift command!"""
    if release_type not in ('patch', 'minor', 'major'):
        raise Exit('The release type parameter is invalid.\nMust be one of: major, minor, patch.')

    # Run checks
    lint(ctx)
    test(ctx)

    # Bump version and git tag it
    ctx.run('bump-my-version bump %s --verbose' % release_type)

    # Prepare the change log for the next release
    prepare_changelog(ctx)


@contextlib.contextmanager
def chdir(dirname=None):
    current_dir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(current_dir)
