"""Fake astrometry-engine executable recording successful startup checks."""

from pathlib import Path
import subprocess
import sys


__test__ = False


def main():
    assert sys.argv[1:] == ['--inputs-from', '-']
    assert sys.stdin.read() == ''
    subprocess.run(['pouakai-test-helper'], check=True)
    with Path(__file__).with_name('index-checks').open('a') as stream:
        stream.write('checked\n')


if __name__ == '__main__':
    main()
