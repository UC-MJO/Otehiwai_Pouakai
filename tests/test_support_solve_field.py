"""Fake solve-field executable for testing subprocess and output handling."""

from pathlib import Path
import gzip
import shutil
import subprocess
import sys


__test__ = False


def main():
    args = sys.argv[1:]
    output = Path(args[args.index('--dir') + 1])
    name = args[args.index('-o') + 1]
    subprocess.run(['pouakai-test-helper'], check=True)
    source = Path(args[-1])
    # solve-field always writes uncompressed FITS, even for compressed inputs.
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as input_file, (output / (name + '.new')).open('wb') as new_file:
        shutil.copyfileobj(input_file, new_file)


if __name__ == '__main__':
    main()
