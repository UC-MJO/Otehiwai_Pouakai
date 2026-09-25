"""
File discovery and cataloging for the B&C reduction pipeline.

`organise_fli_files` walks the raw data archive for FITS files, inspects
any not already present in the master image catalog (`open_and_inspect`),
classifies each by image type (dark/flat/bias/science) from its FITS
header, and writes the updated catalog CSVs used by the rest of the
pipeline (`bc_all_image_list.csv` plus per-type subsets).
"""

from astropy.io import fits
import pandas as pd
import numpy as np
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm

from astropy.time import Time

from pathlib import Path

import os
import fnmatch
import logging

logger = logging.getLogger(__name__)

from . import config

# FLI_DIR has no safe default (see config.raw_archive_dir docstring) --
# set POUAKAI_RAW_ARCHIVE_DIR, or pass fli_dir= explicitly to
# organise_fli_files(). CAL_LIST_PATH follows POUAKAI_CAL_LIST_DIR.
FLI_DIR = config.raw_archive_dir()
CAL_LIST_PATH = config.cal_list_dir()

_REQUIRED_KEYS = ['IMAGETYP', 'EXPTIME', 'JD', 'DATE-OBS', 'READOUTM']

def open_and_inspect(file):
    """
    Inspect a single FITS file and return a one-row DataFrame describing
    it: image type (dark/flat/bias/science), object name, exposure time,
    Julian date, readout mode, band/filter, detector shape, and median
    pixel value.

    Any file that can't be fully characterised (missing required header
    keys, unreadable, etc.) is still recorded -- so it isn't silently
    re-scanned on every future run -- but flagged `telescope='bad'`, with
    the reason logged.

    Parameters
    ----------
    file : str or Path
        Path to the FITS file to inspect.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame describing the file.
    """
    entry = {}
    file = str(file)

    try:
        with fits.open(file, memmap=False) as hdul:
            header = hdul[0].header
            data = hdul[0].data

            missing = [k for k in _REQUIRED_KEYS if k not in header]
            if missing:
                raise KeyError(f'missing header keys: {missing}')

            imagetype = header['IMAGETYP'].lower()

            entry['name'] = file.split('/')[-1].split('.')[0]
            entry['telescope'] = 'bc'
            entry['exptime'] = header['EXPTIME']
            entry['jd'] = header['JD']
            entry['date'] = header['DATE-OBS']
            entry['readout'] = header['READOUTM']
            entry['shape'] = data.shape[0] if data is not None else -1
            entry['median'] = float(np.nanmedian(data)) if data is not None else np.nan
            entry['filename'] = file

            if 'dark' in imagetype:
                entry['imagetype'] = 'dark'
                entry['object'] = 'dark'
            elif 'flat' in imagetype:
                entry['imagetype'] = 'flat'
                entry['object'] = 'flat'
            elif 'bias' in imagetype:
                entry['imagetype'] = 'bias'
                entry['object'] = 'bias'
            else:
                entry['imagetype'] = 'science'
                obj = header.get('OBJECT', None)
                entry['object'] = obj.replace(' ', '') if obj else 'unknown'

            entry['band'] = header.get('FILTER', 'Clear')

    except Exception as e:
        logger.warning(f'{file}: failed to inspect ({e}); marking telescope=bad')
        entry = {'name': file.split('/')[-1].split('.')[0],
                 'telescope': 'bad', 'filename': file}

    return pd.DataFrame([entry])

class organise_fli_files():
    """
    Discover new FITS files under `FLI_DIR`, inspect and classify them,
    and (re)write the master image catalog CSVs under `CAL_LIST_PATH`.

    Running this repeatedly (e.g. as a nightly cron job) is cheap: only
    files not already present in the catalog by filename are inspected
    each time.
    """

    def __init__(self, num_cores=1, retry_bad=False, fli_dir=None):
        """
        Parameters
        ----------
        num_cores : int
            If greater than 1, inspect newly discovered files in
            parallel threads.
        fli_dir : str or None
            Root of the raw FITS archive to walk recursively for new
            files. Defaults to the module-level `FLI_DIR` (itself
            resolved from the `POUAKAI_RAW_ARCHIVE_DIR` environment
            variable -- see config.py) if not given here. Raises
            `ValueError` if neither is set, rather than silently
            scanning nothing/the wrong directory.
        retry_bad : bool
            A file that fails inspection is recorded with
            `telescope='bad'` in `bc_all_image_list.csv`; since its
            filename is then present in the catalog, it is excluded from
            `updated_files` on every subsequent run, so bad files are
            never silently re-attempted (good for cron efficiency). To
            deliberately retry previously bad-marked files (e.g. after
            fixing a transient I/O issue, or a header-parsing bug), set
            `retry_bad=True`: this removes existing `telescope='bad'`
            rows before diffing against the current file list, so those
            files are treated as "new" again and re-inspected.
        """
        self.num_cores = num_cores

        resolved_fli_dir = fli_dir or FLI_DIR
        if not resolved_fli_dir:
            raise ValueError(
                'No raw archive directory to scan: pass fli_dir=... explicitly, '
                'or set the POUAKAI_RAW_ARCHIVE_DIR environment variable.'
            )
        root = Path(resolved_fli_dir)

        files = list(root.rglob("*.fit*"))
        self.files = files
        print(f"Found {len(files)} files")

        self._loading_dataframes()

        if retry_bad and 'telescope' in self.all_image_df.columns:
            n_bad = (self.all_image_df['telescope'] == 'bad').sum()
            if n_bad > 0:
                logger.info(f'retry_bad=True: removing {n_bad} previously bad-marked entries for re-inspection')
                self.all_image_df = self.all_image_df[self.all_image_df['telescope'] != 'bad'].copy()

        filenames = set(self.all_image_df['filename'].astype(str).to_list())
        set_files = set([str(file) for file in files])

        updated_files = list(set_files - filenames)
        print(f"Found {len(updated_files)} new files to process")

        self.process_files(updated_files)

        print(f"Total of {len(self.all_image_df)} files in the database")
        self.all_image_df.to_csv(CAL_LIST_PATH + 'bc_all_image_list.csv', index=False)

        for saving in ['dark', 'science', 'flat']:
            self.all_image_df[self.all_image_df['imagetype'] == saving].to_csv(CAL_LIST_PATH + f'bc_{saving}_image_list.csv', index=False)

    def _loading_dataframes(self):
        """
        Load the existing master image catalog from disk, or create an
        empty one with the expected columns if it doesn't exist yet.
        Result is stored in `self.all_image_df`.
        """
        if not os.path.exists(CAL_LIST_PATH):
            os.makedirs(CAL_LIST_PATH)

        if not os.path.exists(CAL_LIST_PATH + 'bc_all_image_list.csv'):
            all_image_df = pd.DataFrame(columns=['name', 'telescope', 'imagetype', 'exptime', 'jd',
                                                  'date', 'band', 'readout', 'shape', 'median',
                                                  'object', 'filename'])
        else:
            all_image_df = pd.read_csv(CAL_LIST_PATH + 'bc_all_image_list.csv')

        self.all_image_df = all_image_df

    def process_files(self, updated_files):
        """
        Inspect each newly discovered file in `updated_files` (via
        `open_and_inspect`) and append the results to
        `self.all_image_df`.

        Runs in parallel threads if `self.num_cores > 1`, otherwise
        sequentially -- either way, every file in `updated_files` is
        inspected and added.

        Parameters
        ----------
        updated_files : list of str
            Paths of files not yet present in the catalog.
        """
        if len(updated_files) == 0:
            return

        if self.num_cores > 1:
            entries = Parallel(n_jobs=self.num_cores, backend="threading", prefer="threads")(
                delayed(open_and_inspect)(updated_file)
                for updated_file in tqdm(updated_files, desc='Processing files'))
        else:
            entries = [open_and_inspect(updated_file) for updated_file in tqdm(updated_files, desc='Processing files')]

        for entry in entries:
            if entry is not None:
                self.all_image_df = pd.concat([self.all_image_df, entry], ignore_index=True)