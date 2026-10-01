# Otehiwai Pouakai

End-to-end FLI/B&C astronomical image reduction, astrometric (WCS), and
photometric calibration pipeline: dark/flat master-building, advanced structure background
subtraction, ePSF-based photometry, Gaia/`calibrimbore`-based zeropoint
calibration (accounting for atmospheric corrections), and cron-friendly orchestration with persistent
failure-tracking so re-runs don't re-attempt known-bad frames.

This package was developed for the University of Canterbury's Whetu server.
Configure data locations for your deployment using environment variables (see
[Configuring pipeline data locations](#configuring-pipeline-data-locations)).
Importing the package and running `otehiwai-pouakai --help` require no data setup.

## Quickstart

Python 3.10 and 3.11 are supported. The examples below use Python 3.11.

### 1. Clone the repository

```bash
git clone https://github.com/UC-MJO/Otehiwai_Pouakai.git
cd Otehiwai_Pouakai
```

### 2. Install

If Astrometry.net is already installed, use either **uv** or **pip**.

**uv**

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python .
source .venv/bin/activate
```

**pip**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

To install the pipeline **and Astrometry.net** together, use **conda**:

```bash
conda env create --solver libmamba -f environment.yml
conda activate Pouakai
```

### 3. Configure your data

Before processing observations, configure [pipeline data locations](#configuring-pipeline-data-locations) and [Astrometry.net indices](#solve-field-astrometrynet). See the [CDBS reference data notes](#pysynphot-cdbs-reference-data) for workflows that require additional pysynphot data.

### 4. Run the end-to-end example (on Whetu)

First adjust `SAVE_LOCATION`, `TEST_NIGHT_GLOB` and `NUM_CORES` in `scripts/run_test_20250914.py`, then run:

```bash
source scripts/whetu_env.sh
python scripts/run_test_20250914.py
```

See [Running the full pipeline](#running-the-full-pipeline) to adapt the script to your needs.

## Repository layout

```
Otehiwai_Pouakai/
├── pyproject.toml          # package metadata + dependencies
├── environment.yml         # complete environment including Astrometry.net
├── src/otehiwai_pouakai/   # the installable library
│   ├── pipeline.py         #   orchestration (was otehiwai_pouakai.py) -- Pouakai class, setup_logging, main()
│   ├── config.py           #   all filesystem locations, via environment variables
│   ├── organise_files.py   #   raw-archive discovery/cataloguing
│   ├── dark_masters.py, flat_masters.py   #   master calibration frames
│   ├── core_reduction.py, background_subtraction.py
│   ├── wcs_compute.py      #   astrometry.net solve-field wrapper
│   ├── calibration_saurus.py, psf_photometry.py, spatial_epsf.py, frame_quality.py
│   ├── gaia_query.py, cross_process_semaphore.py
│   ├── failure_ledger.py, provenance.py, matau.py, running_stats.py, worker_logging.py
│   └── suppress_warnings.py
└── scripts/                 # standalone examples / one-off diagnostic CLIs
    ├── run_test_20250914.py
    ├── run_psf_photometry_example.py
    ├── calibration_diagnostics.py
    ├── diagnose_calibration_tolerances.py
    ├── open_image.py
    ├── whetu_env.sh         # source to select the existing Whetu data paths
    └── setup_cdbs_data.sh
```

`src/otehiwai_pouakai/` is what gets installed and imported as
`otehiwai_pouakai`. `scripts/` are personal/example entry points meant to
be copied and adapted (they still reference site-specific paths like
`/home/phys/astro8/MJArchive/octans/` in a couple of places, by design --
see the comments in each) rather than installed as part of the package.

## Installation

## solve-field (astrometry.net)

The pipeline invokes `solve-field` as a subprocess. The conda recipe installs
Astrometry.net 0.97; an existing installation can also be used if it supports NumPy 1.24.4.

The pipeline uses `solve-field` from your current `PATH`, including an active
conda environment. To select a different installation explicitly:

```bash
export POUAKAI_ASTROMETRY_BIN=/path/to/astrometry/bin
```

That directory must contain an executable `solve-field`. It is added to the
solver subprocess's `PATH` so its helper programs can be found; the pipeline
leaves the parent process's `PATH` unchanged. Leave the variable unset or empty
to use your current `PATH`.

**Index files are separate from the software.** Configure
`$CONDA_PREFIX/etc/astrometry.cfg` for the conda installation (or the config
used by your external solver), for example:

```text
cpulimit 120
add_path /absolute/path/to/astrometry-indices
autoindex
```

See the [Astrometry.net instructions](https://astrometry.net/use.html) for choosing and
configuring indices.

## pysynphot CDBS reference data

Calibrimbore uses pysynphot, but its normal saved-state calibration workflow
uses bundled spectra and passbands and does not require the full CDBS tree.
For workflows that use additional STScI reference data, set the standard
`PYSYN_CDBS` variable **before importing calibrimbore or pysynphot**:

```bash
export PYSYN_CDBS=/absolute/path/to/cdbs
```

Pouakai does not set this variable automatically or create the directory. If
set, it must point to an existing, readable directory when calibration starts.

If unset, pysynphot attempts remote reference-table lookups during import.
Failed lookups can cause startup delays and warnings, but do not prevent
saved-state calibration. The Whetu setup script points to the existing CDBS
directory to avoid these remote lookups.

`scripts/setup_cdbs_data.sh <source> [dest]` can copy an existing CDBS tree and
print the corresponding export.

## Configuring pipeline data locations

Set `POUAKAI_HOME` to choose a root for catalogues, calibration states and master
frames. It defaults to `pouakai_data/` in the **current working directory**: the
directory you run Python from, even if the calling script lives elsewhere.
This keeps generated catalogues and masters alongside your project for inspection.
The repository ignores `pouakai_data/`; add the same entry to your own project's
`.gitignore` if needed. Individual variables override the root:

| Variable | Default | Used for |
|----------|---------|----------|
| `POUAKAI_RAW_ARCHIVE_DIR` | None; required for archive discovery | Existing raw FITS archive to scan |
| `POUAKAI_CAL_LIST_DIR` | `$POUAKAI_HOME/cal_lists` | Image and master catalogue CSVs |
| `POUAKAI_CAL_FILES_DIR` | `$POUAKAI_HOME/cal_files` | Existing calibrimbore sauron state files |
| `POUAKAI_MASTER_DARK_DIR` | `$POUAKAI_HOME/masters/darks` | Master dark outputs |
| `POUAKAI_MASTER_FLAT_DIR` | `$POUAKAI_HOME/masters/flats` | Master flat outputs |
| `POUAKAI_ASTROMETRY_BIN` | Use `PATH` | Optional Astrometry.net bin directory |
| `PYSYN_CDBS` | Unset | Optional external pysynphot reference data |

For example:

```bash
export POUAKAI_HOME=/path/to/pouakai-data
export POUAKAI_RAW_ARCHIVE_DIR=/path/to/existing/raw-archive
export POUAKAI_CAL_FILES_DIR=/path/to/existing/calibrimbore-states
```

Archive discovery requires an existing, readable input directory (`fli_dir=...` overrides the archive variable). Catalogue readers require existing CSV files; calibration requires existing state files.

On Whetu, source the deployment settings after activating your Python environment
and before starting the pipeline:

```bash
source /path/to/Otehiwai_Pouakai/scripts/whetu_env.sh
```

## Running the full pipeline

There are two equivalent ways to drive the pipeline: importing the
`Pouakai` class from Python (a script, a REPL, a Jupyter notebook), or
the `otehiwai-pouakai` command installed on `PATH`. Both go through the
exact same code -- the command-line tool is a thin `argparse` wrapper
(`build_arg_parser()`/`main()` in `pipeline.py`) that just builds the
same `Pouakai(...)` call for you. Pick whichever fits how you work;
see [Running individual stages](#running-individual-stages-red--wcs--cal-only)
below for the one place the two methods use different names for the
same option (input-file overrides for the wcs/cal stages).

### From Python

```python
from otehiwai_pouakai import Pouakai, setup_logging

logger, log_file = setup_logging('/path/to/save_location/')
Pouakai(files=[...], save_location='/path/to/save_location/',
        organise_files=True, make_masters=True, run=True, mode='modulo')
```

`files` can be any list of file paths -- build it however's convenient,
e.g. with `otehiwai_pouakai.matau.get_file_paths(glob_pattern)` (what
the CLI uses internally for `--glob`) or `glob.glob(...)` directly.

### From the command line

Once installed (`pip install -e .` registers the `otehiwai-pouakai`
console script -- see [Installation](#installation)):

```bash
otehiwai-pouakai --mode modulo --glob "/path/to/archive/20260714*/*.fit" \
    --save-location /path/to/save_location/
```

`--glob` and `--files` are mutually exclusive, alternative ways of
specifying the same `files` argument Python callers pass directly:

```bash
# equivalent to Python's files=[...]
otehiwai-pouakai --mode modulo --files /path/to/frame1.fit /path/to/frame2.fit \
    --save-location /path/to/save_location/
```

Every other `Pouakai(...)` keyword argument has a matching `--flag`
(e.g. `num_cores` → `--num-cores`, `wcs_order` → `--wcs-order`) -- see
the [full parameter table](#pouakai-parameters) below, or run
`otehiwai-pouakai --help` for the authoritative, always-current list
straight from `argparse`. Boolean flags that default `True` in Python
(`organise_files`, `make_masters`) are switched off on the CLI with a
`--no-...` flag instead of e.g. `--organise-files false` (i.e.
`--no-organise`, `--no-masters`).

A cron-friendly nightly run (safe to re-run -- already-processed files
are skipped per-stage, and known-bad frames are skipped too unless
retried, see [Quickstart](#quickstart-from-scratch)):

```bash
0 14 * * * /path/to/envs/Pouakai/bin/otehiwai-pouakai --mode modulo \
    --glob "/home/phys/astro8/MJArchive/octans/$(date +%Y%m%d)*/*.fit" \
    --save-location /home/users/<you>/Otehiwai_Nightly/ \
    >> /home/users/<you>/logs/pouakai_cron.log 2>&1
```

Use the env's own interpreter/console-script path explicitly in cron
(as above) rather than relying on `conda activate`, since cron doesn't
run an interactive login shell by default.

See `scripts/run_test_20250914.py` for a fuller worked example
(organise → build masters → reduce → WCS-solve → calibrate → per-stage
failure summary).

### `Pouakai(...)` parameters

The full constructor signature:

```python
Pouakai(files, save_location, num_cores=1,
        dark_exp_tol=1, dark_date_tol=3, dark_delta_t=7,
        flat_exp_tol=3, flat_date_tol=30, flat_delta_t=15,
        wcs_order=3, wcs_cpulimit=300, wcs_subprocess_timeout=330,
        make_masters=True, organise_files=True,
        run=True, mode='modulo', overwrite=True,
        bkg_box_size=100, bkg_filter_size=3,
        match_tol_px=2.5, isolation_radius_px=21.0,
        max_contamination_frac=0.05, max_calibration_stars=150,
        group_min_separation_px=None, group_min_separation_fwhm_factor=2.0,
        max_group_size=25,
        use_grouping=True,
        psf_error_inflation_max_scale=8.0,
        epsf_sampling_candidates=(3, 2),
        assess_spatial_variation=True, subtract_background=True)
```

**Setup / control flow**

| Parameter | Default | Meaning |
|---|---|---|
| `files` | *required* | List of input science file paths for this run (e.g. from `matau.get_file_paths(glob_pattern)`). |
| `save_location` | *required* | Output root -- `red/`, `wcs/`, `cal/`, `fig/`, `zp/`, `phot_table/`, `logs/` are created under here. |
| `num_cores` | `1` | Parallelism (joblib) for organising files, building masters, reducing, WCS-solving, and calibrating. |
| `organise_files` | `True` | If True, run `organise_fli_files` first to catalog any new raw files (see `organise_files.py`). |
| `make_masters` | `True` | If True, build any master dark/flat frames not yet present before reduction (skipped/cheap if already built). |
| `run` | `True` | If False, stop after the organise/master-building setup stages -- useful for a "just refresh the catalogs and masters" run without processing any science frames. |
| `mode` | `'modulo'` | Which stage(s) to run: `'modulo'` = reduction → WCS → calibration (the full chain); `'red'` = reduction only; `'wcs'` = WCS-solving only; `'cal'` = calibration only. |
| `overwrite` | `True` | Whether reduction is allowed to overwrite an existing output file for a frame. |

**Dark/flat master matching tolerances** -- `_date_tol` controls how far
(in days) a science frame may be from a candidate master's timestamp to
be considered a match at lookup time; `_delta_t` controls how wide a
time window individual calibration frames are clustered into when
*building* a master in the first place; `_exp_tol` is the allowed
exposure-time mismatch (seconds), used both when clustering and when
matching. See `diagnose_calibration_tolerances.py` if you're unsure
what values are appropriate for your archive's actual calibration
cadence.

| Parameter | Default | Meaning |
|---|---|---|
| `dark_exp_tol` | `1` | Max exposure-time difference (s) for dark clustering/matching. |
| `dark_date_tol` | `3` | Max days between a science frame and its matched master dark. |
| `dark_delta_t` | `7` | Time window (days) individual dark frames are clustered into one master-build group. |
| `flat_exp_tol` | `3` | Max exposure-time difference (s) for flat clustering/matching. |
| `flat_date_tol` | `30` | Max days between a science frame and its matched master flat (flats drift with dust/illumination, so this is looser than darks). |
| `flat_delta_t` | `15` | Time window (days) individual flat frames are clustered into one master-build group. |

**Background subtraction** (`background_subtraction.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `subtract_background` | `True` | Whether to estimate and subtract sky background during reduction at all. |
| `bkg_box_size` | `100` | Tile size (px) for the tiled background estimate -- smaller tracks finer spatial structure but is noisier per-tile. |
| `bkg_filter_size` | `3` | Box-to-box smoothing (in tiles) applied on top of the tiled background estimate. |

**WCS solving** (`wcs_compute.py`, via `solve-field`)

| Parameter | Default | Meaning |
|---|---|---|
| `wcs_order` | `3` | SIP tweak polynomial order passed to `solve-field`. |
| `wcs_cpulimit` | `300` | Seconds passed to `solve-field --cpulimit`; astrometry.net gives up gracefully after this many CPU-seconds. |
| `wcs_subprocess_timeout` | `330` | Hard wall-clock backstop (s) enforced by Python's own `subprocess`, in case `--cpulimit` alone doesn't bound wall-clock time. Should stay somewhat larger than `wcs_cpulimit`. |

**Photometric calibration** (`calibration_saurus.py` / `psf_photometry.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `match_tol_px` | `2.5` | Positional match tolerance (px) between detected sources and the reference (Gaia) catalog. |
| `isolation_radius_px` | `21.0` | Search radius (px) used to estimate flux contamination from neighbours for each calibration star. |
| `max_contamination_frac` | `0.05` | Calibration stars with more than this fraction of contaminating flux within `isolation_radius_px` are excluded. |
| `max_calibration_stars` | `150` | Caps the number of (brightest) stars fed into ePSF building/PSF photometry, for numerical stability. |
| `use_grouping` | `True` | If True, stars close enough together are fit simultaneously in groups (more accurate for blends); if False, every source is fit independently, which structurally avoids a rare `astropy.modeling` recursion failure at the cost of some blended-star accuracy. |
| `group_min_separation_px` | `None` | Minimum separation (px) for `SourceGrouper`'s simultaneous-fit groups. If `None` (default), derived per-frame from the frame's own measured FWHM (see `group_min_separation_fwhm_factor`) instead of a fixed value, so grouping tracks each frame's actual PSF width. Ignored if `use_grouping=False`. |
| `group_min_separation_fwhm_factor` | `2.0` | Multiplier on the frame's measured FWHM used to derive `group_min_separation_px` when that's `None`. |
| `max_group_size` | `25` | Hard ceiling on simultaneous PSF-fit group size, regardless of `group_min_separation_px` -- protects against pathologically large groups in dense fields. Ignored if `use_grouping=False`. |
| `epsf_sampling_candidates` | `(3, 2)` | ePSF oversampling factors to try, most preferred first, falling back with a logged warning if a frame's star sample can't support the preferred value. |
| `psf_error_inflation_max_scale` | `8.0` | Ceiling on the empirical inflation applied to PSF flux errors (formal PSF-fit covariance is typically optimistic) -- see `psf_photometry.inflate_psf_errors`. If errors are consistently hitting this ceiling, check `calibration_diagnostics.py --full` to see whether that particular field needs a different scale. |
| `assess_spatial_variation` | `True` | Diagnostic-only check for spatially varying zeropoint bias across the frame -- does not modify the image or the zeropoint used; purely informational logging. |

## Running individual stages: red / wcs / cal only

`mode` (Python) / `--mode` (CLI) controls how much of the chain runs:

| `mode` | Runs | Typical use |
|---|---|---|
| `'modulo'` (default) | reduction → WCS → calibration, in order | a normal full run |
| `'red'` | reduction only | just bias/dark/flat-correct and background-subtract raw frames |
| `'wcs'` | WCS-solving only | astrometrically solve some already-reduced frames |
| `'cal'` | calibration only | photometrically calibrate some already-WCS-solved frames |

`organise_files` and `make_masters` (Python) / the absence of
`--no-organise` and `--no-masters` (CLI) still run before whichever
stage `mode` selects, since reduction depends on having up-to-date
catalogs and master calibration frames available -- pass
`organise_files=False, make_masters=False` (`--no-organise
--no-masters`) too if you want to skip straight to the selected stage
with no setup work at all.

Each single-stage mode has its own dedicated input-override option, so
you don't need the frames to already sit inside a `save_location/red/`
or `save_location/wcs/` folder from a prior run of *this* pipeline --
useful for re-running just one stage after a parameter change, or for
feeding in frames prepared some other way entirely. Priority when more
than one is given: explicit file list > glob pattern > directory
(globbed for `*.fits.gz`) > the stage's own default location.

| Stage | Default input | Python override(s) | CLI override(s) |
|---|---|---|---|
| `red` | the `files` list you pass in | *(none needed -- `files` already is the input)* | *(none needed -- `--glob`/`--files` already is the input)* |
| `wcs` | `save_location/red/*.fits.gz` | `wcs_input_files`, `wcs_input_dir`, `wcs_input_glob` | `--wcs-input-dir`, `--wcs-input-glob` |
| `cal` | `save_location/wcs/*.fits.gz` | `cal_input_files`, `cal_input_dir`, `cal_input_glob` | `--cal-input-dir`, `--cal-input-glob` |

Outputs always land under *this run's* `save_location` (`wcs/` for the
WCS stage, `cal/`/`phot_table/`/`zp/` for calibration) regardless of
where the input frames came from.

### Reduction only (`red`)

**Python:**
```python
from otehiwai_pouakai import Pouakai, setup_logging

logger, log_file = setup_logging('/path/to/save_location/')
Pouakai(files=[...], save_location='/path/to/save_location/', mode='red')
```

**Command line:**
```bash
otehiwai-pouakai --mode red --glob "/path/to/archive/20260714*/*.fit" \
    --save-location /path/to/save_location/
```

`files`/`--glob`/`--files` is still how you tell the reduction stage
*what to reduce* -- there's no separate override for it, since it's
already exactly the input list.

### WCS-solving only (`wcs`)

Defaults to solving whatever's already in `save_location/red/` from a
prior reduction run:

**Python:**
```python
Pouakai(files=[], save_location='/path/to/save_location/', mode='wcs',
        organise_files=False, make_masters=False)
```

**Command line:**
```bash
otehiwai-pouakai --mode wcs --save-location /path/to/save_location/ \
    --no-organise --no-masters
```

(`files=[]` / omitting `--glob`/`--files` is fine here -- `red` isn't
running, so there's nothing for the raw-frame list to feed. The CLI
only requires `--glob`/`--files` when it can't otherwise tell what a
`mode=wcs`/`mode=cal` run should act on -- see `stage_input_given` in
`pipeline.py`'s `main()` -- so an explicit input override, as below,
also satisfies it.)

To WCS-solve a specific folder or file subset instead of
`save_location/red/`:

**Python:**
```python
Pouakai(files=[], save_location='/path/to/save_location/', mode='wcs',
        organise_files=False, make_masters=False,
        wcs_input_dir='/some/other/folder/of/reduced_frames/')
# or: wcs_input_glob='/some/other/folder/*_reduced.fits.gz'
# or: wcs_input_files=['/path/a.fits.gz', '/path/b.fits.gz']
```

**Command line:**
```bash
otehiwai-pouakai --mode wcs --save-location /path/to/save_location/ \
    --wcs-input-dir /some/other/folder/of/reduced_frames/
# or: --wcs-input-glob '/some/other/folder/*_reduced.fits.gz'
```

(There's no `--wcs-input-files` flag for an explicit list on the CLI --
use `--wcs-input-glob` with a pattern that matches just those files, or
drive it from Python instead.)

### Calibration only (`cal`)

Defaults to calibrating whatever's already in `save_location/wcs/`:

**Python:**
```python
Pouakai(files=[], save_location='/path/to/save_location/', mode='cal',
        organise_files=False, make_masters=False)
```

**Command line:**
```bash
otehiwai-pouakai --mode cal --save-location /path/to/save_location/ \
    --no-organise --no-masters
```

To calibrate a specific folder of already WCS-solved frames instead
(they don't need to follow this pipeline's `_wcs` filename convention
or live under this run's `save_location/wcs/` at all):

**Python:**
```python
Pouakai(files=[], save_location='/path/to/save_location/', mode='cal',
        organise_files=False, make_masters=False,
        cal_input_dir='/home/users/<you>/Pouakai_Test_20250914/wcs')
# or: cal_input_glob='/some/folder/*_wcs.fits.gz'
# or: cal_input_files=['/path/a_wcs.fits.gz', '/path/b_wcs.fits.gz']
```

**Command line:**
```bash
otehiwai-pouakai --mode cal --save-location /path/to/save_location/ \
    --cal-input-dir /home/users/<you>/Pouakai_Test_20250914/wcs
# or: --cal-input-glob '/some/folder/*_wcs.fits.gz'
```

Calibration-specific tuning parameters (`--match-tol-px`,
`--max-calibration-stars`, `--no-grouping`, etc.) apply here the same
as in a full `modulo` run -- see the
[photometric calibration parameter table](#pouakai-parameters) above,
or `otehiwai-pouakai --help`.

## Photometry only (no full pipeline)

If you just want PSF photometry on an already-reduced frame -- without
running organise/master-building/reduction/WCS-solving/calibration --
`psf_photometry.py` (and its ePSF-building and spatial-variation
support in `spatial_epsf.py`) can be used directly.
`scripts/run_psf_photometry_example.py` is a worked, copy-and-adapt
example wrapping this in a small `PSFPhotometryRunner` class.

**This is a Python class, not a `Pouakai`-style installed CLI tool** --
there's no `otehiwai-pouakai`-equivalent console script for photometry
alone, and no `--flag` options to look up. `scripts/` is explicitly
"copy and adapt", not installed as part of the package (see
[Repository layout](#repository-layout)), so it isn't on `PATH` or
importable by name from just anywhere either way -- run it from inside
`scripts/`, or copy `run_psf_photometry_example.py` next to your own
code first. Below are both ways to actually run it.

### Non-command-line: interactive Python / a notebook

The normal way to use this -- import the class and call `.run(...)`
directly, from a REPL, a Jupyter notebook, or your own script:

```python
from run_psf_photometry_example import PSFPhotometryRunner

runner = PSFPhotometryRunner('reduced_frame.fits')

result = runner.run(x=1024.3, y=987.1)                      # single target, pixel coords
result = runner.run(ra=83.6331, dec=-5.3911)                 # single target, sky coords
result = runner.run(x=[10, 20, 30], y=[15, 25, 35])          # several targets, pixel coords
result = runner.run(targets='my_targets.csv')                # many targets, from a CSV (columns x,y OR ra,dec)
result = runner.run(snr_min=10)                               # every detected source above this SNR
result = runner.run(snr_min=10, spatial=True, nx=3, ny=3)    # spatially varying ePSF instead of one global model

runner.save(result, 'my_output.csv')
```

A single target and a list of targets go through the identical code
path (a scalar is just the N=1 case of "a list"), so there's nothing
different to learn between photometering one star and a few thousand.

### From the command line

Two options, depending on whether you want a reusable script or a
one-off result:

**Option A -- edit and run the example script directly.** Open
`scripts/run_psf_photometry_example.py`, edit the `if __name__ ==
'__main__':` block at the bottom (set `FITS_FILE` and whichever
`runner.run(...)` calls you want, then:

```bash
cd scripts
python run_psf_photometry_example.py
```

This is the same approach the [Quickstart](#quickstart-from-scratch)
uses for `run_test_20250914.py` -- these scripts are meant to be
edited in place for your own target/frame, not passed arguments.

**Option B -- a one-off, no-file-editing terminal command**, useful for
a quick check without touching the script itself:

```bash
cd scripts
python -c "
from run_psf_photometry_example import PSFPhotometryRunner
runner = PSFPhotometryRunner('/path/to/reduced_frame.fits')
result = runner.run(ra=83.6331, dec=-5.3911)
print(result[['x_fit', 'y_fit', 'ra', 'dec', 'flux_fit', 'mag_inst']])
runner.save(result, 'my_output.csv')
"
```

Both options must be run **from inside `scripts/`** (or with
`scripts/` on `PYTHONPATH`), since `run_psf_photometry_example.py` is a
standalone module, not something `pip install -e .` puts on `PATH` or
makes importable by name.

**`PSFPhotometryRunner(fits_file, ...)` setup options**

| Parameter | Default | Meaning |
|---|---|---|
| `fits_file` | *required* | Path to an already-reduced (background-subtracted, WCS-solved) FITS frame. |
| `fwhm_guess` | `3.2` | Starting FWHM guess (px) -- only used for the initial reference-star SNR cut before the real FWHM is measured from the frame itself. |
| `snr_min_reference` | `20` | SNR threshold for the reference-star sample the ePSF is built from (independent of any per-target SNR cut used later in `.run()`). |
| `max_reference_stars` | `200` | Cap on how many (brightest) reference stars are used to build the ePSF. |
| `sampling_candidates` | `(3, 2)` | ePSF oversampling factors to try, most preferred first -- see `psf_photometry.build_epsf_adaptive`. |

**`.run(...)` options** -- which target-selection mode is used is
decided by which of `targets` / `x,y` / `ra,dec` you pass (checked in
that order); with none of those, every detected source passing
`snr > snr_min` is photometered.

| Parameter | Default | Meaning |
|---|---|---|
| `x`, `y` | `None` | Pixel coordinates -- scalar or list/array (same length). |
| `ra`, `dec` | `None` | Sky coordinates (deg) -- scalar or list/array; converted to pixel coordinates via this frame's WCS. |
| `targets` | `None` | CSV path, DataFrame, or `(N, 2)` array-like; CSV/DataFrame needs columns `x,y` or `ra,dec`. |
| `snr_min` | `10` | SNR threshold used only when no explicit targets are given (detect-everything mode). |
| `spatial` | `False` | If True, use a spatially varying ePSF grid (`spatial_epsf.build_spatial_epsf`) instead of one global ePSF -- each target uses its nearest grid node's own model. See `nx`/`ny`/`min_stars_per_node`. |
| `nx`, `ny` | `3`, `3` | Spatial ePSF grid dimensions (only used if `spatial=True`). |
| `min_stars_per_node` | `15` | Minimum calibration stars required for a grid node to get its own ePSF; nodes with fewer fall back to the global ePSF (only used if `spatial=True`). |
| `zeropoint` | `None` | Optional zeropoint (mag) to also report an approximate calibrated magnitude -- a convenience for this example script, not a substitute for the full `calibration_saurus.cal_photom` zeropoint pipeline. |

Returns a `pandas.DataFrame` with fitted position, flux, sky
coordinates, and instrumental magnitude, one row per target.

## Development

```bash
python -m pip install -e ".[dev]"
python -m build
```

To check the distribution, install the resulting wheel into a fresh environment
and run `python -m pip check`. Configure data locations before processing data.

Tests use temporary directories and synthetic inputs:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Installation troubleshooting

- Use Python 3.10 or 3.11; calibrimbore/pysynphot currently prevent Python 3.12+.
- Keep NumPy at 1.24.4 and runtime setuptools below 81. Calibrimbore supplies the setuptools bound because pysynphot imports `pkg_resources`.
- pysynphot 2.0.0 falls back to Python for spectral binning: isolated builds can select incompatible NumPy 2, and an upstream import bug also prevents the optional C extension from loading. Enabling it requires compatible build-time NumPy and an import fix. The fallback is slower for binning, but calibrimbore's main synthetic-photometry calculation uses its own NumPy integration.
- If pip says it is defaulting to a user installation, check `python -c "import sys; print(sys.executable)"` and use the intended environment's `python -m pip`. Do not install into a shared base environment.
- If conda's classic solver stalls, use the documented `--solver libmamba` option.
- A `ConfigurationError` names the missing setting or input path. Check the relevant environment variable.
- A solver error about missing indices requires configuring `astrometry.cfg`;
  reinstalling the Python package will not supply index files.
