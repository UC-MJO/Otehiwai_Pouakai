# Otehiwai Pouakai

End-to-end FLI/B&C astronomical image reduction, astrometric (WCS), and
photometric calibration pipeline: dark/flat master-building, advanced structure background
subtraction, ePSF-based photometry, Gaia/`calibrimbore`-based zeropoint
calibration (accounting for atmospheric corrections), and cron-friendly orchestration with persistent
failure-tracking so re-runs don't re-attempt known-bad frames.

This package was designed for and is deployed on the **University of
Canterbury's Whetu server** -- the shared-storage paths documented below
(`/home/phys/astro8/...`, `/home/phys/astronomy/...`) are that
deployment's specific layout. If you're installing this elsewhere, the
same paths are all overridable via environment variables (see
[Configuring pipeline data locations](#configuring-pipeline-data-locations)).

## Quickstart: from scratch

Every command needed to go from a bare checkout to a first successful
test run, in order. Each step links to fuller detail further down if
something goes wrong.

```bash
# 1. Clone and install (see Install below for what each line does).
#    conda is used only for the Python interpreter -- pip installs
#    everything else. numpy<1.24 (pinned in pyproject.toml) avoids a
#    numpy/solve-field incompatibility (see step 2's note below); no
#    manual patching of any shared install required.
git clone https://github.com/UC-MJO/Otehiwai_Pouakai.git
cd Otehiwai_Pouakai
conda create -n Pouakai python=3.11.15 -c conda-forge -y --copy && conda activate Pouakai
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
conda update ca-certificates -y
pip install -e . 
pip install -e ".[calibrimbore]" 
pip install --upgrade certifi

# 2. Confirm the install itself is sound before running anything real
python -c "import otehiwai_pouakai; print(otehiwai_pouakai.__version__)"
which python; which pip          # both should be inside .../envs/Pouakai/...
pip list | wc -l                 # should be a short, mostly-this-project's-deps list --
                                  # if it's huge (100+), see Known issues
echo $PYSYN_CDBS                 # should print /home/phys/astronomy/Pysynphot_Files/
solve-field --help | head -1     # should succeed with no PATH/import errors

# 3. Run the worked example end-to-end (organise -> masters -> reduce ->
#    WCS-solve -> calibrate -> per-stage failure summary)
cd scripts
python run_test_20250914.py
```

`run_test_20250914.py` is a real, adapt-before-reusing example (it
targets one specific night's data as a smoke test) -- see
[Running the full pipeline](#running-the-full-pipeline) for how to
point the `Pouakai` class at your own file list once this succeeds.

If step 3 reports other failures, its printed summary already breaks
them down by stage and reason (pulled from `failure_ledger.py`) --
check that first. If step 1 or 2 itself fails, jump to
[Known issues](#known-issues); every failure mode hit so far while
setting this up is documented there with its exact fix.

`run_test_20250914.py` has `RETRY_KNOWN_FAILURES = True` by default, so
if you're re-running it after fixing something upstream (e.g. after
step 2 above, or after any other fix), previously-recorded failures at
every stage are retried automatically rather than skipped as
"known-bad" -- no need to clear `logs/failed_files.csv` by hand first.

## Repository layout

```
Otehiwai_Pouakai/
├── pyproject.toml          # package metadata + dependencies
├── environment.yml         # full-conda-solve alternative (see Install)
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
    └── setup_cdbs_data.sh
```

`src/otehiwai_pouakai/` is what gets installed and imported as
`otehiwai_pouakai`. `scripts/` are personal/example entry points meant to
be copied and adapted (they still reference site-specific paths like
`/home/phys/astro8/MJArchive/octans/` in a couple of places, by design --
see the comments in each) rather than installed as part of the package.

## Install (recommended)

```bash
git clone https://github.com/UC-MJO/Otehiwai_Pouakai.git
cd Otehiwai_Pouakai
conda create -n Pouakai python=3.11.15 -c conda-forge -y --copy && conda activate Pouakai
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
conda update ca-certificates -y
pip install -e . 
pip install -e ".[calibrimbore]"
pip install --upgrade certifi
```

This uses conda only for the Python interpreter (`--copy` avoids
inheriting read-only permissions from a shared package cache -- see
[Known issues](#known-issues) if you still see this), then lets `pip`
install everything else from `pyproject.toml` as prebuilt wheels from
PyPI. This is deliberately **not** `conda env create -f
environment.yml` -- conda's own dependency solver can take a very long
time (multiple hours isn't unusual) resolving this many pinned
packages, whereas `conda create -n Pouakai python=3.11` only has to
solve for a single package. See
[Alternative: full conda solve](#alternative-full-conda-solve) below
if you'd rather have conda manage every binary itself anyway.

A few things worth knowing about that sequence:

- **Python >=3.10 is required, not just preferred.** `calibrimbore`
  requires astroquery's GitHub `main` branch (see below), which has
  dropped Python <3.10 support -- Python 3.9 will fail during
  calibration even if the install itself succeeds. 3.11 is the default
  here; Python 3.9 is no longer a safe substitute despite earlier
  versions of this README saying otherwise (Python 3.9 also reached
  end-of-life in October 2025, no more security patches, which was the
  original reasoning for preferring 3.11 anyway).
- **`export PYTHONNOUSERSITE=1` before the `pip install` steps** makes
  Python ignore your personal `~/.local` site-packages entirely for
  those installs -- keep it set, since it protects against `pip`
  silently landing packages outside the env on a shared/misconfigured
  install.
- **`python -m pip install --upgrade pip setuptools wheel` before `pip
  install -e .`.** A freshly created env's bundled `pip` can be old
  enough to predate PEP 660 (editable installs from a
  `pyproject.toml`-only project, no `setup.py`), which fails with
  `Directory cannot be installed in editable mode ... editable mode
  currently requires a setuptools-based build`. Upgrading `pip` first
  avoids this.
- `numpy` is deliberately pinned `<1.24` in `pyproject.toml` -- see
  [solve-field: numpy compatibility](#solve-field-fails-with-attributeerror-module-numpy-has-no-attribute-bool)
  for why. This works fine on Python 3.11 (numpy added 3.11 support
  back in the 1.23.x series).
- **`calibrimbore` is a separate step on purpose.** It isn't on PyPI, so
  `pip install -e .` alone won't pull it in -- it's an opt-in extra
  (`pip install -e ".[calibrimbore]"`) rather than a hard dependency, so
  a plain install doesn't silently reach out to GitHub and build a
  third-party package. **Run it from inside the repo directory,
  referencing the local checkout with `.`** -- `pip install
  "otehiwai-pouakai[calibrimbore]"` (by package name alone) will always
  fail, since this package isn't published on PyPI. For a reproducible
  build, pin to a specific commit instead of tracking `main`:
  ```bash
  pip install git+https://github.com/CheerfulUser/calibrimbore.git@<commit_hash>
  ```
- **`astroquery`, installed from GitHub `main`, not the last PyPI
  release.** This is required, not just preferred: calibrimbore's own
  README states it needs astroquery's master branch, and using an
  older release causes `sauron.estimate_mag` to fail during
  calibration with cryptic errors (confirmed empirically). For a reproducible
  build, pin to a specific commit instead of tracking `main`:
  ```
  "astroquery @ git+https://github.com/astropy/astroquery.git@<commit_hash>"
  ```

### Alternative: full conda solve

```bash
git clone https://github.com/UC-MJO/Otehiwai_Pouakai.git
cd Otehiwai_Pouakai
conda env create -f environment.yml
conda activate Pouakai
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[calibrimbore]"
```

Lets conda resolve and install every dependency's binary itself
(`astroscrappy`, `sep`, `scikit-image`, `pysynphot` all have compiled
C/Cython extensions), which is more thorough but can mean a very long
solve time for this many pinned packages -- see
[Known issues](#known-issues) for ways to speed that up (the
`libmamba` solver, or `micromamba` as a standalone alternative that
doesn't touch a shared `base` env at all) if you go this route.
`environment.yml` pins Python 3.11 (same as the recommended route) --
Python 3.9 is not a safe substitute here either, since `calibrimbore`
needs astroquery's `main` branch regardless of which install route you
use (see the astroquery bullet above).

### Alternative: plain venv + pip (not recommended and ill-tested)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[calibrimbore]"
```

You'll additionally need, on the system itself (not pip-installable):
- A C/C++ compiler (for `astroscrappy`, `sep` if no prebuilt wheel
  exists for your platform/Python version).
- `solve-field` (astrometry.net) -- see the next section.

## solve-field (astrometry.net)

`wcs_compute.py` shells out to astrometry.net's `solve-field` binary
(as a subprocess, not a Python import) to compute each frame's
astrometric WCS solution.

This site already has a manually-installed, correctly index-configured
astrometry.net build at `/usr/local/astrometry/`. `config.py` ensures
`solve-field` resolves to *that* build every time, ahead of anything
else that might also be on `PATH`: on import, it **prepends**
`/usr/local/astrometry/bin` (overridable via the `POUAKAI_ASTROMETRY_BIN`
environment variable) to `os.environ['PATH']` for the current process
and anything it spawns as a subprocess -- so this happens automatically,
without any shell profile edit, and without being able to be silently
shadowed by another `solve-field` earlier on `PATH`.

If you ever call `solve-field` by hand at a terminal, outside this
package, you'd still want the shell export too:
```bash
export PATH=$PATH:/usr/local/astrometry/bin
```
since the automatic fix only patches the environment as seen by the
Python process and its subprocesses.

### `solve-field` fails with `AttributeError: module 'numpy' has no attribute 'bool'`

This is a separate, unrelated issue from the PATH-shadowing one above --
it means `solve-field` is now correctly using this site's manual
astrometry.net build, but that build's own bundled Python helper
script (`removelines.py`, via `util/fits.py`) references deprecated
numpy scalar aliases (`np.bool`, `np.int`, etc.) that numpy's own
1.24.0 release notes confirm were fully removed in that version. This
is a bug in astrometry.net's own bundled code on disk at
`/usr/local/astrometry/`, entirely outside this repo.

**The fix used here: pin numpy `<1.24` in this project's own
environment, not the astrometry.net install.** The traceback for this
error shows `removelines` running with *this env's* Python/numpy
(`.../envs/Pouakai/lib/python3.X/site-packages/numpy/...`), not some
separate system Python -- so an environment we fully control is enough
to avoid the bug entirely, with no filesystem write access to
`/usr/local/astrometry/` needed. This is already the default in both
`pyproject.toml` and `environment.yml` (`numpy<1.24`, alongside Python
3.9 to match this site's known-working deployment).

**Alternative, if you want a modern numpy and have write access to
`/usr/local/astrometry/`:** patch the astrometry.net install directly
instead of pinning numpy down.

## pysynphot CDBS reference data

`calibration_saurus.py` calls into `calibrimbore`'s `sauron`, which

depends on `pysynphot`, which in turn needs the CDBS reference-file
tree pointed to by the standard `PYSYN_CDBS` environment variable. The
default for this deployment is:

```bash
export PYSYN_CDBS=/home/phys/astronomy/Pysynphot_Files/
```

`otehiwai_pouakai.config` also sets this exact path as a Python-level
default (`os.environ.setdefault('PYSYN_CDBS', ...)`) the moment the
package is imported, so it works even if you forget the shell export --
but add the export to a shell profile anyway if anything outside this
package (or outside Python entirely) also needs `PYSYN_CDBS` set.

If you'd rather keep your own personal copy of the CDBS files instead
of using the shared one above (e.g. for isolated testing), point
`PYSYN_CDBS` at that folder instead -- any directory with the expected
CDBS layout works, this doesn't have to be the shared location:

```bash
export PYSYN_CDBS=/home/<you>/Pysynphot_Files/
```

`scripts/setup_cdbs_data.sh <source> [dest]` is available if you ever
need to relocate an existing CDBS tree to a new location and want the
matching `export` line printed for you.

## Configuring pipeline data locations

Every filesystem location the pipeline reads from or writes to (raw
archive root, catalog CSVs, master dark/flat directories, `calibrimbore`
state files) is centralised in `src/otehiwai_pouakai/config.py`, with
defaults pointed at this deployment's actual shared storage:

| Variable                    | Default (this site)                                | Used for |
|------------------------------|-----------------------------------------------------|----------|
| `POUAKAI_RAW_ARCHIVE_DIR`    | `/home/phys/astro8/MJArchive/octans/`                | root of the raw FITS archive `organise_files.py` scans |
| `POUAKAI_CAL_LIST_DIR`       | `/home/phys/astronomy/Pouakai_cal_Lists/`            | master image/dark/flat/science catalog CSVs |
| `POUAKAI_CAL_FILES_DIR`      | `/home/phys/astronomy/Pouakai_cal_Files/`            | `calibrimbore` sauron state files |
| `POUAKAI_MASTER_DARK_DIR`    | `/home/phys/astronomy/Pouakai_Masters/Master_Darks/` | combined master dark frames |
| `POUAKAI_MASTER_FLAT_DIR`    | `/home/phys/astronomy/Pouakai_Masters/Master_Flats/` | combined master flat frames |
| `PYSYN_CDBS`                 | `/home/phys/astronomy/Pysynphot_Files/`              | pysynphot/calibrimbore reference data, see above |

Every variable above is optional at this site -- the defaults already
point at the shared folders in use here. Set the matching environment
variable only to override one for a different run or a different
deployment entirely (e.g. testing on a laptop with a local archive
copy). `POUAKAI_HOME` is a separate, lower-priority fallback used only
if you strip out a default above without setting its environment
variable; see `config.py`'s docstring if you need it.

These are **data** locations, unrelated to wherever you `git clone`d or
`pip install`ed the package itself -- there's no dependency between the
two.

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
console script -- see [Install](#install-recommended)):

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
pip install -e ".[dev]"
```

## Known issues

- **`pip install -e .` fails with `Directory cannot be installed in
  editable mode ... editable mode currently requires a setuptools-based
  build`.** The env's freshly-installed `pip` is too old to support
  PEP 660 editable installs from a `pyproject.toml`-only project (needs
  pip >=21.3) -- already why the recommended
  [Install](#install-recommended) commands run `python -m pip install
  --upgrade pip setuptools wheel` before `pip install -e .`. If you hit
  this, run that upgrade and retry. (This also explains a
  `ModuleNotFoundError: No module named 'otehiwai_pouakai'` on a later
  run, if `pip install -e .` silently failed this way earlier without
  you noticing.)

- **`conda env create -f environment.yml` runs for hours without
  finishing.** Conda's *classic* dependency solver can be extremely
  slow on a large pinned package set like this one -- this isn't
  specific to this repo, and the recommended
  [Install](#install-recommended) path avoids it entirely by not using
  `environment.yml` for a normal install. If you're deliberately using
  [Alternative: full conda solve](#alternative-full-conda-solve)
  anyway, two ways to speed it up:
  - **`libmamba` solver** (stays within `conda` itself):
    ```bash
    conda install -n base -c conda-forge conda-libmamba-solver -y
    conda config --set solver libmamba
    ```
    Note this one-off setup step itself still uses the *classic*
    solver (libmamba isn't active until after it's installed) -- if
    your `base` env already has hundreds of packages in it (common on
    a shared university install), even installing the solver plugin
    can be slow, for the same underlying reason.
  - **`micromamba`**, a standalone binary that doesn't touch `base` (or
    any shared conda install) at all, sidestepping that chicken-and-egg
    problem entirely:
    ```bash
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
    micromamba create -n Pouakai -f environment.yml -c conda-forge -y
    micromamba activate Pouakai
    ```
  Either way, re-run environment creation as normal afterwards -- same
  file, same result, just resolved in minutes instead of hours. If
  you're already stuck: `Ctrl+C`, then `conda env remove -n Pouakai`
  before retrying with the solver switched.

- **`solve-field` fails with `AttributeError: module 'numpy' has no
  attribute 'bool'`.** Astrometry.net's own bundled Python helper
  script uses a deprecated numpy scalar alias that numpy 1.24+ removed
  entirely -- a bug in that external code, not this repo, but one this
  project works around by default: `numpy<1.24` is pinned in both
  `pyproject.toml` and `environment.yml`, since the bundled script runs
  using *this env's* numpy (confirmed from its own traceback), so
  pinning it here needs no write access to the astrometry.net install
  itself. If you deliberately use a newer numpy instead, patch that
  install directly with `./scripts/patch_astrometry_numpy_compat.sh`.
  See [that section above](#solve-field-fails-with-attributeerror-module-numpy-has-no-attribute-bool)
  for details.

- **WCS stage fails with `no solution (.new file not produced)` across
  most/all frames, where it previously worked.** Almost certainly means
  `solve-field` is resolving to an unconfigured install (e.g.
  conda-forge's `astrometry` package, which has no index files by
  default) instead of this site's working manual install -- run
  `solve-field <any_reduced_frame.fits.gz>` by hand and look for "You
  must list at least one index in the config file" in the output to
  confirm. See [solve-field (astrometry.net)](#solve-field-astrometrynet)
  above; `environment.yml` no longer installs the conda-forge package
  for exactly this reason, and `config.py` now prepends (not just
  appends) the known-good bin directory to `PATH` so it can't be
  shadowed again.

- **`setuptools` must stay below 81.** `pysynphot` imports `pkg_resources`
  at its own import time; `setuptools>=81` breaks that import with
  `ModuleNotFoundError: No module named 'pkg_resources'`, confirmed
  empirically (2026-07) -- `setuptools==83.0.0` fails, `80.10.2` works.
  This is already pinned (`setuptools<81`) in both `pyproject.toml` and
  `environment.yml`, so a normal install won't hit it -- only relevant
  if something else in your environment forces a newer `setuptools`. If
  you hit the error anyway: `pip install "setuptools<81"`.

- **`pip install "otehiwai-pouakai[...]"` (by package name) will always
  fail with "No matching distribution found".** This package is not
  published on PyPI -- it only exists as your local checkout. Install
  extras with `pip install -e ".[calibrimbore]"` run from inside the
  repo directory instead (the leading `.` means "this checkout", not "a
  package named `.`").

- **`pip` reports "Defaulting to user installation because normal
  site-packages is not writeable" while a conda env is active.** This
  means `pip`/`python` on your `PATH` are not actually the active
  conda env's copies (they resolved to a different, non-writable
  Python install instead -- e.g. a shared system Anaconda base
  install). Everything will still appear to "work" but silently install
  into `~/.local` rather than the env, and later imports/extras
  resolution will behave inconsistently. Check with:
  ```bash
  conda activate Pouakai
  which python; which pip
  # both paths should contain .../envs/Pouakai/... -- if they instead
  # point at a shared/system anaconda install, conda isn't being
  # activated correctly in this shell (common on cluster login nodes
  # with an old shell or a PATH set before `conda init` runs). Try a
  # fresh login shell, or use the env's interpreter explicitly:
  ~/.conda/envs/Pouakai/bin/pip install -e .
  ```