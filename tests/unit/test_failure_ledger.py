"""Unit tests for failure-ledger persistence and filesystem behaviour."""

import pytest


pytestmark = pytest.mark.unit


def test_failure_ledger_reads_do_not_create_outputs(tmp_path):
    from otehiwai_pouakai import failure_ledger as ledger

    output = tmp_path / 'output'
    assert ledger.is_known_failure(output, 'wcs', 'frame.fits.gz') is None
    assert ledger.load_known_failures(output, 'wcs') == set()
    assert ledger.load_all_failures(output).empty
    assert ledger.summarize(output).empty
    ledger.clear_failure(output, 'wcs', 'frame.fits.gz')
    assert not output.exists()

    ledger.record_failure(output, 'wcs', 'frame.fits.gz', 'test failure')
    assert ledger.is_known_failure(output, 'wcs', 'frame.fits.gz') == 'test failure'
    ledger.clear_failure(output, 'wcs', 'frame.fits.gz')
    assert ledger.is_known_failure(output, 'wcs', 'frame.fits.gz') is None
