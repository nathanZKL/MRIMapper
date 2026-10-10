"""Checks the fast metadata reader against the full loader on real files."""

import pytest

from config import MAT_DIR
from mat_loader import load_metadata, load_slice, pid_base


# Real IDs from the dataset (reports/inventory_report.md) and the group each must
# map to. Written out by hand so a change to pid_base() can't go unnoticed.
@pytest.mark.parametrize("pid, group", [
    ("100360", "100360"),          # plain numeric ID: unchanged
    ("MR040240", "MR040240"),      # plain MR ID: unchanged
    ("MR040240B", "MR040240"),
    ("MR040240D", "MR040240"),
    ("MR024780G", "MR024780"),     # group whose plain ID never appears
    ("MR029209I", "MR029209"),
    ("MR050687C", "MR050687"),
])
def test_pid_base_on_real_ids(pid, group):
    assert pid_base(pid) == group

# First, last, and one of the 15 files that are 256x256 (1073.mat).
SAMPLE_FILES = ["1.mat", "1073.mat", "2453.mat", "3064.mat"]


@pytest.mark.skipif(not MAT_DIR.exists(), reason="dataset not extracted (run src/prepare_data.py)")
@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_load_metadata_matches_load_slice(name):
    full = load_slice(MAT_DIR / name)
    assert load_metadata(MAT_DIR / name) == (full.label, full.pid)
