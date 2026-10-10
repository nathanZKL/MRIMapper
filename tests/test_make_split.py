"""Tests for src/make_split.py: create once, then always reuse."""

import csv
import json

import h5py
import numpy as np
import pytest

from config import EXPECTED_FILE_NUMBERS, SPLIT_CSV, SPLIT_JSON
from make_split import ensure_split, read_records
from splitting import ImageRecord, SplitError, assign_splits, load_split
from test_splitting import FRACTIONS, make_records


def test_first_run_creates_and_second_run_reuses(tmp_path):
    records = make_records()
    csv_path, json_path = tmp_path / "s.csv", tmp_path / "s.json"

    first, status1 = ensure_split(records, csv_path, json_path, seed=42, fractions=FRACTIONS)
    saved = csv_path.read_bytes(), json_path.read_bytes()
    # A different seed on the second run must NOT reshuffle: the saved split wins.
    second, status2 = ensure_split(records, csv_path, json_path, seed=7, fractions=FRACTIONS)

    assert (status1, status2) == ("created", "reused")
    assert first == second
    assert (csv_path.read_bytes(), json_path.read_bytes()) == saved


def test_half_written_split_is_an_error(tmp_path):
    records = make_records()
    csv_path, json_path = tmp_path / "s.csv", tmp_path / "s.json"
    ensure_split(records, csv_path, json_path, seed=42, fractions=FRACTIONS)
    json_path.unlink()

    with pytest.raises(SplitError):
        ensure_split(records, csv_path, json_path, seed=42, fractions=FRACTIONS)


# --- Missing or bad data stops the script (nothing is silently skipped) ---------

def write_fake_mat(path, label, pid):
    """A tiny file with the same layout as the real ones: cjdata/label and cjdata/PID."""
    with h5py.File(path, "w") as f:
        cj = f.create_group("cjdata")
        cj["label"] = np.array([[float(label)]])
        cj["PID"] = np.array([[ord(c)] for c in pid], dtype=np.uint16).reshape(-1, 1)


def fake_dataset(folder, n=3):
    for i in range(1, n + 1):
        write_fake_mat(folder / f"{i}.mat", label=i % 3 + 1, pid=f"P{i}")
    return range(1, n + 1)


def test_read_records_reads_every_file(tmp_path):
    numbers = fake_dataset(tmp_path)
    records = read_records(tmp_path, numbers)
    assert records == [ImageRecord("1.mat", "P1", 2), ImageRecord("2.mat", "P2", 3), ImageRecord("3.mat", "P3", 1)]


def test_missing_folder_stops(tmp_path):
    with pytest.raises(SystemExit, match="prepare_data"):
        read_records(tmp_path / "nope", range(1, 4))


def test_missing_file_stops_and_names_it(tmp_path):
    numbers = fake_dataset(tmp_path)
    (tmp_path / "2.mat").unlink()
    with pytest.raises(SystemExit, match=r"2\.mat"):
        read_records(tmp_path, numbers)


def test_unreadable_file_stops_and_names_it(tmp_path):
    numbers = fake_dataset(tmp_path)
    (tmp_path / "2.mat").write_bytes(b"this is not a MATLAB file")
    with pytest.raises(SystemExit, match=r"2\.mat"):
        read_records(tmp_path, numbers)


def test_invalid_label_stops_and_names_it(tmp_path):
    numbers = fake_dataset(tmp_path)
    write_fake_mat(tmp_path / "2.mat", label=4, pid="P2")
    with pytest.raises(SystemExit, match=r"2\.mat"):
        read_records(tmp_path, numbers)


def test_empty_patient_id_stops_and_names_it(tmp_path):
    numbers = fake_dataset(tmp_path)
    write_fake_mat(tmp_path / "2.mat", label=1, pid="")
    with pytest.raises(SystemExit, match=r"2\.mat"):
        read_records(tmp_path, numbers)


def test_dataset_change_is_caught_on_reuse(tmp_path):
    records = make_records()
    csv_path, json_path = tmp_path / "s.csv", tmp_path / "s.json"
    ensure_split(records, csv_path, json_path, seed=42, fractions=FRACTIONS)
    with pytest.raises(SplitError, match="dataset"):
        ensure_split(records[:-1], csv_path, json_path, seed=42, fractions=FRACTIONS)  # one image gone


# --- The real, saved split -------------------------------------------------------

needs_real_split = pytest.mark.skipif(not SPLIT_CSV.exists(),
                                      reason="real split not created yet (run src/make_split.py)")

# The 18 look-alike ID groups found in milestone 1 (reports/inventory_report.md),
# written out by hand. "" means the plain ID, e.g. MR040240 itself.
KNOWN_SUFFIX_GROUPS = {
    "MR024780": "BDEG", "MR026175": "CE", "MR029209": "EGI", "MR037458": "BC",
    "MR039473": ["", "B"], "MR040240": ["", "B", "C", "D"], "MR048294": ["", "B"],
    "MR048944": ["", "B"], "MR048994": ["", "B"], "MR049019": "CG", "MR049104": ["", "B"],
    "MR049453": ["", "B"], "MR050687": ["", "C"], "MR051461": ["", "B"],
    "MR051644": ["", "B", "C"], "MR051651": ["", "B"], "MR051796": ["", "B"], "MR053110": "BE",
}


def read_split_csv():
    with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@needs_real_split
def test_saved_real_split_loads_and_is_intact():
    assignment, _ = load_split(SPLIT_CSV, SPLIT_JSON)  # raises if edited or rule-breaking
    assert sorted(assignment) == sorted(f"{n}.mat" for n in EXPECTED_FILE_NUMBERS)


@needs_real_split
def test_saved_real_split_has_no_patient_in_two_splits():
    """Checked without pid_base(): known groups are spelled out, every other ID is its own patient."""
    rows = read_split_csv()
    patient_of = {base + s: base for base, suffixes in KNOWN_SUFFIX_GROUPS.items() for s in suffixes}
    present = {r["pid"] for r in rows}
    assert set(patient_of) <= present, sorted(set(patient_of) - present)

    splits_per_patient = {}
    for r in rows:
        splits_per_patient.setdefault(patient_of.get(r["pid"], r["pid"]), set()).add(r["split"])
    assert {p: s for p, s in splits_per_patient.items() if len(s) > 1} == {}
    assert len(splits_per_patient) == 209
    # The CSV's patient_group column must group rows exactly like the hand-written
    # list: one value per patient, and no value shared by two patients. (Its text can
    # differ: a lone suffixed ID like MR017260F is stored as group MR017260.)
    group_values = {}
    for r in rows:
        group_values.setdefault(patient_of.get(r["pid"], r["pid"]), set()).add(r["patient_group"])
    assert all(len(v) == 1 for v in group_values.values())
    assert len({r["patient_group"] for r in rows}) == 209


@needs_real_split
def test_saved_real_split_can_be_rebuilt_from_its_seed():
    """If this fails, the split code or Python's shuffle changed. Keep using the
    saved file (never regenerate split_v1) and investigate before any split_v2."""
    rows = read_split_csv()
    meta = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    records = [ImageRecord(r["file"], r["pid"], int(r["label"])) for r in rows]
    rebuilt = assign_splits(records, meta["fractions"], meta["seed"])
    assert rebuilt == {r["file"]: r["split"] for r in rows}
