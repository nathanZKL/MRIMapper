"""Tests for src/splitting.py.

They use small made-up datasets, so they run in well under a second and do
not need the real .mat files.
"""

import dataclasses
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from mat_loader import pid_base
from splitting import (SPLIT_NAMES, ImageRecord, SplitError, assign_splits,
                       check_label_consistency, input_fingerprint, load_split,
                       save_split, summarize_split, validate_split)

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


# Suffix patterns seen in the real data (docs/DATASET.md): "" is the plain ID.
# Some real groups have no plain ID at all (MR024780B/D/E/G), so neither do some here.
SUFFIX_PATTERNS = [["", "B"], ["", "B", "C"], ["B", "D", "E", "G"], ["C", "E"], ["E", "G", "I"], ["", "C"]]


def make_dataset(seed=0, groups_per_label=30):
    """A fake dataset shaped like Cheng's, plus the answer key.

    Returns (records, truth) where truth[file] is the patient each image really
    belongs to. The answer key is written down while generating, NOT computed
    with pid_base(), so tests using it would catch a broken pid_base().
    """
    rng = random.Random(seed)
    records, truth, number = [], {}, 1
    for label in (1, 2, 3):
        for g in range(groups_per_label):
            base = f"MR{label}{g:03d}"
            # Every 5th patient has several IDs that differ only by a trailing letter.
            suffixes = SUFFIX_PATTERNS[(g // 5) % len(SUFFIX_PATTERNS)] if g % 5 == 0 else [""]
            for suffix in suffixes:
                for _ in range(rng.randint(1, 40)):
                    records.append(ImageRecord(file=f"{number}.mat", pid=base + suffix, label=label))
                    truth[f"{number}.mat"] = f"patient-{label}-{g}"
                    number += 1
    return records, truth


def make_records(seed=0, groups_per_label=30):
    return make_dataset(seed, groups_per_label)[0]


def test_every_image_is_assigned_exactly_once():
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)

    assert sorted(assignment) == sorted(r.file for r in records)  # none missing, none extra
    assert set(assignment.values()) <= set(SPLIT_NAMES)            # only train / val / test


def test_patients_never_overlap_between_splits():
    records, truth = make_dataset()
    assignment = assign_splits(records, FRACTIONS, seed=42)

    # Checked against the answer key, not against pid_base(): a test that used the
    # same function as the code would agree with it even when both are wrong.
    splits_per_patient = {}
    for r in records:
        splits_per_patient.setdefault(truth[r.file], set()).add(assignment[r.file])
    leaking = {p: s for p, s in splits_per_patient.items() if len(s) > 1}
    assert leaking == {}
    # The made-up data must contain every suffix pattern, or this test proves less.
    pids = {r.pid for r in records}
    for suffix in "BCDEGI":
        assert any(p.endswith(suffix) for p in pids), suffix
    assert "MR1010B" in pids and "MR1010" not in pids  # a group with no plain ID


def test_same_seed_gives_identical_assignment():
    records = make_records()
    first = assign_splits(records, FRACTIONS, seed=42)
    second = assign_splits(list(reversed(records)), FRACTIONS, seed=42)  # input order must not matter

    assert first == second


def test_different_seed_gives_different_assignment():
    records = make_records()
    assert assign_splits(records, FRACTIONS, seed=1) != assign_splits(records, FRACTIONS, seed=2)


# Run in a brand-new Python process. Python orders sets of strings differently in
# each process (controlled by PYTHONHASHSEED), so this catches order bugs that a
# repeat inside one process can't.
_FRESH_PROCESS_CODE = """
import hashlib
from splitting import assign_splits
from test_splitting import FRACTIONS, make_records
a = assign_splits(make_records(), FRACTIONS, seed=42)
print(hashlib.sha256(repr(sorted(a.items())).encode()).hexdigest())
"""


def test_same_seed_gives_identical_assignment_in_fresh_processes():
    tests_dir = Path(__file__).parent
    pythonpath = os.pathsep.join([str(tests_dir.parent / "src"), str(tests_dir)])
    results = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": pythonpath}
        out = subprocess.run([sys.executable, "-c", _FRESH_PROCESS_CODE], env=env,
                             capture_output=True, text=True, check=True)
        results.add(out.stdout.strip())
    assert len(results) == 1, results


# --- Class balance ------------------------------------------------------------

def test_each_tumor_type_is_split_close_to_target_fractions():
    records = make_records(groups_per_label=60)
    summary = summarize_split(records, assign_splits(records, FRACTIONS, seed=42))

    for label in (1, 2, 3):
        total = sum(summary[s][label]["images"] for s in SPLIT_NAMES)
        for split in SPLIT_NAMES:
            assert summary[split][label]["groups"] > 0, f"no label {label} patients in {split}"
            share = summary[split][label]["images"] / total
            assert abs(share - FRACTIONS[split]) < 0.05, (label, split, share)


def test_summary_counts_add_up():
    records = make_records()
    summary = summarize_split(records, assign_splits(records, FRACTIONS, seed=42))

    assert sum(summary[s][lab]["images"] for s in SPLIT_NAMES for lab in (1, 2, 3)) == len(records)
    assert sum(summary[s][lab]["groups"] for s in SPLIT_NAMES for lab in (1, 2, 3)) == \
        len({pid_base(r.pid) for r in records})


# --- Label consistency --------------------------------------------------------

def test_consistent_labels_pass():
    check_label_consistency(make_records())  # must not raise


def test_patient_id_with_two_labels_is_rejected():
    records = make_records() + [ImageRecord("9998.mat", "P1", 1), ImageRecord("9999.mat", "P1", 3)]
    with pytest.raises(SplitError, match="P1"):
        check_label_consistency(records)


def test_patient_group_with_two_labels_is_rejected():
    # Different exact IDs, but the same group after removing the letter suffix.
    records = make_records() + [ImageRecord("9998.mat", "P7", 1), ImageRecord("9999.mat", "P7B", 2)]
    with pytest.raises(SplitError, match="P7"):
        check_label_consistency(records)


def test_assign_splits_refuses_inconsistent_labels():
    records = make_records() + [ImageRecord("9998.mat", "P1", 1), ImageRecord("9999.mat", "P1", 3)]
    with pytest.raises(SplitError):
        assign_splits(records, FRACTIONS, seed=42)


# --- Validation catches broken splits ------------------------------------------

def test_validate_accepts_a_correct_split():
    records = make_records()
    assert validate_split(records, assign_splits(records, FRACTIONS, seed=42)) == []


def test_validate_catches_missing_extra_and_unknown_entries():
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)
    del assignment[records[0].file]
    assignment["0.mat"] = "train"
    assignment[records[1].file] = "holdout"

    problems = " | ".join(validate_split(records, assignment))
    assert records[0].file in problems and "missing" in problems
    assert "0.mat" in problems
    assert "holdout" in problems


def test_validate_catches_a_patient_spread_over_two_splits():
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)
    group_members = [r for r in records if pid_base(r.pid) == pid_base(records[0].pid)]
    assert len(group_members) > 1
    current = assignment[group_members[0].file]
    assignment[group_members[0].file] = "test" if current != "test" else "train"

    problems = " | ".join(validate_split(records, assignment))
    assert pid_base(records[0].pid) in problems


# --- Saving, reloading and reusing ----------------------------------------------

def paths(tmp_path):
    return tmp_path / "split.csv", tmp_path / "split.json"


def test_save_then_load_returns_the_same_assignment(tmp_path):
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)
    csv_path, json_path = paths(tmp_path)
    save_split(csv_path, json_path, records, assignment, seed=42, fractions=FRACTIONS)

    loaded, meta = load_split(csv_path, json_path, records)
    assert loaded == assignment
    assert meta["seed"] == 42
    assert meta["fractions"] == FRACTIONS


def test_save_refuses_to_overwrite(tmp_path):
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)
    csv_path, json_path = paths(tmp_path)
    save_split(csv_path, json_path, records, assignment, seed=42, fractions=FRACTIONS)
    before = csv_path.read_bytes()

    with pytest.raises(FileExistsError):
        save_split(csv_path, json_path, records, assign_splits(records, FRACTIONS, seed=7),
                   seed=7, fractions=FRACTIONS)
    assert csv_path.read_bytes() == before


def test_load_detects_a_hand_edited_split_file(tmp_path):
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)
    csv_path, json_path = paths(tmp_path)
    save_split(csv_path, json_path, records, assignment, seed=42, fractions=FRACTIONS)

    # Move a single-image patient to another split: no group rule breaks, so only
    # the saved fingerprint can notice.
    single = next(r for r in records if sum(pid_base(x.pid) == pid_base(r.pid) for x in records) == 1)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(single.file + ","):
            old = assignment[single.file]
            lines[i] = line.rsplit(",", 1)[0] + "," + ("test" if old != "test" else "val")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SplitError, match="fingerprint"):
        load_split(csv_path, json_path, records)


def test_load_detects_that_the_dataset_changed(tmp_path):
    records = make_records()
    assignment = assign_splits(records, FRACTIONS, seed=42)
    csv_path, json_path = paths(tmp_path)
    save_split(csv_path, json_path, records, assignment, seed=42, fractions=FRACTIONS)

    changed = list(records)
    changed[0] = dataclasses.replace(changed[0], label=changed[0].label % 3 + 1)
    with pytest.raises(SplitError, match="dataset"):
        load_split(csv_path, json_path, changed)


def test_fingerprint_ignores_order_but_not_content():
    records = make_records()
    assert input_fingerprint(records) == input_fingerprint(list(reversed(records)))
    changed = [dataclasses.replace(records[0], pid="OTHER")] + records[1:]
    assert input_fingerprint(records) != input_fingerprint(changed)
