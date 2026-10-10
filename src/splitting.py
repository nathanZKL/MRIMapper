"""Patient-grouped train / validation / test split (milestone 2).

The unit we split is the *patient group*: all images whose patient ID is the
same after mat_loader.pid_base() strips a trailing letter (MR040240,
MR040240B, ... count as one group). A group is never divided between splits,
so the model can't be tested on a patient it has already seen in training.

The split is stratified by tumor type: each type is split on its own, so all
three splits get roughly the same mix of meningioma / glioma / pituitary.

This module only computes, checks, saves and loads splits. make_split.py is
the script that runs it on the real dataset.
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from config import LABEL_NAMES, SOURCE
from mat_loader import pid_base

SPLIT_NAMES = ("train", "val", "test")
# Bump this if assign_splits() ever changes, so old and new splits can't be confused.
ALGORITHM_VERSION = 1
CSV_COLUMNS = ["file", "number", "pid", "patient_group", "label", "label_name", "split"]


class SplitError(Exception):
    """The data or a saved split breaks a rule; the message says which."""


@dataclass(frozen=True)
class ImageRecord:
    file: str   # e.g. "1.mat"
    pid: str    # patient ID exactly as stored in the file
    label: int  # 1 meningioma, 2 glioma, 3 pituitary tumor


def build_groups(records):
    """Return {patient_group: [ImageRecord, ...]}."""
    groups = defaultdict(list)
    for r in records:
        groups[pid_base(r.pid)].append(r)
    return dict(groups)


def check_label_consistency(records):
    """Raise SplitError if a patient ID, or a patient group, has more than one tumor type."""
    problems = []
    for kind, key in (("patient ID", lambda r: r.pid), ("patient group", lambda r: pid_base(r.pid))):
        labels = defaultdict(set)
        for r in records:
            labels[key(r)].add(r.label)
        for name, labs in sorted(labels.items()):
            if len(labs) > 1:
                problems.append(f"{kind} {name} has labels {sorted(labs)}")
    if problems:
        raise SplitError("Inconsistent tumor labels:\n  " + "\n  ".join(problems))


def assign_splits(records, fractions, seed):
    """Return {file: split_name}, keeping each patient group in one split.

    For each tumor type: sort its groups by name, shuffle them with `seed`,
    then give each group to the split furthest below its target number of
    images. Sorting first makes the result independent of input order (and of
    Python's per-run set ordering), so the same seed always gives the same split.
    """
    check_label_consistency(records)
    groups = build_groups(records)
    rng = random.Random(seed)
    assignment = {}

    groups_by_label = defaultdict(list)
    for name, members in groups.items():
        groups_by_label[members[0].label].append(name)

    for label in sorted(groups_by_label):
        names = sorted(groups_by_label[label])
        rng.shuffle(names)
        total = sum(len(groups[n]) for n in names)
        target = {s: fractions[s] * total for s in SPLIT_NAMES}
        placed = dict.fromkeys(SPLIT_NAMES, 0)
        for name in names:
            # Furthest below target; ties go to the earlier name in SPLIT_NAMES.
            split = max(SPLIT_NAMES, key=lambda s: target[s] - placed[s])
            placed[split] += len(groups[name])
            for r in groups[name]:
                assignment[r.file] = split
    return assignment


def validate_split(records, assignment):
    """Return a list of rule violations (empty list = the split is valid)."""
    problems = []
    files = {r.file for r in records}
    if len(files) != len(records):
        problems.append("the same file appears more than once in the dataset")
    missing = sorted(files - assignment.keys())
    extra = sorted(assignment.keys() - files)
    if missing:
        problems.append(f"{len(missing)} images missing from the split, e.g. {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unknown files in the split, e.g. {extra[:5]}")
    unknown = sorted({s for s in assignment.values() if s not in SPLIT_NAMES})
    if unknown:
        problems.append(f"unknown split names {unknown}")

    try:
        check_label_consistency(records)
    except SplitError as e:
        problems.append(str(e))

    splits_by_group = defaultdict(set)
    for r in records:
        if r.file in assignment:
            splits_by_group[pid_base(r.pid)].add(assignment[r.file])
    for group, splits in sorted(splits_by_group.items()):
        if len(splits) > 1:
            problems.append(f"patient group {group} is in several splits: {sorted(splits)}")

    present = {(r.label, assignment.get(r.file)) for r in records}
    for label in sorted({r.label for r in records}):
        for split in SPLIT_NAMES:
            if (label, split) not in present:
                problems.append(f"no images of label {label} in {split}")
    return problems


def summarize_split(records, assignment):
    """Return {split: {label: {"images": n, "groups": n}}} for every split and label."""
    images = defaultdict(int)
    groups = defaultdict(set)
    for r in records:
        key = (assignment[r.file], r.label)
        images[key] += 1
        groups[key].add(pid_base(r.pid))
    labels = sorted({r.label for r in records})
    return {s: {lab: {"images": images[(s, lab)], "groups": len(groups[(s, lab)])} for lab in labels}
            for s in SPLIT_NAMES}


def _sha256_of_lines(lines):
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def input_fingerprint(records):
    """SHA-256 of every (file, pid, label). Changes if the dataset's metadata changes."""
    return _sha256_of_lines(f"{r.file},{r.pid},{r.label}" for r in records)


def assignment_fingerprint(assignment):
    """SHA-256 of every (file, split). Changes if any image moves to another split."""
    return _sha256_of_lines(f"{f},{s}" for f, s in assignment.items())


def _file_number(name):
    """12 for "12.mat"; None for a name that isn't a number."""
    stem = name.rsplit(".", 1)[0]
    return int(stem) if stem.isdigit() else None


def save_split(csv_path, json_path, records, assignment, seed, fractions):
    """Write the split CSV and its metadata JSON. Never overwrites existing files."""
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; delete it yourself to make a new split.")
    problems = validate_split(records, assignment)
    if problems:
        raise SplitError("Refusing to save an invalid split:\n  " + "\n  ".join(problems))

    summary = summarize_split(records, assignment)
    meta = {
        "split_file": csv_path.name,
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "fractions": fractions,
        "grouping": "mat_loader.pid_base (trailing letters removed from patient IDs)",
        "stratified_by": "label",
        "source": SOURCE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_fingerprint": input_fingerprint(records),
        "assignment_fingerprint": assignment_fingerprint(assignment),
        "counts": {
            "images": len(records),
            "patient_ids": len({r.pid for r in records}),
            "patient_groups": len(build_groups(records)),
            "patient_groups_in_multiple_splits": 0,  # validate_split above guarantees this
            "per_split": {s: {LABEL_NAMES.get(lab, str(lab)): c for lab, c in by_label.items()}
                          for s, by_label in summary.items()},
        },
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Rows in file-number order (1.mat, 2.mat, ...), so the CSV is easy to read and diff.
    rows = sorted(records, key=lambda r: (_file_number(r.file) is None, _file_number(r.file) or 0, r.file))
    # Write to temporary names first, so an interrupted run never leaves half a split.
    tmp_csv, tmp_json = csv_path.with_suffix(".csv.partial"), json_path.with_suffix(".json.partial")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            number = _file_number(r.file)
            writer.writerow([r.file, "" if number is None else number, r.pid, pid_base(r.pid),
                             r.label, LABEL_NAMES.get(r.label, ""), assignment[r.file]])
    tmp_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp_csv.replace(csv_path)
    tmp_json.replace(json_path)
    return meta


def load_split(csv_path, json_path, records=None):
    """Read a saved split, check it, and return ({file: split}, metadata).

    Checks that the CSV still matches the fingerprints saved next to it (so a
    hand edit is caught) and that it obeys every split rule. If `records` from
    the current dataset are given, also checks the dataset hasn't changed since
    the split was made.
    """
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise SplitError(f"{csv_path.name}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    saved_records = [ImageRecord(row["file"], row["pid"], int(row["label"])) for row in rows]
    assignment = {row["file"]: row["split"] for row in rows}

    if len(assignment) != len(rows):
        raise SplitError(f"{csv_path.name}: a file is listed more than once")
    if input_fingerprint(saved_records) != meta["input_fingerprint"] or \
            assignment_fingerprint(assignment) != meta["assignment_fingerprint"]:
        raise SplitError(f"{csv_path.name} does not match the fingerprint in {json_path.name}: "
                         "the split file was edited or damaged. Restore it from Git.")
    problems = validate_split(saved_records, assignment)
    if problems:
        raise SplitError(f"{csv_path.name} breaks split rules:\n  " + "\n  ".join(problems))
    if records is not None and input_fingerprint(records) != meta["input_fingerprint"]:
        raise SplitError("The dataset's files, patient IDs or labels differ from when this split was "
                         f"made ({meta['created_utc']}). Check the data before making a new split.")
    return assignment, meta
