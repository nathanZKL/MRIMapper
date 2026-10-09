"""Build an inventory report of the extracted dataset.

Reads every .mat file with mat_loader and writes, into reports/:
  inventory_report.md       human-readable summary (start here)
  inventory_summary.json    the same numbers in machine-readable form
  inventory_per_image.csv   one row per file

Nothing is trained and nothing is split here. This only describes the data.

Run:  python src\\inventory.py
"""

import csv
import json
import re
from collections import Counter, defaultdict

import numpy as np
from matplotlib.path import Path as PolygonPath

from config import (CVIND_PATH, EXPECTED_FILE_NUMBERS, LABEL_NAMES, MAT_DIR,
                    README_EXPECTED, REPORTS_DIR, SOURCE)
from mat_loader import load_cv_folds, load_slice

# If the mask drawn from tumorBorder overlaps the stored tumorMask less than
# this (IoU, 0..1), we flag the image for a human to look at.
BORDER_IOU_WARN = 0.8


def border_to_mask(border, shape):
    """Fill the tumorBorder polygon into a boolean image of the given shape.

    Border points are MATLAB-style (x=column, y=row, 1-based), so pixel
    [r, c] has its centre at (x=c+1, y=r+1).
    """
    out = np.zeros(shape, dtype=bool)
    if len(border) < 3:
        return out
    polygon = PolygonPath(border)
    # Only test pixels inside the polygon's bounding box (much faster).
    c0, r0 = np.floor(border.min(axis=0)).astype(int) - 2
    c1, r1 = np.ceil(border.max(axis=0)).astype(int) + 1
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0]), min(c1, shape[1])
    if r1 <= r0 or c1 <= c0:
        return out
    rows, cols = np.mgrid[r0:r1, c0:c1]
    points = np.column_stack([cols.ravel() + 1, rows.ravel() + 1])
    out[r0:r1, c0:c1] = polygon.contains_points(points).reshape(rows.shape)
    return out


def pid_base(pid):
    """Strip a trailing letter suffix: "MR040240B" -> "MR040240".

    Some IDs differ only by such a suffix. The README does not say whether they
    are the same person, so we report them as *possibly* the same patient.
    """
    return re.sub(r"(?<=\d)[A-Za-z]+$", "", pid)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def inspect_file(path, number):
    """Return one inventory row (dict) for a .mat file."""
    row = {"file": path.name, "number": number, "status": "ok", "problems": ""}
    try:
        s = load_slice(path)
    except (ValueError, KeyError) as e:
        row.update(status="invalid", problems=f"{type(e).__name__}: {e}")
        return row
    except Exception as e:  # corrupt or not a readable HDF5/.mat file
        row.update(status="unreadable", problems=f"{type(e).__name__}: {e}")
        return row

    problems = []
    if s.label not in LABEL_NAMES:
        problems.append(f"invalid label {s.label!r}")
    if not s.pid:
        problems.append("empty patient ID")
    if s.image.ndim != 2:
        problems.append(f"image is {s.image.ndim}D")
    if s.image.shape != s.mask.shape:
        problems.append(f"image/mask shape mismatch {s.image.shape} vs {s.mask.shape}")
    if not np.isfinite(s.image).all():
        problems.append("image contains NaN/inf")
    mask_pixels = int(s.mask.sum())
    if mask_pixels == 0:
        problems.append("empty mask")

    border_iou = border_iou_transposed = float("nan")
    if s.image.shape == s.mask.shape and s.mask.shape[0] == s.mask.shape[1]:
        from_border = border_to_mask(s.border, s.mask.shape)
        border_iou = iou(from_border, s.mask)
        border_iou_transposed = iou(from_border, s.mask.T)
        if not border_iou >= BORDER_IOU_WARN:
            problems.append(f"tumorBorder/tumorMask IoU only {border_iou:.2f}")

    image = s.image.astype(np.float64)
    row.update(
        label=s.label,
        label_name=LABEL_NAMES.get(s.label, "INVALID"),
        pid=s.pid,
        height=s.image.shape[0],
        width=s.image.shape[1] if s.image.ndim > 1 else "",
        dtype=str(s.image.dtype),
        intensity_min=float(image.min()),
        intensity_max=float(image.max()),
        intensity_mean=round(float(image.mean()), 2),
        mask_pixels=mask_pixels,
        mask_fraction=round(mask_pixels / s.mask.size, 5),
        border_points=len(s.border),
        border_mask_iou=round(border_iou, 4),
        border_mask_iou_if_transposed=round(border_iou_transposed, 4),
    )
    if problems:
        row.update(status="invalid", problems="; ".join(problems))
    return row


def describe(values):
    v = np.asarray(values, dtype=float)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def summarize(rows, missing, extra, folds):
    loaded = [r for r in rows if "label" in r]
    valid_label = [r for r in loaded if r["label"] in LABEL_NAMES]

    labels_by_pid = defaultdict(set)
    for r in loaded:
        labels_by_pid[r["pid"]].add(r["label"])
    inconsistent = {pid: sorted(map(str, labs)) for pid, labs in labels_by_pid.items() if len(labs) > 1}

    pids_by_base = defaultdict(set)
    for pid in labels_by_pid:
        pids_by_base[pid_base(pid)].add(pid)
    suffix_groups = {base: sorted(p) for base, p in sorted(pids_by_base.items()) if len(p) > 1}

    per_label = {}
    for lab, name in LABEL_NAMES.items():
        rs = [r for r in valid_label if r["label"] == lab]
        per_label[name] = {
            "label_value": lab,
            "images": len(rs),
            "patients": len({r["pid"] for r in rs}),
            "readme_says_images": README_EXPECTED["per_label"][lab],
            "intensity_max": describe([r["intensity_max"] for r in rs]) if rs else None,
            "mask_fraction": describe([r["mask_fraction"] for r in rs]) if rs else None,
        }

    # cvind.mat: does any patient appear in more than one fold?
    cv = {"available": folds is not None}
    if folds is not None:
        cv["length"] = int(len(folds))
        cv["images_per_fold"] = {int(k): int(v) for k, v in sorted(Counter(folds.tolist()).items())}
        folds_by_pid = defaultdict(set)
        for r in loaded:
            if 1 <= r["number"] <= len(folds):
                folds_by_pid[r["pid"]].add(int(folds[r["number"] - 1]))
        leaking = {pid: sorted(f) for pid, f in folds_by_pid.items() if len(f) > 1}
        cv["patients_in_multiple_folds"] = len(leaking)
        cv["images_from_those_patients"] = sum(1 for r in loaded if r["pid"] in leaking)
        cv["examples"] = dict(sorted(leaking.items())[:15])
        cv["patient_separated"] = not leaking
        folds_by_base = defaultdict(set)
        for pid, f in folds_by_pid.items():
            folds_by_base[pid_base(pid)] |= f
        cv["suffix_groups_in_multiple_folds"] = sum(
            1 for base in suffix_groups if len(folds_by_base[base]) > 1)

    transposed_better = sum(
        1 for r in loaded
        if r["border_mask_iou_if_transposed"] > r["border_mask_iou"]
    )
    ious = [r["border_mask_iou"] for r in loaded if not np.isnan(r["border_mask_iou"])]

    return {
        "source": SOURCE,
        "files_expected": len(EXPECTED_FILE_NUMBERS),
        "files_found": len(rows),
        "missing_files": missing,
        "unexpected_files": extra,
        "unreadable_files": [{"file": r["file"], "error": r["problems"]} for r in rows if r["status"] == "unreadable"],
        "invalid_files": [{"file": r["file"], "problems": r["problems"]} for r in rows if r["status"] == "invalid"],
        "images_loaded": len(loaded),
        "unique_patients": len(labels_by_pid),
        "readme_says": {"images": README_EXPECTED["images"], "patients": README_EXPECTED["patients"]},
        "per_label": per_label,
        "invalid_label_values": dict(Counter(str(r["label"]) for r in loaded if r["label"] not in LABEL_NAMES)),
        "image_shapes": {f"{h}x{w}": n for (h, w), n in Counter((r["height"], r["width"]) for r in loaded).most_common()},
        "image_dtypes": dict(Counter(r["dtype"] for r in loaded)),
        "intensity": {
            "global_min": min(r["intensity_min"] for r in loaded),
            "global_max": max(r["intensity_max"] for r in loaded),
            "per_image_min": describe([r["intensity_min"] for r in loaded]),
            "per_image_max": describe([r["intensity_max"] for r in loaded]),
        },
        "patients_with_inconsistent_labels": inconsistent,
        "possible_same_patient_id_groups": suffix_groups,
        "unique_patients_if_groups_merged": len(pids_by_base),
        "orientation_check": {
            "border_mask_iou": describe(ious) if ious else None,
            "images_where_transposed_mask_fits_border_better": transposed_better,
        },
        "cvind": cv,
    }


def fmt_range(d):
    return f"{d['min']:g} / {d['median']:g} / {d['max']:g}"


def write_markdown(summary, path):
    s = summary
    L = []
    L.append("# Dataset inventory: Jun Cheng brain tumor dataset\n")
    L.append(f"Source: {s['source']['url']} (DOI {s['source']['doi']}, Figshare version "
             f"{s['source']['figshare_version']}, {s['source']['license']}).  ")
    L.append("Generated by `src/inventory.py` from the real extracted .mat files. "
             "These numbers describe the data only; no model is involved.\n")

    L.append("## Files")
    L.append(f"- Expected files: {s['files_expected']} (1.mat ... 3064.mat)")
    L.append(f"- Found: {s['files_found']}")
    L.append(f"- Missing: {len(s['missing_files'])} {s['missing_files'][:20] if s['missing_files'] else ''}")
    L.append(f"- Unexpected extra files: {len(s['unexpected_files'])} {s['unexpected_files'][:20] if s['unexpected_files'] else ''}")
    L.append(f"- Unreadable: {len(s['unreadable_files'])}")
    L.append(f"- Loaded but failing a check: {len(s['invalid_files'])}")
    for item in s["invalid_files"][:30] + s["unreadable_files"][:30]:
        L.append(f"  - `{item['file']}`: {item.get('problems') or item.get('error')}")
    L.append("")

    L.append("## Images and patients")
    L.append(f"- Images loaded: **{s['images_loaded']}** (README says {s['readme_says']['images']})")
    L.append(f"- Unique patient IDs: **{s['unique_patients']}** (README says {s['readme_says']['patients']})\n")
    L.append("| Label value | Tumor type (from README) | Images | README says | Patients |")
    L.append("|---|---|---|---|---|")
    for name, d in s["per_label"].items():
        flag = "" if d["images"] == d["readme_says_images"] else " ⚠️"
        L.append(f"| {d['label_value']} | {name} | {d['images']}{flag} | {d['readme_says_images']} | {d['patients']} |")
    L.append(f"\nInvalid label values: {s['invalid_label_values'] or 'none'}\n")

    L.append("## Image dimensions and intensities")
    L.append("| Shape (rows x cols) | Images |")
    L.append("|---|---|")
    for shape, n in s["image_shapes"].items():
        L.append(f"| {shape} | {n} |")
    L.append(f"\nStored data types: {s['image_dtypes']}  ")
    L.append(f"Intensity range across all images: {s['intensity']['global_min']:g} to {s['intensity']['global_max']:g}  ")
    L.append(f"Per-image minimum (min / median / max over images): {fmt_range(s['intensity']['per_image_min'])}  ")
    L.append(f"Per-image maximum (min / median / max over images): {fmt_range(s['intensity']['per_image_max'])}\n")
    L.append("| Tumor type | Per-image max intensity (min / median / max) | Tumor area as fraction of image (min / median / max) |")
    L.append("|---|---|---|")
    for name, d in s["per_label"].items():
        if d["intensity_max"]:
            L.append(f"| {name} | {fmt_range(d['intensity_max'])} | {fmt_range(d['mask_fraction'])} |")
    L.append("\nNote: intensities are raw MRI scanner values with no fixed scale, so ranges differ "
             "between images. We will need per-image normalization before training.\n")

    L.append("## Mask checks")
    oc = s["orientation_check"]
    if oc["border_mask_iou"]:
        L.append("Each mask was compared with a mask redrawn from the separate `tumorBorder` outline. "
                 "IoU (overlap / union, 1.0 = identical) min / median / max: "
                 f"{fmt_range(oc['border_mask_iou'])}.  ")
        L.append(f"Images where the *transposed* mask would fit the border better: "
                 f"{oc['images_where_transposed_mask_fits_border_better']} "
                 "(0 means our row/column orientation is consistent).\n")

    L.append("## Patients with inconsistent tumor labels")
    if s["patients_with_inconsistent_labels"]:
        for pid, labs in s["patients_with_inconsistent_labels"].items():
            L.append(f"- PID `{pid}`: labels {labs}")
    else:
        L.append("None: every patient ID has exactly one tumor-type label.")
    L.append("")

    L.append("## Patient IDs that may belong to the same person")
    groups = s["possible_same_patient_id_groups"]
    if groups:
        L.append(f"{len(groups)} groups of IDs differ only by a trailing letter (e.g. `MR040240` / `MR040240B`). "
                 "The README does not explain these suffixes, so they may be repeat scans of one person. "
                 f"Treating each group as one patient gives **{s['unique_patients_if_groups_merged']}** patients "
                 f"instead of {s['unique_patients']}. To be safe, the patient split should keep each group together.\n")
        for base, members in groups.items():
            L.append(f"- `{base}`: {', '.join(members)}")
    else:
        L.append("None found.")
    L.append("")

    L.append("## Is cvind.mat patient-separated?")
    cv = s["cvind"]
    if not cv["available"]:
        L.append("cvind.mat was not available.")
    else:
        L.append(f"cvind.mat assigns each of {cv['length']} images to a fold: {cv['images_per_fold']}.  ")
        if cv["patient_separated"]:
            L.append("Every exact patient ID's images fall in a single fold, so the folds are separated "
                     "**by exact patient ID**.")
            if groups:
                L.append(f"\n**But** {cv['suffix_groups_in_multiple_folds']} of the {len(groups)} possible "
                         "same-patient ID groups above are spread across different folds. If those IDs are the "
                         "same person, cvind.mat would leak that person between training and testing. "
                         "Recommendation: do not rely on cvind.mat; make our own split by grouped ID.")
        else:
            L.append(f"**No.** {cv['patients_in_multiple_folds']} patients have images in more than one fold "
                     f"({cv['images_from_those_patients']} images). Using these folds as-is would let the "
                     "same patient appear in both training and testing. Examples (PID: folds): "
                     f"{cv['examples']}")
        L.append("\n(Assumes entry *i* of cvind belongs to file *i*.mat; the README does not state the ordering explicitly.)")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    if not MAT_DIR.exists():
        raise SystemExit(f"No extracted data at {MAT_DIR}. Run: python src\\prepare_data.py")

    present = {p.name: p for p in MAT_DIR.glob("*.mat")}
    expected_names = {f"{n}.mat" for n in EXPECTED_FILE_NUMBERS}
    missing = sorted((n for n in expected_names if n not in present), key=lambda n: int(n[:-4]))
    extra = sorted(n for n in present if n not in expected_names)

    rows = []
    ordered = sorted(present.values(), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.name))
    for i, path in enumerate(ordered, 1):
        number = int(path.stem) if path.stem.isdigit() else -1
        rows.append(inspect_file(path, number))
        if i % 500 == 0:
            print(f"  inspected {i}/{len(ordered)}")

    folds = load_cv_folds(CVIND_PATH) if CVIND_PATH.exists() else None
    summary = summarize(rows, missing, extra, folds)
    if folds is not None:
        for r in rows:
            r["cvind_fold"] = int(folds[r["number"] - 1]) if 1 <= r["number"] <= len(folds) else ""

    fieldnames = []
    for r in rows:
        fieldnames += [k for k in r if k not in fieldnames]
    with open(REPORTS_DIR / "inventory_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (REPORTS_DIR / "inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, REPORTS_DIR / "inventory_report.md")

    print(f"Images loaded: {summary['images_loaded']}, patients: {summary['unique_patients']}")
    print(f"Missing: {len(missing)}, unreadable: {len(summary['unreadable_files'])}, invalid: {len(summary['invalid_files'])}")
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
