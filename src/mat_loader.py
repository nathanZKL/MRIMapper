"""Load one slice from Jun Cheng's brain tumor dataset (.mat, MATLAB v7.3).

What we found by inspecting the real files:
  * They are MATLAB v7.3 files, which are HDF5 containers -> read with h5py.
  * Everything lives in a group called "cjdata" with fields
    label, PID, image, tumorBorder, tumorMask (matches the README).
  * MATLAB stores arrays column-by-column, so h5py returns them transposed.
    We transpose image and mask back so that array[row, col] matches MATLAB.
    (inventory.py double-checks this against the independent tumorBorder.)
  * PID is stored as MATLAB characters (uint16 codes), e.g. "100360".
"""

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TumorSlice:
    file: str           # e.g. "1.mat"
    label: int          # 1 meningioma, 2 glioma, 3 pituitary tumor
    pid: str            # patient ID
    image: np.ndarray   # 2D int16 MRI intensities, original values (no rescaling)
    mask: np.ndarray    # 2D bool, True = tumor (expert annotation)
    border: np.ndarray  # (N, 2) float, columns are (x, y) = (column, row), 1-based like MATLAB


def _decode_matlab_string(dataset):
    codes = np.asarray(dataset[()]).ravel()
    return "".join(chr(int(c)) for c in codes).strip()


def _read_label(cj):
    label_values = np.asarray(cj["label"][()]).ravel()
    if label_values.size != 1:
        raise ValueError(f"label has {label_values.size} values, expected 1")
    label_float = float(label_values[0])
    return int(label_float) if label_float.is_integer() else label_float  # keep weird values visible


def load_slice(path):
    """Read one .mat file and return a TumorSlice. Raises on unreadable files."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        label = _read_label(cj)

        image = np.asarray(cj["image"][()]).T
        mask_raw = np.asarray(cj["tumorMask"][()]).T
        border_flat = np.asarray(cj["tumorBorder"][()]).ravel()
        pid = _decode_matlab_string(cj["PID"])

    unique_mask_values = np.unique(mask_raw)
    if not set(unique_mask_values.tolist()) <= {0, 1}:
        raise ValueError(f"mask is not binary: values {unique_mask_values[:10].tolist()}")
    if border_flat.size % 2:
        raise ValueError(f"tumorBorder has odd length {border_flat.size}")

    return TumorSlice(
        file=path.name,
        label=label,
        pid=pid,
        image=image,
        mask=mask_raw.astype(bool),
        border=border_flat.reshape(-1, 2),
    )


def load_metadata(path):
    """Return (label, pid) only. Much faster than load_slice: skips image and mask."""
    with h5py.File(Path(path), "r") as f:
        cj = f["cjdata"]
        return _read_label(cj), _decode_matlab_string(cj["PID"])


def pid_base(pid):
    """Strip a trailing letter suffix: "MR040240B" -> "MR040240".

    Some IDs differ only by such a suffix. The README does not say whether they
    are the same person, so we treat them as *possibly* the same patient.
    """
    return re.sub(r"(?<=\d)[A-Za-z]+$", "", pid)


def load_cv_folds(cvind_path):
    """Return a 1D int array: fold number for image i is folds[i - 1]."""
    with h5py.File(cvind_path, "r") as f:
        return np.asarray(f["cvind"][()]).ravel().astype(int)
