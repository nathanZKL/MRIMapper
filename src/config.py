"""Shared paths and dataset facts for MRIMapper.

Every script imports from here so folder locations live in one place.
"""

from pathlib import Path

# Project root = the folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ID = "figshare_1512427_v8"

# Original downloads, kept exactly as Figshare served them (never modified).
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_ID
# Files unpacked from the original zips.
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted" / DATASET_ID
MAT_DIR = EXTRACTED_DIR / "mat"
# Generated reports and figures.
REPORTS_DIR = PROJECT_ROOT / "reports"

CVIND_PATH = RAW_DIR / "cvind.mat"
README_PATH = RAW_DIR / "README 2024.txt"

# Provenance, checked against the Figshare API on 2026-10-09:
# https://api.figshare.com/v2/articles/1512427
SOURCE = {
    "title": "brain tumor dataset",
    "author": "Jun Cheng",
    "doi": "10.6084/m9.figshare.1512427.v8",
    "url": "https://doi.org/10.6084/m9.figshare.1512427",
    "figshare_version": 8,
    "published": "2024-12-21",
    "license": "CC BY 4.0",
}

# Official MD5 checksums published by Figshare for version 8.
EXPECTED_MD5 = {
    "brainTumorDataPublic_1-766.zip": "74b949ad33f042e6e103523091cd1428",
    "brainTumorDataPublic_767-1532.zip": "7e8a875500d2c8a346f270538e29890e",
    "brainTumorDataPublic_1533-2298.zip": "8227bf6080cb71f15a88be8d25c79ae7",
    "brainTumorDataPublic_2299-3064.zip": "b378a80d6174e5317d59eb28430c6652",
    "cvind.mat": "5ea82eb3212c9fcb84290ccf0beb486d",
    "README 2024.txt": "4188f7c30957e9673c3611613071bd6e",
}
ZIP_NAMES = [name for name in EXPECTED_MD5 if name.endswith(".zip")]

# Label meanings, copied from "README 2024.txt" section 1:
#   "cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor"
LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary tumor"}

# Counts stated in the README, used as a cross-check (not as ground truth).
README_EXPECTED = {
    "images": 3064,
    "patients": 233,
    "per_label": {1: 708, 2: 1426, 3: 930},
}
EXPECTED_FILE_NUMBERS = range(1, 3065)  # 1.mat ... 3064.mat
