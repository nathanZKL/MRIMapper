# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

2D brain tumor **segmentation** (predict a tumor mask from one MRI slice) using Jun Cheng's original Figshare dataset (DOI 10.6084/m9.figshare.1512427, **version 8**, CC BY 4.0). Use only this dataset, not Kaggle re-uploads or JPEG conversions. Every image contains a tumor, so the dataset cannot support healthy-vs-tumor or early-detection claims. Don't frame results that way.

Status: milestone 1 (setup, loader, inventory, preview) and milestone 2 (patient-grouped train/val/test split, `splits/split_v1.*`) are done. Next comes training, which needs its own milestone plan and approval.

## Working with the owner

- The owner is a beginner in both machine learning and Claude Code. Briefly explain unfamiliar concepts, and say what each change is for.
- Work on one milestone at a time. Don't start training, or work that belongs to a later milestone, unless the current milestone asks for it.
- Run the relevant checks after changes. In reports, keep what was verified (command run, output seen) separate from what is assumed.
- Don't commit or push unless the owner asks.

## Commands (Windows, PowerShell, from repo root)

The venv is called directly instead of being activated, which avoids execution-policy errors:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\prepare_data.py   # verify MD5s of data/raw, extract .mat files (idempotent)
.\.venv\Scripts\python.exe src\inventory.py      # ~30 s over all 3064 files -> reports/inventory_report.md (+ .json, .csv)
.\.venv\Scripts\python.exe src\preview.py --per-type 2 --seed 0   # -> reports/preview_tumor_types.png
.\.venv\Scripts\python.exe src\make_split.py     # reuse + verify the saved split -> reports/split_report.md
.\.venv\Scripts\python.exe -m pytest -v          # all tests (~7 s); one test: -m pytest tests\test_splitting.py -k overlap
```

Tests live in `tests/`. `pytest.ini` puts `src/` on the import path. Unit tests use made-up records. Tests that need the real `.mat` files or the saved split skip when those are missing. There is no linter. Other checks: `python -m py_compile src/*.py`, and `inventory.py` should report 0 missing / unreadable / invalid.

## Architecture

- `src/` is a flat folder of scripts, not a package. Scripts are run as `python src\<name>.py`, which puts `src/` on `sys.path`, so modules import each other by bare name (`from config import ...`). Keep that convention, or add packaging deliberately.
- `src/config.py` is the single source of truth for paths (`RAW_DIR`, `MAT_DIR`, `REPORTS_DIR`), label names, README-stated counts, and Figshare's official MD5 checksums. Don't hard-code paths elsewhere.
- Data flow: `data/raw/figshare_1512427_v8/` (original zips, read-only) → `prepare_data.py` → `data/extracted/figshare_1512427_v8/mat/N.mat` + `SOURCE.json` (provenance) → `mat_loader.load_slice()` → `inventory.py` / `preview.py` → `reports/`.
- Keep these out of Git: datasets, model weights, credentials (`.env`), environments (`.venv/`), and generated outputs. `data/`, `*.mat`, `*.zip`, `reports/`, `.env` and `.venv` are already ignored. Before saving model weights for the first time, add ignore patterns for them (e.g. `*.pt`, `*.pth`, a checkpoints folder). Reports are regenerated, never hand-edited. **Exception:** `splits/` *is* committed, on purpose. It defines the experiment, so every run and computer must use the same split.

## Data facts (verified on the real files; don't re-assume)

- `.mat` files are **MATLAB v7.3 (HDF5)**: read with `h5py`, not `scipy.io.loadmat`. Fields are under `cjdata`: `label` (1 meningioma, 2 glioma, 3 pituitary), `PID` (uint16 char codes), `image` (int16), `tumorMask` (0/1), `tumorBorder` (flat `[x1,y1,...]`, 1-based, x = column).
- h5py returns MATLAB arrays **transposed**. `load_slice` applies `.T`. Orientation was proven by rasterizing `tumorBorder` against the mask (IoU ≥ 0.97 on all files, transposed never better). Any new reader must keep this.
- 3064 images and 233 raw PIDs, matching the README. But **15 images are 256×256**, not 512×512 as the README claims. Resize masks with nearest-neighbour only.
- Intensities are raw int16 with no common scale (per-image max 238–12728), so normalize per image. Never convert the training source to JPEG/8-bit. The README's JPEG snippet is for viewing only.

## Splitting rules (milestone 2 and beyond)

- Split by patient, never by individual image: all images from one patient go in the same split.
- 18 glioma PID groups differ only by a trailing letter (`MR040240`, `MR040240B`, ...). Treat them as possibly the same person. Group with `mat_loader.pid_base()`, which yields 209 groups.
- **Don't rely on `cvind.mat`.** It separates exact PIDs, but 16 of those 18 groups span multiple folds.
- The split is `splits/split_v1.csv` (+ `.json` with seed 42, fractions and SHA-256 fingerprints). Load it with `splitting.load_split()`, which verifies it. Never regenerate or hand-edit it. A new split means a new version file (`split_v2`) and a bump to `splitting.ALGORITHM_VERSION`.
- Glioma test has only 5 patient groups (val 8), so treat per-class test scores as noisy.
- 13 other glioma IDs have a letter suffix but no sibling (e.g. `MR017260F`). They are single patients, and their `patient_group` drops the letter.
- Tests about grouping or leakage must not check results with `pid_base()` itself: use hand-written expectations (see `KNOWN_SUFFIX_GROUPS` in `tests/test_make_split.py` and the answer key from `make_dataset()` in `tests/test_splitting.py`).

## Safety conventions

- Preserve the original dataset files: never modify, move, or delete `data/raw/`. Extraction must not silently overwrite: identical files are skipped, differing files stop with an error.
- Only `^\d+\.mat$` archive members are extracted (zip-slip guard).
- Disk space on C: is tight (~20 GB free). Check before large installs such as PyTorch, or before writing derived datasets.
