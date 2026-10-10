# Windows setup and running milestone 1

All commands are for **PowerShell** (Start menu → type "PowerShell").
Run them one at a time, in order.

## 0. Go to the project folder

```powershell
cd "C:\Users\natha\Desktop\MRI Machine Learning\MRIMapper"
```

## 1. Create the virtual environment (one time only)

A *virtual environment* is a private copy of Python just for this project, so
the packages we install here can't break (or be broken by) other projects.
It lives in the `.venv` folder, which Git ignores.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

We call `.\.venv\Scripts\python.exe` directly instead of "activating" the
environment. It does the same thing and avoids Windows script-permission errors.

## 2. Put the original downloads in place

From https://doi.org/10.6084/m9.figshare.1512427 download these six files and
copy them, **unchanged and unrenamed**, into:

```
C:\Users\natha\Desktop\MRI Machine Learning\MRIMapper\data\raw\figshare_1512427_v8\
```

| File | Size (bytes) |
|---|---|
| brainTumorDataPublic_1-766.zip | 214,401,279 |
| brainTumorDataPublic_767-1532.zip | 217,848,429 |
| brainTumorDataPublic_1533-2298.zip | 215,563,856 |
| brainTumorDataPublic_2299-3064.zip | 231,679,762 |
| cvind.mat | 5,736 |
| README 2024.txt | 3,303 |

(If you downloaded the single bundle `1512427.zip`, open it and copy those six
files out of it. This was already done on this computer on 2026-10-09.)

## 3. Verify and extract

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py
```

This checks every file against Figshare's official MD5 checksum (a fingerprint
that changes if even one byte is different), then unpacks the 3,064 `.mat`
files into `data\extracted\figshare_1512427_v8\mat\`. Running it again is safe:
identical files are skipped, and it refuses to overwrite anything that differs.

## 4. Build the inventory report

```powershell
.\.venv\Scripts\python.exe src\inventory.py
```

Takes about 30 seconds. Then open `reports\inventory_report.md`.

## 5. Make the preview picture

```powershell
.\.venv\Scripts\python.exe src\preview.py
```

Then open `reports\preview_tumor_types.png`. To see different examples:

```powershell
.\.venv\Scripts\python.exe src\preview.py --per-type 4 --seed 7
```

## 6. The train / validation / test split (milestone 2)

```powershell
.\.venv\Scripts\python.exe src\make_split.py
```

The split is already saved in `splits\split_v1.csv` (one row per image) and
`splits\split_v1.json` (seed, settings, counts and fingerprints). These files are
in Git, so this command **reuses** them: it checks they are intact and still match
the dataset, then rewrites `reports\split_report.md`. It never reshuffles. To make
a brand-new split you would have to delete both files yourself.

## 7. Run the automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All tests should say `PASSED` (about 7 seconds). They check, among other things,
that no patient is in two splits and that every image is assigned exactly once.

## Folder layout

```
MRIMapper/
├── src/                     code (in Git)
│   ├── config.py            paths, label names, checksums
│   ├── mat_loader.py        reads one .mat file -> image, mask, label, patient ID
│   ├── prepare_data.py      verify downloads + safe extraction
│   ├── inventory.py         dataset statistics and sanity checks
│   ├── preview.py           image / mask / overlay figure
│   ├── splitting.py         split logic: group, assign, validate, save, load
│   └── make_split.py        create once / reuse the train-val-test split
├── tests/                   automated tests, run with pytest (in Git)
├── splits/                  saved split assignments (in Git, on purpose)
├── docs/                    documentation (in Git)
├── data/                    NOT in Git
│   ├── raw/figshare_1512427_v8/        original downloads, never modified
│   └── extracted/figshare_1512427_v8/  mat/ (3,064 files) + SOURCE.json
├── reports/                 generated outputs, NOT in Git
└── .venv/                   virtual environment, NOT in Git
```

## Troubleshooting

- **`py` is not recognized**: install Python 3.13 from python.org and tick
  "Add python.exe to PATH" during installation.
- **"Missing folder" or "MISSING" from prepare_data.py**: a download is not in
  the `data\raw\figshare_1512427_v8\` folder, or it was renamed.
- **"MISMATCH"**: that file is corrupted or a different version. Download it again.
- **"Refusing to overwrite"**: a file in `data\extracted\...\mat\` differs from the
  original archive. Delete the `data\extracted` folder and run step 3 again.
