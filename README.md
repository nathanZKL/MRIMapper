# MRIMapper
Brain tumor segmentation from MRI images using deep learning and Cheng’s Figshare dataset.

**Goal:** given a 2D MRI slice, predict a mask that outlines the tumor.
The dataset contains only tumor cases, so this project does **not** do
healthy-vs-tumor detection or early detection.

## Status

- [x] **Milestone 1:** environment, dataset verification and extraction, `.mat` loader, inventory report, preview figure
- [x] **Milestone 2:** patient-grouped train/validation/test split (70/15/15, saved in `splits/`), with automated tests
- [ ] Later: training a segmentation model

## Quick start (Windows PowerShell)

```powershell
cd "C:\Users\natha\Desktop\MRI Machine Learning\MRIMapper"
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\inventory.py
.\.venv\Scripts\python.exe src\preview.py
.\.venv\Scripts\python.exe src\make_split.py     # reuses the saved split in splits/
.\.venv\Scripts\python.exe -m pytest -v          # automated tests
```

Full step-by-step instructions: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md).
Dataset source, format and findings: [docs/DATASET.md](docs/DATASET.md).

Data: Jun Cheng, *brain tumor dataset*, Figshare, https://doi.org/10.6084/m9.figshare.1512427 (v8), CC BY 4.0.
