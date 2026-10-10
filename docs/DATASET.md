# Dataset notes

## Source

- **Title:** brain tumor dataset, by Jun Cheng
- **DOI:** 10.6084/m9.figshare.1512427 (we use **version 8**, published 2024-12-21)
- **License:** CC BY 4.0 (you may reuse it with attribution)
- **Cite:** Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via
  Tumor Region Augmentation and Partition." PLoS ONE 10.10 (2015).

Our downloaded files were verified against the MD5 checksums from the Figshare
API on 2026-10-09 (see `src/config.py`). Each run of `prepare_data.py` writes a
provenance record to `data/extracted/figshare_1512427_v8/SOURCE.json`.

## What is in it

3,064 T1-weighted contrast-enhanced MRI slices (2D images) from patients who
**all have a tumor**. There are no healthy scans, so this dataset can't teach
healthy-vs-tumor detection or early detection. It supports **tumor
segmentation** (outlining the tumor) and tumor-type classification.

## File format (verified by inspecting the real files)

Each `N.mat` is a **MATLAB v7.3** file, which is an HDF5 container, so we read it
with `h5py` (`scipy.io.loadmat` cannot open v7.3). Inside is a group `cjdata`:

| Field | Stored as | Meaning (from README) |
|---|---|---|
| `label` | 1×1 float | 1 = meningioma, 2 = glioma, 3 = pituitary tumor |
| `PID` | uint16 character codes | patient ID, e.g. `100360` or `MR040240B` |
| `image` | int16, 512×512 (15 files are 256×256) | raw MRI intensities |
| `tumorMask` | uint8 0/1, same size as image | expert tumor annotation |
| `tumorBorder` | [x1, y1, x2, y2, ...] | the hand-drawn outline the mask was made from |

MATLAB stores arrays column-first, so h5py gives them back **transposed**.
`mat_loader.py` transposes them back. This was verified on all 3,064 files: the
mask redrawn from `tumorBorder` matches `tumorMask` (IoU ≥ 0.97), and the
transposed alternative never fits better.

We keep the original `.mat` files as the training source and never convert them
to JPEG. JPEG compression is lossy, and rescaling to 8-bit throws away intensity detail.

## Findings that matter for the next milestone

1. **README says all images are 512×512. 15 are actually 256×256.** We must
   resize consistently (the image smoothly, the mask with nearest-neighbour so it stays 0/1).
2. **Intensities have no fixed scale** (per-image maximum ranges from 238 to 12,728),
   so each image needs normalizing before training.
3. **Possible duplicate patients.** 18 groups of glioma IDs differ only by a letter
   suffix (e.g. `MR040240`, `MR040240B`, `MR040240C`, `MR040240D`). The README doesn't
   explain this. Treating each group as one person gives 209 patients instead of 233.
   The safe choice is to keep each group in the same split.
4. **cvind.mat** keeps each exact ID in one fold, but 16 of those 18 groups are split
   across folds. We therefore plan to make our own patient-grouped split instead of
   using cvind.mat.

## The patient-grouped split (milestone 2)

Made by `src/make_split.py`, saved in `splits/split_v1.csv` + `splits/split_v1.json`,
summarized in `reports/split_report.md`.

- **Unit:** the patient group from `mat_loader.pid_base()` (209 groups). A group is never
  divided, so no patient appears in two splits.
- **Stratified by tumor type:** each type is split on its own. Groups are sorted, shuffled
  with seed 42, and each goes to the split furthest below its target share of images.
- **Result:** train 2151 / val 468 / test 445 images (70.2 / 15.3 / 14.5 %), 151 / 28 / 30
  patient groups. Every tumor type is within 1 point of 70/15/15 in images.
- **Caveat:** glioma patients have many images each, so test contains only 5 glioma patient
  groups (val 8). Glioma scores on test will rest on few people.
- **Reuse, not reshuffle:** the JSON stores SHA-256 fingerprints of the dataset metadata
  (file, PID, label) and of the assignments. Later runs reuse the saved split and stop if the
  file was edited or the dataset changed.
- **Lone suffixed IDs:** besides the 18 groups, 13 more glioma IDs carry a letter suffix
  but have no sibling ID (e.g. `MR017260F`). Each is its own patient; its
  `patient_group` in the split CSV just drops the letter.
