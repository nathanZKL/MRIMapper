# Milestone 2 TDD evidence: patient-grouped split

**Source plan:** the approved `/plan` for milestone 2 in the 2026-10-10 session (no plan file).
**Commits:** none. The owner asked for no commits, so the skill's checkpoint commits were skipped and this file records the evidence instead.

## User journeys

- As the project owner, I want each patient's images in exactly one split, so test scores aren't inflated by patients the model has already seen.
- As the project owner, I want the same split on every run and computer, so results from different runs are comparable.
- As the project owner, I want the split to stop me if the data or the saved split file changes, so I never train on a silently different split.

## RED / GREEN record (commands actually run)

| Round | What | Command | Result |
|---|---|---|---|
| 1 RED | 4 core tests vs. a placeholder `assign_splits` that raises `NotImplementedError` | `python -m pytest -q` | `4 failed in 0.27s` (all `NotImplementedError`) |
| 1 GREEN | `assign_splits` implemented | `python -m pytest -v` | `4 passed in 0.20s` |
| 2 RED | 21 more tests vs. placeholder functions | `python -m pytest -q` | `20 failed, 4 passed, 1 skipped` (real-split test skipped: file not created yet) |
| 2 GREEN | `splitting.py`, `load_metadata`, `make_split.py` implemented | `python -m pytest -q` | `24 passed, 1 skipped in 4.11s` |
| Real data | Create split, then rerun | `python src\make_split.py` (twice) | `Split created`, then `Split reused`; file hashes identical |
| Refactor | Tidy CSV row sorting; computed report note | `python -m pytest -v` | `25 passed in 4.08s`; saved split byte-identical |
| Sabotage | Grouping replaced by exact-ID grouping (in memory only) | ad-hoc script | overlap test **failed as expected** |

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | Every image is assigned once: none missing, none extra, only train/val/test | `test_splitting.py::test_every_image_is_assigned_exactly_once` | unit | PASS |
| 2 | No patient group (incl. `X` / `XB` / `XC`) is in more than one split | `::test_patient_groups_never_overlap_between_splits` | unit | PASS |
| 3 | Same seed gives identical assignment, whatever the input order | `::test_same_seed_gives_identical_assignment` | unit | PASS |
| 4 | Different seed gives a different assignment | `::test_different_seed_gives_different_assignment` | unit | PASS |
| 5 | Each tumor type is in every split, within 5 points of 70/15/15 | `::test_each_tumor_type_is_split_close_to_target_fractions` | unit | PASS |
| 6 | Summary counts add up to all images and all groups | `::test_summary_counts_add_up` | unit | PASS |
| 7 | A patient ID or patient group with two tumor types is rejected | `::test_patient_id_with_two_labels_is_rejected`, `::test_patient_group_with_two_labels_is_rejected`, `::test_assign_splits_refuses_inconsistent_labels`, `::test_consistent_labels_pass` | unit | PASS |
| 8 | Validation catches missing, extra and unknown entries, and a patient spread over splits | `::test_validate_*` (3 tests) | unit | PASS |
| 9 | Save + load round-trips; saving never overwrites | `::test_save_then_load_returns_the_same_assignment`, `::test_save_refuses_to_overwrite` | unit | PASS |
| 10 | A hand-edited split file is detected, even when no group rule breaks | `::test_load_detects_a_hand_edited_split_file` | unit | PASS |
| 11 | A changed dataset (label/PID/file) is detected | `::test_load_detects_that_the_dataset_changed`, `::test_fingerprint_ignores_order_but_not_content` | unit | PASS |
| 12 | Script creates once, then reuses even if given another seed; a half-saved split stops it | `test_make_split.py::test_first_run_creates_and_second_run_reuses`, `::test_half_written_split_is_an_error` | integration | PASS |
| 13 | The saved real split: 3064 images, 209 groups, 0 leaking, fingerprints intact | `test_make_split.py::test_saved_real_split_is_valid` | data | PASS |
| 14 | Fast `load_metadata` agrees with full `load_slice` (incl. a 256x256 file) | `test_mat_loader.py` (4 files) | data | PASS |

Independent cross-check, without `splitting.py`: every (file, PID, label) in `splits/split_v1.csv` matches milestone 1's `reports/inventory_per_image.csv`, and 0 of 209 groups span splits.

## Coverage and known gaps

- Line coverage was **not measured**. `pytest-cov` was not in the approved plan, so it wasn't installed.
- `make_split.write_report` and `make_split.main` are exercised only by running the script, not by a unit test.
- `inventory.py` still has its own copy of `pid_base` (milestone 1 code, left unchanged). It is identical to `mat_loader.pid_base`.
- Data tests skip, rather than fail, on a machine without the extracted dataset or the split file.

## Follow-up: review fixes (test quality)

A focused review found that the leakage tests checked the grouping with the same
`pid_base()` the code uses. When grouping was broken on purpose (only a trailing `B`
stripped), all 25 tests still passed.

| Fix | Tests added |
|---|---|
| Leakage checked against an answer key, not `pid_base()` | `test_splitting.py::test_patients_never_overlap_between_splits` (made-up data now records each image's true patient and covers suffixes B/C/D/E/G/I and groups without a plain ID) |
| `pid_base()` pinned to real examples | `test_mat_loader.py::test_pid_base_on_real_ids` (7 cases) |
| Real split checked without `pid_base()` | `test_make_split.py::test_saved_real_split_has_no_patient_in_two_splits` (18 known groups written out by hand; 209 patients; `patient_group` column groups rows the same way) |
| Reproducibility across processes and from the seed | `::test_same_seed_gives_identical_assignment_in_fresh_processes` (PYTHONHASHSEED 0 / 1 / 12345), `test_make_split.py::test_saved_real_split_can_be_rebuilt_from_its_seed` |
| Missing or bad data stops the script | `test_make_split.py::test_read_records_*`, `test_missing_*`, `test_unreadable_*`, `test_invalid_label_*`, `test_empty_patient_id_*`, `test_dataset_change_is_caught_on_reuse` (fake `.mat` files in a temp folder) |
| Assertions that couldn't fail removed | old `test_saved_real_split_is_valid` replaced |

**RED:** the 6 `read_records` tests failed with `TypeError` (it couldn't take a folder yet).
The real-split group test also failed, because of a wrong expectation in the test: 13
suffixed IDs, e.g. `MR017260F`, have no sibling ID, and their `patient_group` drops the
letter. The test was corrected to compare groupings rather than text.
**GREEN:** `read_records(mat_dir, numbers)` added. Result: `42 passed in 6.39s`. The
saved split is byte-identical.

**Deliberate bugs (run only on throwaway copies or in memory):**

| Bug | Caught by |
|---|---|
| Grouping strips only `B` | 6 failures: answer-key leakage test, 4 `pid_base` cases, rebuild-from-seed |
| Shuffle uses per-process set order instead of sorting | 2 failures, incl. the fresh-process test |
| Unreadable files skipped | `test_unreadable_file_stops_and_names_it` |
| Missing files dropped from the list | `test_missing_file_stops_and_names_it` |
| Missing-file check removed only | Not caught, by design: the next step still stops with "Could not read 2.mat" |

Still open from the first review: damaged split files raise raw `KeyError`/`ValueError`
instead of `SplitError`, and the JSON counts are not covered by a fingerprint.
