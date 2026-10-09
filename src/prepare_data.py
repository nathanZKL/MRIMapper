"""Verify the original Figshare downloads and extract the .mat files.

Safe by design:
  * Original files in data/raw/ are only read, never changed.
  * Each file is checked against Figshare's published MD5 checksum.
  * Extraction never silently overwrites: an existing identical file is
    skipped, and an existing *different* file stops the script with an error.

Run:  python src\\prepare_data.py
"""

import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone

from config import (EXPECTED_MD5, EXTRACTED_DIR, MAT_DIR, RAW_DIR, SOURCE,
                    ZIP_NAMES)

# Only plain names like "123.mat" are accepted from the archives. This also
# blocks "zip slip" (a crafted path such as ..\..\evil.exe escaping the folder).
SAFE_MEMBER = re.compile(r"^\d+\.mat$")


def md5_of(path, chunk_size=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_files():
    """Return {filename: md5}. Exits with instructions if anything is wrong."""
    if not RAW_DIR.exists():
        sys.exit(f"Missing folder: {RAW_DIR}\nPut the Figshare downloads there (see docs/SETUP_WINDOWS.md).")

    problems, checksums = [], {}
    for name, expected in EXPECTED_MD5.items():
        path = RAW_DIR / name
        if not path.exists():
            problems.append(f"MISSING   {name}  -> expected at {path}")
            continue
        actual = md5_of(path)
        checksums[name] = actual
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  [{status:8}] {name}  md5={actual}")
        if actual != expected:
            problems.append(f"MISMATCH  {name}: got {actual}, Figshare says {expected}")

    if problems:
        print("\nProblems found:")
        for p in problems:
            print("  " + p)
        sys.exit("Fix the files above (re-download from Figshare) and run again.")
    return checksums


def extract_zip(zip_path):
    """Extract one archive into MAT_DIR. Returns counts per outcome."""
    counts = {"extracted": 0, "already_identical": 0}
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not SAFE_MEMBER.match(member.filename):
                sys.exit(f"Unexpected file {member.filename!r} in {zip_path.name}; stopping for safety.")

            target = MAT_DIR / member.filename
            data = archive.read(member)  # zipfile verifies the CRC on read
            if target.exists():
                if target.read_bytes() == data:
                    counts["already_identical"] += 1
                    continue
                sys.exit(f"Refusing to overwrite {target}: it differs from the copy in {zip_path.name}.\n"
                         "Move or delete it yourself if you are sure, then run again.")
            # Write to a temp name first so an interrupted run never leaves a half-written .mat
            tmp = target.with_suffix(".mat.partial")
            tmp.write_bytes(data)
            tmp.replace(target)
            counts["extracted"] += 1
    return counts


def main():
    print(f"Checking original downloads in {RAW_DIR}")
    checksums = verify_raw_files()

    MAT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ZIP_NAMES:
        counts = extract_zip(RAW_DIR / name)
        summary[name] = counts
        print(f"  {name}: {counts['extracted']} extracted, {counts['already_identical']} already present (identical)")

    total = len(list(MAT_DIR.glob("*.mat")))
    print(f"\n{total} .mat files now in {MAT_DIR}")

    # Provenance record so we always know where these files came from.
    record = {
        "source": SOURCE,
        "raw_md5": checksums,
        "md5_verified_against": "Figshare API computed_md5",
        "extraction": summary,
        "mat_file_count": total,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (EXTRACTED_DIR / "SOURCE.json").write_text(json.dumps(record, indent=2))
    print(f"Provenance written to {EXTRACTED_DIR / 'SOURCE.json'}")


if __name__ == "__main__":
    main()
