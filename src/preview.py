"""Draw a labeled preview grid: original image | expert mask | overlay.

For each tumor type we pick a few examples from *different* patients
(fixed random seed, so the same picks every run). The masks shown are the
dataset's expert annotations, NOT predictions from any model.

Display notes:
  * Contrast is stretched per image (0.5th-99.5th percentile) only for viewing;
    the source data on disk is never changed or converted.
  * Images are drawn pixel-for-pixel (no interpolation, square pixels), and the
    mask is drawn on the same pixel grid, so alignment and aspect ratio are kept.

Run:  python src\\preview.py            (2 examples per type)
      python src\\preview.py --per-type 4 --seed 7
"""

import argparse
import random

import matplotlib

matplotlib.use("Agg")  # write files without needing a display window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import LABEL_NAMES, MAT_DIR, REPORTS_DIR
from mat_loader import load_slice, pid_base

MASK_COLOR = (1.0, 0.15, 0.15)  # red


def pick_examples(per_type, seed):
    """Return {label: [TumorSlice, ...]} with one slice per distinct patient."""
    files = sorted(MAT_DIR.glob("*.mat"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"No .mat files in {MAT_DIR}. Run: python src\\prepare_data.py")
    random.Random(seed).shuffle(files)

    chosen = {label: [] for label in LABEL_NAMES}
    used_pids = set()
    for path in files:
        if all(len(v) >= per_type for v in chosen.values()):
            break
        s = load_slice(path)
        if s.label in chosen and len(chosen[s.label]) < per_type and pid_base(s.pid) not in used_pids:
            chosen[s.label].append(s)
            used_pids.add(pid_base(s.pid))
    return chosen


def display_window(image):
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip((image.astype(np.float64) - lo) / max(hi - lo, 1e-9), 0, 1)


def draw(chosen, out_path):
    rows = [(label, s) for label, slices in chosen.items() for s in slices]
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.7 * len(rows)), squeeze=False)
    mask_cmap = ListedColormap([(0, 0, 0, 0), (*MASK_COLOR, 0.45)])  # transparent where no tumor

    for col, title in enumerate(["Original MRI (T1 contrast-enhanced)",
                                 "Expert tumor mask (dataset annotation)",
                                 "Mask overlaid on MRI"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=22)

    for i, (label, s) in enumerate(rows):
        shown = display_window(s.image)
        h, w = s.image.shape
        common = dict(interpolation="nearest", aspect="equal", origin="upper")

        axes[i, 0].imshow(shown, cmap="gray", vmin=0, vmax=1, **common)
        axes[i, 1].imshow(s.mask, cmap="gray", vmin=0, vmax=1, **common)
        axes[i, 2].imshow(shown, cmap="gray", vmin=0, vmax=1, **common)
        axes[i, 2].imshow(s.mask.astype(int), cmap=mask_cmap, vmin=0, vmax=1, **common)
        axes[i, 2].contour(s.mask, levels=[0.5], colors=[MASK_COLOR], linewidths=0.8)

        axes[i, 0].set_ylabel(f"{LABEL_NAMES[label]} (label {label})\n{s.file} | patient {s.pid}\n{h}x{w} px",
                              fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)

    fig.suptitle("Jun Cheng brain tumor dataset (Figshare 1512427 v8): examples per tumor type\n"
                 "Masks are EXPERT ANNOTATIONS from the dataset, NOT model predictions. "
                 "Contrast stretched for display only.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-type", type=int, default=2, help="examples per tumor type (default 2)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for which examples are picked")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    chosen = pick_examples(args.per_type, args.seed)
    out_path = REPORTS_DIR / "preview_tumor_types.png"
    draw(chosen, out_path)
    for label, slices in chosen.items():
        print(f"  {LABEL_NAMES[label]}: " + ", ".join(f"{s.file} (patient {s.pid})" for s in slices))
    print(f"Preview saved to {out_path}")


if __name__ == "__main__":
    main()
