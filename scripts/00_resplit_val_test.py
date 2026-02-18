#!/usr/bin/env python3

#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#
#  Licensed under the MIT License.
#  For details: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   RIAWELC — Welding Defect Classification & Segmentation Pipeline

"""Re-split validation set into val/test and remove leaked test set.

The original testing partition is an exact duplicate of a training subset.
This script deletes it, then performs a stratified 50/50 split of the
validation set into new validation and test partitions.

Usage:
    python scripts/00_resplit_val_test.py
    python scripts/00_resplit_val_test.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 9
DATA_ROOT = Path("Dataset_partitioned")
CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-split val into val/test, remove leaked test.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without moving files.")
    return parser.parse_args()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect_hashes(directory: Path) -> dict[str, list[Path]]:
    """Map md5 hash -> list of file paths."""
    hashes: dict[str, list[Path]] = defaultdict(list)
    for png in sorted(directory.rglob("*.png")):
        hashes[file_md5(png)].append(png)
    return hashes


def verify_no_duplicates(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
) -> bool:
    """Verify zero checksum overlap between all split pairs."""
    print("\nComputing checksums for verification...")
    train_h = set(collect_hashes(train_dir).keys())
    val_h = set(collect_hashes(val_dir).keys())
    test_h = set(collect_hashes(test_dir).keys())

    ok = True
    for name_a, set_a, name_b, set_b in [
        ("train", train_h, "val", val_h),
        ("train", train_h, "test", test_h),
        ("val", val_h, "test", test_h),
    ]:
        overlap = set_a & set_b
        if overlap:
            print(f"  FAIL: {len(overlap)} checksum duplicates between {name_a} and {name_b}")
            ok = False
        else:
            print(f"  OK: 0 duplicates between {name_a} ({len(set_a)}) and {name_b} ({len(set_b)})")

    return ok


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(SEED)

    train_dir = DATA_ROOT / "training"
    val_dir = DATA_ROOT / "validation"
    test_dir = DATA_ROOT / "testing"

    # --- Step 1: Delete leaked test set ---
    print("Step 1: Deleting leaked test set...")
    test_count = len(list(test_dir.rglob("*.png")))
    if args.dry_run:
        print(f"  [DRY RUN] Would delete {test_count} files from {test_dir}")
    else:
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                shutil.rmtree(class_dir)
        print(f"  Deleted {test_count} files from {test_dir}")

    # --- Step 2: Stratified 50/50 split of validation ---
    print("\nStep 2: Stratified 50/50 split of validation set...")
    total_moved = 0

    for class_name in CLASS_NAMES:
        val_class_dir = val_dir / class_name
        test_class_dir = test_dir / class_name

        images = sorted(val_class_dir.glob("*.png"))
        n = len(images)
        n_test = n // 2
        n_val = n - n_test

        # Shuffle deterministically, then split
        indices = rng.permutation(n)
        test_indices = set(indices[:n_test].tolist())

        test_files = [images[i] for i in range(n) if i in test_indices]

        print(f"  {class_name}: {n} total -> {n_val} val + {n_test} test")

        if args.dry_run:
            print(f"    [DRY RUN] Would move {len(test_files)} files to {test_class_dir}")
        else:
            test_class_dir.mkdir(parents=True, exist_ok=True)
            for f in test_files:
                shutil.move(str(f), test_class_dir / f.name)
            total_moved += len(test_files)

    if not args.dry_run:
        print(f"\n  Moved {total_moved} files to {test_dir}")

    # --- Step 3: Print final counts ---
    print("\nStep 3: Final split counts")
    splits = [("training", train_dir), ("validation", val_dir), ("testing", test_dir)]
    for split_name, split_dir in splits:
        total = 0
        print(f"  {split_name}:")
        for class_name in CLASS_NAMES:
            count = len(list((split_dir / class_name).glob("*.png")))
            total += count
            print(f"    {class_name}: {count}")
        print(f"    TOTAL: {total}")

    # --- Step 4: Checksum verification ---
    if not args.dry_run:
        passed = verify_no_duplicates(train_dir, val_dir, test_dir)
        if passed:
            print("\nAll checks passed. No duplicate images across splits.")
        else:
            print("\nWARNING: Duplicate images detected! Review the output above.")
    else:
        print("\n[DRY RUN] Skipping checksum verification.")


if __name__ == "__main__":
    main()
