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

"""Re-split dataset by source radiograph to prevent data leakage.

Merges all images into a temporary pool, then redistributes them into
training/validation/testing based on a hardcoded radiograph-to-split
assignment. This ensures no patches from the same radiograph appear
in multiple splits.

Usage:
    python scripts/00_resplit_by_radiograph.py
    python scripts/00_resplit_by_radiograph.py --dry-run
    python scripts/00_resplit_by_radiograph.py --data-root Dataset_partitioned
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]
SPLITS = ["training", "validation", "testing"]
SPLIT_MAP = {"train": "training", "val": "validation", "test": "testing"}
POOL_DIR_NAME = "_pool"

RADIOGRAPH_ASSIGNMENT: dict[str, str] = {
    # TRAIN (15 radiographs, ~71% of patches)
    "RRT-09R": "train",
    "RRT-107R": "train",
    "RRT-11R": "train",
    "RRT-12R": "train",
    "RRT-13R": "train",
    "RRT-15R": "train",
    "RRT-20R": "train",
    "RRT-26R": "train",
    "RRT-27R": "train",
    "RRT-28R": "train",
    "RRT-39R": "train",
    "RRT-40R": "train",
    "RRT-88R": "train",
    "RRT-94R": "train",
    "RRT-99R": "train",
    # VAL (8 radiographs, ~16% of patches)
    "RRT-24R": "val",
    "bam5": "val",
    "RRT-23R": "val",
    "RRT-101R": "val",
    "RRT-105R": "val",
    "RRT-97R": "val",
    "RRT-31R": "val",
    "RRT-102R": "val",
    # TEST (6 radiographs, ~12% of patches)
    "RRT-90R": "test",
    "RRT-29R": "test",
    "RRT-42R": "test",
    "RRT-22R": "test",
    "RRT-30R": "test",
    "RRT-98R": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split dataset by source radiograph to prevent data leakage.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Dataset_partitioned"),
        help="Root directory containing training/validation/testing splits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without moving files.",
    )
    return parser.parse_args()


def extract_radiograph_id(filename: str) -> str:
    """Extract radiograph ID from filename (everything before '_Img')."""
    parts = filename.split("_Img")
    if len(parts) < 2:
        return filename.rsplit(".", 1)[0]
    return parts[0]


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_hashes(directory: Path) -> dict[str, Path]:
    """Map md5 hash -> file path for all PNGs under directory."""
    hashes: dict[str, Path] = {}
    for png in sorted(directory.rglob("*.png")):
        hashes[file_md5(png)] = png
    return hashes


def verify_no_cross_split_overlap(data_root: Path) -> bool:
    """Verify zero checksum overlap between all split pairs."""
    print("\nVerification: computing checksums...")
    split_hashes: dict[str, set[str]] = {}
    for split in SPLITS:
        split_dir = data_root / split
        if split_dir.exists():
            hashes = collect_hashes(split_dir)
            split_hashes[split] = set(hashes.keys())
            print(f"  {split}: {len(hashes)} unique files")
        else:
            split_hashes[split] = set()

    ok = True
    names = list(split_hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = split_hashes[a] & split_hashes[b]
            if overlap:
                print(f"  FAIL: {len(overlap)} checksum duplicates between {a} and {b}")
                ok = False
            else:
                print(
                    f"  OK: 0 duplicates between {a} ({len(split_hashes[a])}) "
                    f"and {b} ({len(split_hashes[b])})"
                )
    return ok


def verify_no_radiograph_overlap(data_root: Path) -> bool:
    """Verify no radiograph ID appears in multiple splits."""
    print("\nVerification: checking radiograph-level split integrity...")
    split_radiographs: dict[str, set[str]] = {}
    for split in SPLITS:
        split_dir = data_root / split
        ids: set[str] = set()
        if split_dir.exists():
            for png in split_dir.rglob("*.png"):
                ids.add(extract_radiograph_id(png.name))
        split_radiographs[split] = ids
        print(f"  {split}: {len(ids)} unique radiographs")

    ok = True
    names = list(split_radiographs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = split_radiographs[a] & split_radiographs[b]
            if overlap:
                print(f"  FAIL: radiograph(s) in both {a} and {b}: {sorted(overlap)}")
                ok = False
            else:
                print(f"  OK: 0 shared radiographs between {a} and {b}")
    return ok


def print_split_counts(data_root: Path) -> None:
    """Print per-class, per-split file counts."""
    print("\nFinal split counts:")
    header = f"  {'class':<25}" + "".join(f"{s:>12}" for s in SPLITS)
    print(header)
    print("  " + "-" * (25 + 12 * len(SPLITS)))

    totals = {s: 0 for s in SPLITS}
    for class_name in CLASS_NAMES:
        row = f"  {class_name:<25}"
        for split in SPLITS:
            class_dir = data_root / split / class_name
            count = len(list(class_dir.glob("*.png"))) if class_dir.exists() else 0
            totals[split] += count
            row += f"{count:>12}"
        print(row)

    row = f"  {'TOTAL':<25}"
    for split in SPLITS:
        row += f"{totals[split]:>12}"
    print(row)

    grand_total = sum(totals.values())
    print(f"\n  Grand total: {grand_total} images")
    if grand_total > 0:
        for split in SPLITS:
            pct = 100.0 * totals[split] / grand_total
            print(f"  {split}: {totals[split]} ({pct:.1f}%)")


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    dry_run: bool = args.dry_run
    pool_dir = data_root / POOL_DIR_NAME

    if not data_root.exists():
        print(f"ERROR: Data root {data_root} does not exist.")
        sys.exit(1)

    # --- Step 1: Move all images into the pool ---
    print("Step 1: Collecting all images into temporary pool...")
    pool_count = 0
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            src_dir = data_root / split / class_name
            if not src_dir.exists():
                continue
            dst_dir = pool_dir / class_name
            images = sorted(src_dir.glob("*.png"))
            if not images:
                continue

            if dry_run:
                print(
                    f"  [DRY RUN] Would move {len(images)} files from {split}/{class_name} to pool"
                )
                pool_count += len(images)
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img in images:
                    target = dst_dir / img.name
                    # Handle duplicate filenames from different splits
                    if target.exists():
                        # Skip if identical content
                        if file_md5(img) == file_md5(target):
                            img.unlink()
                            continue
                        # Rename if different content (shouldn't happen, but be safe)
                        stem = img.stem
                        suffix = img.suffix
                        counter = 1
                        while target.exists():
                            target = dst_dir / f"{stem}_dup{counter}{suffix}"
                            counter += 1
                    shutil.move(str(img), target)
                    pool_count += 1

    print(f"  Pooled {pool_count} images")

    if not dry_run:
        # Clear split directories
        for split in SPLITS:
            for class_name in CLASS_NAMES:
                class_dir = data_root / split / class_name
                if class_dir.exists():
                    shutil.rmtree(class_dir)
                class_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 2: Redistribute by radiograph assignment ---
    print("\nStep 2: Redistributing images by radiograph assignment...")
    unknown_ids: set[str] = set()
    moved_counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASS_NAMES} for s in SPLITS}
    radiograph_splits: dict[str, set[str]] = defaultdict(set)

    if dry_run:
        # In dry-run, scan the existing split dirs since we didn't move anything
        all_images: list[tuple[str, Path]] = []
        for split in SPLITS:
            for class_name in CLASS_NAMES:
                src_dir = data_root / split / class_name
                if src_dir.exists():
                    for img in sorted(src_dir.glob("*.png")):
                        all_images.append((class_name, img))
    else:
        all_images = []
        for class_name in CLASS_NAMES:
            class_pool = pool_dir / class_name
            if class_pool.exists():
                for img in sorted(class_pool.glob("*.png")):
                    all_images.append((class_name, img))

    for class_name, img in all_images:
        radiograph_id = extract_radiograph_id(img.name)

        if radiograph_id not in RADIOGRAPH_ASSIGNMENT:
            unknown_ids.add(radiograph_id)
            continue

        split_key = RADIOGRAPH_ASSIGNMENT[radiograph_id]
        split_dirname = SPLIT_MAP[split_key]
        radiograph_splits[radiograph_id].add(split_key)
        dst_dir = data_root / split_dirname / class_name

        if dry_run:
            moved_counts[split_dirname][class_name] += 1
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), dst_dir / img.name)
            moved_counts[split_dirname][class_name] += 1

    if unknown_ids:
        print(f"\n  ERROR: Unknown radiograph IDs found: {sorted(unknown_ids)}")
        print("  These images cannot be assigned to a split. Aborting.")
        # Clean up pool if we created it
        if not dry_run and pool_dir.exists():
            print(f"  WARNING: Pool directory {pool_dir} left in place for inspection.")
        sys.exit(1)

    for split in SPLITS:
        total = sum(moved_counts[split].values())
        per_class = ", ".join(f"{c}={moved_counts[split][c]}" for c in CLASS_NAMES)
        print(f"  {split}: {total} images ({per_class})")

    # --- Step 3: Clean up pool ---
    if not dry_run:
        if pool_dir.exists():
            shutil.rmtree(pool_dir)
            print(f"\nCleaned up temporary pool directory: {pool_dir}")
    else:
        print(f"\n  [DRY RUN] Would clean up pool directory: {pool_dir}")

    # --- Step 4: Print final counts ---
    if not dry_run:
        print_split_counts(data_root)

        # --- Step 5: Verification ---
        checksum_ok = verify_no_cross_split_overlap(data_root)
        radiograph_ok = verify_no_radiograph_overlap(data_root)

        if checksum_ok and radiograph_ok:
            print("\nAll checks passed. Dataset is cleanly split by radiograph.")
        else:
            print("\nWARNING: Verification failed! Review the output above.")
            sys.exit(1)
    else:
        print("\n[DRY RUN] Skipping verification. No files were moved.")
        # Still print what the counts would be
        print("\nProjected split counts:")
        for split in SPLITS:
            total = sum(moved_counts[split].values())
            per_class = ", ".join(f"{c}={moved_counts[split][c]}" for c in CLASS_NAMES)
            print(f"  {split}: {total} images ({per_class})")


if __name__ == "__main__":
    main()
