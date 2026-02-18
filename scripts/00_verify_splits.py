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

"""Verify dataset splits have no duplicate images (by checksum) and no radiograph-level leakage.

Run this before training to ensure data integrity. Exits with code 1
if any cross-split duplicates, within-split duplicates, or radiograph-level
leakage across splits are found.

Usage:
    python scripts/00_verify_splits.py
    python scripts/00_verify_splits.py --data-root Dataset_partitioned
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]
SPLITS = ["training", "validation", "testing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify no checksum duplicates across splits.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Dataset_partitioned"),
        help="Root directory containing training/validation/testing splits.",
    )
    return parser.parse_args()


def compute_hashes(directory: Path) -> dict[str, Path]:
    """Map md5 hash -> file path for all PNGs in directory."""
    hashes: dict[str, Path] = {}
    dupes: list[tuple[Path, Path]] = []
    for png in sorted(directory.rglob("*.png")):
        h = hashlib.md5(png.read_bytes()).hexdigest()
        if h in hashes:
            dupes.append((hashes[h], png))
        else:
            hashes[h] = png
    return hashes, dupes


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    failed = False

    # Collect hashes per split
    split_hashes: dict[str, dict[str, Path]] = {}
    print("Computing checksums...")
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  WARNING: {split_dir} does not exist, skipping.")
            continue
        hashes, within_dupes = compute_hashes(split_dir)
        split_hashes[split] = hashes
        count = len(hashes) + len(within_dupes)
        print(f"  {split}: {count} files, {len(hashes)} unique")

        if within_dupes:
            failed = True
            print(f"  FAIL: {len(within_dupes)} within-split duplicates in {split}:")
            for orig, dupe in within_dupes[:5]:
                print(f"    {orig.relative_to(data_root)}")
                print(f"    {dupe.relative_to(data_root)}")
            if len(within_dupes) > 5:
                print(f"    ... and {len(within_dupes) - 5} more")

    # Cross-split checks
    print("\nCross-split duplicate check:")
    splits = list(split_hashes.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            name_a, name_b = splits[i], splits[j]
            overlap = set(split_hashes[name_a]) & set(split_hashes[name_b])
            if overlap:
                failed = True
                print(f"  FAIL: {len(overlap)} duplicates between {name_a} and {name_b}")
                examples = list(overlap)[:3]
                for h in examples:
                    print(f"    {split_hashes[name_a][h].relative_to(data_root)}")
                    print(f"    {split_hashes[name_b][h].relative_to(data_root)}")
                if len(overlap) > 3:
                    print(f"    ... and {len(overlap) - 3} more")
            else:
                print(f"  OK: {name_a} ({len(split_hashes[name_a])}) vs "
                      f"{name_b} ({len(split_hashes[name_b])}): 0 duplicates")

    # Radiograph-level leakage check
    print("\nRadiograph-level leakage check:")
    split_radiographs: dict[str, set[str]] = {}
    for split in splits:
        split_dir = data_root / split
        radiograph_ids: set[str] = set()
        for png in split_dir.rglob("*.png"):
            parts = png.stem.split("_Img")
            if len(parts) >= 2:
                radiograph_ids.add(parts[0])
        split_radiographs[split] = radiograph_ids
        print(f"  {split}: {len(radiograph_ids)} unique radiographs")

    radiograph_leakage_found = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            name_a, name_b = splits[i], splits[j]
            overlap = split_radiographs[name_a] & split_radiographs[name_b]
            if overlap:
                radiograph_leakage_found = True
                failed = True
                print(
                    f"  FAIL: {len(overlap)} radiographs appear in both "
                    f"{name_a} and {name_b}"
                )
                examples = sorted(overlap)[:5]
                for rid in examples:
                    print(f"    {rid}")
                if len(overlap) > 5:
                    print(f"    ... and {len(overlap) - 5} more")
            else:
                print(f"  OK: {name_a} vs {name_b}: 0 shared radiographs")

    if not radiograph_leakage_found:
        print("  PASSED: No radiograph-level leakage detected.")

    # Radiograph-to-split mapping summary
    print("\nRadiograph-to-split mapping:")
    all_radiographs: dict[str, list[str]] = {}
    for split, rids in split_radiographs.items():
        for rid in rids:
            all_radiographs.setdefault(rid, []).append(split)
    for rid in sorted(all_radiographs):
        assigned = ", ".join(all_radiographs[rid])
        print(f"  {rid}: {assigned}")

    # Class distribution summary
    print("\nClass distribution:")
    header = f"  {'class':<25}" + "".join(f"{s:>12}" for s in splits)
    print(header)
    print("  " + "-" * (25 + 12 * len(splits)))
    for class_name in CLASS_NAMES:
        row = f"  {class_name:<25}"
        for split in splits:
            split_dir = data_root / split / class_name
            count = len(list(split_dir.glob("*.png"))) if split_dir.exists() else 0
            row += f"{count:>12}"
        print(row)
    # Totals
    row = f"  {'TOTAL':<25}"
    for split in splits:
        row += f"{len(split_hashes.get(split, {})):>12}"
    print(row)

    if failed:
        print("\nFAILED: Duplicate images detected. Fix before training.")
        sys.exit(1)
    else:
        print("\nPASSED: All splits are clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
