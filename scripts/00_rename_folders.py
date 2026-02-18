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

"""Rename Italian dataset folder names to English.

Maps: Difetto1 → lack_of_penetration, Difetto2 → porosity,
      Difetto4 → crack, NoDifetto → no_defect

Usage:
    python scripts/00_rename_folders.py
    python scripts/00_rename_folders.py --data-root Dataset_partitioned --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAPPING = {
    "Difetto1": "lack_of_penetration",
    "Difetto2": "porosity",
    "Difetto4": "crack",
    "NoDifetto": "no_defect",
}

SPLITS = ["training", "validation", "testing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename Italian folders to English.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Dataset_partitioned"),
        help="Root of the partitioned dataset (default: Dataset_partitioned)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print renames without executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renamed = 0

    for split in SPLITS:
        split_dir = args.data_root / split
        if not split_dir.exists():
            print(f"  SKIP {split_dir} (not found)")
            continue

        for italian, english in MAPPING.items():
            src = split_dir / italian
            dst = split_dir / english
            if src.exists():
                if args.dry_run:
                    print(f"  WOULD RENAME {src} → {dst}")
                else:
                    src.rename(dst)
                    print(f"  RENAMED {src} → {dst}")
                renamed += 1
            elif dst.exists():
                print(f"  OK {dst} (already renamed)")

    action = "would rename" if args.dry_run else "renamed"
    print(f"\nDone: {action} {renamed} folders.")


if __name__ == "__main__":
    main()
