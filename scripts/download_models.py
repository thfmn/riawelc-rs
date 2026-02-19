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

"""Download production model files into the models/ directory.

If you cloned the repository with Git LFS the models are already present.
Run this script as a fallback when LFS is unavailable or the files are
missing.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# ---- Configuration --------------------------------------------------------

_MODELS: list[dict[str, str | int]] = [
    {
        "name": "classifier_efficientnetb0_v1.keras",
        "url": "https://github.com/thfmn/riawelc-rs/releases/download/v1.0.0/classifier_efficientnetb0_v1.keras",
        "size": 46_470_910,
    },
    {
        "name": "segmentation_unet_v2.keras",
        "url": "https://github.com/thfmn/riawelc-rs/releases/download/v1.0.0/segmentation_unet_v2.keras",
        "size": 102_831_694,
    },
]

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---- Helpers --------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    print(f"  Saved to {dest}")


def _verify_size(path: Path, expected: int) -> bool:
    actual = path.stat().st_size
    if actual != expected:
        print(
            f"  WARNING: Size mismatch for {path.name} "
            f"(expected {expected:,} bytes, got {actual:,} bytes)"
        )
        return False
    print(f"  Verified {path.name} ({expected:,} bytes)")
    return True


# ---- Main -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RIAWELC production models."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the files already exist.",
    )
    args = parser.parse_args()

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for model in _MODELS:
        dest = _MODELS_DIR / str(model["name"])
        if dest.exists() and not args.force:
            print(f"  {model['name']} already exists, skipping (use --force to re-download).")
            _verify_size(dest, int(model["size"]))
            continue
        try:
            _download(str(model["url"]), dest)
            if not _verify_size(dest, int(model["size"])):
                all_ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR downloading {model['name']}: {exc}")
            all_ok = False

    if not all_ok:
        print("\nSome downloads failed or had size mismatches.")
        sys.exit(1)

    print("\nAll models are ready in models/.")


if __name__ == "__main__":
    main()
