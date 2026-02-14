#!/usr/bin/env python
"""
Download Volve field data from Equinor's public data portal.

The Volve dataset is hosted on Azure Blob Storage. This script downloads
the seismic SEG-Y and well LAS files needed by build_volve_project.py.

Downloaded files are stored in examples/volve/data/ (gitignored).

Usage:
    python examples/volve/download_volve.py

Requirements:
    requests (pip install requests)

Notes:
    - Total download size is ~500MB–2GB depending on selected files.
    - The Volve data portal: https://data.equinor.com/dataset/Volve
    - If URLs become unavailable, download manually from the portal
      and place files in the data/ subdirectory.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
SEISMIC_DIR = DATA_DIR / "seismic"
WELLS_DIR = DATA_DIR / "wells"

# Equinor's Azure blob storage base URL for public Volve data.
# These URLs are based on the known Volve data distribution structure.
# If they become stale, download manually from https://data.equinor.com/dataset/Volve
AZURE_BASE = "https://dataplatformblvolve.blob.core.windows.net/pub"

# Files to download.  Each entry: (relative_url, local_subdir, local_filename)
# Seismic: ST0202 near-stack 3D OBC survey
DOWNLOAD_MANIFEST = [
    # -- Seismic (the main ST0202 3D OBC survey) --
    {
        "url": f"{AZURE_BASE}/Seismic/ST0202/Stacked/ST0202R08_PSTM_FULL_OFFSET_DEPTH.sgy",
        "dest": "seismic/ST0202.sgy",
        "description": "ST0202 3D OBC survey — full offset stack",
        "optional": False,
    },
    # -- Wells (LAS files for key F-wells) --
    # The Volve well data is distributed as a zip or individual files.
    # Adjust URLs based on actual portal structure.
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-1/15_9-F-1.las",
        "dest": "wells/15_9-F-1.las",
        "description": "Well 15/9-F-1 log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-1A/15_9-F-1A.las",
        "dest": "wells/15_9-F-1A.las",
        "description": "Well 15/9-F-1A log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-4/15_9-F-4.las",
        "dest": "wells/15_9-F-4.las",
        "description": "Well 15/9-F-4 log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-5/15_9-F-5.las",
        "dest": "wells/15_9-F-5.las",
        "description": "Well 15/9-F-5 log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-11/15_9-F-11.las",
        "dest": "wells/15_9-F-11.las",
        "description": "Well 15/9-F-11 log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-12/15_9-F-12.las",
        "dest": "wells/15_9-F-12.las",
        "description": "Well 15/9-F-12 log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-14/15_9-F-14.las",
        "dest": "wells/15_9-F-14.las",
        "description": "Well 15/9-F-14 log data",
        "optional": True,
    },
    {
        "url": f"{AZURE_BASE}/Well_logs/15_9-F-15/15_9-F-15.las",
        "dest": "wells/15_9-F-15.las",
        "description": "Well 15/9-F-15 log data",
        "optional": True,
    },
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url, dest_path, description="", chunk_size=8192):
    """Download a file with progress reporting."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  [SKIP] {dest_path.name} already exists")
        return True

    print(f"  Downloading: {description or dest_path.name}")
    print(f"    URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    print(f"\r    {mb:.1f} MB ({pct}%)", end='', flush=True)

        print(f"\r    Done: {downloaded / (1024*1024):.1f} MB")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n    [ERROR] Download failed: {e}")
        # Clean up partial file
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    """Download all Volve data files."""
    print("=" * 60)
    print("Volve Field Data Downloader")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    skipped = 0

    for item in DOWNLOAD_MANIFEST:
        dest = DATA_DIR / item["dest"]
        ok = download_file(item["url"], dest, item["description"])
        if ok:
            if dest.exists():
                success += 1
            else:
                skipped += 1
        else:
            if item.get("optional", True):
                print(f"    (optional — continuing)")
                skipped += 1
            else:
                print(f"    [CRITICAL] Required file failed to download!")
                failed += 1

    print()
    print("-" * 60)
    print(f"Downloaded: {success}  Skipped: {skipped}  Failed: {failed}")
    print()

    if failed > 0:
        print("Some required files failed to download.")
        print("Try downloading manually from: https://data.equinor.com/dataset/Volve")
        print(f"Place files in: {DATA_DIR}")
        sys.exit(1)

    print("Download complete. Next step:")
    print("  python examples/volve/build_volve_project.py")


if __name__ == "__main__":
    main()
