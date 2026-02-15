#!/usr/bin/env python
"""
Build a SeisTrans-compatible project from Teapot Dome field data.

This script reads the Teapot Dome (NPR-3) SEG-Y and LAS files from the
extracted RMOTC tar archive, then uses GeoAgent's ProjectBuilder to create
a project that can be opened in SeisTrans.

Prerequisites:
    1. Download rmotc.tar from S3 and extract it into data/:
       curl -o data/rmotc.tar http://s3.amazonaws.com/open.source.geoscience/open_data/teapot/rmotc.tar
       tar xf data/rmotc.tar -C data/
    2. Ensure geoagent is installed: pip install -e .

Usage:
    python examples/teapot/build_teapot_project.py

Output:
    G:/2026/download_opendata/examples/teapot/teapot_project/   — project data folder
    G:/2026/download_opendata/examples/teapot/teapot_project.str — SeisTrans project pointer

Data source:
    US DOE / Rocky Mountain Oilfield Testing Center (RMOTC)
    http://s3.amazonaws.com/open.source.geoscience/open_data/teapot/rmotc.tar
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

try:
    import segyio
except ImportError:
    print("ERROR: segyio is required. Install with: pip install segyio")
    sys.exit(1)

try:
    import lasio
except ImportError:
    print("ERROR: lasio is required. Install with: pip install lasio")
    sys.exit(1)

from pathlib import Path

# Add project root to path if running as standalone script
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geoagent.io.project_builder import ProjectBuilder
from geoagent.core.data_manager import CoreDataManager
from geoagent.synthetic.wavelet_functions import ricker_wavelet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("G:/2026/download_opendata/examples/teapot/data/DataSets")
OUTPUT_DIR = Path("G:/2026/download_opendata/examples/teapot")
PROJECT_FOLDER = str(OUTPUT_DIR / "teapot_project")
PROJECT_NAME = "TeapotDome"

# Teapot Dome 3D SEG-Y byte positions (from header analysis):
# Inline = byte 181 (CDP_X field, values 1-345)
# Crossline = byte 185 (CDP_Y field, values 1-188)
# Coordinates = bytes 73/77 (SourceX/SourceY)
# Coordinate scalar = -10 (divide by 10)
TEAPOT_BYTE_POSITIONS = {
    'CDP_X': 73,              # SourceX = State Plane X (scaled by -10)
    'CDP_Y': 77,              # SourceY = State Plane Y (scaled by -10)
    'Inline': 181,            # CDP_X field = inline number (1-345)
    'Crossline': 185,         # CDP_Y field = crossline number (1-188)
    'Coord_Mult_Factor': 0.1, # Scalar is -10 → divide by 10 = multiply by 0.1
}

# Maximum number of LAS files to import (set to None for all)
MAX_WELLS = None


# ---------------------------------------------------------------------------
# Well Data Parsing
# ---------------------------------------------------------------------------

def parse_well_headers_xlsx(xlsx_path):
    """
    Parse TeapotDomeWellHeaders02-09-10.xlsx into a SeisTrans-compatible DataFrame.

    The XLSX has a descriptive header in row 0, column names in row 0
    (merged with title text), and data starting at row 1.
    """
    print(f"  Parsing well headers: {os.path.basename(xlsx_path)}")

    # Read with header=0 (the first row contains column-like labels)
    df_raw = pd.read_excel(xlsx_path, header=0)

    # The actual column names are in the first data row
    # Row 0 has: API Number, Operator, Well Name, Well Number, Northing, Easting, etc.
    col_names = df_raw.iloc[0].tolist()
    df = df_raw.iloc[1:].copy()
    df.columns = col_names
    df = df.reset_index(drop=True)

    # Filter out rows with no well number
    df = df.dropna(subset=['Well Number'])

    records = []
    for _, row in df.iterrows():
        try:
            well_num = str(row.get('Well Number', '')).strip()
            if not well_num:
                continue

            api = str(row.get('API Number', '')).strip()
            northing = _safe_float(row.get('Northing', 0))
            easting = _safe_float(row.get('Easting', 0))
            kb = _safe_float(row.get('Datum Elevation', 0))
            td = _safe_float(row.get('Total Depth', 0))

            well_name = well_num  # Use well number as name (e.g., "34-A-21")

            records.append({
                'Name': well_name,
                'UWI': api,
                'Well symbol': 'OIL',
                'Surface X': easting,
                'Surface Y': northing,
                'Latitude': '',
                'Latitude_dd': 0.0,
                'Longitude': '',
                'Longitude_dd': 0.0,
                'Drilling structure': '',
                'Well datum name': 'KB',
                'Well datum value': kb,
                'Well datum description': 'Kelly Bushing (feet)',
                'TD (MD)': td,
                'Cost': '',
                'Spud date': str(row.get('Spud Date', '')),
                'Operator': 'US DOE / RMOTC',
                'TWT auto': 0.0,
                'Bottom hole X': easting,
                'Bottom hole Y': northing,
            })
        except Exception as e:
            continue

    result = pd.DataFrame(records)
    result = result.set_index('Name')

    # Remove duplicates (keep first)
    result = result[~result.index.duplicated(keep='first')]

    print(f"    {len(result)} wells parsed")
    return result


def _safe_float(val, default=0.0):
    """Convert value to float, returning default on failure."""
    try:
        v = float(val)
        if np.isnan(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def extract_well_name_from_las(las_path):
    """Extract well identifier from LAS WELL header."""
    try:
        las = lasio.read(las_path)
        well_str = str(las.well['WELL'].value).strip()
        # LAS WELL field is like "NAVAL PETROLEUM RESERVE 3  #62-S-14"
        # Extract the well number after the #
        if '#' in well_str:
            well_num = well_str.split('#')[-1].strip()
            return well_num
        return well_str
    except Exception:
        return os.path.splitext(os.path.basename(las_path))[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Teapot Dome Project Builder")
    print("=" * 60)

    # Check for data
    segy_path = DATA_DIR / "Seismic" / "CD files" / "3D_Seismic" / "filt_mig.sgy"
    wells_dir = DATA_DIR / "Well Log" / "CD Files"
    headers_path = wells_dir / "TeapotDomeWellHeaders02-09-10.xlsx"

    if not segy_path.exists():
        print(f"\nSEG-Y file not found: {segy_path}")
        print("Extract from rmotc.tar first.")
        sys.exit(1)

    # Collect LAS files from both Shallow and Deeper directories
    las_dirs = [
        wells_dir / "LAS_log_files" / "Shallow_LAS_files",
        wells_dir / "LAS_log_files" / "Deeper_LAS_files",
    ]
    las_files = []
    for d in las_dirs:
        if d.exists():
            las_files.extend(sorted(set(
                glob.glob(str(d / "*.LAS")) +
                glob.glob(str(d / "*.las"))
            )))
    # Remove macOS resource fork files
    las_files = [f for f in las_files if not os.path.basename(f).startswith('._')]

    print(f"\nFound: SEG-Y={segy_path.exists()}, "
          f"Headers={headers_path.exists()}, "
          f"LAS files={len(las_files)}")

    # Create builder
    print(f"\nCreating project: {PROJECT_FOLDER}")
    builder = ProjectBuilder(PROJECT_FOLDER, PROJECT_NAME)

    # --- Inspect and import seismic ---
    print(f"\n--- Seismic Import ---")
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        print(f"  Traces:       {f.tracecount}")
        print(f"  Samples:      {len(f.samples)}")
        print(f"  Sample range: {f.samples[0]} - {f.samples[-1]} ms")
        if len(f.samples) > 1:
            print(f"  Sample rate:  {f.samples[1] - f.samples[0]} ms")

        h = f.header[0]
        scalar = h[71]
        print(f"  Coord scalar: {scalar} (multiply by {1.0/abs(scalar) if scalar < 0 else scalar})")
        print(f"  First trace: IL(b181)={h[181]}, XL(b185)={h[185]}, "
              f"X={h[73]*0.1:.1f}, Y={h[77]*0.1:.1f}")

    byte_pos = dict(TEAPOT_BYTE_POSITIONS)
    print(f"\n  Byte positions: {byte_pos}")
    print(f"  Importing as survey 'TeapotDome3D', attribute 'Amplitude'...")

    result = builder.import_segy(
        str(segy_path),
        survey_name="TeapotDome3D",
        attribute_name="Amplitude",
        byte_positions=byte_pos,
        copy_segy=True,
    )
    print(f"  Imported: {result.get('num_samples', '?')} samples/trace")

    # --- Import well headers ---
    well_heads_df = None
    if headers_path.exists():
        print(f"\n--- Well Headers ---")
        well_heads_df = parse_well_headers_xlsx(str(headers_path))

    # --- Import well logs ---
    if las_files:
        print(f"\n--- Well Logs ---")
        n_to_import = len(las_files) if MAX_WELLS is None else min(MAX_WELLS, len(las_files))
        print(f"  Importing {n_to_import} of {len(las_files)} LAS files...")

        imported_wells = []
        failed = 0
        for las_path in las_files[:n_to_import]:
            try:
                well_name = extract_well_name_from_las(las_path)
                builder.well_log_handler.import_well_logs(well_name, las_path)
                imported_wells.append(well_name)
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"    Warning: Failed to import {os.path.basename(las_path)}: {e}")

        print(f"  Imported: {len(imported_wells)} wells, {failed} failed")
        if imported_wells:
            print(f"  First 10: {imported_wells[:10]}")

    # Set well heads
    if well_heads_df is not None:
        # Match well heads to imported wells where possible
        builder.set_well_heads(well_heads_df)
        print(f"  Well heads set: {len(well_heads_df)} entries")

    # --- Add wavelets ---
    print(f"\n--- Wavelets ---")
    ricker_25 = ricker_wavelet(25.0, 128, 0.002)
    builder.add_wavelet("Ricker_25Hz", ricker_25, {
        'type': 'Ricker',
        'frequency': 25.0,
        'sample_rate': 0.002,
        'length': 128,
    })
    print("  Added Ricker 25 Hz wavelet")

    ricker_40 = ricker_wavelet(40.0, 128, 0.002)
    builder.add_wavelet("Ricker_40Hz", ricker_40, {
        'type': 'Ricker',
        'frequency': 40.0,
        'sample_rate': 0.002,
        'length': 128,
    })
    print("  Added Ricker 40 Hz wavelet")

    # --- Build project ---
    print(f"\n--- Building Project ---")
    str_path = builder.build()
    print(f"  .str file: {str_path}")

    # --- Summary ---
    summary = builder.summary()
    print(f"\n--- Project Summary ---")
    print(f"  Name:      {summary['project_name']}")
    print(f"  Surveys:   {summary['surveys']}")
    print(f"  Volumes:   {summary['volumes']}")
    print(f"  Wells:     {summary['n_wells']}")
    print(f"  Well logs: {summary['n_well_logs']}")
    print(f"  Wavelets:  {summary['n_wavelets']}")
    print(f"  Horizons:  {summary['n_horizons']}")

    # --- Validate round-trip ---
    print(f"\n--- Validation ---")
    try:
        dm = CoreDataManager(PROJECT_FOLDER)
        surveys = dm.get_available_surveys()
        wells = dm.get_available_wells()
        print(f"  CoreDataManager loaded successfully")
        print(f"  Surveys: {surveys}")
        print(f"  Wells:   {len(wells)} total")
        if wells:
            print(f"  First 10: {wells[:10]}")

        wavelets = dm.seismic_handler.loaded_data.get('wavelets', [])
        print(f"  Wavelets: {[w['name'] for w in wavelets]}")

        well_logs = dm.well_log_handler.loaded_data.get('well_logs', {})
        print(f"  Well logs: {len(well_logs)} wells with log data")

        print(f"\n  PASS: Project is valid and loadable")
    except Exception as e:
        print(f"  FAIL: Could not load project: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Project ready at: {PROJECT_FOLDER}")
    print(f"Open in SeisTrans: File > Open Project > {str_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
