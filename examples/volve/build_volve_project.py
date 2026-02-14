#!/usr/bin/env python
"""
Build a SeisTrans-compatible project from Volve field data.

This script reads the downloaded Volve SEG-Y and LAS files, then uses
GeoAgent's ProjectBuilder to create a project that can be opened in SeisTrans.

Prerequisites:
    1. Run download_volve.py first (or manually place files in data/).
    2. Ensure geoagent is installed: pip install -e .

Usage:
    python examples/volve/build_volve_project.py

Output:
    examples/volve/output/volve_project/   — project data folder
    examples/volve/output/volve_project.str — SeisTrans project pointer
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

DATA_DIR = script_dir / "data"
OUTPUT_DIR = script_dir / "output"
PROJECT_FOLDER = str(OUTPUT_DIR / "volve_project")
PROJECT_NAME = "Volve"

# Known Volve ST0202 survey SEG-Y byte positions.
# These may need adjustment depending on the specific SEG-Y variant downloaded.
# Use segyio.tools.metadata() or segyio.tools.wrap() to inspect if needed.
VOLVE_BYTE_POSITIONS = {
    'CDP_X': 181,        # CDP X coordinate
    'CDP_Y': 185,        # CDP Y coordinate
    'Inline': 189,       # Inline number
    'Crossline': 193,    # Crossline number
    'Coord_Mult_Factor': 0.01,  # Adjust based on coordinate scalar
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_coord_scalar(segy_path):
    """
    Auto-detect coordinate scalar from a SEG-Y file.

    In standard SEG-Y, the scalar at bytes 71-72 or 181-185 headers
    determines how to interpret coordinate values:
      - Positive scalar: multiply
      - Negative scalar: divide by abs(scalar)
    """
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        if scalar == 0:
            scalar = 1
        if scalar < 0:
            return 1.0 / abs(scalar)
        return float(scalar)


def inspect_segy(segy_path):
    """Print basic SEG-Y file information for debugging."""
    print(f"\nInspecting: {os.path.basename(segy_path)}")
    print("-" * 50)

    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        print(f"  Traces:       {f.tracecount}")
        print(f"  Samples:      {len(f.samples)}")
        print(f"  Sample range: {f.samples[0]} - {f.samples[-1]}")
        if len(f.samples) > 1:
            print(f"  Sample rate:  {f.samples[1] - f.samples[0]} ms")

        # Read first trace header for coordinate info
        h = f.header[0]
        print(f"  First trace:")
        print(f"    Inline:     {h[segyio.TraceField.INLINE_3D]}")
        print(f"    Crossline:  {h[segyio.TraceField.CROSSLINE_3D]}")
        print(f"    CDP_X:      {h[segyio.TraceField.CDP_X]}")
        print(f"    CDP_Y:      {h[segyio.TraceField.CDP_Y]}")
        print(f"    Scalar:     {h[segyio.TraceField.SourceGroupScalar]}")
        print(f"    Delay:      {h[segyio.TraceField.DelayRecordingTime]}")

    return True


def build_well_heads_from_las(las_dir):
    """
    Build a well_heads DataFrame from LAS file headers.

    Extracts well name, coordinates, and other metadata from the LAS
    well information section.
    """
    las_files = sorted(
        glob.glob(os.path.join(las_dir, '*.las')) +
        glob.glob(os.path.join(las_dir, '*.LAS'))
    )

    if not las_files:
        print(f"  No LAS files found in {las_dir}")
        return pd.DataFrame()

    records = []
    for las_path in las_files:
        try:
            las = lasio.read(las_path)
            well_info = las.well

            name = str(well_info.get('WELL', {}).get('value', '') or
                       os.path.splitext(os.path.basename(las_path))[0]).strip()

            # Try to extract coordinates from LAS header
            x = _get_las_param(well_info, ['XWELL', 'X', 'XCOORD', 'EASTING'])
            y = _get_las_param(well_info, ['YWELL', 'Y', 'YCOORD', 'NORTHING'])
            kb = _get_las_param(well_info, ['KB', 'ELEV', 'EKB', 'EKBE'])
            td = _get_las_param(well_info, ['TD', 'TDEP', 'TDD'])

            records.append({
                'Name': name,
                'UWI': str(well_info.get('UWI', {}).get('value', '') or ''),
                'Well symbol': 'OIL',
                'Surface X': x if x else 0.0,
                'Surface Y': y if y else 0.0,
                'Latitude': '',
                'Latitude_dd': 0.0,
                'Longitude': '',
                'Longitude_dd': 0.0,
                'Drilling structure': '',
                'Well datum name': 'KB',
                'Well datum value': kb if kb else 0.0,
                'Well datum description': 'Kelly Bushing',
                'TD (MD)': td if td else 0.0,
                'Cost': '',
                'Spud date': '',
                'Operator': 'Equinor',
                'TWT auto': 0.0,
                'Bottom hole X': x if x else 0.0,
                'Bottom hole Y': y if y else 0.0,
            })
            print(f"  Read well header: {name}")

        except Exception as e:
            print(f"  Warning: Could not parse {las_path}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index('Name')
    return df


def _get_las_param(well_info, keys):
    """Try multiple LAS parameter names, return first valid float."""
    for key in keys:
        try:
            val = well_info[key].value
            if val is not None and str(val).strip():
                return float(val)
        except (KeyError, ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Volve Project Builder")
    print("=" * 60)

    # Check for data
    seismic_dir = DATA_DIR / "seismic"
    wells_dir = DATA_DIR / "wells"

    segy_files = list(seismic_dir.glob("*.sgy")) + list(seismic_dir.glob("*.segy"))
    las_files = list(wells_dir.glob("*.las")) + list(wells_dir.glob("*.LAS"))

    if not segy_files and not las_files:
        print(f"\nNo data found in {DATA_DIR}")
        print("Run download_volve.py first, or place files manually:")
        print(f"  SEG-Y files: {seismic_dir}/")
        print(f"  LAS files:   {wells_dir}/")
        sys.exit(1)

    print(f"\nFound {len(segy_files)} SEG-Y files, {len(las_files)} LAS files")

    # Create builder
    print(f"\nCreating project: {PROJECT_FOLDER}")
    builder = ProjectBuilder(PROJECT_FOLDER, PROJECT_NAME)

    # --- Import seismic ---
    if segy_files:
        segy_path = str(segy_files[0])  # Use first SEG-Y file
        print(f"\n--- Seismic Import ---")
        inspect_segy(segy_path)

        # Auto-detect coordinate scalar
        scalar = detect_coord_scalar(segy_path)
        byte_pos = dict(VOLVE_BYTE_POSITIONS)
        byte_pos['Coord_Mult_Factor'] = scalar
        print(f"\n  Using coordinate scalar: {scalar}")

        survey_name = "ST0202"
        attribute_name = "Amplitude"

        print(f"  Importing as survey '{survey_name}', attribute '{attribute_name}'...")
        result = builder.import_segy(segy_path, survey_name, attribute_name,
                                     byte_pos)
        print(f"  Imported {result.get('num_samples', '?')} samples/trace, "
              f"time range {result.get('start_time', '?')}-{result.get('end_time', '?')} ms")
    else:
        print("\n  No SEG-Y files found — skipping seismic import")

    # --- Import wells ---
    if las_files:
        print(f"\n--- Well Import ---")
        print(f"  Building well headers from {len(las_files)} LAS files...")

        well_heads_df = build_well_heads_from_las(str(wells_dir))
        if not well_heads_df.empty:
            print(f"  {len(well_heads_df)} wells with header info")

        imported = builder.import_wells_from_las(
            str(wells_dir),
            well_heads_df=well_heads_df if not well_heads_df.empty else None
        )
        print(f"  Imported logs for {len(imported)} wells: {imported}")
    else:
        print("\n  No LAS files found — skipping well import")

    # --- Add default wavelet ---
    print(f"\n--- Wavelets ---")
    ricker_25 = ricker_wavelet(25.0, 128, 0.002)
    builder.add_wavelet("Ricker_25Hz", ricker_25, {
        'type': 'Ricker',
        'frequency': 25.0,
        'sample_rate': 0.002,
        'length': 128,
    })
    print("  Added Ricker 25 Hz wavelet")

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
        print(f"  Wells:   {wells[:10]}{'...' if len(wells) > 10 else ''}")
        print(f"\n  PASS: Project is valid and loadable")
    except Exception as e:
        print(f"  FAIL: Could not load project: {e}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Project ready at: {PROJECT_FOLDER}")
    print(f"Open in SeisTrans: File > Open Project > {str_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
