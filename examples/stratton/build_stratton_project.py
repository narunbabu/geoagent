#!/usr/bin/env python
"""
Build a SeisTrans-compatible project from Stratton 3D field data.

This script reads the downloaded Stratton SEG-Y and well TXT files, then uses
GeoAgent's ProjectBuilder to create a project that can be opened in SeisTrans.

Prerequisites:
    1. Download the data (see download instructions in README).
    2. Ensure geoagent is installed: pip install -e .

Usage:
    python examples/stratton/build_stratton_project.py

Output:
    examples/stratton/output/stratton_project/   — project data folder
    examples/stratton/output/stratton_project.str — SeisTrans project pointer

Data source:
    Bureau of Economic Geology, UT Austin
    http://s3.amazonaws.com/open.source.geoscience/open_data/stratton/
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import segyio
except ImportError:
    print("ERROR: segyio is required. Install with: pip install segyio")
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
PROJECT_FOLDER = str(OUTPUT_DIR / "stratton_project")
PROJECT_NAME = "Stratton"

# Stratton 3D SEG-Y byte positions (non-standard locations)
# Inline numbers stored at byte 21 (CDP field)
# Crossline numbers stored at byte 9 (FieldRecord field)
# Coordinates at bytes 73/77 (SourceX/SourceY) in State Plane feet
STRATTON_BYTE_POSITIONS = {
    'CDP_X': 73,             # SourceX = State Plane X (feet)
    'CDP_Y': 77,             # SourceY = State Plane Y (feet)
    'Inline': 21,            # CDP number = inline
    'Crossline': 9,          # FieldRecord = crossline
    'Coord_Mult_Factor': 1.0,  # No scaling (scalar byte is 0)
}

# Grid geometry (from SEG-Y header analysis):
# Inlines 1-230, Crosslines 2-310, spacing 55 ft both directions
# Origin (inline=1, xline=2): X=2193745, Y=705550 (State Plane feet)
GRID_ORIGIN_X = 2193745.0
GRID_ORIGIN_Y = 705550.0
GRID_SPACING = 55.0  # feet


# ---------------------------------------------------------------------------
# Well Data Parsing (TXT format, not LAS)
# ---------------------------------------------------------------------------

def parse_table2(table2_path):
    """
    Parse TABLE2.TXT — well header data with coordinates, KB, and formation tops.

    Returns:
        DataFrame with well headers (SeisTrans-compatible columns).
    """
    print(f"  Parsing {os.path.basename(table2_path)}...")

    with open(table2_path, 'r') as f:
        lines = f.readlines()

    # First line is the header
    # Data lines follow (tab-separated)
    records = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 6:
            continue

        well_no = int(parts[0])
        inline = int(parts[1])
        crossline = int(parts[2])
        kb_elev = float(parts[3])  # KB elevation in feet
        local_x = float(parts[4])  # Local grid X (feet)
        local_y = float(parts[5])  # Local grid Y (feet)

        # Convert local grid coordinates to State Plane
        # From SEG-Y analysis: inline increases → X decreases, crossline increases → Y increases
        # X_state = GRID_ORIGIN_X - (inline - 1) * GRID_SPACING
        # Y_state = GRID_ORIGIN_Y + (crossline - 2) * GRID_SPACING
        state_x = GRID_ORIGIN_X - (inline - 1) * GRID_SPACING
        state_y = GRID_ORIGIN_Y + (crossline - 2) * GRID_SPACING

        well_name = f"Stratton-{well_no}"

        records.append({
            'Name': well_name,
            'UWI': f'STRATTON-{well_no:02d}',
            'Well symbol': 'OIL',
            'Surface X': state_x,
            'Surface Y': state_y,
            'Latitude': '',
            'Latitude_dd': 0.0,
            'Longitude': '',
            'Longitude_dd': 0.0,
            'Drilling structure': '',
            'Well datum name': 'KB',
            'Well datum value': kb_elev,
            'Well datum description': 'Kelly Bushing (feet)',
            'TD (MD)': 0.0,  # Will be filled from well logs
            'Cost': '',
            'Spud date': '',
            'Operator': 'Bureau of Economic Geology',
            'TWT auto': 0.0,
            'Bottom hole X': state_x,
            'Bottom hole Y': state_y,
        })
        print(f"    Well {well_name}: inline={inline}, xline={crossline}, "
              f"KB={kb_elev} ft, X={state_x:.0f}, Y={state_y:.0f}")

    df = pd.DataFrame(records)
    # SeisTrans format: integer index, Name as column (NOT as index)
    df = df.drop_duplicates(subset=['Name'], keep='first')
    df = df.reset_index(drop=True)
    return df


def parse_well_log_txt(txt_path, well_name):
    """
    Parse a Stratton well log TXT file into {mnemonic: ndarray} format.

    The TXT files have a header line with curve names, then space-separated data.
    Depth is in feet. Null values are -99999.

    Returns:
        dict of {curve_name: 1D numpy array}
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # First line has curve names
    header_line = lines[0].strip()
    curve_names = header_line.split()

    # Skip blank/descriptor lines, find first data line
    data_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Check if line starts with a number (data line)
        parts = stripped.split()
        try:
            float(parts[0])
            data_lines.append(parts)
        except ValueError:
            continue  # Skip descriptor lines

    if not data_lines:
        print(f"    Warning: No data in {txt_path}")
        return {}

    # Parse into numpy arrays
    n_rows = len(data_lines)
    n_cols = len(curve_names)
    data = np.full((n_rows, n_cols), np.nan)

    for i, parts in enumerate(data_lines):
        for j in range(min(len(parts), n_cols)):
            try:
                val = float(parts[j])
                if val == -99999.0 or val == -99999:
                    data[i, j] = np.nan
                else:
                    data[i, j] = val
            except ValueError:
                data[i, j] = np.nan

    curves = {}
    for j, name in enumerate(curve_names):
        curves[name] = data[:, j]

    return curves


def parse_vsp_table3(table3_path):
    """
    Parse TABLE3.TXT — time-depth calibration from VSP in well 9.

    Returns:
        dict suitable for checkshot data: {twt_ms: array, depth_ft: array}
    """
    print(f"  Parsing {os.path.basename(table3_path)} (VSP/TDR for Well 9)...")

    with open(table3_path, 'r') as f:
        lines = f.readlines()

    twt_values = []
    depth_values = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('Table') or stripped.startswith('('):
            continue
        parts = stripped.split('\t')
        # Skip header lines
        if 'Two-way' in stripped or 'time' in stripped or 'ms' in stripped:
            continue

        try:
            base_twt = float(parts[0])
        except (ValueError, IndexError):
            continue

        # Each row has: base_twt, depth_at+0, depth_at+1, ..., depth_at+9
        for offset in range(min(10, len(parts) - 1)):
            try:
                depth = float(parts[1 + offset])
                twt = base_twt + offset
                twt_values.append(twt)
                depth_values.append(depth)
            except (ValueError, IndexError):
                continue

    return {
        'twt_ms': np.array(twt_values),
        'depth_ft': np.array(depth_values),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stratton 3D Project Builder")
    print("=" * 60)

    # Check for data
    seismic_dir = DATA_DIR / "seismic"
    wells_dir = DATA_DIR / "wells"

    segy_path = seismic_dir / "Stratton3D_32bit.sgy"
    table2_path = wells_dir / "TABLE2.TXT"
    table3_path = wells_dir / "TABLE3.TXT"

    if not segy_path.exists():
        print(f"\nSEG-Y file not found: {segy_path}")
        print("Download from: http://s3.amazonaws.com/open.source.geoscience/"
              "open_data/stratton/segy/processed/Stratton3D_32bit.sgy")
        sys.exit(1)

    if not table2_path.exists():
        print(f"\nWell data not found: {table2_path}")
        print("Download from: http://s3.amazonaws.com/open.source.geoscience/"
              "open_data/stratton/Stratton_wells.zip")
        sys.exit(1)

    # Create builder
    print(f"\nCreating project: {PROJECT_FOLDER}")
    builder = ProjectBuilder(PROJECT_FOLDER, PROJECT_NAME)

    # --- Inspect and import seismic ---
    print(f"\n--- Seismic Import ---")
    print(f"  File: {segy_path}")

    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        print(f"  Traces:       {f.tracecount}")
        print(f"  Samples:      {len(f.samples)}")
        print(f"  Sample range: {f.samples[0]} - {f.samples[-1]} ms")
        if len(f.samples) > 1:
            print(f"  Sample rate:  {f.samples[1] - f.samples[0]} ms")

        h = f.header[0]
        print(f"  First trace:")
        print(f"    Inline (byte 21):    {h[21]}")
        print(f"    Crossline (byte 9):  {h[9]}")
        print(f"    SourceX (byte 73):   {h[73]}")
        print(f"    SourceY (byte 77):   {h[77]}")

    byte_pos = dict(STRATTON_BYTE_POSITIONS)
    print(f"\n  Byte positions: {byte_pos}")

    result = builder.import_segy(
        str(segy_path),
        survey_name="Stratton3D",
        attribute_name="Amplitude",
        byte_positions=byte_pos,
        copy_segy=True,
    )
    print(f"  Imported: {result.get('num_traces', '?')} traces, "
          f"{result.get('num_samples', '?')} samples/trace")

    # --- Import well headers ---
    print(f"\n--- Well Headers ---")
    well_heads_df = parse_table2(str(table2_path))
    print(f"  {len(well_heads_df)} wells parsed")

    # --- Import well logs ---
    print(f"\n--- Well Logs ---")
    well_log_files = sorted(wells_dir.glob("WELL_*.TXT"))
    print(f"  Found {len(well_log_files)} well log files")

    for well_file in well_log_files:
        # Extract well number from filename (WELL_1.TXT → 1)
        well_num = int(well_file.stem.split('_')[1])
        well_name = f"Stratton-{well_num}"

        curves = parse_well_log_txt(str(well_file), well_name)
        if curves:
            # Store as well_logs in the handler
            if 'well_logs' not in builder.well_log_handler.loaded_data:
                builder.well_log_handler.loaded_data['well_logs'] = {}
            builder.well_log_handler.loaded_data['well_logs'][well_name] = curves

            # Update TD from DEPTH curve
            if 'DEPTH' in curves:
                valid_depth = curves['DEPTH'][~np.isnan(curves['DEPTH'])]
                if len(valid_depth) > 0:
                    td = float(valid_depth.max())
                    if well_name in well_heads_df.index:
                        well_heads_df.loc[well_name, 'TD (MD)'] = td

            n_curves = len([k for k in curves if k != 'DEPTH'])
            n_samples = len(curves.get('DEPTH', []))
            print(f"    {well_name}: {n_curves} curves, {n_samples} samples, "
                  f"depth {curves.get('DEPTH', [0])[0]:.0f}-"
                  f"{curves.get('DEPTH', [0])[-1]:.0f} ft")

    # Set well heads (after TD update)
    builder.set_well_heads(well_heads_df)

    # --- Parse VSP/checkshot data ---
    if table3_path.exists():
        print(f"\n--- VSP/Time-Depth Relationship ---")
        tdr = parse_vsp_table3(str(table3_path))
        print(f"  {len(tdr['twt_ms'])} TDR points, "
              f"TWT {tdr['twt_ms'][0]:.0f}-{tdr['twt_ms'][-1]:.0f} ms, "
              f"Depth {tdr['depth_ft'][0]:.0f}-{tdr['depth_ft'][-1]:.0f} ft")

        # Store as checkshot for Well 9
        checkshot_data = {
            'Stratton-9': {
                'twt': tdr['twt_ms'],
                'depth': tdr['depth_ft'],
                'units': 'feet',
            }
        }
        builder.set_checkshots(checkshot_data)

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
        print(f"  Wells:   {wells[:10]}{'...' if len(wells) > 10 else ''}")

        # Check wavelets
        wavelets = dm.seismic_handler.loaded_data.get('wavelets', [])
        print(f"  Wavelets: {[w['name'] for w in wavelets]}")

        # Check well logs
        well_logs = dm.well_log_handler.loaded_data.get('well_logs', {})
        print(f"  Well logs: {len(well_logs)} wells")

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
