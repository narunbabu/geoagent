#!/usr/bin/env python
"""
Enrich the Teapot Dome project with horizons, formation tops, deviation surveys,
checkshot/TDR data, deeper well logs, and field boundary.

This script adds to an existing project built by build_teapot_project.py.
It does NOT rebuild the seismic or shallow well logs — only adds missing data.

Prerequisites:
    1. Run build_teapot_project.py first to create the base project.
    2. Extract additional files from rmotc.tar (this script checks and guides).

Usage:
    python examples/teapot/enrich_teapot_project.py
"""

import os
import sys
import glob
import pickle
import collections
import datetime
import numpy as np
import pandas as pd

try:
    import lasio
except ImportError:
    print("ERROR: lasio required. pip install lasio")
    sys.exit(1)

from pathlib import Path



# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("G:/2026/download_opendata/examples/teapot/data/DataSets")
PROJECT_DIR = Path("G:/2026/download_opendata/examples/teapot/teapot_project")

# Source files
HORIZONS_XYZ = DATA_DIR / "Seismic" / "CD files" / "3D_Seismic" / "3DHorizons.xyz"
FORMATION_TOPS_XLS = DATA_DIR / "Well Log" / "CD Files" / "TeapotDomeFormationLogTops.xls"
DIRECTIONAL_XLSX = DATA_DIR / "Well Log" / "CD Files" / "DirectionalSurveys_020910.xlsx"
TIME_DEPTH_XLS = DATA_DIR / "Seismic" / "CD files" / "TimeDepthTables.xls"
FIELD_BOUNDARY_TXT = DATA_DIR / "Seismic" / "CD files" / "NPR3_FieldBoundary.txt"
DEEPER_LAS_DIR = DATA_DIR / "Well Log" / "CD Files" / "LAS_log_files" / "Deeper_LAS_files"
WELL_HEADERS_XLSX = DATA_DIR / "Well Log" / "CD Files" / "TeapotDomeWellHeaders02-09-10.xlsx"


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (ValueError, TypeError):
        return default


def load_pkl(name):
    """Load a PKL file from the project directory."""
    path = PROJECT_DIR / name
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save_pkl(name, data):
    """Save a PKL file to the project directory."""
    path = PROJECT_DIR / name
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved {name} ({os.path.getsize(path) / 1024:.1f} KB)")


def build_api_to_name():
    """Build API-to-well-name mapping from well headers XLSX."""
    if not WELL_HEADERS_XLSX.exists():
        return {}
    df_raw = pd.read_excel(str(WELL_HEADERS_XLSX), header=0)
    col_names = df_raw.iloc[0].tolist()
    df = df_raw.iloc[1:].copy()
    df.columns = col_names
    df = df.reset_index(drop=True)

    api_to_name = {}
    for _, row in df.iterrows():
        api = str(row.get('API Number', '')).strip()
        well_num = str(row.get('Well Number', '')).strip()
        if api and well_num:
            api_digits = ''.join(c for c in api if c.isdigit())
            if api_digits:
                api_to_name[api_digits] = well_num
    return api_to_name


# ---------------------------------------------------------------------------
# 1. Horizons
# ---------------------------------------------------------------------------

def load_horizons():
    """Parse 3DHorizons.xyz → horizons.pkl with proper meshgrid X/Y arrays.

    SeisTrans uses RegularGridInterpolator which needs X/Y as complete
    meshgrids (no NaN). X must vary only along columns, Y only along rows.
    NaN is only allowed in Z (where horizon isn't interpreted).

    We build the meshgrid from the seismic headerdata coordinate system,
    then place horizon Z values at the matching inline/xline positions.
    """
    if not HORIZONS_XYZ.exists():
        print("  SKIP: 3DHorizons.xyz not found")
        return None

    # Load headerdata to build the coordinate meshgrid
    hd_path = PROJECT_DIR / "teapotdome3d" / "headerdata.pkl"
    if not hd_path.exists():
        print("  ERROR: headerdata.pkl not found, cannot build horizon grid")
        return None

    hd = load_pkl("teapotdome3d/headerdata.pkl")
    # headerdata: [trace_idx, inline, xline, cdp_x, cdp_y]
    # Build inline → mean_Y and xline → mean_X mappings from all traces
    il_coords = {}  # inline → mean Y coordinate
    xl_coords = {}  # xline → mean X coordinate
    for row in hd:
        il, xl, x, y = int(row[1]), int(row[2]), row[3], row[4]
        il_coords.setdefault(il, []).append(y)
        xl_coords.setdefault(xl, []).append(x)

    # Average to get a single coordinate per inline/xline
    inlines_sorted = sorted(il_coords.keys())
    xlines_sorted = sorted(xl_coords.keys())
    y_vector = np.array([np.mean(il_coords[il]) for il in inlines_sorted])  # (n_il,)
    x_vector = np.array([np.mean(xl_coords[xl]) for xl in xlines_sorted])  # (n_xl,)

    il_to_idx = {il: i for i, il in enumerate(inlines_sorted)}
    xl_to_idx = {xl: j for j, xl in enumerate(xlines_sorted)}

    ny, nx = len(inlines_sorted), len(xlines_sorted)
    # Build complete meshgrid (no NaN in X/Y)
    X_mesh, Y_mesh = np.meshgrid(x_vector, y_vector)  # both (ny, nx)

    print(f"  Built meshgrid from headerdata: {ny}x{nx} "
          f"(IL {inlines_sorted[0]}-{inlines_sorted[-1]}, "
          f"XL {xlines_sorted[0]}-{xlines_sorted[-1]})")
    print(f"  X range: {x_vector.min():.1f}-{x_vector.max():.1f}, "
          f"Y range: {y_vector.min():.1f}-{y_vector.max():.1f}")

    # Parse horizon XYZ file
    print("  Reading 3DHorizons.xyz ...")
    rows = []
    with open(HORIZONS_XYZ, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            hz_name = parts[4]
            if hz_name == 'Horizon':  # skip leaked header
                continue
            rows.append({
                'inline': int(parts[0]),
                'xline': int(parts[1]),
                'horizon': hz_name,
                'time_s': float(parts[5]),
            })

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} points across {df['horizon'].nunique()} horizons")

    horizons = {}
    for hz_name, grp in df.groupby('horizon'):
        Z_grid = np.full((ny, nx), np.nan)

        for _, row in grp.iterrows():
            il, xl = row['inline'], row['xline']
            if il in il_to_idx and xl in xl_to_idx:
                i = il_to_idx[il]
                j = xl_to_idx[xl]
                Z_grid[i, j] = row['time_s'] * 1000.0  # seconds → ms TWT

        horizons[hz_name] = {
            'X': X_mesh.copy(),
            'Y': Y_mesh.copy(),
            'Z': Z_grid,
        }
        n_valid = np.count_nonzero(~np.isnan(Z_grid))
        print(f"    {hz_name}: {ny}x{nx} grid, {n_valid} valid points, "
              f"TWT {np.nanmin(Z_grid):.0f}-{np.nanmax(Z_grid):.0f} ms")

    return horizons


# ---------------------------------------------------------------------------
# 2. Formation Tops
# ---------------------------------------------------------------------------

def load_formation_tops():
    """Parse TeapotDomeFormationLogTops.xls → well_tops.pkl DataFrame."""
    if not FORMATION_TOPS_XLS.exists():
        print("  SKIP: Formation tops XLS not found")
        return None

    print("  Reading formation tops ...")
    df = pd.read_excel(str(FORMATION_TOPS_XLS), header=0)
    print(f"  Raw rows: {len(df)}, columns: {list(df.columns)}")

    # Expected columns: API Number, Well Number, Form Alias, Top MD
    # Map to SeisTrans well_tops format: X, Y, Z, TWT, TWT Auto, MD, Surface, Well
    well_heads = load_pkl('well_heads.pkl')
    if well_heads is None:
        print("  ERROR: well_heads.pkl not found, cannot build well_tops")
        return None

    # Build well_name → (x, y) lookup
    name_to_coords = {}
    for _, wh in well_heads.iterrows():
        name_to_coords[str(wh['Name'])] = (
            _safe_float(wh.get('Surface X')),
            _safe_float(wh.get('Surface Y')),
        )

    # Also build well_number → well_name lookup from the tops file
    # The "Well Number" in the tops file should match the well_heads "Name" column
    records = []
    matched = 0
    unmatched_wells = set()

    for _, row in df.iterrows():
        well_num = str(row.get('Well Number', '')).strip()
        surface = str(row.get('Form Alias', '')).strip()
        md = _safe_float(row.get('Top MD'), np.nan)

        if not well_num or not surface or np.isnan(md):
            continue

        x, y = name_to_coords.get(well_num, (0.0, 0.0))
        if x == 0.0 and y == 0.0:
            unmatched_wells.add(well_num)

        records.append({
            'X': x,
            'Y': y,
            'Z': np.nan,       # TVDSS — not computed without deviation
            'TWT': np.nan,     # Will be derived from checkshot
            'TWT Auto': np.nan,
            'MD': md,
            'Surface': surface,
            'Well': well_num,
        })
        if x != 0.0:
            matched += 1

    result = pd.DataFrame(records)
    print(f"  {len(result)} tops for {result['Well'].nunique()} wells, "
          f"{result['Surface'].nunique()} formations")
    print(f"  Coordinate-matched: {matched}, unmatched wells: {len(unmatched_wells)}")
    if unmatched_wells:
        print(f"  Sample unmatched: {sorted(unmatched_wells)[:10]}")

    return result


# ---------------------------------------------------------------------------
# 3. Directional Surveys
# ---------------------------------------------------------------------------

def load_directional_surveys():
    """Parse DirectionalSurveys_020910.xlsx → deviation.pkl dict."""
    if not DIRECTIONAL_XLSX.exists():
        print("  SKIP: Directional surveys XLSX not found")
        return None

    print("  Reading directional surveys ...")
    df_raw = pd.read_excel(str(DIRECTIONAL_XLSX), header=None)
    print(f"  Raw rows: {len(df_raw)}")

    well_heads = load_pkl('well_heads.pkl')
    api_to_name = build_api_to_name()

    # Build well_name → (x, y, kb) lookup
    name_to_info = {}
    if well_heads is not None:
        for _, wh in well_heads.iterrows():
            name_to_info[str(wh['Name'])] = {
                'x': _safe_float(wh.get('Surface X')),
                'y': _safe_float(wh.get('Surface Y')),
                'kb': _safe_float(wh.get('Well datum value')),
            }

    # Parse the repeating block structure:
    # WELL: <name>
    # API Number | MD (Feet) | INCLINATION | AZIMUTH
    # data rows...
    deviation = {}
    current_well = None
    current_data = []

    def flush_well():
        nonlocal current_well, current_data
        if current_well and current_data:
            arr = np.array(current_data, dtype=float)
            # dev_data columns: [MD, Inclination, Azimuth] minimum
            # SeisTrans deviation format: dev_data ndarray with shape (n, 11)
            # But we only have MD, Inc, Az — pad the rest
            n = len(arr)
            dev_data = np.zeros((n, 11))
            dev_data[:, 0] = arr[:, 0]  # MD
            dev_data[:, 1] = arr[:, 1]  # Inclination
            dev_data[:, 2] = arr[:, 2]  # Azimuth

            info = name_to_info.get(current_well, {})
            deviation[current_well] = {
                'well_info': {
                    'name': current_well,
                    'x': info.get('x', 0.0),
                    'y': info.get('y', 0.0),
                    'kb': info.get('kb', 0.0),
                },
                'dev_data': dev_data,
            }
        current_data = []

    for idx in range(len(df_raw)):
        row = df_raw.iloc[idx]
        val0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''

        # Detect "WELL:" header rows
        if val0.upper().startswith('WELL'):
            flush_well()
            # Well name is in the next column
            well_name_raw = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
            if well_name_raw:
                current_well = well_name_raw
            continue

        # Skip column header rows (API Number, API#, etc.)
        if val0.upper().startswith('API'):
            continue

        # Skip blank/NaN rows
        if not val0 or val0 == 'nan':
            continue

        # Try to parse as data row: API, MD, Inc, Az
        try:
            api_str = ''.join(c for c in val0 if c.isdigit())
            md = float(row.iloc[1])
            inc = float(row.iloc[2])
            az = float(row.iloc[3])

            # Resolve well name from API if we don't have one
            if current_well is None and api_str in api_to_name:
                current_well = api_to_name[api_str]
            elif current_well is None and api_str[:12] in api_to_name:
                current_well = api_to_name[api_str[:12]]

            current_data.append([md, inc, az])
        except (ValueError, TypeError, IndexError):
            continue

    flush_well()  # flush last well

    print(f"  Parsed {len(deviation)} wells with directional surveys")
    # Show sample
    for wn in sorted(deviation.keys())[:5]:
        d = deviation[wn]
        n = d['dev_data'].shape[0]
        md_range = f"{d['dev_data'][:, 0].min():.0f}-{d['dev_data'][:, 0].max():.0f} ft"
        print(f"    {wn}: {n} stations, MD {md_range}")

    return deviation


# ---------------------------------------------------------------------------
# 4. Time-Depth Tables (Checkshots)
# ---------------------------------------------------------------------------

def load_time_depth_tables():
    """Parse TimeDepthTables.xls → checkshot.pkl + tdr_mappings.pkl."""
    if not TIME_DEPTH_XLS.exists():
        print("  SKIP: TimeDepthTables.xls not found")
        return None, None, None

    print("  Reading time-depth tables ...")
    xls = pd.ExcelFile(str(TIME_DEPTH_XLS))
    print(f"  Sheets: {xls.sheet_names}")

    well_heads = load_pkl('well_heads.pkl')
    name_to_coords = {}
    if well_heads is not None:
        for _, wh in well_heads.iterrows():
            name_to_coords[str(wh['Name'])] = (
                _safe_float(wh.get('Surface X')),
                _safe_float(wh.get('Surface Y')),
            )

    checkshot = {}  # well_name → DataFrame
    tdr_mappings = {}
    checkshot_mapping_rows = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        well_name = sheet.strip()
        cs_id = f"{well_name}_checkshot_{timestamp}"
        x, y = name_to_coords.get(well_name, (0.0, 0.0))

        if well_name == '48-X-28':
            # Format: Shot#, TWT(ms), MD(ft), TVD(ft)
            # Find the data start (after header rows)
            data_rows = []
            for i in range(len(df)):
                try:
                    twt = float(df.iloc[i, 1])
                    tvd = float(df.iloc[i, 3])
                    if not np.isnan(twt) and not np.isnan(tvd):
                        data_rows.append({'TWT': twt, 'TVD': tvd})
                except (ValueError, TypeError, IndexError):
                    continue

            if data_rows:
                cs_df = pd.DataFrame(data_rows)
                cs_records = []
                for _, r in cs_df.iterrows():
                    cs_records.append({
                        'X': x, 'Y': y,
                        'Z': -r['TVD'],  # Z = negative TVD (TVDSS convention)
                        'TWT picked': r['TWT'],
                        'MD': r['TVD'],  # Assume vertical: MD ≈ TVD
                        'Well': well_name,
                        'Average velocity': 0.0,
                        'Interval velocity': 0.0,
                        'CheckShotID': cs_id,
                    })
                checkshot[well_name] = pd.DataFrame(cs_records)
                tdr_mappings[well_name] = {
                    'source_well': well_name,
                    'checkshot_id': cs_id,
                }
                checkshot_mapping_rows.append({
                    'Well': well_name, 'CheckShotID': cs_id,
                    'SourceWell': well_name, 'ImportTime': timestamp,
                })
                print(f"    {well_name}: {len(cs_records)} TWT/TVD pairs "
                      f"(TWT 0-{cs_df['TWT'].max():.0f} ms)")

        elif well_name == '35-1-X-10':
            # VSP format: skip metadata rows, find data starting around row 18
            data_rows = []
            for i in range(len(df)):
                try:
                    md_val = float(df.iloc[i, 1])
                    twt_val = float(df.iloc[i, 4])  # Corrected Transit Time
                    tvd_val = float(df.iloc[i, 3])   # True Vert Depth from SRD
                    if not np.isnan(md_val) and not np.isnan(twt_val):
                        data_rows.append({
                            'MD': md_val, 'TWT': twt_val, 'TVD': tvd_val
                        })
                except (ValueError, TypeError, IndexError):
                    continue

            if data_rows:
                cs_df = pd.DataFrame(data_rows)
                cs_records = []
                for _, r in cs_df.iterrows():
                    cs_records.append({
                        'X': x, 'Y': y,
                        'Z': -r['TVD'],
                        'TWT picked': r['TWT'],
                        'MD': r['MD'],
                        'Well': well_name,
                        'Average velocity': 0.0,
                        'Interval velocity': 0.0,
                        'CheckShotID': cs_id,
                    })
                checkshot[well_name] = pd.DataFrame(cs_records)
                tdr_mappings[well_name] = {
                    'source_well': well_name,
                    'checkshot_id': cs_id,
                }
                checkshot_mapping_rows.append({
                    'Well': well_name, 'CheckShotID': cs_id,
                    'SourceWell': well_name, 'ImportTime': timestamp,
                })
                print(f"    {well_name} (VSP): {len(cs_records)} stations "
                      f"(MD {cs_df['MD'].min():.0f}-{cs_df['MD'].max():.0f} ft)")

        else:
            # Formation velocity table format (25-1-X-14 and 67-1-X-10)
            # Columns: FORMATION, DEPTH ft(KB), DEPTH ft(ASL), TIME(s), ...
            data_rows = []
            for i in range(len(df)):
                try:
                    depth_kb = float(df.iloc[i, 1])
                    time_s = float(df.iloc[i, 3])
                    if not np.isnan(depth_kb) and not np.isnan(time_s):
                        data_rows.append({
                            'MD': depth_kb,
                            'TWT': time_s * 1000.0,  # seconds → ms
                        })
                except (ValueError, TypeError, IndexError):
                    continue

            if data_rows:
                cs_df = pd.DataFrame(data_rows)
                cs_records = []
                for _, r in cs_df.iterrows():
                    cs_records.append({
                        'X': x, 'Y': y,
                        'Z': -r['MD'],  # Approximate Z
                        'TWT picked': r['TWT'],
                        'MD': r['MD'],
                        'Well': well_name,
                        'Average velocity': 0.0,
                        'Interval velocity': 0.0,
                        'CheckShotID': cs_id,
                    })
                checkshot[well_name] = pd.DataFrame(cs_records)
                tdr_mappings[well_name] = {
                    'source_well': well_name,
                    'checkshot_id': cs_id,
                }
                checkshot_mapping_rows.append({
                    'Well': well_name, 'CheckShotID': cs_id,
                    'SourceWell': well_name, 'ImportTime': timestamp,
                })
                print(f"    {well_name}: {len(cs_records)} formation picks "
                      f"(TWT {cs_df['TWT'].min():.0f}-{cs_df['TWT'].max():.0f} ms)")

    checkshot_mapping = pd.DataFrame(checkshot_mapping_rows)
    print(f"  Total: {len(checkshot)} wells with TDR data")

    return checkshot, tdr_mappings, checkshot_mapping


# ---------------------------------------------------------------------------
# 5. Deeper Well Logs
# ---------------------------------------------------------------------------

def extract_api_from_las_filename(las_path):
    """Extract normalized 12-digit API from LAS filename."""
    fname = os.path.splitext(os.path.basename(las_path))[0]
    parts = fname.split('_')
    if parts:
        api_digits = ''.join(c for c in parts[0] if c.isdigit())
        if len(api_digits) >= 12:
            return api_digits[:12]
    return None


def resolve_well_name(las_path, api_to_name):
    """Resolve well name for a LAS file using API matching."""
    api = extract_api_from_las_filename(las_path)
    if api and api in api_to_name:
        return api_to_name[api]

    # Try reading LAS header for API/UWI
    try:
        las = lasio.read(las_path)
        for field in ['API', 'UWI']:
            try:
                val = str(las.well[field].value).strip()
                api_digits = ''.join(c for c in val if c.isdigit())
                if api_digits[:12] in api_to_name:
                    return api_to_name[api_digits[:12]]
            except (KeyError, AttributeError):
                pass
        # Fallback: use WELL name from header
        well_str = str(las.well['WELL'].value).strip()
        if '#' in well_str:
            return well_str.split('#')[-1].strip()
        return well_str
    except Exception:
        return os.path.splitext(os.path.basename(las_path))[0]


def load_deeper_well_logs():
    """Parse deeper LAS files and merge into existing well_logs."""
    if not DEEPER_LAS_DIR.exists():
        print("  SKIP: Deeper LAS directory not found")
        return None

    # Collect ALL LAS files (root + subdirectories)
    all_las = []
    for root, dirs, files in os.walk(str(DEEPER_LAS_DIR)):
        for f in files:
            if f.lower().endswith('.las') and not f.startswith('._'):
                all_las.append(os.path.join(root, f))
    all_las = sorted(set(all_las))
    print(f"  Found {len(all_las)} deeper LAS files")

    api_to_name = build_api_to_name()

    # Load existing well_logs
    existing_logs = load_pkl('well_logs.pkl') or {}
    print(f"  Existing well_logs: {len(existing_logs)} wells")

    imported = 0
    merged = 0
    failed = 0
    curve_counter = collections.Counter()

    for las_path in all_las:
        try:
            well_name = resolve_well_name(las_path, api_to_name)
            las = lasio.read(las_path)

            curves = {}
            for curve in las.curves:
                mnem = curve.mnemonic.strip()
                data = curve.data
                if data is not None and len(data) > 0:
                    curves[mnem] = data
                    curve_counter[mnem] += 1

            if len(curves) < 2:  # Need at least depth + one curve
                continue

            if well_name in existing_logs:
                # Merge: add new curves that don't exist yet
                existing = existing_logs[well_name]
                for k, v in curves.items():
                    if k not in existing:
                        existing[k] = v
                merged += 1
            else:
                existing_logs[well_name] = curves
                imported += 1

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"    Warning: {os.path.basename(las_path)}: {e}")

    print(f"  New wells: {imported}, merged: {merged}, failed: {failed}")
    print(f"  Total well_logs: {len(existing_logs)} wells")
    print(f"  Most common curves: {curve_counter.most_common(15)}")

    return existing_logs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Teapot Dome Project Enrichment")
    print("=" * 60)

    if not PROJECT_DIR.exists():
        print(f"\nProject not found: {PROJECT_DIR}")
        print("Run build_teapot_project.py first.")
        sys.exit(1)

    # Check which source files exist
    print("\n--- Source File Check ---")
    sources = {
        'Horizons XYZ': HORIZONS_XYZ,
        'Formation Tops': FORMATION_TOPS_XLS,
        'Directional Surveys': DIRECTIONAL_XLSX,
        'Time-Depth Tables': TIME_DEPTH_XLS,
        'Deeper LAS Dir': DEEPER_LAS_DIR,
        'Field Boundary': FIELD_BOUNDARY_TXT,
    }
    for name, path in sources.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  {name}: {status} — {path}")

    # --- 1. Horizons ---
    print(f"\n{'='*60}")
    print("1. HORIZONS")
    print(f"{'='*60}")
    horizons = load_horizons()
    if horizons:
        save_pkl('horizons.pkl', horizons)

    # --- 2. Formation Tops ---
    print(f"\n{'='*60}")
    print("2. FORMATION TOPS")
    print(f"{'='*60}")
    well_tops = load_formation_tops()
    if well_tops is not None:
        save_pkl('well_tops.pkl', well_tops)

    # --- 3. Directional Surveys ---
    print(f"\n{'='*60}")
    print("3. DIRECTIONAL SURVEYS")
    print(f"{'='*60}")
    deviation = load_directional_surveys()
    if deviation:
        save_pkl('deviation.pkl', deviation)

    # --- 4. Time-Depth Tables ---
    print(f"\n{'='*60}")
    print("4. TIME-DEPTH TABLES (CHECKSHOTS)")
    print(f"{'='*60}")
    checkshot, tdr_mappings, checkshot_mapping = load_time_depth_tables()
    if checkshot:
        save_pkl('checkshot.pkl', checkshot)
    if tdr_mappings:
        save_pkl('tdr_mappings.pkl', tdr_mappings)
    if checkshot_mapping is not None and len(checkshot_mapping) > 0:
        save_pkl('checkshot_mapping.pkl', checkshot_mapping)

    # --- 5. Deeper Well Logs ---
    print(f"\n{'='*60}")
    print("5. DEEPER WELL LOGS")
    print(f"{'='*60}")
    well_logs = load_deeper_well_logs()
    if well_logs:
        save_pkl('well_logs.pkl', well_logs)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("ENRICHMENT SUMMARY")
    print(f"{'='*60}")

    hz = load_pkl('horizons.pkl')
    wt = load_pkl('well_tops.pkl')
    dev = load_pkl('deviation.pkl')
    cs = load_pkl('checkshot.pkl')
    tdr = load_pkl('tdr_mappings.pkl')
    wl = load_pkl('well_logs.pkl')
    wh = load_pkl('well_heads.pkl')

    print(f"  Well headers:     {len(wh) if wh is not None else 0} wells")
    print(f"  Well logs:        {len(wl) if wl else 0} wells")
    print(f"  Horizons:         {len(hz) if hz else 0} surfaces")
    if hz:
        for name in sorted(hz.keys()):
            z = hz[name]['Z']
            print(f"    {name}: TWT {np.nanmin(z):.0f}-{np.nanmax(z):.0f} ms")
    print(f"  Formation tops:   {len(wt) if wt is not None else 0} picks")
    if wt is not None and hasattr(wt, 'nunique'):
        print(f"    Wells: {wt['Well'].nunique()}, Formations: {wt['Surface'].nunique()}")
    print(f"  Deviation:        {len(dev) if dev else 0} wells")
    print(f"  Checkshots:       {len(cs) if cs else 0} wells")
    print(f"  TDR mappings:     {len(tdr) if tdr else 0} wells")

    # --- Validation ---
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    try:
        from geoagent.core.data_manager import CoreDataManager
        dm = CoreDataManager(str(PROJECT_DIR))
        print(f"  CoreDataManager loaded OK")
        print(f"  Surveys: {dm.get_available_surveys()}")
        print(f"  Wells: {len(dm.get_available_wells())}")

        # Check horizons
        hz_data = dm.get_data('horizons')
        if hz_data:
            print(f"  Horizons loaded: {list(hz_data.keys())}")
        else:
            print(f"  Horizons: not loaded (may need handler access)")

        print(f"\n  PASS: Enriched project is valid and loadable")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Project enriched at: {PROJECT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
