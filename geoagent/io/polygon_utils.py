"""
Polygon utilities: lat/lon→X,Y affine transform, polygon storage.

Uses well_heads (Latitude_dd, Longitude_dd, Surface X, Surface Y) as control
points for a least-squares affine fit.  Stores polygons in polygons.pkl inside
the project directory for reuse across jobs.
"""
import os
import pickle
import numpy as np


def fit_latlon_to_xy_transform(well_heads):
    """
    Fit an affine transform (lat,lon) → (x,y) from well_heads.

    Uses np.linalg.lstsq on all wells with valid Latitude_dd/Longitude_dd and
    Surface X/Surface Y.

    Returns:
        (A, b) where x_xy = A @ [lat, lon] + b
        A is (2,2), b is (2,).
    """
    import pandas as pd

    df = well_heads.copy()

    # Coerce coordinate columns to numeric (handles string "nan", string numbers)
    for col in ['Surface X', 'Surface Y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Resolve and coerce latitude
    lat_col = _resolve_col(df, ['Latitude_dd', 'LAT'])
    if lat_col is not None:
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        if df[lat_col].notna().sum() == 0:
            lat_col = None  # all NaN after coercion

    # Fallback: extract lat_dd from Longitude column (Petrel format: "lat_dd DMS_string")
    if lat_col is None and 'Longitude' in df.columns:
        df['_lat_dd_parsed'] = df['Longitude'].apply(_extract_leading_float)
        if df['_lat_dd_parsed'].notna().sum() >= 3:
            lat_col = '_lat_dd_parsed'

    # Fallback: parse DMS from Latitude column
    if lat_col is None and 'Latitude' in df.columns:
        df['_lat_dd_parsed2'] = df['Latitude'].apply(_parse_dms)
        if df['_lat_dd_parsed2'].notna().sum() >= 3:
            lat_col = '_lat_dd_parsed2'

    # Resolve and coerce longitude
    lon_col = _resolve_col(df, ['Longitude_dd', 'LON', 'LONG'])
    if lon_col is not None:
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        if df[lon_col].notna().sum() == 0:
            lon_col = None

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"well_heads missing usable lat/lon data. "
            f"Columns: {list(well_heads.columns)}")

    df = df.dropna(subset=[lat_col, lon_col, 'Surface X', 'Surface Y'])
    if len(df) < 3:
        raise ValueError(f"Need >= 3 control points, got {len(df)}")

    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)
    sx = df['Surface X'].values.astype(float)
    sy = df['Surface Y'].values.astype(float)

    # Solve: [x] = A @ [lat, lon]^T + b  in least-squares sense
    # Rewrite as: [x] = [lat, lon, 1] @ [a11, a21; a12, a22; b1, b2]
    M = np.column_stack([lat, lon, np.ones(len(lat))])  # (N, 3)
    targets = np.column_stack([sx, sy])                  # (N, 2)

    result, residuals, rank, sv = np.linalg.lstsq(M, targets, rcond=None)
    # result is (3, 2): rows = [lat_coeff, lon_coeff, intercept]
    A = result[:2, :].T   # (2, 2)
    b = result[2, :]      # (2,)

    return A, b


def convert_polygon_latlon_to_xy(corners_latlon, well_heads):
    """
    Convert polygon corners from (lat, lon) to (x, y) using well_heads affine.

    Args:
        corners_latlon: list of (lat, lon) tuples in decimal degrees
        well_heads: DataFrame with lat/lon and Surface X/Y

    Returns:
        list of (x, y) tuples
    """
    A, b = fit_latlon_to_xy_transform(well_heads)

    result = []
    for lat, lon in corners_latlon:
        xy = A @ np.array([lat, lon]) + b
        result.append((float(xy[0]), float(xy[1])))
    return result


def save_polygon(polygon_xy, project_dir, name='block_boundary'):
    """
    Save a named polygon to polygons.pkl in the project directory.

    Args:
        polygon_xy: list of (x, y) tuples
        project_dir: path to project directory
        name: polygon identifier string
    """
    pkl_path = os.path.join(project_dir, 'polygons.pkl')
    polygons = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            polygons = pickle.load(f)

    polygons[name] = polygon_xy
    with open(pkl_path, 'wb') as f:
        pickle.dump(polygons, f)
    print(f"  Saved polygon '{name}' ({len(polygon_xy)} vertices) to {pkl_path}")


def load_polygon(project_dir, name='block_boundary'):
    """
    Load a named polygon from polygons.pkl.

    Returns:
        list of (x, y) tuples, or None if not found.
    """
    pkl_path = os.path.join(project_dir, 'polygons.pkl')
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        polygons = pickle.load(f)
    return polygons.get(name)


def _resolve_col(df, candidates):
    """Find first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_leading_float(val):
    """Extract leading float from a string like '22.98 "72 45\'20"'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    parts = s.split()
    if parts:
        try:
            return float(parts[0])
        except (ValueError, TypeError):
            pass
    return np.nan


def _parse_dms(val):
    """Parse DMS string like '"22 58\'51.7243" N"' to decimal degrees."""
    import re
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().strip('"').strip()
    match = re.match(r'(\d+)\s+(\d+)[\'′]\s*([\d.]+)["\″]?\s*([NSEW])?', s)
    if match:
        deg = float(match.group(1))
        minutes = float(match.group(2))
        sec = float(match.group(3))
        dd = deg + minutes / 60 + sec / 3600
        direction = match.group(4)
        if direction in ('S', 'W'):
            dd = -dd
        return dd
    return np.nan
