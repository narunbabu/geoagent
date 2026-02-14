"""
Horizon interpolation utilities.

Provides functions for extracting horizon values at arbitrary (x, y) locations
using scipy's RegularGridInterpolator — a Qt-free replacement for the
coordinate_cache + interpolation_manager combo in SeisTrans.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_horizon_at_points(horizon_data, x_points, y_points, method='linear'):
    """
    Interpolate horizon Z values at arbitrary (x, y) locations.

    Args:
        horizon_data: dict with keys 'X' (1D or 2D), 'Y' (1D or 2D), 'Z' (2D)
        x_points: 1D array or scalar of X coordinates to query
        y_points: 1D array or scalar of Y coordinates to query
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        1D array of interpolated Z values (NaN where extrapolated)
    """
    x_vals = np.asarray(horizon_data['X'])
    y_vals = np.asarray(horizon_data['Y'])
    z_vals = np.asarray(horizon_data['Z'])

    # Extract 1D axes from meshgrid if needed
    if x_vals.ndim == 2:
        x_axis = x_vals[0, :]
    else:
        x_axis = x_vals

    if y_vals.ndim == 2:
        y_axis = y_vals[:, 0]
    else:
        y_axis = y_vals

    # Ensure axes are sorted (RegularGridInterpolator requires it)
    x_sorted = np.sort(x_axis)
    y_sorted = np.sort(y_axis)

    # Flip Z if axes were reversed
    if not np.array_equal(x_axis, x_sorted):
        z_vals = z_vals[:, ::-1]
    if not np.array_equal(y_axis, y_sorted):
        z_vals = z_vals[::-1, :]

    interpolator = RegularGridInterpolator(
        (y_sorted, x_sorted), z_vals,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    x_pts = np.atleast_1d(np.asarray(x_points, dtype=float))
    y_pts = np.atleast_1d(np.asarray(y_points, dtype=float))

    points = np.column_stack([y_pts, x_pts])
    result = interpolator(points)

    return result


def interpolate_horizon_at_wells(horizon_data, well_coords, method='linear'):
    """
    Interpolate horizon Z values at well surface coordinates.

    Args:
        horizon_data: dict with 'X', 'Y', 'Z' arrays
        well_coords: list of (x, y) tuples or dict {well_name: (x, y)}
        method: 'linear' or 'nearest'

    Returns:
        If well_coords is a list: 1D array of Z values
        If well_coords is a dict: dict {well_name: z_value}
    """
    if isinstance(well_coords, dict):
        names = list(well_coords.keys())
        coords = list(well_coords.values())
        x_pts = [c[0] for c in coords]
        y_pts = [c[1] for c in coords]
        z_vals = interpolate_horizon_at_points(horizon_data, x_pts, y_pts, method)
        return dict(zip(names, z_vals))
    else:
        x_pts = [c[0] for c in well_coords]
        y_pts = [c[1] for c in well_coords]
        return interpolate_horizon_at_points(horizon_data, x_pts, y_pts, method)


def extract_horizon_along_line(horizon_data, x_line, y_line, method='linear'):
    """
    Extract horizon values along an arbitrary line (e.g., a seismic section line).

    Args:
        horizon_data: dict with 'X', 'Y', 'Z' arrays
        x_line: 1D array of X coordinates along the line
        y_line: 1D array of Y coordinates along the line
        method: 'linear' or 'nearest'

    Returns:
        1D array of Z values along the line
    """
    return interpolate_horizon_at_points(horizon_data, x_line, y_line, method)
