"""
Deviation survey utilities — MD→TVDSS conversion, well path XY extraction.
Handles both plain DataFrame and dict {'well_info': ..., 'dev_data': DataFrame} formats.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def get_dev_dataframe(deviation, well_name):
    """
    Extract deviation DataFrame from either dict-of-dicts or dict-of-DataFrames.

    Args:
        deviation: dict keyed by well name. Values may be:
            - pd.DataFrame directly
            - dict with 'dev_data' key containing a DataFrame
        well_name: well name string

    Returns:
        pd.DataFrame or None
    """
    if well_name not in deviation:
        return None
    dev = deviation[well_name]
    if isinstance(dev, pd.DataFrame):
        return dev
    if isinstance(dev, dict) and 'dev_data' in dev:
        dd = dev['dev_data']
        if isinstance(dd, pd.DataFrame):
            return dd
    return None


def compute_tvdss(deviation, well_name, md_array, kb):
    """
    Convert MD array to TVDSS using deviation survey.

    TVDSS = TVD - KB  (positive values = depth below sea level).
    Falls back to None if deviation data is missing or KB is None.

    Args:
        deviation: deviation dict (see get_dev_dataframe)
        well_name: well name string
        md_array: numpy array of measured depths
        kb: Kelly Bushing elevation (meters above sea level)

    Returns:
        numpy array of TVDSS values, or None
    """
    if kb is None:
        return None

    dev_df = get_dev_dataframe(deviation, well_name)
    if dev_df is None:
        return None

    dev_md = dev_df['MD'].values if 'MD' in dev_df.columns else None
    dev_tvd = None
    for col in ['TVD', 'Z', 'TVDSS']:
        if col in dev_df.columns:
            dev_tvd = np.abs(dev_df[col].values)
            break
    if dev_md is None or dev_tvd is None:
        return None

    sort_idx = np.argsort(dev_md)
    dev_md = dev_md[sort_idx]
    dev_tvd = dev_tvd[sort_idx]

    try:
        tvd_interp = interp1d(dev_md, dev_tvd, kind='linear',
                              fill_value='extrapolate', bounds_error=False)
        tvd_at_md = tvd_interp(md_array)
        tvdss = tvd_at_md - kb
        return tvdss
    except Exception:
        return None


def get_well_path_xy(deviation, well_name):
    """
    Get well path X, Y coordinates from deviation survey for map plotting.

    Returns:
        (x_array, y_array) tuple, or None if well is essentially vertical
        or has no deviation data.
    """
    dev_df = get_dev_dataframe(deviation, well_name)
    if dev_df is None:
        return None
    if 'X' not in dev_df.columns or 'Y' not in dev_df.columns:
        return None
    x = dev_df['X'].values
    y = dev_df['Y'].values
    if len(x) < 2:
        return None
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    if x_range < 1.0 and y_range < 1.0:
        return None  # Essentially vertical
    return x, y
