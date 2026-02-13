"""
SeisTrans project pickle loader.
Loads well_heads, well_tops, well_logs, checkshot, deviation from .pkl files.
"""
import os
import pickle
from typing import Dict, List, Any, Optional


# Default pickle names matching SeisTrans project structure
DEFAULT_PICKLE_NAMES = ['well_heads', 'well_tops', 'well_logs', 'checkshot', 'deviation']


def load_pickles(project_dir, pickle_names=None):
    """
    Load pickle files from a SeisTrans project directory.

    Args:
        project_dir: path to project directory containing .pkl files
        pickle_names: list of pickle basenames to load (without .pkl extension).
            Defaults to DEFAULT_PICKLE_NAMES.

    Returns:
        dict mapping pickle name → loaded Python object

    Raises:
        FileNotFoundError: if a required pickle file is missing
    """
    if pickle_names is None:
        pickle_names = DEFAULT_PICKLE_NAMES

    data = {}
    for name in pickle_names:
        path = os.path.join(project_dir, f'{name}.pkl')
        with open(path, 'rb') as f:
            data[name] = pickle.load(f)
    return data
