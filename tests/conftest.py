"""
Shared test fixtures for GeoAgent test suite.

All fixtures use in-memory or tmpdir data — no external project required.
"""

import os
import pickle
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_well_heads():
    """DataFrame of well head data (5 wells)."""
    return pd.DataFrame({
        'Well Name': ['W-1', 'W-2', 'W-3', 'W-4', 'W-5'],
        'Surface X': [270100, 270200, 270300, 270400, 270500],
        'Surface Y': [2541100, 2541200, 2541300, 2541400, 2541500],
        'KB': [25.0, 28.0, 22.0, 30.0, 24.0],
    }).set_index('Well Name')


@pytest.fixture
def sample_well_logs():
    """Dict of well log data keyed by well name."""
    np.random.seed(42)
    n = 500
    depth = np.linspace(800, 1200, n)

    def make_logs():
        return {
            'DEPT': depth,
            'GR': np.random.uniform(20, 120, n),
            'DTC': np.random.uniform(60, 130, n),
            'RHOB': np.random.uniform(2.0, 2.7, n),
            'LLD': 10 ** np.random.uniform(-0.5, 2.5, n),
            'NPHI': np.random.uniform(0.05, 0.45, n),
        }

    return {
        'W-1': make_logs(),
        'W-2': make_logs(),
        'W-3': make_logs(),
    }


@pytest.fixture
def sample_well_tops():
    """Dict of formation tops keyed by well name."""
    return {
        'W-1': {
            'Top-A': {'MD': 900, 'TVD': 898, 'Z': 873, 'TWT Auto': 1450},
            'Top-B': {'MD': 1000, 'TVD': 998, 'Z': 973, 'TWT Auto': 1480},
            'Top-C': {'MD': 1100, 'TVD': 1098, 'Z': 1073, 'TWT Auto': 1510},
        },
        'W-2': {
            'Top-A': {'MD': 910, 'TVD': 908, 'Z': 880, 'TWT Auto': 1455},
            'Top-B': {'MD': 1010, 'TVD': 1008, 'Z': 980, 'TWT Auto': 1485},
        },
    }


@pytest.fixture
def sample_deviation():
    """Dict of deviation data keyed by well name."""
    n = 100
    md = np.linspace(0, 1200, n)
    return {
        'W-1': {
            'well_info': {'well_name': 'W-1', 'KB': 25.0},
            'dev_data': pd.DataFrame({
                'MD': md,
                'TVD': md * 0.998,
                'X': 270100 + np.cumsum(np.random.uniform(-0.5, 0.5, n)),
                'Y': 2541100 + np.cumsum(np.random.uniform(-0.5, 0.5, n)),
            })
        },
    }


@pytest.fixture
def sample_impedance():
    """Simple acoustic impedance array for synthetic testing."""
    np.random.seed(42)
    base = np.linspace(5000, 8000, 200)
    noise = np.random.normal(0, 200, 200)
    return base + noise


@pytest.fixture
def sample_horizon():
    """Horizon grid data dict."""
    x = np.arange(270000, 270600, 10.0)
    y = np.arange(2541000, 2541600, 10.0)
    X, Y = np.meshgrid(x, y)
    Z = 1500 + np.sin(X / 100) * 20 + np.cos(Y / 80) * 15
    return {'X': x, 'Y': y, 'Z': Z}


@pytest.fixture
def project_dir(tmp_path, sample_well_heads, sample_well_logs, sample_well_tops, sample_deviation):
    """
    Create a minimal project directory with pickle files.

    Returns the path to the temp project folder.
    """
    with open(tmp_path / 'well_heads.pkl', 'wb') as f:
        pickle.dump(sample_well_heads, f)

    with open(tmp_path / 'well_logs.pkl', 'wb') as f:
        pickle.dump(sample_well_logs, f)

    with open(tmp_path / 'well_tops.pkl', 'wb') as f:
        pickle.dump(sample_well_tops, f)

    with open(tmp_path / 'deviation.pkl', 'wb') as f:
        pickle.dump(sample_deviation, f)

    return str(tmp_path)
