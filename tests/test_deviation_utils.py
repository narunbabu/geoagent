"""Tests for deviation utilities."""
import numpy as np
import pandas as pd
import pytest
from geoagent.well.deviation_utils import compute_tvdss, get_well_path_xy


class TestComputeTVDSS:

    def test_vertical_well(self, sample_deviation):
        """Vertical well: TVDSS = TVD - KB."""
        md = np.array([0, 100, 200, 300])
        kb = 25.0
        tvdss = compute_tvdss(sample_deviation, 'W-1', md, kb)
        assert tvdss is not None
        assert len(tvdss) == len(md)
        # All TVDSS values should be less than MD (because KB>0)
        assert np.all(tvdss < md + 1)  # TVD ~ 0.998*MD, so TVDSS ~ 0.998*MD - 25

    def test_zero_kb(self, sample_deviation):
        md = np.array([100, 200])
        tvdss = compute_tvdss(sample_deviation, 'W-1', md, 0.0)
        assert tvdss is not None
        assert len(tvdss) == 2

    def test_missing_well(self, sample_deviation):
        md = np.array([100, 200])
        tvdss = compute_tvdss(sample_deviation, 'NONEXISTENT', md, 25.0)
        assert tvdss is None


class TestGetWellPathXY:

    def test_dict_format(self, sample_deviation):
        """Deviation in dict format with 'dev_data' key."""
        result = get_well_path_xy(sample_deviation, 'W-1')
        assert result is not None
        x, y = result
        assert len(x) > 0
        assert len(y) > 0
        assert len(x) == len(y)

    def test_missing_well(self, sample_deviation):
        result = get_well_path_xy(sample_deviation, 'NONEXISTENT')
        assert result is None
