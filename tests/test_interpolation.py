"""Tests for horizon interpolation utilities."""
import numpy as np
import pytest
from geoagent.utils.interpolation import (
    interpolate_horizon_at_points,
    interpolate_horizon_at_wells,
    extract_horizon_along_line,
)


class TestInterpolateHorizon:

    def test_single_point(self, sample_horizon):
        z = interpolate_horizon_at_points(sample_horizon, [270050], [2541050])
        assert len(z) == 1
        assert np.isfinite(z[0])

    def test_multiple_points(self, sample_horizon):
        x = [270100, 270200, 270300]
        y = [2541100, 2541200, 2541300]
        z = interpolate_horizon_at_points(sample_horizon, x, y)
        assert len(z) == 3
        assert all(np.isfinite(z))

    def test_out_of_bounds_returns_nan(self, sample_horizon):
        z = interpolate_horizon_at_points(sample_horizon, [999999], [999999])
        assert np.isnan(z[0])

    def test_wells_dict(self, sample_horizon):
        wells = {'W1': (270100, 2541100), 'W2': (270300, 2541300)}
        result = interpolate_horizon_at_wells(sample_horizon, wells)
        assert 'W1' in result
        assert 'W2' in result
        assert np.isfinite(result['W1'])

    def test_wells_list(self, sample_horizon):
        coords = [(270100, 2541100), (270200, 2541200)]
        result = interpolate_horizon_at_wells(sample_horizon, coords)
        assert len(result) == 2

    def test_along_line(self, sample_horizon):
        x_line = np.linspace(270050, 270550, 30)
        y_line = np.full(30, 2541300.0)
        z = extract_horizon_along_line(sample_horizon, x_line, y_line)
        assert len(z) == 30
        assert all(np.isfinite(z))

    def test_2d_meshgrid_input(self, sample_horizon):
        """Test with 2D X,Y arrays (meshgrid format)."""
        x1d = sample_horizon['X']
        y1d = sample_horizon['Y']
        X2d, Y2d = np.meshgrid(x1d, y1d)

        horizon_2d = {'X': X2d, 'Y': Y2d, 'Z': sample_horizon['Z']}
        z = interpolate_horizon_at_points(horizon_2d, [270100], [2541100])
        assert np.isfinite(z[0])
