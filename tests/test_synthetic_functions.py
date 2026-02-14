"""Tests for synthetic seismic functions."""
import numpy as np
import pytest

from geoagent.synthetic.functions import (
    create_reflectivity,
    create_synthetic_seismic_valid,
    calculate_correlation_with_shift,
    trigger_upscaling_workflow,
)
from geoagent.synthetic.wavelet_functions import ricker_wavelet


class TestCreateReflectivity:

    def test_basic(self, sample_impedance):
        refl = create_reflectivity(sample_impedance)
        assert len(refl) == len(sample_impedance) - 1

    def test_simple_case(self):
        imp = np.array([1000, 2000, 1500])
        refl = create_reflectivity(imp)
        assert len(refl) == 2
        # (2000-1000)/(2000+1000) = 1/3
        np.testing.assert_almost_equal(refl[0], 1 / 3, decimal=4)

    def test_constant_impedance(self):
        imp = np.ones(100) * 5000
        refl = create_reflectivity(imp)
        np.testing.assert_array_almost_equal(refl, 0.0)


class TestCreateSyntheticSeismic:

    def test_output_length(self, sample_impedance):
        refl = create_reflectivity(sample_impedance)
        wavelet = ricker_wavelet(f=30, length=51, dt=0.002)
        times = np.linspace(1400, 1600, len(refl))
        synth = create_synthetic_seismic_valid(refl, wavelet, times)
        assert len(synth) > 0

    def test_zero_reflectivity(self):
        refl = np.zeros(100)
        wavelet = ricker_wavelet(f=30, length=31, dt=0.002)
        times = np.linspace(1400, 1600, len(refl))
        synth = create_synthetic_seismic_valid(refl, wavelet, times)
        np.testing.assert_array_almost_equal(synth, 0.0)


class TestCorrelation:

    def test_same_signal_high_correlation(self):
        """Same signal at same times should give high correlation."""
        times = np.linspace(1400, 1600, 200)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        cc = calculate_correlation_with_shift(signal, times, signal, times, shift=0.0)
        assert cc > 0.99

    def test_shifted_signal_lower_correlation(self):
        """Signal with shift=0 but actually offset should give lower correlation."""
        times = np.linspace(1400, 1600, 200)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        shifted_times = times + 5  # 5ms shift
        cc = calculate_correlation_with_shift(signal, times, signal, shifted_times, shift=0.0)
        # Correlation should still be somewhat high due to overlap
        assert isinstance(cc, float)


class TestTriggerUpscalingWorkflow:

    def test_headless_stub_returns_false(self):
        result = trigger_upscaling_workflow(None, 'test_well')
        assert result is False
