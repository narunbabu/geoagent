"""Tests for visualization modules (non-interactive, Agg backend)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest


class TestSeismicPlotter:

    def test_variable_density(self):
        from geoagent.plotting.seismic_plotter import plot_seismic_section
        traces = np.random.randn(100, 20)
        time_axis = np.linspace(1400, 1600, 100)
        fig, ax = plot_seismic_section(traces, time_axis, display='variable_density')
        assert fig is not None
        plt.close(fig)

    def test_wiggle(self):
        from geoagent.plotting.seismic_plotter import plot_seismic_section
        traces = np.random.randn(100, 10)
        time_axis = np.linspace(1400, 1600, 100)
        fig, ax = plot_seismic_section(traces, time_axis, display='wiggle')
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        from geoagent.plotting.seismic_plotter import plot_seismic_section
        traces = np.random.randn(50, 10)
        time_axis = np.linspace(1400, 1500, 50)
        out = str(tmp_path / 'seismic.png')
        fig, ax = plot_seismic_section(traces, time_axis, save_path=out)
        assert os.path.exists(out)
        plt.close(fig)

    def test_trace_gather(self):
        from geoagent.plotting.seismic_plotter import plot_trace_gather
        traces = np.random.randn(100, 5)
        time_axis = np.linspace(1400, 1600, 100)
        fig, ax = plot_trace_gather(traces, time_axis, well_name='TEST')
        assert fig is not None
        plt.close(fig)

    def test_with_horizons_and_markers(self):
        from geoagent.plotting.seismic_plotter import plot_seismic_section
        traces = np.random.randn(100, 30)
        time_axis = np.linspace(1400, 1600, 100)
        positions = np.arange(30)

        horizons = {
            'Top-A': (positions, np.full(30, 1480.0), 'red'),
        }
        markers = [{'position': 15, 'name': 'W-1', 'color': 'blue'}]

        fig, ax = plot_seismic_section(
            traces, time_axis,
            trace_positions=positions,
            horizons=horizons,
            well_markers=markers,
        )
        assert fig is not None
        plt.close(fig)


class TestWellPanel:

    def test_basic_panel(self):
        from geoagent.plotting.well_panel import plot_well_panel
        depth = np.linspace(1000, 1200, 300)
        logs = {
            'GR': np.random.uniform(20, 120, 300),
            'LLD': 10 ** np.random.uniform(-0.5, 2.5, 300),
            'NPHI': np.random.uniform(0.05, 0.45, 300),
            'RHOB': np.random.uniform(2.0, 2.8, 300),
            'DT': np.random.uniform(50, 130, 300),
        }
        fig, axes = plot_well_panel(depth, logs, well_name='TEST')
        assert len(axes) == 5
        plt.close(fig)

    def test_missing_logs(self):
        """Should handle missing log curves gracefully."""
        from geoagent.plotting.well_panel import plot_well_panel
        depth = np.linspace(1000, 1200, 100)
        logs = {'GR': np.random.uniform(20, 120, 100)}  # Only GR
        fig, axes = plot_well_panel(depth, logs, well_name='SPARSE')
        assert len(axes) == 5
        plt.close(fig)

    def test_with_formation_tops(self):
        from geoagent.plotting.well_panel import plot_well_panel
        depth = np.linspace(1000, 1200, 200)
        logs = {'GR': np.random.uniform(20, 120, 200)}
        tops = {'Top-A': {'md': 1050, 'color': 'red'}}
        fig, axes = plot_well_panel(depth, logs, formation_tops=tops)
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        from geoagent.plotting.well_panel import plot_well_panel
        depth = np.linspace(1000, 1200, 100)
        logs = {'GR': np.random.uniform(20, 120, 100)}
        out = str(tmp_path / 'panel.png')
        fig, _ = plot_well_panel(depth, logs, save_path=out)
        assert os.path.exists(out)
        plt.close(fig)
