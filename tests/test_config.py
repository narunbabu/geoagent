"""Tests for plotting config dataclasses."""
import pytest
from geoagent.plotting.config import SectionPlotConfig, FormationTop


class TestFormationTop:

    def test_defaults(self):
        ft = FormationTop(color='red')
        assert ft.color == 'red'
        assert ft.linestyle == '-'
        assert ft.linewidth == 1.5
        assert ft.label == ''

    def test_custom(self):
        ft = FormationTop(color='blue', linestyle='--', linewidth=2.0, label='My Top')
        assert ft.linestyle == '--'
        assert ft.label == 'My Top'


class TestSectionPlotConfig:

    def test_default_construction(self):
        cfg = SectionPlotConfig()
        assert cfg.gr_range == (0, 150)
        assert cfg.lld_range == (0.2, 2000)
        assert cfg.datum_surface == ''
        assert 'depth' in cfg.track_widths

    def test_custom_construction(self):
        tops = {
            'Top-A': FormationTop(color='red', label='A'),
            'Top-B': FormationTop(color='blue', label='B'),
        }
        cfg = SectionPlotConfig(
            formation_tops=tops,
            datum_surface='Top-A',
            window_above=50,
            window_below=30,
        )
        assert len(cfg.formation_tops) == 2
        assert cfg.datum_surface == 'Top-A'
        assert cfg.window_above == 50
