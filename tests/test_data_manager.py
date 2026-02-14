"""Tests for CoreDataManager."""
import pytest
from geoagent.core.data_manager import CoreDataManager


class TestCoreDataManager:

    def test_init_empty(self):
        dm = CoreDataManager()
        assert dm.well_handler is not None
        assert dm.seismic_handler is not None
        assert dm.well_log_handler is not None
        assert dm.horizons_handler is not None
        assert dm.project_folder == ''

    def test_repr(self):
        dm = CoreDataManager()
        r = repr(dm)
        assert 'CoreDataManager' in r
        assert 'wells=0' in r

    def test_get_data_from_dm(self):
        dm = CoreDataManager()
        dm.loaded_data['test_key'] = 'test_value'
        assert dm.get_data('test_key') == 'test_value'

    def test_get_data_fallback_to_handler(self):
        dm = CoreDataManager()
        dm.well_handler.loaded_data['handler_key'] = 'handler_value'
        assert dm.get_data('handler_key') == 'handler_value'

    def test_get_data_missing(self):
        dm = CoreDataManager()
        assert dm.get_data('nonexistent') is None

    def test_load_project(self, project_dir):
        dm = CoreDataManager(project_dir)
        assert dm.project_folder == project_dir

        # Well heads should be loaded
        well_heads = dm.get_data('well_heads')
        assert well_heads is not None
        assert len(well_heads) == 5

    def test_get_available_wells(self, project_dir):
        dm = CoreDataManager(project_dir)
        wells = dm.get_available_wells()
        assert len(wells) == 5
        assert 'W-1' in wells

    def test_find_log_column(self, project_dir):
        dm = CoreDataManager(project_dir)
        sonic = dm.find_log_column('sonic_log', 'W-1')
        assert sonic is not None  # DTC should resolve to sonic_log

    def test_find_log_column_missing_well(self, project_dir):
        dm = CoreDataManager(project_dir)
        assert dm.find_log_column('sonic_log', 'NONEXISTENT') is None

    def test_synthetic_settings(self):
        dm = CoreDataManager()
        assert dm.get_synthetic_settings_all() is None

    def test_well_settings(self):
        dm = CoreDataManager()
        assert dm.load_well_settings('any') is None

    def test_default_well_settings(self):
        dm = CoreDataManager()
        d = dm.get_default_well_settings('test')
        assert d['bulk_shift'] == 0.0
        assert d['sampling_interval'] == 2.0

    def test_synthetic_result_cache(self):
        dm = CoreDataManager()
        dm.store_synthetic_result('W-1', {'cc': 0.95})
        result = dm.get_synthetic_result('W-1')
        assert result['cc'] == 0.95
        assert dm.get_synthetic_result('W-2') is None
