"""Tests for mnemonic resolution and log naming settings."""
import pytest
from geoagent.settings.log_naming_settings import LogNamingSettings


class TestLogNamingSettings:

    def test_default_log_names(self):
        lns = LogNamingSettings()
        assert 'DTC' in lns.get_log_names_for_type('sonic_log')
        assert 'RHOB' in lns.get_log_names_for_type('density_log')

    def test_find_direct_match(self):
        lns = LogNamingSettings()
        assert lns.find_log_column('sonic_log', ['DTC', 'GR', 'NPHI']) == 'DTC'

    def test_find_case_insensitive(self):
        lns = LogNamingSettings()
        assert lns.find_log_column('density_log', ['gr', 'rhob', 'nphi']) == 'rhob'

    def test_find_partial_match(self):
        lns = LogNamingSettings()
        # 'SONIC_FILTERED' contains 'SONIC' which is in the sonic_log aliases
        assert lns.find_log_column('sonic_log', ['SONIC_FILTERED']) == 'SONIC_FILTERED'

    def test_find_no_match(self):
        lns = LogNamingSettings()
        assert lns.find_log_column('sonic_log', ['CALIPER', 'SP']) is None

    def test_find_none_columns(self):
        lns = LogNamingSettings()
        assert lns.find_log_column('sonic_log', None) is None

    def test_find_empty_columns(self):
        lns = LogNamingSettings()
        assert lns.find_log_column('sonic_log', []) is None

    def test_add_variant(self):
        lns = LogNamingSettings()
        lns.add_log_name_variant('sonic_log', 'MY_SONIC')
        assert 'MY_SONIC' in lns.get_log_names_for_type('sonic_log')
        assert lns.find_log_column('sonic_log', ['MY_SONIC']) == 'MY_SONIC'

    def test_serialization_roundtrip(self):
        lns = LogNamingSettings()
        lns.add_log_name_variant('sonic_log', 'CUSTOM_DT')
        data = lns.to_dict()

        lns2 = LogNamingSettings()
        lns2.from_dict(data)
        assert 'CUSTOM_DT' in lns2.get_log_names_for_type('sonic_log')

    def test_validate_log_configuration(self):
        lns = LogNamingSettings()
        result = lns.validate_log_configuration({'DTC': [1, 2], 'RHOB': [3, 4]})
        assert result['valid'] is True
        assert 'sonic_log' in result['found_logs']
        assert 'density_log' in result['found_logs']

    def test_validate_missing_essential(self):
        lns = LogNamingSettings()
        result = lns.validate_log_configuration({'GR': [1, 2]})
        assert result['valid'] is False
        assert 'sonic_log' in result['missing_logs']

    def test_reset_to_defaults(self):
        lns = LogNamingSettings()
        lns.set_log_names_for_type('sonic_log', ['CUSTOM'])
        lns.reset_to_defaults()
        assert 'DTC' in lns.get_log_names_for_type('sonic_log')
