"""Tests for ProjectBuilder — SeisTrans-compatible project creation."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
import segyio

from geoagent.io.project_builder import ProjectBuilder
from geoagent.core.data_manager import CoreDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_segy(path, n_inlines=5, n_crosslines=8, n_samples=50,
                           dt_ms=2.0, start_time_ms=1000.0,
                           origin_x=270000.0, origin_y=2540000.0,
                           dx=25.0, dy=25.0):
    """
    Create a minimal SEG-Y file with known geometry for testing.

    Returns the path and the byte positions dict needed by ProjectBuilder.
    """
    spec = segyio.spec()
    spec.sorting = 2  # crossline sorting
    spec.format = 1   # IBM float
    spec.samples = np.arange(n_samples) * dt_ms + start_time_ms
    spec.ilines = np.arange(1, n_inlines + 1)
    spec.xlines = np.arange(1, n_crosslines + 1)
    spec.tracecount = n_inlines * n_crosslines

    with segyio.create(path, spec) as f:
        trace_idx = 0
        for il in spec.ilines:
            for xl in spec.xlines:
                # Synthetic trace: low-freq sinusoid
                f.trace[trace_idx] = np.sin(
                    np.linspace(0, 4 * np.pi, n_samples)
                ).astype(np.float32)

                # Write trace headers
                x = origin_x + (xl - 1) * dx
                y = origin_y + (il - 1) * dy

                f.header[trace_idx] = {
                    segyio.TraceField.INLINE_3D: il,
                    segyio.TraceField.CROSSLINE_3D: xl,
                    segyio.TraceField.CDP_X: int(x * 100),
                    segyio.TraceField.CDP_Y: int(y * 100),
                    segyio.TraceField.DelayRecordingTime: int(start_time_ms),
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(dt_ms * 1000),
                }
                trace_idx += 1

    byte_positions = {
        'CDP_X': 181,
        'CDP_Y': 185,
        'Inline': 189,
        'Crossline': 193,
        'Coord_Mult_Factor': 0.01,
    }
    return path, byte_positions


def _create_synthetic_las(path, well_name="TEST-1", n_points=200,
                          depth_start=500.0, depth_end=1500.0):
    """Create a minimal LAS file for testing."""
    import lasio

    las = lasio.LASFile()
    las.well['WELL'].value = well_name

    depth = np.linspace(depth_start, depth_end, n_points)
    las.append_curve('DEPT', depth, unit='M', descr='Measured Depth')
    las.append_curve('GR', np.random.uniform(20, 120, n_points),
                     unit='API', descr='Gamma Ray')
    las.append_curve('DT', np.random.uniform(60, 140, n_points),
                     unit='US/FT', descr='Sonic')
    las.append_curve('RHOB', np.random.uniform(2.0, 2.65, n_points),
                     unit='G/CC', descr='Bulk Density')

    with open(path, 'w') as f:
        las.write(f)

    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def segy_file(tmp_path):
    """Create a synthetic SEG-Y file and return (path, byte_positions)."""
    path = str(tmp_path / 'test_survey.sgy')
    return _create_synthetic_segy(path)


@pytest.fixture
def las_dir(tmp_path):
    """Create a directory with two synthetic LAS files."""
    las_folder = tmp_path / 'las_files'
    las_folder.mkdir()
    _create_synthetic_las(str(las_folder / 'well_A.las'), well_name='WELL-A')
    _create_synthetic_las(str(las_folder / 'well_B.las'), well_name='WELL-B')
    return str(las_folder)


@pytest.fixture
def builder(tmp_path):
    """Create a ProjectBuilder instance in a temp directory."""
    project_folder = str(tmp_path / 'test_project')
    return ProjectBuilder(project_folder, 'TestProject')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProjectBuilderInit:

    def test_creates_project_folder(self, tmp_path):
        folder = str(tmp_path / 'new_project')
        assert not os.path.exists(folder)
        builder = ProjectBuilder(folder, 'New')
        assert os.path.isdir(folder)

    def test_handlers_initialized(self, builder):
        assert builder.seismic_handler is not None
        assert builder.well_handler is not None
        assert builder.well_log_handler is not None
        assert builder.horizons_handler is not None

    def test_project_name_stored(self, builder):
        assert builder.project_name == 'TestProject'

    def test_repr(self, builder):
        r = repr(builder)
        assert 'TestProject' in r
        assert 'ProjectBuilder' in r


class TestSegyImport:

    def test_import_segy_creates_survey_folder(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)

        survey_folder = os.path.join(builder.project_folder, 'testsurvey')
        assert os.path.isdir(survey_folder)

    def test_import_segy_creates_headerdata(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)

        hdr_path = os.path.join(builder.project_folder, 'testsurvey',
                                'headerdata.pkl')
        assert os.path.exists(hdr_path)

        with open(hdr_path, 'rb') as f:
            headerdata = pickle.load(f)
        assert headerdata.shape == (40, 5)  # 5 inlines * 8 crosslines
        assert headerdata.shape[1] == 5     # trace_idx, il, xl, cdp_x, cdp_y

    def test_import_segy_creates_endpoints(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)

        ep_path = os.path.join(builder.project_folder, 'testsurvey',
                               'endpoints.pkl')
        assert os.path.exists(ep_path)

        with open(ep_path, 'rb') as f:
            endpoints = pickle.load(f)
        assert 'inlines' in endpoints
        assert 'crosslines' in endpoints
        assert len(endpoints['inlines']) == 5
        assert len(endpoints['crosslines']) == 8

    def test_import_segy_saves_seis_params(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)

        sp_path = os.path.join(builder.project_folder, 'testsurvey',
                               'Amplitude_seis_params.pkl')
        assert os.path.exists(sp_path)

        with open(sp_path, 'rb') as f:
            seis_params = pickle.load(f)
        assert seis_params['CDP_X'] == 181
        assert seis_params['CDP_Y'] == 185
        assert 'Sampling Rate' in seis_params
        assert 'Number of Samples' in seis_params
        assert seis_params['Number of Samples'] == 50

    def test_import_segy_copies_file(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        result = builder.import_segy(segy_path, 'TestSurvey', 'Amplitude',
                                     byte_pos)
        assert os.path.exists(result['segy_path'])

    def test_import_segy_no_copy(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        result = builder.import_segy(segy_path, 'TestSurvey', 'Amplitude',
                                     byte_pos, copy_segy=False)
        # File may or may not exist at segy_path depending on naming
        # but the project structure should still be valid
        assert result['survey_name'] == 'TestSurvey'

    def test_import_segy_missing_file_raises(self, builder):
        with pytest.raises(FileNotFoundError):
            builder.import_segy('/nonexistent/file.sgy', 'S', 'A', {})

    def test_import_multiple_volumes(self, builder, tmp_path):
        """Import two volumes into the same survey."""
        segy1 = str(tmp_path / 'vol1.sgy')
        segy2 = str(tmp_path / 'vol2.sgy')
        _, byte_pos = _create_synthetic_segy(segy1)
        _create_synthetic_segy(segy2)

        builder.import_segy(segy1, 'Survey1', 'Amplitude', byte_pos)
        builder.import_segy(segy2, 'Survey1', 'Phase', byte_pos)

        # Project structure should have one survey with two volumes
        surveys = builder.seismic_handler.project_files
        assert len(surveys) == 1
        assert len(surveys[0]['survey_volumes']) == 2
        names = {v['attribute_name'] for v in surveys[0]['survey_volumes']}
        assert names == {'Amplitude', 'Phase'}

    def test_import_segy_updates_project_structure(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)

        ps_path = os.path.join(builder.project_folder,
                               'seismic_project_structure.pkl')
        assert os.path.exists(ps_path)

        with open(ps_path, 'rb') as f:
            project_structure = pickle.load(f)
        assert len(project_structure) == 1
        assert project_structure[0]['survey_name'] == 'TestSurvey'


class TestWellData:

    def test_set_well_heads(self, builder):
        df = pd.DataFrame({
            'Name': ['W-1', 'W-2'],
            'Surface X': [270100, 270200],
            'Surface Y': [2541100, 2541200],
        })
        builder.set_well_heads(df)
        stored = builder.well_handler.loaded_data['well_heads']
        assert list(stored['Name']) == ['W-1', 'W-2']
        assert 'Name' in stored.columns  # Must be a column, not index

    def test_set_well_heads_normalizes_name_index(self, builder):
        """If Name is the DataFrame index, set_well_heads resets it to a column."""
        df = pd.DataFrame({
            'Name': ['W-1', 'W-2'],
            'Surface X': [270100, 270200],
        })
        df = df.set_index('Name')  # Simulate the old buggy format
        builder.set_well_heads(df)
        stored = builder.well_handler.loaded_data['well_heads']
        assert 'Name' in stored.columns
        assert list(stored['Name']) == ['W-1', 'W-2']
        assert stored.index.tolist() == [0, 1]  # Integer index

    def test_set_deviation(self, builder):
        dev = np.zeros((10, 3))
        builder.set_deviation('W-1', {'name': 'W-1', 'x': 0, 'y': 0, 'kb': 25},
                              dev)
        assert 'W-1' in builder.well_handler.loaded_data['deviation']

    def test_set_well_tops(self, builder):
        tops = {'W-1': {'TopA': {'MD': 900}}}
        builder.set_well_tops(tops)
        assert builder.well_handler.loaded_data['well_tops'] == tops

    def test_set_checkshots(self, builder):
        cs = {'W-1': {'depths': [100, 200], 'times': [50, 100]}}
        builder.set_checkshots(cs, tdr_mappings={'W-1': {'source_well': 'W-1'}})
        assert builder.well_handler.loaded_data['checkshot'] == cs
        assert 'W-1' in builder.well_handler.loaded_data['tdr_mappings']


class TestWellLogImport:

    def test_import_single_las(self, builder, tmp_path):
        las_path = str(tmp_path / 'test.las')
        _create_synthetic_las(las_path, well_name='W-1')
        builder.import_well_logs('W-1', las_path)

        logs = builder.well_log_handler.loaded_data.get('well_logs', {})
        assert 'W-1' in logs
        assert 'GR' in logs['W-1']
        assert 'DT' in logs['W-1']

    def test_import_wells_from_las_dir(self, builder, las_dir):
        imported = builder.import_wells_from_las(las_dir)
        assert len(imported) == 2
        assert 'WELL-A' in imported
        assert 'WELL-B' in imported

        logs = builder.well_log_handler.loaded_data.get('well_logs', {})
        assert 'WELL-A' in logs
        assert 'WELL-B' in logs

    def test_import_wells_with_heads(self, builder, las_dir):
        df = pd.DataFrame({
            'Name': ['WELL-A', 'WELL-B'],
            'Surface X': [270100, 270200],
            'Surface Y': [2541100, 2541200],
        })
        builder.import_wells_from_las(las_dir, well_heads_df=df)
        stored = builder.well_handler.loaded_data['well_heads']
        assert list(stored['Name']) == ['WELL-A', 'WELL-B']
        assert 'Name' in stored.columns


class TestWavelets:

    def test_add_wavelet(self, builder):
        data = np.sin(np.linspace(0, 2 * np.pi, 64))
        builder.add_wavelet('Ricker_25Hz', data, {'frequency': 25.0})

        wavelets = builder.seismic_handler.loaded_data['wavelets']
        assert len(wavelets) == 1
        assert wavelets[0]['name'] == 'Ricker_25Hz'
        assert wavelets[0]['metadata']['frequency'] == 25.0

    def test_add_multiple_wavelets(self, builder):
        builder.add_wavelet('W1', np.zeros(32))
        builder.add_wavelet('W2', np.ones(64))
        assert len(builder.seismic_handler.loaded_data['wavelets']) == 2


class TestHorizons:

    def test_set_horizons(self, builder):
        hz = {
            'TopA': {
                'X': np.arange(10.0),
                'Y': np.arange(10.0),
                'Z': np.random.rand(10, 10),
            }
        }
        builder.set_horizons(hz)
        assert 'TopA' in builder.horizons_handler.loaded_data['horizons']


class TestBuild:

    def test_build_creates_str_file(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)
        str_path = builder.build()

        assert os.path.exists(str_path)
        with open(str_path, 'r') as f:
            assert f.read() == "Seismic Data Viewer Project"

    def test_build_saves_well_heads(self, builder):
        df = pd.DataFrame({
            'Name': ['W-1'],
            'Surface X': [270100],
            'Surface Y': [2541100],
        }).set_index('Name')
        builder.set_well_heads(df)
        builder.build()

        wh_path = os.path.join(builder.project_folder, 'well_heads.pkl')
        assert os.path.exists(wh_path)
        with open(wh_path, 'rb') as f:
            loaded = pickle.load(f)
        assert len(loaded) == 1

    def test_build_saves_well_logs(self, builder, tmp_path):
        las_path = str(tmp_path / 'test.las')
        _create_synthetic_las(las_path, well_name='W-1')
        builder.import_well_logs('W-1', las_path)
        builder.build()

        wl_path = os.path.join(builder.project_folder, 'well_logs.pkl')
        assert os.path.exists(wl_path)

    def test_build_saves_wavelets(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)
        builder.add_wavelet('TestWav', np.zeros(32))
        builder.build()

        wav_path = os.path.join(builder.project_folder, 'wavelets.pkl')
        assert os.path.exists(wav_path)
        with open(wav_path, 'rb') as f:
            wavelets = pickle.load(f)
        assert len(wavelets) == 1
        assert wavelets[0]['name'] == 'TestWav'

    def test_build_saves_project_structure(self, builder, segy_file):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)
        builder.build()

        ps_path = os.path.join(builder.project_folder,
                               'seismic_project_structure.pkl')
        assert os.path.exists(ps_path)

    def test_build_minimal_wells_only(self, builder):
        """Build a project with only well data (no seismic)."""
        df = pd.DataFrame({
            'Name': ['W-1'],
            'Surface X': [270100],
            'Surface Y': [2541100],
        }).set_index('Name')
        builder.set_well_heads(df)
        str_path = builder.build()
        assert os.path.exists(str_path)


class TestRoundTrip:

    def test_build_then_load(self, builder, segy_file, tmp_path):
        """Build a project, then load it with CoreDataManager."""
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)

        # Add well heads
        df = pd.DataFrame({
            'Name': ['W-1', 'W-2'],
            'Surface X': [270050, 270100],
            'Surface Y': [2540050, 2540100],
        }).set_index('Name')
        builder.set_well_heads(df)

        # Add well logs
        las_path = str(tmp_path / 'w1.las')
        _create_synthetic_las(las_path, well_name='W-1')
        builder.import_well_logs('W-1', las_path)

        # Add wavelet
        builder.add_wavelet('Ricker_30', np.sin(np.linspace(0, np.pi, 64)))

        str_path = builder.build()

        # Load with CoreDataManager
        dm = CoreDataManager(builder.project_folder)

        # Verify surveys
        surveys = dm.get_available_surveys()
        assert 'TestSurvey' in surveys

        # Verify wells
        wells = dm.get_available_wells()
        assert 'W-1' in wells
        assert 'W-2' in wells

        # Verify well logs
        w1_logs = dm.well_log_handler.get_well_logs('W-1')
        assert w1_logs is not None

        # Verify wavelets (stored in seismic_handler.loaded_data)
        wavelets = dm.seismic_handler.loaded_data.get('wavelets', [])
        assert len(wavelets) == 1
        assert wavelets[0]['name'] == 'Ricker_30'

    def test_build_then_load_seismic_attributes(self, builder, tmp_path):
        """Verify seismic attributes are accessible after round-trip."""
        segy_path = str(tmp_path / 'amp.sgy')
        _, byte_pos = _create_synthetic_segy(segy_path)
        builder.import_segy(segy_path, 'Survey1', 'Amplitude', byte_pos)
        builder.build()

        dm = CoreDataManager(builder.project_folder)
        attrs = dm.seismic_handler.get_available_attributes('Survey1')
        assert 'Amplitude' in attrs


class TestSummary:

    def test_summary_empty(self, builder):
        s = builder.summary()
        assert s['n_wells'] == 0
        assert s['n_well_logs'] == 0
        assert s['n_wavelets'] == 0
        assert s['surveys'] == []

    def test_summary_with_data(self, builder, segy_file, las_dir):
        segy_path, byte_pos = segy_file
        builder.import_segy(segy_path, 'TestSurvey', 'Amplitude', byte_pos)
        builder.import_wells_from_las(las_dir)
        builder.add_wavelet('W1', np.zeros(32))

        s = builder.summary()
        assert 'TestSurvey' in s['surveys']
        assert s['n_well_logs'] == 2
        assert s['n_wavelets'] == 1
