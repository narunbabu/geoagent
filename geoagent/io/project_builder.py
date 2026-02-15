"""
ProjectBuilder — Create SeisTrans-compatible projects from raw geoscience data.

Orchestrates the existing handlers (SeismicHandler, WellHandler, WellLogHandler,
HorizonsHandler) to build a complete project that can be opened in SeisTrans or
loaded by CoreDataManager.

Usage:
    from geoagent.io.project_builder import ProjectBuilder

    builder = ProjectBuilder("C:/Projects/volve", "Volve")
    builder.import_segy("path/to/survey.sgy", "MySurvey", "Amplitude", {...})
    builder.import_wells_from_las("path/to/las_files/")
    builder.add_wavelet("Ricker_25Hz", ricker_data)
    builder.build()
"""

import os
import glob
import shutil
import logging

import numpy as np

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False

try:
    import segyio
    HAS_SEGYIO = True
except ImportError:
    HAS_SEGYIO = False

from geoagent.core.seismic_handler import SeismicHandler
from geoagent.core.well_handler import WellHandler
from geoagent.core.well_log_handler import WellLogHandler
from geoagent.core.horizons_handler import HorizonsHandler
from geoagent.utils.safe_print import safe_print

logger = logging.getLogger(__name__)


class ProjectBuilder:
    """
    Build a SeisTrans-compatible project from raw geoscience data.

    Delegates to the existing handler classes for all heavy lifting:
    - SeismicHandler for SEG-Y import, headerdata/endpoints calculation
    - WellHandler for well heads, deviation, checkshots
    - WellLogHandler for LAS file import
    - HorizonsHandler for ZMAP horizon import

    The build() method writes the .str pointer file and flushes all
    handler data to disk as pickle files.
    """

    def __init__(self, project_folder, project_name):
        """
        Initialize a new project.

        Args:
            project_folder: Directory where project data will be stored.
                           Created if it does not exist.
            project_name: Display name for the project.
        """
        self.project_folder = os.path.abspath(project_folder)
        self.project_name = project_name
        os.makedirs(self.project_folder, exist_ok=True)

        # Instantiate handlers pointed at the project folder
        self.seismic_handler = SeismicHandler(self.project_folder)
        self.well_handler = WellHandler(self.project_folder)
        self.well_log_handler = WellLogHandler(self.project_folder)
        self.horizons_handler = HorizonsHandler(self.project_folder)

    # ------------------------------------------------------------------
    # Seismic
    # ------------------------------------------------------------------

    def import_segy(self, file_path, survey_name, attribute_name,
                    byte_positions, copy_segy=True):
        """
        Import a SEG-Y file into the project.

        Creates the survey subfolder, calculates headerdata/endpoints/spatial
        index, saves seis_params, and optionally copies the SEG-Y file into
        the project directory.

        Args:
            file_path: Path to the source SEG-Y file.
            survey_name: Display name for the seismic survey.
            attribute_name: Seismic attribute name (e.g. "Amplitude").
            byte_positions: Dict with at minimum:
                - CDP_X: int — trace header byte for CDP X
                - CDP_Y: int — trace header byte for CDP Y
                - Inline: int — trace header byte for inline number
                - Crossline: int — trace header byte for crossline number
                - Coord_Mult_Factor: float — coordinate multiplier
                Additional keys (Sampling Rate, Number of Samples, Format,
                Start Time, End Time) are auto-detected from the SEG-Y file
                if not provided.
            copy_segy: If True (default), copy the SEG-Y file into the
                      project's survey subfolder. Set False if the file
                      is already in place.

        Returns:
            dict with import result metadata.

        Raises:
            FileNotFoundError: If the SEG-Y file does not exist.
            ImportError: If segyio is not installed.
        """
        if not HAS_SEGYIO:
            raise ImportError("segyio is required for SEG-Y import. "
                              "Install with: pip install segyio")

        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SEG-Y file not found: {file_path}")

        # Build complete seis_params by merging user-provided byte positions
        # with auto-detected parameters from the SEG-Y file
        seis_params = dict(byte_positions)
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            dt = f.samples[1] - f.samples[0] if len(f.samples) > 1 else 2.0
            start_time = f.header[0][segyio.TraceField.DelayRecordingTime]
            num_samples = len(f.samples)
            end_time = start_time + (num_samples - 1) * dt

            seis_params.setdefault('Sampling Rate', dt * 1000)  # ms → µs
            seis_params.setdefault('Number of Samples', num_samples)
            seis_params.setdefault('Start Time', float(start_time))
            seis_params.setdefault('End Time', float(end_time))
            seis_params.setdefault('Format', 'IBM')

        # Delegate to SeismicHandler — does header extraction, endpoints,
        # spatial index, seis_params save, and project_files update
        result = self.seismic_handler.import_segy_file(
            file_path, seis_params, survey_name, attribute_name
        )

        # Copy SEG-Y file into the project survey subfolder
        if copy_segy:
            segy_dest = result['segy_path']
            if not os.path.exists(segy_dest):
                shutil.copy2(file_path, segy_dest)
                logger.info("Copied SEG-Y to %s", segy_dest)

        return result

    # ------------------------------------------------------------------
    # Wells
    # ------------------------------------------------------------------

    def set_well_heads(self, well_heads_df):
        """
        Set well header data directly.

        Args:
            well_heads_df: pandas DataFrame with well header information.
                          Should have columns like 'Name', 'Surface X',
                          'Surface Y', 'Well datum name', 'Well datum value',
                          'TD (MD)', etc.  Can be indexed by well name or
                          have a 'Name' / 'Well Name' column.

        The DataFrame is normalized to SeisTrans format: integer index
        with 'Name' as a regular column.  If 'Name' is currently the
        index, it is reset to a column automatically.
        """
        import pandas as pd

        df = well_heads_df.copy()

        # Normalize: ensure 'Name' is a column, not the index
        if 'Name' not in df.columns:
            if df.index.name == 'Name':
                df = df.reset_index()
            elif hasattr(df.index, 'names') and 'Name' in df.index.names:
                df = df.reset_index()

        # Ensure integer index
        if df.index.name is not None:
            df = df.reset_index(drop=True)

        self.well_handler.loaded_data['well_heads'] = df

    def set_deviation(self, well_name, well_info, dev_data):
        """
        Set deviation survey data for a single well.

        Args:
            well_name: Well identifier string.
            well_info: Dict with keys like 'name', 'x', 'y', 'kb'.
            dev_data: numpy ndarray or DataFrame with deviation stations.
        """
        if 'deviation' not in self.well_handler.loaded_data:
            self.well_handler.loaded_data['deviation'] = {}
        self.well_handler.loaded_data['deviation'][well_name] = {
            'well_info': well_info,
            'dev_data': dev_data,
        }

    def set_well_tops(self, well_tops):
        """
        Set formation tops data.

        Args:
            well_tops: Dict or DataFrame of well tops, matching the
                      WellHandler format.
        """
        self.well_handler.loaded_data['well_tops'] = well_tops

    def set_checkshots(self, checkshot_data, checkshot_mapping=None,
                       tdr_mappings=None):
        """
        Set checkshot / time-depth relationship data.

        Args:
            checkshot_data: Dict of checkshot records.
            checkshot_mapping: Optional DataFrame mapping wells to checkshots.
            tdr_mappings: Optional dict mapping wells to TDR sources.
        """
        self.well_handler.loaded_data['checkshot'] = checkshot_data
        if checkshot_mapping is not None:
            self.well_handler.loaded_data['checkshot_mapping'] = checkshot_mapping
        if tdr_mappings is not None:
            self.well_handler.loaded_data['tdr_mappings'] = tdr_mappings

    # ------------------------------------------------------------------
    # Well logs
    # ------------------------------------------------------------------

    def import_well_logs(self, well_name, las_path):
        """
        Import well logs from a single LAS file.

        Args:
            well_name: Well identifier to store the logs under.
            las_path: Path to the LAS file.

        Returns:
            True on success.
        """
        return self.well_log_handler.import_well_logs(well_name, las_path)

    def import_wells_from_las(self, las_dir, well_heads_df=None):
        """
        Batch-import all LAS files from a directory.

        Optionally sets well_heads from a provided DataFrame. Well names
        are extracted from the LAS WELL header; falls back to filename.

        Args:
            las_dir: Directory containing .las files.
            well_heads_df: Optional DataFrame to set as well heads.

        Returns:
            List of imported well names.
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required for LAS import. "
                              "Install with: pip install lasio")

        if well_heads_df is not None:
            self.set_well_heads(well_heads_df)

        # Collect LAS files; deduplicate for case-insensitive filesystems
        las_files = sorted(set(
            glob.glob(os.path.join(las_dir, '*.las')) +
            glob.glob(os.path.join(las_dir, '*.LAS'))
        ))

        imported = []
        for las_file in las_files:
            well_name = self._extract_well_name(las_file)
            self.well_log_handler.import_well_logs(well_name, las_file)
            imported.append(well_name)
            logger.info("Imported well logs: %s from %s", well_name, las_file)

        return imported

    @staticmethod
    def _extract_well_name(las_path):
        """Extract well name from LAS WELL header, falling back to filename."""
        try:
            las = lasio.read(las_path)
            name = las.well['WELL'].value
            if name and str(name).strip():
                return str(name).strip()
        except Exception:
            pass
        return os.path.splitext(os.path.basename(las_path))[0]

    # ------------------------------------------------------------------
    # Wavelets
    # ------------------------------------------------------------------

    def add_wavelet(self, name, data, metadata=None):
        """
        Add a wavelet to the project.

        Args:
            name: Wavelet display name (e.g. "Ricker_25Hz").
            data: 1-D numpy array (amplitudes) or 2-D array (time, amplitude).
            metadata: Optional dict with wavelet metadata (type, frequency,
                     sample_rate, length, etc.).
        """
        self.seismic_handler.loaded_data['wavelets'].append({
            'name': name,
            'data': np.asarray(data),
            'metadata': metadata or {},
        })

    # ------------------------------------------------------------------
    # Horizons
    # ------------------------------------------------------------------

    def import_horizons(self, file_paths):
        """
        Import horizons from ZMAP grid files.

        Args:
            file_paths: List of paths to .zmap files.

        Returns:
            Dict of imported horizons.
        """
        return self.horizons_handler.read_horizons(file_paths)

    def set_horizons(self, horizons_dict):
        """
        Set horizon data directly.

        Args:
            horizons_dict: Dict mapping horizon names to
                          {'X': array, 'Y': array, 'Z': array}.
        """
        self.horizons_handler.loaded_data['horizons'] = horizons_dict

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """
        Finalize the project: write .str file and save all handler data.

        The .str file is placed alongside the project folder (same parent
        directory) with the same name plus '.str' extension.

        Returns:
            Path to the generated .str file.
        """
        # Write the .str pointer file
        str_path = self.project_folder + '.str'
        with open(str_path, 'w') as f:
            f.write("Seismic Data Viewer Project")

        # Flush all handler data to disk
        self.well_handler.save_project(self.project_folder)
        self.well_log_handler.save_project(self.project_folder)
        self.seismic_handler.save_project(self.project_folder)
        self.horizons_handler.save_project(self.project_folder)

        logger.info("Project built: %s", str_path)
        safe_print(f"Project built successfully: {str_path}")

        return str_path

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self):
        """Return a summary dict of what's in the project so far."""
        surveys = [s['survey_name'] for s in self.seismic_handler.project_files]
        volumes = {}
        for s in self.seismic_handler.project_files:
            volumes[s['survey_name']] = [
                v['attribute_name'] for v in s['survey_volumes']
            ]

        well_heads = self.well_handler.loaded_data.get('well_heads')
        n_wells = len(well_heads) if well_heads is not None else 0

        well_logs = self.well_log_handler.loaded_data.get('well_logs', {})
        n_logs = len(well_logs)

        n_wavelets = len(self.seismic_handler.loaded_data.get('wavelets', []))

        horizons = self.horizons_handler.loaded_data.get('horizons', {})
        n_horizons = len(horizons)

        deviation = self.well_handler.loaded_data.get('deviation', {})
        n_deviations = len(deviation)

        return {
            'project_name': self.project_name,
            'project_folder': self.project_folder,
            'surveys': surveys,
            'volumes': volumes,
            'n_wells': n_wells,
            'n_well_logs': n_logs,
            'n_wavelets': n_wavelets,
            'n_horizons': n_horizons,
            'n_deviations': n_deviations,
        }

    def __repr__(self):
        s = self.summary()
        return (
            f"ProjectBuilder('{s['project_name']}', "
            f"surveys={len(s['surveys'])}, wells={s['n_wells']}, "
            f"logs={s['n_well_logs']}, wavelets={s['n_wavelets']}, "
            f"horizons={s['n_horizons']})"
        )
