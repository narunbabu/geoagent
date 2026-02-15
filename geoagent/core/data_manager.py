"""
CoreDataManager - Qt-free data management facade for GeoAgent.

This is the central data access layer. It wraps the four specialized handlers
(seismic, well, well_log, horizons) and provides a unified API for loading
projects, querying data, and resolving log mnemonics.

Unlike SeisTrans's DataManager (which inherits QObject and uses pyqtSignal),
this implementation is pure Python with no GUI dependencies.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd

from geoagent.core.seismic_handler import SeismicHandler
from geoagent.core.well_handler import WellHandler
from geoagent.core.well_log_handler import WellLogHandler
from geoagent.core.horizons_handler import HorizonsHandler
from geoagent.settings.log_naming_settings import LogNamingSettings
from geoagent.utils.safe_print import safe_print

logger = logging.getLogger(__name__)


# Project-level pickle files managed by CoreDataManager directly
_DEFAULT_PROJECT_FILES = {
    'selection_indices': 'selection_indices.pkl',
    'well_settings': 'well_settings.pkl',
    'synthetic_settings': 'synthetic_settings.pkl',
    'import_settings': 'import_settings.pkl',
    'log_naming_settings': 'log_naming_settings.pkl',
}


class CoreDataManager:
    """
    Qt-free central data facade for GeoAgent.

    Wraps SeismicHandler, WellHandler, WellLogHandler, and HorizonsHandler.
    Provides the same data-access interface consumed by synthetic_functions
    and bulk_shift without any PyQt6 dependency.
    """

    def __init__(self, project_folder=None):
        """
        Initialize CoreDataManager.

        Args:
            project_folder: Path to the project directory. If provided,
                           calls load_project() automatically.
        """
        self.project_folder = project_folder or ''
        self.loaded_data = {}
        self.project_files = dict(_DEFAULT_PROJECT_FILES)

        # Initialize specialized handlers
        self.seismic_handler = SeismicHandler(self.project_folder)
        self.well_handler = WellHandler(self.project_folder)
        self.well_log_handler = WellLogHandler(self.project_folder)
        self.horizons_handler = HorizonsHandler(self.project_folder)

        # Synthetic results cache (in-memory)
        self.synthetic_results = {}

        if project_folder:
            self.load_project(project_folder)

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def load_project(self, project_folder):
        """
        Load a project from disk.

        Delegates to each handler's load_project() and loads
        CoreDataManager's own pickle files (well_settings, synthetic_settings, etc.).
        """
        self.project_folder = project_folder

        # Load handler data
        self.well_handler.load_project(project_folder)
        self.well_log_handler.load_project(project_folder)
        self.seismic_handler.load_project(project_folder)
        self.horizons_handler.load_project(project_folder)

        # Load CoreDataManager-level pickles
        for data_type, file_name in self.project_files.items():
            file_path = os.path.join(project_folder, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        self.loaded_data[data_type] = pickle.load(f)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", data_type, e)

        logger.info("Project loaded from %s", project_folder)

    def save_project(self, project_folder=None):
        """Save project state to disk."""
        folder = project_folder or self.project_folder
        if not folder:
            logger.warning("No project folder set — cannot save.")
            return

        # Save handler data
        self.well_handler.save_project(folder)
        self.well_log_handler.save_project(folder)
        self.seismic_handler.save_project(folder)
        self.horizons_handler.save_project(folder)

        # Save CoreDataManager-level pickles
        for data_type, file_name in self.project_files.items():
            data = self.loaded_data.get(data_type)
            if data:
                file_path = os.path.join(folder, file_name)
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                except Exception as e:
                    logger.warning("Failed to save %s: %s", data_type, e)

        self.project_folder = folder
        logger.info("Project saved to %s", folder)

    # ------------------------------------------------------------------
    # Unified data access (multi-handler fallback chain)
    # ------------------------------------------------------------------

    def get_data(self, data_type):
        """
        Retrieve data by type, searching across all handlers.

        Search order: CoreDataManager → well_handler → well_log_handler
                      → seismic_handler → horizons_handler
        """
        if data_type in self.loaded_data:
            return self.loaded_data[data_type]
        if data_type in self.well_handler.loaded_data:
            return self.well_handler.get_data(data_type)
        if data_type in self.well_log_handler.loaded_data:
            return self.well_log_handler.get_data(data_type)
        if data_type in self.seismic_handler.loaded_data:
            return self.seismic_handler.get_data(data_type)
        if data_type in self.horizons_handler.loaded_data:
            return self.horizons_handler.get_data(data_type)
        return self.loaded_data.get(data_type)

    # ------------------------------------------------------------------
    # Log naming / mnemonic resolution
    # ------------------------------------------------------------------

    def get_log_naming_settings(self):
        """Get the LogNamingSettings instance, loaded from project data if available."""
        log_settings = LogNamingSettings()
        stored = self.loaded_data.get('log_naming_settings', {})
        if stored:
            log_settings.from_dict(stored)
        return log_settings

    def set_log_naming_settings(self, log_settings):
        """Persist log naming settings."""
        self.loaded_data['log_naming_settings'] = log_settings.to_dict()

    def find_log_column(self, log_type, well_name):
        """
        Find the column name for a specific log type in a well's data.

        Args:
            log_type: e.g. 'sonic_log', 'density_log'
            well_name: well identifier

        Returns:
            Column name string, or None
        """
        log_settings = self.get_log_naming_settings()
        well_logs = self.well_log_handler.get_well_logs(well_name)
        if well_logs is None or (hasattr(well_logs, 'empty') and well_logs.empty):
            return None
        columns = well_logs.columns.tolist() if hasattr(well_logs, 'columns') else list(well_logs.keys())
        return log_settings.find_log_column(log_type, columns)

    # ------------------------------------------------------------------
    # Synthetic settings
    # ------------------------------------------------------------------

    def get_synthetic_settings_all(self):
        """Return the full synthetic_settings dict (or None)."""
        return self.get_data('synthetic_settings')

    def get_synthetic_settings(self, well_name):
        """Return synthetic settings for a single well."""
        settings = self.get_data('synthetic_settings')
        if settings and 'well_settings' in settings:
            return settings['well_settings'].get(well_name)
        return None

    # ------------------------------------------------------------------
    # Well settings
    # ------------------------------------------------------------------

    def load_well_settings(self, well_name):
        """Load stored settings for a specific well."""
        ws = self.loaded_data.get('well_settings', {})
        return ws.get(well_name) if isinstance(ws, dict) else None

    def get_default_well_settings(self, well_name):
        """Return sensible default settings for a well."""
        return {
            'wavelet': None,
            'bulk_shift': 0.0,
            'sampling_interval': 2.0,
            'time_range': (1400, 1600),
        }

    # ------------------------------------------------------------------
    # Synthetic result caching
    # ------------------------------------------------------------------

    def store_synthetic_result(self, well_name, result):
        """Cache a synthetic generation result in memory."""
        self.synthetic_results[well_name] = result

    def get_synthetic_result(self, well_name):
        """Retrieve a cached synthetic result."""
        return self.synthetic_results.get(well_name)

    def save_synthetic_results(self, file_path):
        """Persist all synthetic results to a pickle file."""
        if not self.synthetic_results:
            return False
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.synthetic_results, f)
            return True
        except Exception as e:
            logger.error("Error saving synthetic results: %s", e)
            return False

    def load_synthetic_results(self, file_path):
        """Load synthetic results from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                self.synthetic_results = pickle.load(f)
            return True
        except Exception as e:
            logger.error("Error loading synthetic results: %s", e)
            return False

    # ------------------------------------------------------------------
    # Seismic data extraction
    # ------------------------------------------------------------------

    def get_seismic_data(self, well_name, required_attributes,
                         use_spatial_sampling=False,
                         well_trajectory_data=None,
                         spatial_sampling_params=None,
                         parent_widget=None):
        """
        Extract seismic data for a well.

        Finds the survey containing the requested attributes, locates the
        nearest trace to the well's surface coordinates, and returns a
        DataFrame with seismic amplitudes indexed by TWT.

        Args:
            well_name: Well identifier
            required_attributes: List of seismic attribute names
            use_spatial_sampling: Reserved for spatial sampling (not yet implemented in OSS)
            well_trajectory_data: DataFrame with well trajectory
            spatial_sampling_params: Dict with spatial sampling config
            parent_widget: Ignored (kept for API compatibility)

        Returns:
            pandas DataFrame with TWT column + attribute columns, or empty DataFrame
        """
        # Find survey with requested attributes
        survey_name, available_attributes, missing_attributes = \
            self.find_survey_with_attributes(required_attributes)

        if missing_attributes:
            safe_print(f"Warning: Survey '{survey_name}' missing attributes: {missing_attributes}")

        all_survey_attributes = self.seismic_handler.get_available_attributes(survey_name)
        if not all_survey_attributes:
            raise ValueError(f"No attributes available for survey '{survey_name}'")

        well_data = self.well_handler.get_well_data(well_name)
        if well_data.empty:
            safe_print(f"No well data available for well {well_name}")
            return pd.DataFrame()

        well_x = float(well_data['Surface X'].iloc[0])
        well_y = float(well_data['Surface Y'].iloc[0])

        # Find nearest inline/crossline for this well
        inline, crossline, _trace_index = \
            self.seismic_handler.xy_to_inline_crossline(survey_name, well_x, well_y)

        if inline is None or crossline is None:
            safe_print(f"Could not find inline/crossline for well {well_name}")
            return pd.DataFrame()

        # Extract traces for each requested attribute
        seismic_data = {}
        time_array = None

        for attr_name in required_attributes:
            if attr_name not in all_survey_attributes:
                safe_print(f"Warning: attribute '{attr_name}' not in survey '{survey_name}'")
                continue

            trace_data, time_array, _ = self.seismic_handler.get_trace_data(
                survey_name, attr_name, inline, crossline)

            if trace_data is None or time_array is None:
                continue

            seismic_data[attr_name] = trace_data[0]

        if not seismic_data or time_array is None:
            safe_print(f"No seismic data extracted for well {well_name}")
            return pd.DataFrame()

        seismic_df = pd.DataFrame(seismic_data, index=time_array)
        seismic_df.index.name = 'TWT'

        return seismic_df.reset_index()

    def find_survey_with_attributes(self, required_attributes):
        """
        Find the survey containing the most requested attributes.

        Returns:
            (survey_name, available_attributes, missing_attributes)
        """
        surveys = self.seismic_handler.get_available_surveys()
        if not surveys:
            raise ValueError("No seismic surveys available")

        best_survey = None
        best_match_count = 0
        best_available = []
        best_missing = list(required_attributes)

        for survey_name in surveys:
            if 'wavelet' in survey_name.lower():
                continue
            attrs = self.seismic_handler.get_available_attributes(survey_name)
            if not attrs:
                continue
            matched = [a for a in required_attributes if a in attrs]
            if len(matched) > best_match_count:
                best_match_count = len(matched)
                best_survey = survey_name
                best_available = matched
                best_missing = [a for a in required_attributes if a not in attrs]

        if best_survey is None:
            best_survey = surveys[0]
            best_available = []
            best_missing = list(required_attributes)

        return best_survey, best_available, best_missing

    # ------------------------------------------------------------------
    # Convenience delegations
    # ------------------------------------------------------------------

    def get_well_tdr(self, well_name):
        """Get TDR (time-depth relationship) data for a well."""
        return self.well_handler.get_well_tdr(well_name)

    def get_well_data(self, well_name):
        """Get well header data (surface coordinates, KB, etc.)."""
        return self.well_handler.get_well_data(well_name)

    def get_available_wells(self):
        """Get list of available well names."""
        well_heads = self.well_handler.get_data('well_heads')
        if well_heads is not None and not well_heads.empty:
            # SeisTrans format: Name is a column, index is integer
            if 'Name' in well_heads.columns:
                return list(well_heads['Name'])
            return list(well_heads.index)
        return []

    def get_available_surveys(self):
        """Get list of available seismic surveys."""
        return self.seismic_handler.get_available_surveys()

    def __repr__(self):
        wells = len(self.get_available_wells())
        surveys = len(self.get_available_surveys())
        return f"CoreDataManager(folder='{self.project_folder}', wells={wells}, surveys={surveys})"
