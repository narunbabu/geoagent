"""
GeoAgent — Open-Source Geoscience Data Toolkit

Load well data, generate synthetic seismograms, create professional
well log correlation sections, and plot location maps in pure Python.
"""

__version__ = "0.1.0"
__author__ = "Arun Babu Nalamara"
__license__ = "Apache-2.0"

# Convenience imports for common usage
from geoagent.core.data_manager import CoreDataManager
from geoagent.io.project_builder import ProjectBuilder
from geoagent.plotting.config import SectionPlotConfig, FormationTop
