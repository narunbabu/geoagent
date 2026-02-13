"""
Log Naming Settings Module
Handles configurable log names for GeoAgent.
"""


class LogNamingSettings:
    """
    Manages configurable log names for the application.
    Allows users to define custom log names for different projects.
    """

    # Default log name configurations
    DEFAULT_LOG_NAMES = {
        'sonic_log': ['DTC', 'DT', 'SONIC', 'SLOWNESS'],
        'density_log': ['RHOB', 'RHOZ', 'DENSITY', 'BULK_DENSITY'],
        'gamma_ray': ['GR', 'GAMMA', 'GAMMA_RAY', 'GAPI'],
        'deep_resistivity': ['LLD', 'DEEP_RES', 'RT', 'RESISTIVITY'],
        'porosity': ['NPHI', 'NEUTRON', 'PHIN', 'POROSITY']
    }

    # Essential logs required for preprocessing
    ESSENTIAL_LOGS = ['sonic_log', 'density_log']

    # Additional logs for future prediction
    ADDITIONAL_LOGS = ['gamma_ray', 'deep_resistivity', 'porosity']

    def __init__(self):
        """Initialize with default settings"""
        self.log_names = self.DEFAULT_LOG_NAMES.copy()
        self.project_settings = {}

    def get_log_names_for_type(self, log_type):
        """Get the list of possible names for a log type."""
        return self.log_names.get(log_type, [])

    def set_log_names_for_type(self, log_type, names):
        """Set the possible names for a log type."""
        if isinstance(names, str):
            names = [names]
        self.log_names[log_type] = names

    def add_log_name_variant(self, log_type, name):
        """Add a new variant name for a log type."""
        if log_type not in self.log_names:
            self.log_names[log_type] = []
        if name not in self.log_names[log_type]:
            self.log_names[log_type].append(name)

    def find_log_column(self, log_type, available_columns):
        """
        Find the best matching column for a log type from available columns.

        Args:
            log_type: The log type to find (e.g., 'sonic_log', 'density_log')
            available_columns: List of available column names

        Returns:
            The best matching column name, or None if not found
        """
        if available_columns is None:
            return None

        if not isinstance(available_columns, (list, tuple, set)):
            return None

        possible_names = self.get_log_names_for_type(log_type)

        # Direct match first
        for name in possible_names:
            if name in available_columns:
                return name

        # Case-insensitive match
        for name in possible_names:
            for col in available_columns:
                if name.upper() == col.upper():
                    return col

        # Partial match (contains)
        for name in possible_names:
            for col in available_columns:
                if name.upper() in col.upper() or col.upper() in name.upper():
                    return col

        return None

    def get_all_possible_names(self, log_type):
        """Get all possible names for a log type including variations."""
        base_names = self.get_log_names_for_type(log_type)
        all_names = base_names.copy()

        for name in base_names:
            if name.lower() not in [n.lower() for n in all_names]:
                all_names.append(name.lower())
            if not any(name.upper() + '_' in n.upper() for n in all_names):
                all_names.extend([name + '_US', name + '_USFT'])

        return all_names

    def validate_log_configuration(self, log_data):
        """
        Validate that essential logs are available in the data.

        Returns:
            dict with 'valid', 'found_logs', 'missing_logs', 'warnings'
        """
        results = {
            'valid': True,
            'found_logs': {},
            'missing_logs': [],
            'warnings': []
        }

        if not log_data:
            results['valid'] = False
            results['missing_logs'] = self.ESSENTIAL_LOGS
            return results

        available_columns = list(log_data.keys()) if hasattr(log_data, 'keys') else []

        for log_type in self.ESSENTIAL_LOGS:
            found_column = self.find_log_column(log_type, available_columns)
            if found_column:
                results['found_logs'][log_type] = found_column
            else:
                results['missing_logs'].append(log_type)
                results['valid'] = False

        for log_type in self.ADDITIONAL_LOGS:
            found_column = self.find_log_column(log_type, available_columns)
            if found_column:
                results['found_logs'][log_type] = found_column
            else:
                results['warnings'].append(f"Optional log type '{log_type}' not found")

        return results

    def get_project_settings(self, project_name):
        """Get log naming settings for a specific project."""
        return self.project_settings.get(project_name, self.log_names.copy())

    def set_project_settings(self, project_name, settings):
        """Set log naming settings for a specific project."""
        self.project_settings[project_name] = settings

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.log_names = self.DEFAULT_LOG_NAMES.copy()
        self.project_settings = {}

    def to_dict(self):
        """Convert settings to dictionary for serialization."""
        return {
            'log_names': self.log_names,
            'project_settings': self.project_settings
        }

    def from_dict(self, data):
        """Load settings from dictionary."""
        self.log_names = data.get('log_names', self.DEFAULT_LOG_NAMES.copy())
        self.project_settings = data.get('project_settings', {})

    def get_all_log_types(self):
        """Get all available log types."""
        return self.log_names.copy()
