# datahandlers/well_log_handler.py
import os, sys
import pandas as pd
import numpy as np
import pickle
import lasio
from scipy import interpolate

default_projectfiles = {
    'well_logs': 'well_logs.pkl'
}

class WellLogHandler:
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.loaded_data = {}
        self.project_files = default_projectfiles
        self.mnemonics_map = self.load_mnemonics_map()
        self.loaded_data['upscaled_logs'] = {}
        self.loaded_data['upscale_parameters'] = {}
        self.loaded_data['edited_well_logs'] = {}

    def resource_path(self,relative_path):
        """
        Get absolute path to resource, works for dev and for PyInstaller.

        :param relative_path: Path relative to the script's directory.
        :return: Absolute path to the resource.
        """
        try:
            # PyInstaller creates a temporary folder and stores its path in _MEIPASS
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)
    def save_edited_logs(self, well_name, log_df):
        if 'edited_well_logs' not in self.loaded_data:
            self.loaded_data['edited_well_logs'] = {}
        self.loaded_data['edited_well_logs'][well_name] = log_df
        self.save_project(self.project_folder)
    def get_edited_well_logs_for_well(self, well_name):
        if 'edited_well_logs' in self.loaded_data and well_name in self.loaded_data['edited_well_logs']:
            return self.loaded_data['edited_well_logs'][well_name]
        return None
    def load_mnemonics_map(self):
        mnemonics_map = {}
        mnemonics_file = self.resource_path('assets\\mnemonics.txt')
        
        with open(mnemonics_file, 'r') as f:
            for line in f:
                parts = line.strip().split('=')
                if len(parts) == 2:
                    standard_name = parts[0].strip()
                    aliases = [alias.strip() for alias in parts[1].split()]
                    for alias in aliases:
                        mnemonics_map[alias] = standard_name
        
        return mnemonics_map
    def delete_upscaled_logs(self, well_name):
        upscaled_logs = self.loaded_data.get('upscaled_logs', {})
        upscale_parameters = self.loaded_data.get('upscale_parameters', {})
        
        if well_name in upscaled_logs:
            del upscaled_logs[well_name]
            if well_name in upscale_parameters:
                del upscale_parameters[well_name]
            self.save_project(self.project_folder)
            return True
        return False
    def get_existing_logs(self, well_name):
        return list(self.loaded_data.get('well_logs', {}).get(well_name, {}).keys())
    def remove_well_logs(self, well_name):
        if 'well_logs' in self.loaded_data and well_name in self.loaded_data['well_logs']:
            del self.loaded_data['well_logs'][well_name]
    def import_selected_logs(self, well_name, file_path, selected_logs):
        las_file = lasio.read(file_path)
        new_log_data = {}
        print( selected_logs)
        
        # Find depth curve from the original LAS file
        depth_curve_original = None
        for curve in las_file.curves:
            if curve.mnemonic.upper() in ['DEPT', 'DEPTH', 'MD']:
                depth_curve_original = curve
                break
        
        if depth_curve_original == None:
            curve_names = [curve.mnemonic for curve in las_file.curves]
            raise ValueError(f"No depth curve (DEPT, DEPTH, or MD) found in the LAS file. Available curves: {', '.join(curve_names[:10])}")
        
        # Process selected logs
        for curve in las_file.curves:
            if curve.mnemonic in selected_logs:
                standard_name = selected_logs.get(curve.mnemonic, curve.mnemonic)
                new_log_data[standard_name] = curve.data
        
        print(f"Imported log data keys: {list(new_log_data.keys())}")
        
        # Find depth curve in the processed data
        depth_curve = next((curve for curve in new_log_data.keys() if curve.upper() in ['DEPT', 'DEPTH', 'MD']), None)
        if depth_curve is None:
            raise ValueError("No depth curve found in the processed log data")

        if 'well_logs' not in self.loaded_data:
            self.loaded_data['well_logs'] = {}
        if well_name not in self.loaded_data['well_logs']:
            self.loaded_data['well_logs'][well_name] = {}

        existing_logs = self.loaded_data['well_logs'][well_name]
        existing_depth = existing_logs.get(depth_curve)

        if existing_depth is not None:
            # Extend depth range if necessary
            min_depth = min(existing_depth.min(), new_log_data[depth_curve].min())
            max_depth = max(existing_depth.max(), new_log_data[depth_curve].max())
            new_depth = np.arange(min_depth, max_depth + 0.1, 0.1)  # 0.1 step, adjust as needed

            # Resample existing logs
            for log_name, log_data in existing_logs.items():
                f = interpolate.interp1d(existing_depth, log_data, bounds_error=False, fill_value=np.nan)
                existing_logs[log_name] = f(new_depth)

            # Resample and add new logs
            for log_name, log_data in new_log_data.items():
                if log_name != depth_curve:
                    f = interpolate.interp1d(new_log_data[depth_curve], log_data, bounds_error=False, fill_value=np.nan)
                    new_log_name = self.get_unique_log_name(existing_logs, log_name)
                    existing_logs[new_log_name] = f(new_depth)

            existing_logs[depth_curve] = new_depth
        else:
            # If no existing logs, just add the new logs
            self.loaded_data['well_logs'][well_name] = new_log_data

        self.save_project(self.project_folder)
        return list(new_log_data.keys())  # Return only the newly added logs
    
    def get_wells_with_logs(self):
        """Get a list of wells that have logs"""
        well_logs = self.get_data('well_logs')
        return list(well_logs.keys()) if well_logs else []
    def get_unique_log_name(self, existing_logs, log_name):
        if log_name not in existing_logs:
            return log_name
        i = 1
        while f"{log_name}_{i}" in existing_logs:
            i += 1
        return f"{log_name}_{i}"
    def parse_file(self, file_path, parse_function):
        data = parse_function(file_path)
        return data

    def parse_well_log_file(self, file_path):
        las_file = lasio.read(file_path)
        well_log_data = {curve.mnemonic: curve.data for curve in las_file.curves}
        return well_log_data


    def get_all_logs_for_well(self, well_name):
        well_logs = self.get_data('well_logs')
        if well_name not in well_logs:
            return []
        return [log for log in well_logs[well_name].keys() if log.upper() not in ['DEPT', 'DEPTH', 'MD']]

    def delete_log(self, well_name, log_name):
        well_logs = self.get_data('well_logs')
        if well_name in well_logs and log_name in well_logs[well_name]:
            del well_logs[well_name][log_name]
            self.save_project(self.project_folder)
            return True
        return False
    def delete_all_logs_for_well(self, well_name):
        if 'well_logs' in self.loaded_data and well_name in self.loaded_data['well_logs']:
            del self.loaded_data['well_logs'][well_name]
            self.save_project(self.project_folder)
            return True
        return False
    def get_well_logs(self, well_name):
        """
        Get well logs for a specific well name.
        This method is used by DataManager and should return a pandas DataFrame.
        """
        return self.get_well_logs_for_well(well_name)

    def get_well_logs_for_well(self, well_name):
        well_logs = self.get_data('well_logs')
        
        if well_name not in well_logs:
            print(f"Well '{well_name}' not found in well logs database")
            return None

        well_log = well_logs[well_name]
        depth_curve = next((curve for curve in well_log.keys() if curve.upper() in ['DEPT', 'DEPTH', 'MD']), None)
	

        if depth_curve is None:
            print(f"No depth curve found for well {well_name}")
            return None
        
        try:
            log_df = pd.DataFrame({
                'DEPTH': well_log[depth_curve],
                **{self.mnemonics_map.get(curve, curve): well_log[curve] for curve in well_log.keys() if curve != depth_curve}
            })
            return log_df
        except:
            print("All arrays must be of the same length")
            return None

    def get_upscaled_well_logs_for_well(self, well_name):
        upscaled_logs = self.get_upscaled_logs(well_name)
        if upscaled_logs is not None:
            return upscaled_logs
        print("No Upscaled data is available")
        return pd.DataFrame({'DEPTH': [0, 1], 'GR': [0, 0]})
        
    def import_well_logs(self, well_name, file_path):
        """Import well logs from LAS file for a specific well"""
        try:
            # Parse the LAS file
            well_log_data = self.parse_well_log_file(file_path)
            
            # Initialize well_logs if not exists
            if 'well_logs' not in self.loaded_data:
                self.loaded_data['well_logs'] = {}
            
            # Store the data for the specific well
            self.loaded_data['well_logs'][well_name] = well_log_data
            
            # Save the project
            self.save_project(self.project_folder)
            
            print(f"Successfully imported well logs for {well_name}")
            return True
            
        except Exception as e:
            print(f"Error importing well logs for {well_name}: {str(e)}")
            raise e

    def import_file(self, data_type, file_path):
        if data_type == 'well_logs':
            self.loaded_data[data_type] = self.parse_well_log_file(file_path)
        return data_type, file_path


    def load_project(self, project_folder):
        self.project_folder = project_folder
        for data_type in ['well_logs', 'upscaled_logs', 'upscale_parameters', 'edited_well_logs']:
            file_path = os.path.join(project_folder, f"{data_type}.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.loaded_data[data_type] = pickle.load(f)


    def store_upscaled_logs(self, well_name, time_upscaled_df, parameters):
        # Initialize upscaled_logs and upscale_parameters if they don't exist
        if 'upscaled_logs' not in self.loaded_data:
            self.loaded_data['upscaled_logs'] = {}
        if 'upscale_parameters' not in self.loaded_data:
            self.loaded_data['upscale_parameters'] = {}
            
        self.loaded_data['upscaled_logs'][well_name] = time_upscaled_df
        self.loaded_data['upscale_parameters'][well_name] = parameters
        self.save_project(self.project_folder)

    def get_upscaled_logs(self, well_name):
        # Safely get upscaled logs, return None if key doesn't exist
        upscaled_logs = self.loaded_data.get('upscaled_logs', {})
        result = upscaled_logs.get(well_name)
        # Uncomment for debugging: print(f"[DEBUG] get_upscaled_logs for {well_name}: {result is not None}")
        return result

    def get_upscale_parameters(self, well_name):
        # Safely get upscale parameters, return None if key doesn't exist
        upscale_parameters = self.loaded_data.get('upscale_parameters', {})
        result = upscale_parameters.get(well_name)
        # Uncomment for debugging: print(f"[DEBUG] get_upscale_parameters for {well_name}: {result is not None}")
        return result


    def get_data(self, data_type):
        return self.loaded_data.get(data_type)

    def get_available_datatypes(self):
        return [data_type for data_type in self.project_files if data_type in self.loaded_data]


    # ----- Unit lookup for LAS export -----
    _LOG_UNITS = {
        'GR': 'API', 'SGR': 'API', 'CGR': 'API',
        'LLD': 'OHMM', 'LLS': 'OHMM', 'MSFL': 'OHMM', 'ILD': 'OHMM', 'ILM': 'OHMM',
        'RHOB': 'G/CC', 'RHOZ': 'G/CC',
        'NPHI': 'V/V', 'TNPH': 'V/V',
        'DT': 'US/FT', 'DTC': 'US/FT', 'DTS': 'US/FT',
        'SP': 'MV', 'CAL': 'IN', 'PE': 'B/E',
        'DEPTH': 'M', 'MD': 'M', 'DEPT': 'M',
    }

    def export_to_las(self, well_name, output_path, well_head_info=None):
        """
        Export raw well logs to LAS 2.0 format.

        Args:
            well_name: Well name as stored in well_logs
            output_path: Full path for the output .las file
            well_head_info: Optional dict or Series with keys like
                           X, Y, KB, Name, Field, Company for header

        Returns:
            output_path on success, None on failure
        """
        well_logs = self.get_data('well_logs')
        if well_logs is None or well_name not in well_logs:
            print(f"Well '{well_name}' not found in well logs")
            return None

        logs = well_logs[well_name]

        # Find depth curve
        depth_key = None
        for k in ('DEPTH', 'MD', 'DEPT'):
            if k in logs:
                depth_key = k
                break
        if depth_key is None:
            print(f"No depth curve found for well {well_name}")
            return None

        depth = logs[depth_key]

        # Build LAS
        las = lasio.LASFile()
        las.well['WELL'].value = well_name
        las.well['STRT'].value = float(np.nanmin(depth))
        las.well['STRT'].unit = 'M'
        las.well['STOP'].value = float(np.nanmax(depth))
        las.well['STOP'].unit = 'M'
        las.well['NULL'].value = -999.25

        # Compute step
        finite_depth = depth[np.isfinite(depth)]
        if len(finite_depth) > 1:
            las.well['STEP'].value = float(np.median(np.diff(finite_depth)))
        else:
            las.well['STEP'].value = 0.0
        las.well['STEP'].unit = 'M'

        # Optional header metadata
        if well_head_info is not None:
            hi = well_head_info if isinstance(well_head_info, dict) else well_head_info.to_dict()
            for las_key, source_keys in [
                ('COMP', ['Company', 'COMP']),
                ('FLD', ['Field', 'FLD']),
            ]:
                for sk in source_keys:
                    if sk in hi and hi[sk] is not None and str(hi[sk]).strip():
                        las.well[las_key].value = str(hi[sk])
                        break
            # Coordinates as params
            for mnem, source_keys, descr in [
                ('XCOORD', ['X', 'x', 'Easting'], 'X coordinate'),
                ('YCOORD', ['Y', 'y', 'Northing'], 'Y coordinate'),
                ('KB', ['KB', 'kb', 'Kelly Bushing'], 'Kelly Bushing elevation'),
            ]:
                for sk in source_keys:
                    if sk in hi and hi[sk] is not None:
                        try:
                            val = float(hi[sk])
                            las.params[mnem] = lasio.HeaderItem(
                                mnemonic=mnem, value=val, unit='M', descr=descr)
                            break
                        except (ValueError, TypeError):
                            pass

        # Add depth curve
        las.append_curve('DEPTH', depth, unit='M', descr='Measured Depth')

        # Add log curves (sorted, skip depth)
        log_curves = sorted(k for k in logs if k != depth_key)
        exported = []
        for curve in log_curves:
            vals = logs[curve]
            if len(vals) != len(depth):
                print(f"  WARNING: {curve} length mismatch ({len(vals)} vs {len(depth)}) — skipped")
                continue
            # Map to standard name via mnemonics
            std_name = self.mnemonics_map.get(curve, curve)
            unit = self._LOG_UNITS.get(std_name, self._LOG_UNITS.get(curve, ''))
            # Replace NaN with null value
            arr = np.where(np.isfinite(vals), vals, -999.25)
            las.append_curve(std_name, arr, unit=unit, descr=curve)
            exported.append(std_name)

        # Write
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        las.write(output_path, version=2.0)
        print(f"Exported {well_name}: {len(exported)} curves -> {output_path}")
        return output_path

    def save_project(self, project_folder):
        if project_folder is None:
            print("Warning: project_folder is None. Cannot save project.")
            return

        for data_type in ['well_logs', 'upscaled_logs', 'upscale_parameters', 'edited_well_logs']:
            data = self.loaded_data.get(data_type, {})
            if data:
                pickle_path = os.path.join(project_folder, f"{data_type}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(data, f)
        
        self.project_folder = project_folder
        print(f"Project saved successfully in {project_folder}")



    
