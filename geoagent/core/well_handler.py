# datahandlers/well_handler.py
import os, re
import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import interp1d
from tqdm import tqdm
from datetime import datetime
import traceback  # Add this line to import the traceback module


# Robust Petrel well head parsing helpers
import shlex
from typing import List, Dict

# Debug configuration — graceful fallback if not available
try:
    from geoagent.utils.debug_config import should_log_debug, mark_debug_logged, debug_config
except ImportError:
    def should_log_debug(category, key=None): return False
    def mark_debug_logged(category, key=None): pass
    class _DebugConfig:
        def get_level(self): return 'MINIMAL'
    debug_config = _DebugConfig()
from geoagent.utils.safe_print import safe_print

# DMS regex pattern for Petrel coordinates: "22 58'26.7807"N"
DMS_RX = re.compile(r'(?P<deg>\d+)\s+(?P<min>\d+)\'(?P<sec>[\d.]+)\"(?P<hem>[NSEW])')

def dms_to_dd(deg: str, min: str, sec: str, hem: str) -> float:
    """Convert DMS components to decimal degrees"""
    d = int(deg)
    m = int(min)
    s = float(sec)
    dd = d + m/60.0 + s/3600.0
    return -dd if hem in "SW" else dd

def split_petrel_row(row: str) -> List[str]:
    """
    Two-pass tokenization for Petrel well head rows:
    1. Use shlex to handle quoted strings properly
    2. Reconstruct DMS coordinates from split tokens
    """
    try:
        tokens = shlex.split(row, posix=False)  # Keep quoted strings intact
        
        # Check if we have enough tokens for latitude/longitude reconstruction
        if len(tokens) < 9:
            safe_print(f"Warning: Insufficient tokens in row, got {len(tokens)}: {row[:100]}...")
            return tokens
        
        # Reconstruct latitude and longitude from split tokens
        # tokens[5] = "22", tokens[6] = "58'26.7807\"N"
        # tokens[7] = "72", tokens[8] = "43'32.1830\"E"
        lat = f'{tokens[5]} {tokens[6]}'
        lon = f'{tokens[7]} {tokens[8]}'
        
        # Rebuild tokens list with reconstructed coordinates
        cleaned = tokens[:5] + [lat, lon] + tokens[9:]
        return cleaned
        
    except Exception as e:
        safe_print(f"Error parsing row with shlex: {e}")
        safe_print(f"Row content: {row[:100]}...")
        # Fallback to simple space splitting if shlex fails
        return row.strip().split()

def parse_dms_coordinate(dms_str: str) -> float:
    """Parse a complete DMS coordinate string to decimal degrees"""
    if not isinstance(dms_str, str):
        return dms_str
    
    dms_str = dms_str.strip()
    
    # Check if it's already a decimal number
    try:
        return float(dms_str)
    except ValueError:
        pass
    
    # Try to match the DMS pattern
    match = DMS_RX.match(dms_str)
    if match:
        return dms_to_dd(**match.groupdict())
    else:
        safe_print(f"Warning: Could not parse DMS coordinate: {dms_str}")
        return dms_str

default_projectfiles = {
    'well_heads': 'well_heads.pkl',
    'well_tops': 'well_tops.pkl',
    'deviation': 'deviation.pkl',
    'checkshot': 'checkshot.pkl',
    'checkshot_mapping': 'checkshot_mapping.pkl',
    'tdr_mappings': 'tdr_mappings.pkl',
    'custom_trace_selections': 'custom_trace_selections.pkl'  # New: Store custom trace selections per well
}
class WellHandler:
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.loaded_data = {}
        self.project_files = default_projectfiles

    def _normalise_sign(self, series, want_down=True):
        """
        Flip the sign if the series is trending the wrong way.
        Returns the corrected series and the multiplier (+1 or -1).
        
        Args:
            series: pandas Series with depth or time values
            want_down: True for positive-down convention, False for positive-up
        
        Returns:
            tuple: (corrected_series, multiplier)
        """
        if len(series) < 2:
            return series, 1
            
        trend = series.iloc[-1] - series.iloc[0]
        need_flip = (trend < 0 and want_down) or (trend > 0 and not want_down)
        
        if need_flip:
            return -series, -1
        else:
            return series, 1
    
    def _standardise_sign_by_max_absolute(self, series, want_positive=True):
        """
        Standardize sign based on absolute maximum value as per project rules:
        - Find absolute max value
        - If max value is negative and we want positive, multiply all by -1
        - If max value is positive and we want positive, no change
        
        Args:
            series: pandas Series with depth or time values
            want_positive: True for positive values, False for negative values
        
        Returns:
            tuple: (corrected_series, multiplier)
        """
        if len(series) == 0:
            return series, 1
        
        # Find the value with maximum absolute magnitude
        abs_series = series.abs()
        max_abs_idx = abs_series.idxmax()
        max_abs_value = series.loc[max_abs_idx]
        
        # safe_print(f"  Max absolute value: {max_abs_value:.1f} at index {max_abs_idx}")
        
        # Determine if we need to flip based on the sign of max absolute value
        if want_positive:
            need_flip = max_abs_value < 0
        else:
            need_flip = max_abs_value > 0
            
        if need_flip:
            # safe_print(f"  Flipping signs: max value {max_abs_value:.1f} -> {-max_abs_value:.1f}")
            return -series, -1
        else:
            # safe_print(f"  Keeping signs: max value {max_abs_value:.1f} is already correct")
            return series, 1

    def _standardise_tdr(self, df, depth_col='Z', time_col='TWT picked'):
        """
        Standardize TDR data to positive-down convention using absolute max rule.
        
        Args:
            df: DataFrame with TDR data
            depth_col: Name of depth column
            time_col: Name of time column
            
        Returns:
            tuple: (standardized_df, z_multiplier, t_multiplier)
        """
        df = df.copy()
        
        # safe_print(f"TDR sign standardization for {depth_col} and {time_col}")
        
        # Standardize depth to positive (downward positive)
        # safe_print(f"  Original {depth_col} range: {df[depth_col].min():.1f} to {df[depth_col].max():.1f}")
        df[depth_col], z_mult = self._standardise_sign_by_max_absolute(df[depth_col], want_positive=True)
        # safe_print(f"  Standardized {depth_col} range: {df[depth_col].min():.1f} to {df[depth_col].max():.1f}")
        
        # Standardize time to positive (increasing with depth)
        # safe_print(f"  Original {time_col} range: {df[time_col].min():.1f} to {df[time_col].max():.1f}")
        df[time_col], t_mult = self._standardise_sign_by_max_absolute(df[time_col], want_positive=True)
        # safe_print(f"  Standardized {time_col} range: {df[time_col].min():.1f} to {df[time_col].max():.1f}")
        
        # Ensure depth is monotonically increasing
        df.sort_values(depth_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df, z_mult, t_mult


    def load_project(self, project_folder):
        self.project_folder = project_folder
        for data_type, file_name in self.project_files.items():
            file_path = os.path.join(project_folder, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        self.loaded_data[data_type] = pickle.load(f)
                        safe_print(f"Loaded data for {data_type}:")
                except Exception as e:
                    safe_print(f"Error loading {file_name}: {e}")
                    traceback.print_exc()
            else:
                safe_print(f"File {file_name} does not exist in {project_folder}.")

        # Initialize checkshot data as a dictionary if it doesn't exist
        if 'checkshot' not in self.loaded_data:
            self.loaded_data['checkshot'] = {}
            safe_print("Initialized empty 'checkshot' dictionary.")

        # Initialize checkshot mapping if it doesn't exist
        if 'checkshot_mapping' not in self.loaded_data:
            self.loaded_data['checkshot_mapping'] = pd.DataFrame(columns=['Well', 'CheckShotID', 'SourceWell'])
            safe_print("Initialized empty 'checkshot_mapping' DataFrame.")

        # Initialize tdr_mappings if it doesn't exist
        if 'tdr_mappings' not in self.loaded_data:
            self.loaded_data['tdr_mappings'] = {}
            safe_print("Initialized empty 'tdr_mappings' dictionary.")
            
        # Initialize custom_trace_coordinates if it doesn't exist
        if 'custom_trace_coordinates' not in self.loaded_data:
            self.loaded_data['custom_trace_coordinates'] = {}
            safe_print("Initialized empty 'custom_trace_coordinates' dictionary.")
    def remove_well_data(self, well_name):
        # Remove deviation data
        if 'deviation' in self.loaded_data and well_name in self.loaded_data['deviation']:
            del self.loaded_data['deviation'][well_name]

        # Remove checkshot data
        if 'checkshot' in self.loaded_data and well_name in self.loaded_data['checkshot']:
            del self.loaded_data['checkshot'][well_name]

        # Remove TDR mappings
        if 'tdr_mappings' in self.loaded_data and well_name in self.loaded_data['tdr_mappings']:
            del self.loaded_data['tdr_mappings'][well_name]

        # Remove from checkshot mapping
        if 'checkshot_mapping' in self.loaded_data:
            self.loaded_data['checkshot_mapping'] = self.loaded_data['checkshot_mapping'][
                self.loaded_data['checkshot_mapping']['Well'] != well_name
            ]

        # Remove well tops data
        if 'well_tops' in self.loaded_data:
            self.loaded_data['well_tops'] = self.loaded_data['well_tops'][
                self.loaded_data['well_tops']['Well'] != well_name
            ]
        
        # Remove custom trace coordinates
        if 'custom_trace_coordinates' in self.loaded_data and well_name in self.loaded_data['custom_trace_coordinates']:
            del self.loaded_data['custom_trace_coordinates'][well_name]

    def update_well_heads(self, updated_df):
        self.loaded_data['well_heads'] = updated_df
    def update_well_tops(self, updated_df):
        self.loaded_data['well_tops'] = updated_df
    def get_data(self, data_type):
        return self.loaded_data.get(data_type)
    
    def set_custom_trace_coordinates(self, well_name, inline, crossline):
        """
        Set custom trace coordinates for a specific well.
        
        Args:
            well_name (str): Name of the well
            inline (int): Custom inline coordinate
            crossline (int): Custom crossline coordinate
        """
        if 'custom_trace_coordinates' not in self.loaded_data:
            self.loaded_data['custom_trace_coordinates'] = {}
        
        self.loaded_data['custom_trace_coordinates'][well_name] = {
            'inline': inline,
            'crossline': crossline
        }
        safe_print(f"Set custom trace coordinates for {well_name}: Inline {inline}, Crossline {crossline}")
    
    def get_custom_trace_coordinates(self, well_name):
        """
        Get custom trace coordinates for a specific well.
        
        Args:
            well_name (str): Name of the well
            
        Returns:
            tuple: (inline, crossline) if custom coordinates exist, None otherwise
        """
        custom_coords = self.loaded_data.get('custom_trace_coordinates', {})
        if well_name in custom_coords:
            coords = custom_coords[well_name]
            return (coords['inline'], coords['crossline'])
        return None
    
    def remove_custom_trace_coordinates(self, well_name):
        """
        Remove custom trace coordinates for a specific well.
        
        Args:
            well_name (str): Name of the well
        """
        if ('custom_trace_coordinates' in self.loaded_data and 
            well_name in self.loaded_data['custom_trace_coordinates']):
            del self.loaded_data['custom_trace_coordinates'][well_name]
            safe_print(f"Removed custom trace coordinates for {well_name}")
    
    def has_custom_trace_coordinates(self, well_name):
        """
        Check if a well has custom trace coordinates set.
        
        Args:
            well_name (str): Name of the well
            
        Returns:
            bool: True if custom coordinates exist, False otherwise
        """
        custom_coords = self.loaded_data.get('custom_trace_coordinates', {})
        result = well_name in custom_coords
        safe_print(f"[WELL_HANDLER DEBUG] has_custom_trace_coordinates({well_name}): {result}")
        safe_print(f"[WELL_HANDLER DEBUG] Available wells with custom coords: {list(custom_coords.keys())}")
        return result


    def get_well_data(self, well_name):
        well_heads = self.get_data('well_heads')
        if well_heads is not None and not well_heads.empty:
            return well_heads[well_heads['Name'] == well_name]
        return pd.DataFrame()

    def get_deviation_data(self, well_name):
        deviation_data = self.get_data('deviation')
        if deviation_data and well_name in deviation_data:
            return deviation_data[well_name]['dev_data']
        return pd.DataFrame()

    def get_checkshot_data(self, well_name):
        checkshot_data = self.loaded_data.get('checkshot', {})
        if well_name in checkshot_data:
            return checkshot_data[well_name]
        return pd.DataFrame()


    def get_all_tdr_mappings(self):
        return self.loaded_data.get('tdr_mappings', {})

    def get_available_datatypes(self):
        return [data_type for data_type in self.project_files if data_type in self.loaded_data]

    def _derive_top_properties(self, well_name, md):
        """
        Derive X, Y, Z, TWT Auto for a well top given its MD.

        Derivation chain:
            MD + deviation → TVD, X, Y at depth (vertical if no deviation: TVD=MD)
            KB + TVD → Z (TVDSS = MD - KB for vertical, TVD - KB for deviated)
            MD + checkshot → TWT Auto (interpolation on MD vs abs(TWT picked))

        Args:
            well_name: Well name
            md: Measured depth (m)

        Returns:
            dict {X, Y, Z, TWT_Auto} with np.nan for any that can't be derived
        """
        result = {'X': np.nan, 'Y': np.nan, 'Z': np.nan, 'TWT_Auto': np.nan}

        # --- KB from well_heads ---
        well_heads = self.get_data('well_heads')
        kb = np.nan
        surface_x = np.nan
        surface_y = np.nan
        if well_heads is not None and not well_heads.empty:
            wh_row = well_heads[well_heads['Name'] == well_name]
            if not wh_row.empty:
                _kb = wh_row['Well datum value'].iloc[0]
                if not pd.isna(_kb):
                    kb = float(_kb)
                if 'Surface X' in wh_row.columns:
                    _sx = wh_row['Surface X'].iloc[0]
                    if not pd.isna(_sx):
                        surface_x = float(_sx)
                if 'Surface Y' in wh_row.columns:
                    _sy = wh_row['Surface Y'].iloc[0]
                    if not pd.isna(_sy):
                        surface_y = float(_sy)

        # --- Deviation survey → TVD, X, Y at MD ---
        deviation = self.get_data('deviation')
        has_deviation = False
        tvd = md  # default: vertical (TVD = MD)

        if deviation is not None and well_name in deviation:
            dev_entry = deviation[well_name]
            if isinstance(dev_entry, dict):
                dd = dev_entry.get('dev_data', pd.DataFrame())
            else:
                dd = dev_entry

            if isinstance(dd, pd.DataFrame) and not dd.empty and 'MD' in dd.columns:
                dev_md = dd['MD'].values
                if len(dev_md) >= 2 and md <= dev_md.max():
                    has_deviation = True
                    if 'TVD' in dd.columns:
                        tvd = float(np.interp(md, dev_md, dd['TVD'].values))
                    if 'X' in dd.columns and 'Y' in dd.columns:
                        result['X'] = float(np.interp(md, dev_md, dd['X'].values))
                        result['Y'] = float(np.interp(md, dev_md, dd['Y'].values))

        # For vertical wells (no deviation), use surface coordinates
        if not has_deviation:
            result['X'] = surface_x
            result['Y'] = surface_y

        # --- Z (TVDSS) = TVD - KB ---
        if not np.isnan(kb):
            result['Z'] = tvd - kb

        # --- TWT Auto from checkshot ---
        checkshot = self.get_data('checkshot')
        if checkshot is not None and well_name in checkshot:
            cs = checkshot[well_name]
            if isinstance(cs, pd.DataFrame) and 'MD' in cs.columns and 'TWT picked' in cs.columns:
                cs_md = cs['MD'].values
                cs_twt = np.abs(cs['TWT picked'].values)
                if len(cs_md) >= 2:
                    result['TWT_Auto'] = float(np.interp(md, cs_md, cs_twt))

        return result

    def update_tops_from_table(self, table_path_or_df, well_col='Well Name',
                               kb_col='KB', save=True, dry_run=False):
        """
        Update well tops MD from a pivoted table (wells as rows, surfaces as columns).

        Complete workflow:
        1. Update MD values in well_tops
        2. Update KB in well_heads if KB column present
        3. Derive X, Y, Z, TWT Auto for all modified/added tops using:
           - KB + deviation survey → Z (TVDSS)
           - Deviation survey → X, Y at depth (surface coords if vertical)
           - Checkshot → TWT Auto

        Geologists provide revised picks in this format:
            Well Name, KB, K-VIII-A-Upper, K-VIII-A-Lower, ...
            BK-10,     57.4, 1402.0,       1414.49, ...

        Args:
            table_path_or_df: CSV/Excel path or DataFrame
            well_col: Column name for well names (default 'Well Name')
            kb_col: Column name for KB values (default 'KB')
            save: Whether to save_project after updates (default True)
            dry_run: If True, only report changes without applying (default False)

        Returns:
            dict with counts: {'updated': int, 'added': int, 'unchanged': int,
                               'kb_updated': int, 'details': list}
        """
        # Load table
        if isinstance(table_path_or_df, pd.DataFrame):
            rev = table_path_or_df
        else:
            ext = os.path.splitext(str(table_path_or_df))[1].lower()
            if ext == '.xlsx':
                rev = pd.read_excel(table_path_or_df)
            else:
                rev = pd.read_csv(table_path_or_df)

        well_tops = self.get_data('well_tops')
        if well_tops is None:
            well_tops = pd.DataFrame(columns=['X', 'Y', 'Z', 'TWT', 'TWT Auto', 'MD', 'Surface', 'Well'])
            self.loaded_data['well_tops'] = well_tops

        # --- Step 1: Update/add KB in well_heads if column present ---
        n_kb_updated = 0
        n_wells_added = 0
        well_heads = self.get_data('well_heads')
        if kb_col in rev.columns and well_heads is not None:
            for _, row in rev.iterrows():
                well = row[well_col]
                new_kb = row.get(kb_col)
                if pd.isna(well) or pd.isna(new_kb):
                    continue
                wh_mask = well_heads['Name'] == well
                if wh_mask.any():
                    # Update existing well's KB
                    old_kb = well_heads.loc[wh_mask, 'Well datum value'].iloc[0]
                    if pd.isna(old_kb) or abs(float(new_kb) - float(old_kb)) > 0.001:
                        if not dry_run:
                            well_heads.loc[wh_mask, 'Well datum value'] = float(new_kb)
                        n_kb_updated += 1
                else:
                    # Add new well to well_heads with KB
                    if not dry_run:
                        new_wh = {col: np.nan for col in well_heads.columns}
                        new_wh['Name'] = well
                        new_wh['Well datum value'] = float(new_kb)
                        new_wh['Well datum name'] = 'KB'
                        well_heads = pd.concat([well_heads, pd.DataFrame([new_wh])],
                                               ignore_index=True)
                        self.loaded_data['well_heads'] = well_heads
                    n_wells_added += 1
                    n_kb_updated += 1

        if n_wells_added > 0:
            print(f"Added {n_wells_added} new wells to well_heads")

        # --- Step 2: Detect surface columns ---
        metadata_cols = {well_col, kb_col, 'Profile', 'profile', 'kb'}
        surface_cols = [c for c in rev.columns if c not in metadata_cols]

        n_updated = 0
        n_added = 0
        n_unchanged = 0
        details = []
        indices_to_derive = []  # track rows needing property derivation

        for _, row in rev.iterrows():
            well = row[well_col]
            if pd.isna(well):
                continue

            for surf in surface_cols:
                new_md = row[surf]
                if pd.isna(new_md):
                    continue

                new_md = float(new_md)
                mask = (well_tops['Well'] == well) & (well_tops['Surface'] == surf)
                existing = well_tops.loc[mask]

                if not existing.empty:
                    old_md = float(existing['MD'].iloc[0])
                    if abs(new_md - old_md) > 0.001:
                        details.append(f"UPD {well} / {surf}: {old_md:.2f} -> {new_md:.2f}")
                        if not dry_run:
                            idx = existing.index[0]
                            well_tops.at[idx, 'MD'] = new_md
                            indices_to_derive.append(idx)
                        n_updated += 1
                    else:
                        n_unchanged += 1
                else:
                    details.append(f"ADD {well} / {surf}: MD={new_md:.2f}")
                    if not dry_run:
                        new_row = {
                            'Well': well, 'Surface': surf, 'MD': new_md,
                            'X': np.nan, 'Y': np.nan, 'Z': np.nan,
                            'TWT': np.nan, 'TWT Auto': np.nan,
                        }
                        well_tops = pd.concat([well_tops, pd.DataFrame([new_row])],
                                              ignore_index=True)
                        self.loaded_data['well_tops'] = well_tops
                        indices_to_derive.append(well_tops.index[-1])
                    n_added += 1

        # --- Step 3: Derive X, Y, Z, TWT Auto for all modified/added rows ---
        # Only overwrite if derived value is not NaN — preserve existing good values
        # (e.g., X, Y from original Petrel import when deviation/well_heads are absent)
        if not dry_run and indices_to_derive:
            n_derived = 0
            for idx in indices_to_derive:
                well = well_tops.at[idx, 'Well']
                md = float(well_tops.at[idx, 'MD'])
                props = self._derive_top_properties(well, md)
                for col, key in [('X', 'X'), ('Y', 'Y'), ('Z', 'Z'), ('TWT Auto', 'TWT_Auto')]:
                    derived_val = props[key]
                    if not pd.isna(derived_val):
                        well_tops.at[idx, col] = derived_val
                n_derived += 1
            print(f"Derived properties for {n_derived} tops")

        if not dry_run:
            self.loaded_data['well_tops'] = well_tops
            if save:
                self.save_project(self.project_folder)

        action = "DRY RUN" if dry_run else "Applied"
        print(f"{action}: {n_updated} updated, {n_added} added, {n_unchanged} unchanged, "
              f"{n_kb_updated} KB updated")
        return {'updated': n_updated, 'added': n_added, 'unchanged': n_unchanged,
                'kb_updated': n_kb_updated, 'details': details}

    def export_well_tops(self, output_path, well_names=None, surfaces=None):
        """
        Export well tops to CSV or Excel.

        Args:
            output_path: Full path (.csv or .xlsx)
            well_names: Optional list of well names to filter (None = all)
            surfaces: Optional list of surface names to filter (None = all)

        Returns:
            output_path on success, None on failure
        """
        well_tops = self.get_data('well_tops')
        if well_tops is None or well_tops.empty:
            print("No well tops data available")
            return None

        df = well_tops.copy()

        if well_names is not None:
            df = df[df['Well'].isin(well_names)]
        if surfaces is not None:
            df = df[df['Surface'].isin(surfaces)]

        if df.empty:
            print("No matching well tops after filtering")
            return None

        # Sort for readability
        df = df.sort_values(['Well', 'MD']).reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.xlsx':
            df.to_excel(output_path, index=False, sheet_name='Well_Tops')
        else:
            df.to_csv(output_path, index=False)

        print(f"Exported {len(df)} tops ({df['Well'].nunique()} wells, "
              f"{df['Surface'].nunique()} surfaces) -> {output_path}")
        return output_path

    def save_project(self, project_folder):
        for data_type, file_name in self.project_files.items():
            data = self.loaded_data.get(data_type, {})
            pickle_path = os.path.join(project_folder, file_name)
            
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(data, f)
            elif isinstance(data, dict):
                if data:  # Check if dictionary is not empty
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(data, f)
            else:
                # Handle other data types if necessary
                safe_print(f"Unsupported data type for {data_type}: {type(data)}")
        
        self.project_folder = project_folder


    def parse_file(self, file_path, parse_function):
        data = parse_function(file_path)
        return data

    def parse_well_head_file(self, file_path):
        """
        Robust Petrel well head file parser using two-pass approach:
        1. Extract header dynamically from BEGIN HEADER ... END HEADER block
        2. Use shlex for proper tokenization, then reconstruct DMS coordinates
        3. Add decimal degree columns alongside original DMS columns
        """
        safe_print(f"Parsing Petrel well head file: {file_path}")
        
        records = []
        headers = []
        
        with open(file_path, 'r') as f:
            # Phase 1: Extract header information
            header_started = False
            for line in f:
                line = line.strip()
                if line == 'BEGIN HEADER':
                    header_started = True
                    continue
                elif line == 'END HEADER':
                    header_started = False
                    break
                elif header_started:
                    headers.append(line)
                elif line.startswith('#') or not line:
                    continue  # Skip comments and empty lines before header
            
            # Add decimal degree columns after Latitude and Longitude
            if 'Latitude' in headers and 'Longitude' in headers:
                lat_idx = headers.index('Latitude')
                lon_idx = headers.index('Longitude') 
                # Insert decimal degree columns
                headers.insert(lat_idx + 1, 'Latitude_dd')
                headers.insert(lon_idx + 2, 'Longitude_dd')  # +2 because we already inserted one
                
            safe_print(f"Extracted {len(headers)} headers from file")
            safe_print(f"Sample headers: {headers[:10]}...")
            
            # Phase 2: Parse data rows with robust tokenization
            for line_num, row in enumerate(f):
                row = row.strip()
                if not row or row.startswith('#'):
                    continue
                
                try:
                    # Use robust tokenization
                    tokens = split_petrel_row(row)
                    
                    if len(tokens) < 7:  # Need at least name + coordinates
                        safe_print(f"Warning: Skipping row {line_num} - insufficient data")
                        continue
                    
                    # Parse DMS coordinates if we have them
                    lat_dd = None
                    lon_dd = None
                    
                    if len(tokens) >= 7:  # We should have reconstructed lat/lon at positions 5,6
                        try:
                            lat_str = tokens[5] if len(tokens) > 5 else ""
                            lon_str = tokens[6] if len(tokens) > 6 else ""
                            
                            lat_dd = parse_dms_coordinate(lat_str)
                            lon_dd = parse_dms_coordinate(lon_str)
                            
                        except Exception as e:
                            safe_print(f"Warning: Error parsing coordinates in row {line_num}: {e}")
                            lat_dd = lat_str
                            lon_dd = lon_str
                    
                    # Insert decimal degree values into tokens at correct positions
                    if 'Latitude_dd' in headers and 'Longitude_dd' in headers:
                        # tokens structure: [name, uwi, symbol, surfx, surfy, lat_dms, lon_dms, ...]
                        # target structure: [name, uwi, symbol, surfx, surfy, lat_dms, lat_dd, lon_dms, lon_dd, ...]
                        if len(tokens) >= 7:
                            tokens_with_dd = (tokens[:6] +     # [name, uwi, symbol, surfx, surfy, lat_dms]
                                            [lat_dd] +         # lat_dd 
                                            [tokens[6]] +      # lon_dms
                                            [lon_dd] +         # lon_dd
                                            tokens[7:])        # [rest...]
                            tokens = tokens_with_dd
                        else:
                            # Not enough tokens, just append decimal degrees at the end
                            tokens.extend([lat_dd, lon_dd])
                    
                    # Ensure tokens match header count
                    while len(tokens) < len(headers):
                        tokens.append('')  # Pad with empty strings
                    if len(tokens) > len(headers):
                        tokens = tokens[:len(headers)]  # Truncate excess
                    
                    # Create record
                    record = dict(zip(headers, tokens))
                    records.append(record)
                    
                except Exception as e:
                    safe_print(f"Error parsing row {line_num}: {e}")
                    safe_print(f"Row content: {row[:100]}...")
                    continue
        
        # Create DataFrame
        df = pd.DataFrame.from_records(records)
        safe_print(f"Successfully parsed {len(df)} well records")
        
        # Clean quoted string fields
        string_columns = ['Name', 'UWI', 'Drilling structure', 'Well datum name', 
                         'Well datum description', 'Spud date', 'Simulation name', 'Operator']
        
        for column in string_columns:
            if column in df.columns:
                # Remove surrounding quotes from string fields
                df[column] = df[column].astype(str).str.strip('"').replace('""', '', regex=False)
        
        # Convert numeric columns
        for column in df.columns:
            if column.endswith('_dd') or column in ['Surface X', 'Surface Y', 'Well datum value', 
                                                   'TD (MD)', 'Cost', 'Well symbol']:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except Exception:
                    pass
            elif column not in ['Name', 'UWI', 'Latitude', 'Longitude', 'Drilling structure', 
                              'Well datum name', 'Well datum description', 'Spud date', 
                              'Simulation name', 'Operator']:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except Exception:
                    pass
        
        # Show sample of parsed coordinates and cleaned names
        if 'Latitude_dd' in df.columns and 'Longitude_dd' in df.columns:
            safe_print("\nSample parsed data:")
            safe_print("Well Name | Latitude_dd | Longitude_dd")
            safe_print("-" * 45)
            for i in range(min(3, len(df))):
                name = df.iloc[i]['Name'] if 'Name' in df.columns else f"Row {i}"
                lat = df.iloc[i]['Latitude_dd']
                lon = df.iloc[i]['Longitude_dd']
                safe_print(f"{name:10s} | {lat:11.6f} | {lon:12.6f}")
                
        # Show sample of cleaned well names
        if 'Name' in df.columns:
            safe_print(f"\nSample cleaned well names:")
            sample_names = df['Name'].head(5).tolist()
            safe_print(f"  {sample_names}")
        
        return df

    def _clean_columns(self, df):
        """
        Clean and standardize deviation data columns with alias mapping and graceful fallback.
        """
        # Handle azimuth columns with priority: Grid North > True North > others
        azim_value = None
        if 'AZIM_GN' in df.columns:
            azim_value = df['AZIM_GN']
            df = df.drop(['AZIM_GN'], axis=1)
        elif 'AZIM_TN' in df.columns:
            azim_value = df['AZIM_TN']
            df = df.drop(['AZIM_TN'], axis=1)
        elif 'AZIM' in df.columns:
            azim_value = df['AZIM']
        elif 'Azimuth' in df.columns:
            azim_value = df['Azimuth']
            df = df.drop(['Azimuth'], axis=1)
        elif 'AZI' in df.columns:
            azim_value = df['AZI']
            df = df.drop(['AZI'], axis=1)
        elif 'Azim' in df.columns:
            azim_value = df['Azim']
            df = df.drop(['Azim'], axis=1)
        
        # Set the final AZIM column
        if azim_value is not None:
            if azim_value.isnull().all():
                df['AZIM'] = 0.0
                safe_print("Warning: Azimuth column exists but all values are null. Using 0.0 (due north) as default.")
            else:
                df['AZIM'] = azim_value
        else:
            df['AZIM'] = 0.0
            safe_print("Warning: No azimuth column found (AZIM_GN, AZIM_TN, AZIM, etc.). Using 0.0 (due north) as default.")

        # Handle inclination columns
        inc_value = None
        if 'INCL' in df.columns:
            inc_value = df['INCL']
            df = df.drop(['INCL'], axis=1)
        elif 'INC' in df.columns:
            inc_value = df['INC']
        elif 'Inclination' in df.columns:
            inc_value = df['Inclination']
            df = df.drop(['Inclination'], axis=1)
        
        # Set the final INC column
        if inc_value is not None:
            if inc_value.isnull().all():
                df['INC'] = 0.0
                safe_print("Warning: Inclination column exists but all values are null. Using 0.0 (vertical) as default.")
            else:
                df['INC'] = inc_value
        else:
            df['INC'] = 0.0
            safe_print("Warning: No inclination column found (INC, INCL, Inclination). Using 0.0 (vertical) as default.")

        return df

    def parse_single_well_deviation_file(self, file_path):
        well_info = {}
        data = []
        headers = []
        data_section = False

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# WELL NAME:'):
                    well_info['name'] = line.split(':')[1].strip()
                elif line.startswith('# WELL HEAD X-COORDINATE:'):
                    well_info['x'] = float(line.split(':')[1].split()[0])
                elif line.startswith('# WELL HEAD Y-COORDINATE:'):
                    well_info['y'] = float(line.split(':')[1].split()[0])
                elif line.startswith('# WELL DATUM (KB'):
                    well_info['kb'] = float(line.split(':')[1].split()[0])
                elif line.startswith('#='):
                    if not data_section:
                        headers = next(f).strip().split()
                        data_section = True
                elif data_section and not line.startswith('#'):
                    data.append(line.split())

        df = pd.DataFrame(data, columns=headers)
        
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
            except (ValueError, TypeError):
                # Keep original values if conversion fails
                pass

        # Clean and standardize columns
        df = self._clean_columns(df)

        return {well_info['name']: {'well_info': well_info, 'dev_data': df}}

    def parse_well_tops_file(self, file_path):
        headers, data, header_started = [], [], False
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == 'BEGIN HEADER':
                    header_started = True
                elif line == 'END HEADER':
                    header_started = False
                elif line.startswith('V'):
                    continue
                elif header_started:
                    headers.append(line)
                elif not line.startswith('#') and line:
                    parts = re.findall(r'"([^"]*)"|(\S+)', line)
                    parts = [part[0] if part[0] != '' else part[1] for part in parts]
                    data.append(parts)
        df = pd.DataFrame(data, columns=headers)
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
            except ValueError:
                pass
        bool_columns = ['Used by dep.conv.', 'Used by geo mod', 'Edited by user', 'Locked to fault']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({'TRUE': True, 'FALSE': False})
        return df

    

    def parse_excel_deviation_file(self, deviation_df):
        """
        Parses a deviation file from an Excel file and computes the missing trajectory data 
        using well heads data ('Surface X', 'Surface Y', 'Well datum value').
        
        Args:
        - file_path (str): Path to the Excel file.
        - header_row (int): Row number where the header is located (0-indexed).
        - skip_rows (list): List of rows to skip before reading the data.
        
        Returns:
        - dict: A dictionary with deviation data for each well.
        """

        df=deviation_df
        
        safe_print(f'In parse_excel_deviation_file df: {df}')
        # Ensure numeric conversion for all columns that can be converted
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
            except (ValueError, TypeError):
                # Keep original values if conversion fails
                pass

        # Dictionary to hold the deviation data for each well
        new_deviation_dict = {}

        # Group the data by well names
        for well_name, group in df.groupby('Well'):
            # Retrieve well head information from self.loaded_data['well_heads']
            well_head = self.loaded_data['well_heads'][self.loaded_data['well_heads']['Name'] == well_name]
            if well_head.empty:
                raise ValueError(f"No well head data found for well {well_name}")

            # Extract well head information
            surface_x = well_head['Surface X'].values[0]
            surface_y = well_head['Surface Y'].values[0]
            kb = well_head['Well datum value'].values[0]

            # Calculate missing columns (X, Y, Z, TVD, DX, DY, DLS)
            if 'X' not in group.columns or 'Y' not in group.columns or 'Z' not in group.columns:
                group = self.calculate_trajectory(group, surface_x, surface_y, kb)

            # Prepare well information
            well_info = {
                'name': well_name,
                'x': surface_x,
                'y': surface_y,
                'kb': kb
            }
            safe_print(f"well_info in parse_excel_deviation_file {well_info}")

            # Store the well info and deviation data for each well
            new_deviation_dict[well_name] = {
                'well_info': well_info,
                'dev_data': group[['MD', 'X', 'Y', 'Z', 'TVD', 'DX', 'DY', 'AZIM', 'INCL', 'DLS']]
            }

        return new_deviation_dict

    def full_construction_of_dev_data(self, deviation_data, well_name):
        """
        Full construction of deviation data for a well, using existing data from deviation file
        and calculating missing trajectory data if necessary.
        
        Args:
        - deviation_data (dict): The deviation data for the well (already processed from Excel).
        - well_name (str): The well name to which the data belongs.
        
        Returns:
        - dict: A dictionary containing fully constructed deviation data for the well.
        """
        # Retrieve well head information from self.loaded_data['well_heads']
        well_head = self.loaded_data['well_heads'][self.loaded_data['well_heads']['Name'] == well_name]
        if well_head.empty:
            raise ValueError(f"No well head data found for well {well_name}")

        # Extract well head information
        surface_x = well_head['Surface X'].values[0]
        surface_y = well_head['Surface Y'].values[0]
        kb = well_head['Well datum value'].values[0]

        # Extract deviation data for the well
        deviation_df = deviation_data['dev_data']

        # Calculate missing columns (X, Y, Z, TVD, DX, DY, DLS) if they are not already in the deviation data
        if 'X' not in deviation_df.columns or 'Y' not in deviation_df.columns or 'Z' not in deviation_df.columns:
            deviation_df = self.calculate_trajectory(deviation_df, surface_x, surface_y, kb)

        # Prepare well information for merging with existing data
        well_info = {
            'name': well_name,
            'x': surface_x,
            'y': surface_y,
            'kb': kb
        }

        # if 'AZIM' not in deviation_df.columns and 'AZIM_TN' in deviation_df.columns:
        #     deviation_df.rename(columns={ 'AZIM_TN':'AZIM'}, inplace=True)
        # else:
        #     return {
        #     'well_info': well_info,
        #     'dev_data': pd.DataFrame()
        # }

        return {
            'well_info': well_info,
            'dev_data': deviation_df[['MD', 'X', 'Y', 'Z', 'TVD', 'DX', 'DY','AZIM', 'INCL', 'DLS']] 
        }

    def calculate_trajectory(self, df, surface_x, surface_y, kb):
        """
        Calculates the missing trajectory data (X, Y, Z, TVD, DX, DY, DLS) using the well head information and deviation survey data.
        
        Args:
        - df (DataFrame): The deviation survey data with 'MD', 'AZIM', 'INCL'.
        - surface_x (float): The surface X coordinate.
        - surface_y (float): The surface Y coordinate.
        - kb (float): Kelly bushing height (well datum value).
        
        Returns:
        - df (DataFrame): Updated dataframe with calculated 'X', 'Y', 'Z', 'TVD', 'DX', 'DY', 'DLS' columns.
        """
        # Initialize trajectory columns
        df['X'] = np.nan
        df['Y'] = np.nan
        df['Z'] = np.nan
        df['TVD'] = np.nan
        df['DX'] = np.nan
        df['DY'] = np.nan
        df['DLS'] = np.nan

        # Initialize starting values
        prev_x = surface_x
        prev_y = surface_y
        prev_tvd = kb  # TVD starts at KB
        prev_md = 0
        prev_dx = 0
        prev_dy = 0
        prev_azim = 0
        prev_incl = 0

        # Loop over each row in the dataframe
        for i, row in df.iterrows():
            md = row['MD']
            azim = np.radians(row['AZIM'])  # Convert azimuth to radians
            incl = np.radians(row['INCL'])  # Convert inclination to radians
            
            # Calculate the step in MD
            delta_md = md - prev_md

            # Calculate true vertical depth (TVD)
            delta_tvd = delta_md * np.cos(incl)
            tvd = prev_tvd + delta_tvd

            # Calculate horizontal displacement (dX, dY)
            horizontal_displacement = delta_md * np.sin(incl)
            dx = horizontal_displacement * np.sin(azim)
            dy = horizontal_displacement * np.cos(azim)
            
            # Calculate X and Y positions
            x = prev_x + dx
            y = prev_y + dy
            
            # Calculate Z (depth below surface)
            z = tvd - kb

            # Calculate dog-leg severity (DLS) using the minimum curvature method
            if i > 0:
                dls = self.calculate_dls(delta_md, prev_azim, prev_incl, azim, incl)
            else:
                dls = 0

            # Store the values in the dataframe
            df.at[i, 'X'] = x
            df.at[i, 'Y'] = y
            df.at[i, 'Z'] = z
            df.at[i, 'TVD'] = tvd
            df.at[i, 'DX'] = dx
            df.at[i, 'DY'] = dy
            df.at[i, 'DLS'] = dls

            # Update previous values for the next iteration
            prev_x = x
            prev_y = y
            prev_tvd = tvd
            prev_md = md
            prev_azim = azim
            prev_incl = incl

        return df

    def calculate_dls(self, delta_md, azim1, incl1, azim2, incl2):
        """
        Calculate Dog-Leg Severity (DLS) using the minimum curvature method.
        
        Args:
        - delta_md (float): Change in measured depth.
        - azim1, incl1 (float): Azimuth and inclination at the previous point (radians).
        - azim2, incl2 (float): Azimuth and inclination at the current point (radians).
        
        Returns:
        - dls (float): Dog-leg severity.
        """
        delta_incl = incl2 - incl1
        delta_azim = azim2 - azim1
        angle = np.sqrt(delta_incl**2 + (np.sin((incl1 + incl2) / 2) * delta_azim)**2)
        
        if angle != 0:
            dls = (2 * angle / delta_md) * (180 / np.pi)  # Convert to degrees/100ft
        else:
            dls = 0
        
        return dls

    def calculate_trajectory_minimum_curvature(self, md, incl, azim, surface_x=0.0, surface_y=0.0):
        """
        Calculate well trajectory using industry-standard minimum curvature method
        Following petroleum industry best practices for directional drilling
        
        Args:
            md: Measured depth array (meters)
            incl: Inclination array (degrees)
            azim: Azimuth array (degrees) 
            surface_x: Surface X coordinate (meters)
            surface_y: Surface Y coordinate (meters)
            
        Returns:
            DataFrame with calculated X, Y coordinates and trajectory data
        """
        # Convert to numpy arrays and radians
        MD = np.array(md)
        I = np.array(incl) * np.pi / 180  # Convert to radians
        A = np.array(azim) * np.pi / 180  # Convert to radians
        
        # Pre-allocate arrays
        dN = np.zeros_like(MD)  # North displacement increments
        dE = np.zeros_like(MD)  # East displacement increments
        tvd = np.zeros_like(MD)  # True vertical depth
        
        # Initialize first point
        tvd[0] = 0.0  # First point at surface
        
        # Calculate incremental displacements using minimum curvature method
        for k in range(len(MD) - 1):
            dmd = MD[k+1] - MD[k]
            
            # Calculate dog-leg angle
            cos_dl = (np.sin(I[k]) * np.sin(I[k+1]) * np.cos(A[k+1] - A[k]) +
                     np.cos(I[k]) * np.cos(I[k+1]))
            
            # Ensure cos_dl is within valid range [-1, 1]
            cos_dl = np.clip(cos_dl, -1, 1)
            theta = np.arccos(cos_dl)
            
            # Calculate ratio factor (RF) for minimum curvature method
            if theta < 1e-6:
                RF = 1.0  # Straight line approximation for very small angles
            else:
                RF = 2 / theta * np.tan(theta / 2)
            
            # Calculate north and east increments
            dN[k+1] = 0.5 * dmd * (np.sin(I[k]) * np.cos(A[k]) +
                                  np.sin(I[k+1]) * np.cos(A[k+1])) * RF
            dE[k+1] = 0.5 * dmd * (np.sin(I[k]) * np.sin(A[k]) +
                                  np.sin(I[k+1]) * np.sin(A[k+1])) * RF
            
            # Calculate TVD increment
            tvd[k+1] = tvd[k] + 0.5 * dmd * (np.cos(I[k]) + np.cos(I[k+1])) * RF
        
        # Calculate cumulative coordinates
        N = np.cumsum(dN)
        E = np.cumsum(dE)
        
        # Add surface coordinates
        X_calc = surface_x + E
        Y_calc = surface_y + N
        
        return pd.DataFrame({
            'MD': MD,
            'X': X_calc,
            'Y': Y_calc,
            'Z': tvd,  # Depth positive down from surface
            'TVD': tvd,  # True vertical depth
            'INCL': incl,
            'AZIM': azim,
            'dN': dN,
            'dE': dE,
            'N_cumulative': N,
            'E_cumulative': E
        })
    
    def merge_deviation_data(self, new_data, overwrite=False):
        if 'deviation' not in self.loaded_data:
            self.loaded_data['deviation'] = {}

        warnings = []
        for well_name, well_data in new_data.items():
            # safe_print(f'well_name, well_data: {well_name, well_data}')
            if well_name in self.loaded_data['deviation'] and not overwrite:
                warnings.append(f"Well {well_name} already exists in deviation data. Skipping.")
            else:
                if well_name in self.loaded_data['deviation'] and overwrite:
                    warnings.append(f"Overwriting existing deviation data for well {well_name}.")
                
                # Standardize signs in deviation data before storing
                if 'dev_data' in well_data:
                    dev_df = well_data['dev_data'].copy()
                    safe_print(f"Standardizing deviation data signs during import for well {well_name}")
                    
                    # Standardize Z coordinates if present
                    if 'Z' in dev_df.columns:
                        # safe_print(f"  Original Z range: {dev_df['Z'].min():.1f} to {dev_df['Z'].max():.1f}")
                        dev_df['Z'], z_mult = self._standardise_sign_by_max_absolute(dev_df['Z'], want_positive=True)
                        # safe_print(f"  Standardized Z range: {dev_df['Z'].min():.1f} to {dev_df['Z'].max():.1f}")
                    
                    # Standardize TVD coordinates if present
                    if 'TVD' in dev_df.columns:
                        # safe_print(f"  Original TVD range: {dev_df['TVD'].min():.1f} to {dev_df['TVD'].max():.1f}")
                        dev_df['TVD'], tvd_mult = self._standardise_sign_by_max_absolute(dev_df['TVD'], want_positive=True)
                        # safe_print(f"  Standardized TVD range: {dev_df['TVD'].min():.1f} to {dev_df['TVD'].max():.1f}")
                    
                    # Update the well_data with standardized dev_data
                    well_data['dev_data'] = dev_df
                
                self.loaded_data['deviation'][well_name] = well_data

        return warnings

    def parse_check_shot_file(self, file_path):
        headers, data, header_started = [], [], False
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == 'BEGIN HEADER':
                    header_started = True
                elif line == 'END HEADER':
                    header_started = False
                elif line.startswith('V'):
                    continue
                elif header_started:
                    headers.append(line)
                elif not line.startswith('#') and line:
                    parts = re.findall(r'\"(.*?)\"|(\S+)', line)
                    parts = [part[0] if part[0] else part[1] for part in parts]
                    data.append(parts)
        df = pd.DataFrame(data, columns=headers)
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
            except ValueError:
                pass
        return df

    def get_wells_from_checkshot_data(self, checkshot_df):
        if 'Well' not in checkshot_df.columns:
            safe_print("Warning: 'Well' column not found in checkshot data")
            return []
        return checkshot_df['Well'].unique().tolist()

    def merge_checkshot_data(self, new_data, well_name, new_checkshot_id=""):
        import_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if 'Well' not in new_data.columns:
            raise ValueError(f"'Well' column not found in checkshot data for {well_name}")
        
        if well_name not in self.loaded_data.get('checkshot', {}):
            self.loaded_data.setdefault('checkshot', {})[well_name] = pd.DataFrame()
        
        new_data['CheckShotID'] = new_checkshot_id
        
        # Standardize sign conventions during import if depth/time columns exist
        if 'Z' in new_data.columns and 'TWT picked' in new_data.columns:
            safe_print(f"Standardizing sign conventions for checkshot data: {well_name}")
            new_data_std, z_mult, t_mult = self._standardise_tdr(new_data, 'Z', 'TWT picked')
            new_data = new_data_std
            # safe_print(f"  Applied multipliers: Z={z_mult}, TWT={t_mult}")
        
        try:
            self.loaded_data['checkshot'][well_name] = pd.concat([self.loaded_data['checkshot'][well_name], new_data], ignore_index=True)
        except Exception as e:
            raise ValueError(f"Error concatenating checkshot data for well {well_name}: {str(e)}")
        
        # Update the mapping table
        self.set_well_tdr(well_name, well_name, new_checkshot_id)

        # safe_print(f"Successfully merged checkshot data for well: {well_name}")

    def set_well_tdr(self, well_name, source_well, checkshot_id):
        if 'tdr_mappings' not in self.loaded_data:
            self.loaded_data['tdr_mappings'] = {}
        self.loaded_data['tdr_mappings'][well_name] = {'source_well': source_well, 'checkshot_id': checkshot_id}

        # Update the checkshot_mapping to reflect this as the most recent TDR
        new_mapping = pd.DataFrame({
            'Well': [well_name],
            'CheckShotID': [checkshot_id],
            'SourceWell': [source_well],
            'ImportTime': [datetime.now().strftime("%Y%m%d_%H%M%S")]
        })
        if 'checkshot_mapping' not in self.loaded_data:
            self.loaded_data['checkshot_mapping'] = new_mapping
        else:
            # Remove any existing mappings for this well
            # self.loaded_data['checkshot_mapping'] = self.loaded_data['checkshot_mapping'][self.loaded_data['checkshot_mapping']['Well'] != well_name]

            # Add the new mapping
            self.loaded_data['checkshot_mapping'] = pd.concat([self.loaded_data['checkshot_mapping'], new_mapping], ignore_index=True)



    def get_well_tdr(self, well_name):
        """
        Get the currently active TDR mapping for the well.

        Args:
            well_name (str): The well name.

        Returns:
            Dict: A dictionary with 'source_well' and 'checkshot_id' of the active TDR.
        """
        return self.get_well_tdrs(well_name)


    def get_well_tdrs(self, well_name):
        """
        Retrieve the current active TDR mapping for the specified well.

        Args:
            well_name (str): The target well name.

        Returns:
            Dict: Dictionary containing 'source_well' and 'checkshot_id' if exists, else empty dict.
        """
        return self.loaded_data.get('tdr_mappings', {}).get(well_name, {})


    def get_wells_with_checkshots(self):
        """Get list of wells that have valid checkshot data with CheckShotID column."""
        wells_with_valid_checkshots = []
        checkshot_data = self.loaded_data.get('checkshot', {})
        
        for well_name, checkshot_df in checkshot_data.items():
            # Only include wells that have non-empty checkshot data with CheckShotID column
            if not checkshot_df.empty and 'CheckShotID' in checkshot_df.columns:
                wells_with_valid_checkshots.append(well_name)
        
        return wells_with_valid_checkshots


    def import_checkshot_file(self, file_path):
        checkshot_df = self.parse_check_shot_file(file_path)

        return checkshot_df

    def get_filtered_data(self, data_type, selected_indices, selected_surfaces=None):
        data = self.get_data(data_type)
        
        # Smart logging for well filtering operations
        should_log_filter = should_log_debug("well_filter", data_type)
        if should_log_filter and debug_config.get_debug_level() == 'VERBOSE':
            safe_print(f"WELLHANDLER DEBUG: get_filtered_data called - data_type='{data_type}', selected_indices={selected_indices}")
            mark_debug_logged("well_filter", data_type)
        
        if data_type == 'well_tops':
            if selected_surfaces is not None:
                result = data[(data['Well'].isin(selected_indices)) & (data['Surface'].isin(selected_surfaces))]
                if should_log_filter:
                    safe_print(f"WELLHANDLER DEBUG: well_tops with surfaces filter - found {len(result)} rows")
                return result
            result = data[data['Well'].isin(selected_indices)] if selected_indices else pd.DataFrame()
            if should_log_filter:
                safe_print(f"WELLHANDLER DEBUG: well_tops filter - found {len(result)} rows")
            return result
            
        elif data_type == 'well_heads':
            if selected_indices:
                if should_log_filter and debug_config.get_debug_level() == 'VERBOSE':
                    safe_print(f"WELLHANDLER DEBUG: well_heads data shape: {data.shape}, columns: {list(data.columns)}")
                
                if not data.empty:
                    available_wells = list(data['Name'].unique()) if 'Name' in data.columns else []
                    if should_log_filter and debug_config.get_debug_level() == 'VERBOSE':
                        safe_print(f"WELLHANDLER DEBUG: Available wells in data: {available_wells}")
                    # else:
                    #     # Always log well selection summary
                    #     safe_print(f"WELLHANDLER: well_heads string filter - found {len(data[data['Name'].isin(selected_indices)])} rows for {len(selected_indices)} requested wells")
                    
                    # Check for mismatches
                    if isinstance(selected_indices[0], str):
                        missing_wells = [w for w in selected_indices if w not in available_wells]
                        matching_wells = [w for w in selected_indices if w in available_wells]
                        if missing_wells:
                            if debug_config.get_debug_level() == 'VERBOSE':
                                safe_print(f"WELLHANDLER DEBUG: ⚠️ Missing wells: {missing_wells}")
                        if debug_config.get_debug_level() == 'VERBOSE':
                            safe_print(f"WELLHANDLER DEBUG: ✓ Matching wells: {matching_wells}")
                        
                        # EMPTY SELECTION FIX: Handle case where no wells match
                        if not matching_wells:
                            if debug_config.get_debug_level() == 'VERBOSE':
                                safe_print(f"WELLHANDLER DEBUG: No matching wells found - returning empty DataFrame")
                            return pd.DataFrame()
                        
                        result = data[data['Name'].isin(selected_indices)]
                        if debug_config.get_debug_level() == 'VERBOSE':
                            safe_print(f"WELLHANDLER DEBUG: well_heads string filter - found {len(result)} rows for {len(selected_indices)} requested wells")
                        return result
                    else:
                        result = data.iloc[selected_indices]
                        if debug_config.get_debug_level() == 'VERBOSE':
                            safe_print(f"WELLHANDLER DEBUG: well_heads index filter - found {len(result)} rows")
                        return result
                else:
                    if debug_config.get_debug_level() == 'VERBOSE':
                        safe_print(f"WELLHANDLER DEBUG: ⚠️ well_heads data is empty!")
            else:
                if debug_config.get_debug_level() == 'VERBOSE':
                    safe_print(f"WELLHANDLER DEBUG: selected_indices is empty - returning empty DataFrame")
                    
        if debug_config.get_debug_level() == 'VERBOSE':
            safe_print(f"WELLHANDLER DEBUG: Returning empty DataFrame for data_type='{data_type}'")
        return pd.DataFrame() 

    def _densify_well_path(self, well_data, target_interval=10.0):
        """
        Densify well path data to ensure sufficient points for visualization.
        Interpolates between existing survey points at specified interval.
        
        Parameters:
        - well_data: DataFrame with MD, X, Y, Z, TVD columns
        - target_interval: Target spacing in meters (default 10m)
        
        Returns:
        - DataFrame with densified well path
        """
        if well_data.empty or len(well_data) < 2:
            # safe_print(f"WellHandler: Cannot densify - insufficient data points ({len(well_data)} points)")
            return well_data
        
        # Check if MD column exists
        if 'MD' not in well_data.columns:
            # safe_print(f"WellHandler: Cannot densify - no MD column found")
            return well_data
            
        # Sort by MD to ensure proper ordering
        well_data = well_data.sort_values('MD').reset_index(drop=True)
        
        md_original = well_data['MD'].values
        md_min, md_max = md_original.min(), md_original.max()
        
        # Calculate current average spacing
        if len(md_original) > 1:
            avg_spacing = (md_max - md_min) / (len(md_original) - 1)
        else:
            avg_spacing = target_interval
            
        # safe_print(f"WellHandler: Well path has {len(well_data)} points, avg spacing: {avg_spacing:.1f}m")
        
        # Only densify if current spacing is larger than target
        if avg_spacing <= target_interval:
            # safe_print(f"WellHandler: Well path already dense enough ({avg_spacing:.1f}m <= {target_interval}m)")
            return well_data
            
        # Create new MD array with target interval
        num_points = int((md_max - md_min) / target_interval) + 1
        md_new = np.linspace(md_min, md_max, num_points)
        
        # safe_print(f"WellHandler: Densifying from {len(well_data)} to {len(md_new)} points (spacing: {target_interval}m)")
        
        # Interpolate all spatial coordinates
        densified_data = {'MD': md_new}
        
        for column in ['X', 'Y', 'Z', 'TVD']:
            if column in well_data.columns:
                # Use linear interpolation for spatial coordinates
                interp_func = interp1d(md_original, well_data[column].values, 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                densified_data[column] = interp_func(md_new)
        
        # Handle angle data if present (use appropriate interpolation)
        for column in ['INCL', 'AZIM']:
            if column in well_data.columns:
                # For angles, use linear interpolation but handle wraparound for azimuth
                if column == 'AZIM':
                    # Handle azimuth wraparound (0-360 degrees)
                    azim_values = well_data[column].values
                    # Simple linear interpolation for now - could be improved for wraparound
                    interp_func = interp1d(md_original, azim_values, 
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
                    densified_data[column] = interp_func(md_new)
                else:
                    # Regular linear interpolation for inclination
                    interp_func = interp1d(md_original, well_data[column].values, 
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
                    densified_data[column] = interp_func(md_new)
        
        densified_df = pd.DataFrame(densified_data)
        
        # safe_print(f"WellHandler: Densified well path: MD {md_min:.1f} to {md_max:.1f}m with {len(densified_df)} points")
        
        return densified_df

    def construct_well_data(self, well_name, enable_caching=True, densification_interval=10.0):
        """
        Enhanced version that uses existing deviation data or calculates trajectory using 
        industry-standard minimum curvature method, then adds time column.
        Follows positive sign convention for both depth and time going downward.
        
        Optimized for spatial seismic sampling with caching and configurable densification.
        
        Args:
            well_name: Name of the well
            enable_caching: Enable trajectory caching for performance
            densification_interval: Interval for trajectory densification (meters)
            
        Returns:
            tuple: (trajectory_df, is_deviated, tdr_well_name)
        """
        
        # Check cache first if enabled
        if enable_caching and hasattr(self, '_trajectory_cache'):
            cache_key = f"{well_name}_{densification_interval}"
            if cache_key in self._trajectory_cache:
                # Reduced verbosity: cache hit logging only for debugging if needed
                # safe_print(f"[TRAJECTORY CACHE] Using cached trajectory for {well_name}")
                return self._trajectory_cache[cache_key]
        elif enable_caching:
            # Initialize cache
            self._trajectory_cache = {}
        wells_df = self.get_data('well_heads')
        deviation_data = self.get_well_deviation(well_name)
        tdr_df = self.get_preferred_tdr(well_name)

        well_header = wells_df[wells_df['Name'] == well_name].iloc[0]
        surface_x = float(well_header['Surface X'])
        surface_y = float(well_header['Surface Y'])
        kb = float(well_header['Well datum value'])
        md_td = float(well_header['TD (MD)'])

        if deviation_data and 'dev_data' in deviation_data and not deviation_data['dev_data'].empty:
            # Use existing deviation data directly
            well_deviation = deviation_data['dev_data'].copy()
            well_deviation = self._clean_columns(well_deviation)
            
            # Check if we have complete coordinate data
            has_coordinates = all(col in well_deviation.columns for col in ['X', 'Y', 'Z', 'TVD'])
            
            if has_coordinates:
                # Use existing surveyed coordinates but standardize signs
                # safe_print(f"Using existing surveyed coordinates for well {well_name}")
                result_coords = well_deviation.copy()
                
                # Standardize Z coordinate signs using absolute max rule
                # safe_print(f"Standardizing Z coordinate signs for well {well_name}")
                # safe_print(f"  Original Z range: {result_coords['Z'].min():.1f} to {result_coords['Z'].max():.1f}")
                result_coords['Z'], z_mult = self._standardise_sign_by_max_absolute(result_coords['Z'], want_positive=True)
                # safe_print(f"  Standardized Z range: {result_coords['Z'].min():.1f} to {result_coords['Z'].max():.1f}")
                
                # Also standardize TVD if needed
                if 'TVD' in result_coords.columns:
                    # safe_print(f"  Original TVD range: {result_coords['TVD'].min():.1f} to {result_coords['TVD'].max():.1f}")
                    result_coords['TVD'], tvd_mult = self._standardise_sign_by_max_absolute(result_coords['TVD'], want_positive=True)
                    # safe_print(f"  Standardized TVD range: {result_coords['TVD'].min():.1f} to {result_coords['TVD'].max():.1f}")
                
                # Densify the well path for spatial sampling optimization
                # safe_print(f"Densifying well path for {well_name} (interval: {densification_interval}m)")
                result_coords = self._densify_well_path(result_coords, target_interval=densification_interval)
                
                is_deviated = True
                tdr_well = "Existing_Survey_Data"
            else:
                # Calculate coordinates using minimum curvature method from survey angles
                safe_print(f"Calculating coordinates using minimum curvature method for well {well_name}")
                
                # Use minimum curvature method for proper trajectory calculation
                trajectory_df = self.calculate_trajectory_minimum_curvature(
                    well_deviation['MD'],
                    well_deviation['INC'], 
                    well_deviation['AZIM'],
                    surface_x,
                    surface_y
                )
                
                # Adjust Z to be relative to KB (positive down)
                trajectory_df['Z'] = trajectory_df['TVD']  # TVD is already positive down from surface
                trajectory_df['TVD'] = trajectory_df['TVD'] + kb  # TVD from KB reference
                
                # Densify the calculated trajectory for spatial sampling optimization
                # safe_print(f"Densifying calculated trajectory for {well_name} (interval: {densification_interval}m)")
                result_coords = self._densify_well_path(trajectory_df, target_interval=densification_interval)
                
                is_deviated = True
                tdr_well = "Minimum_Curvature_Calculated"
                
        else:
            # Fallback for wells without deviation data - create minimal vertical well
            safe_print(f"No deviation data for well {well_name}, creating vertical well trajectory")
            
            # Create simple vertical well data
            md_values = np.arange(0, md_td + 10, 10)
            result_coords = pd.DataFrame({
                'MD': md_values,
                'X': np.full_like(md_values, surface_x, dtype=float),
                'Y': np.full_like(md_values, surface_y, dtype=float),
                'Z': md_values,  # For vertical well: Z = MD (positive down)
                'TVD': md_values + kb,  # TVD from KB reference
                'INCL': np.zeros_like(md_values),
                'AZIM': np.zeros_like(md_values)
            })
            is_deviated = False
            tdr_well = "Vertical_Well"

        # Add time column using TDR data if available
        if not tdr_df.empty:
            try:
                # Get source well info for TDR
                if well_name in self.loaded_data.get('checkshot_mapping', pd.DataFrame()).get('Well', pd.Series()).values:
                    tdr_well = self.loaded_data['checkshot_mapping'][self.loaded_data['checkshot_mapping']['Well'] == well_name]['SourceWell'].iloc[-1]
                else:
                    tdr_well = f"{tdr_well}_with_TDR"
                
                tdr_df = tdr_df.copy()
                
                # Ensure numeric columns
                for col in ['Z', 'TWT picked']:
                    if col in tdr_df.columns:
                        tdr_df[col] = pd.to_numeric(tdr_df[col], errors='coerce')
                
                tdr_df.dropna(subset=['Z', 'TWT picked'], inplace=True)
                
                if not tdr_df.empty:
                    # Standardize TDR data to positive-down convention
                    tdr_df_std, z_mult, t_mult = self._standardise_tdr(tdr_df, 'Z', 'TWT picked')
                    
                    # Use bounded interpolation to prevent extrapolation beyond TDR range
                    tdr_interp = interp1d(
                        tdr_df_std['Z'], 
                        tdr_df_std['TWT picked'], 
                        kind='linear',
                        bounds_error=False,
                        fill_value=(tdr_df_std['TWT picked'].iloc[0], tdr_df_std['TWT picked'].iloc[-1])
                    )
                    
                    # Interpolate time values using Z coordinates from trajectory
                    twt_values = tdr_interp(result_coords['Z'])
                    
                    # Add warning for wells with limited TDR coverage
                    z_min, z_max = tdr_df_std['Z'].min(), tdr_df_std['Z'].max()
                    out_of_range = np.sum((result_coords['Z'] < z_min) | (result_coords['Z'] > z_max))
                    # if out_of_range > 0:
                    #     safe_print(f"Warning: {out_of_range}/{len(result_coords)} points for well {well_name} are outside TDR range ({z_min:.1f} to {z_max:.1f}m)")
                    #     safe_print(f"         Well depth range: {result_coords['Z'].min():.1f} to {result_coords['Z'].max():.1f}m")
                else:
                    twt_values = np.zeros(len(result_coords))
            except Exception as e:
                safe_print(f"Warning: Error processing TDR data for well {well_name}: {e}")
                twt_values = np.zeros(len(result_coords))
        else:
            tdr_well = f"{tdr_well}_No_TDR"
            twt_values = np.zeros(len(result_coords))

        # Create final result dataframe in the expected format
        result_df = pd.DataFrame({
            'Well': well_name,
            'X': result_coords['X'],
            'Y': result_coords['Y'],
            'Dev': result_coords['INCL'] if 'INCL' in result_coords.columns else result_coords.get('INC', 0),
            'Azi': result_coords['AZIM'],
            'Z': result_coords['Z'],      # Depth positive down from surface
            'TWT': twt_values,            # Time positive down  
            'MD': result_coords['MD']     # Measured depth
        })
        
        # Cache the result if caching is enabled
        result = (result_df, is_deviated, tdr_well)
        if enable_caching:
            cache_key = f"{well_name}_{densification_interval}"
            self._trajectory_cache[cache_key] = result
            # safe_print(f"[TRAJECTORY CACHE] Cached trajectory for {well_name} ({len(result_df)} points)")

        return result
    
    def clear_trajectory_cache(self):
        """Clear the trajectory cache to free memory"""
        if hasattr(self, '_trajectory_cache'):
            cache_size = len(self._trajectory_cache)
            self._trajectory_cache.clear()
            # safe_print(f"[TRAJECTORY CACHE] Cleared {cache_size} cached trajectories")
        # else:
        #     safe_print("[TRAJECTORY CACHE] No cache to clear")
    
    def get_trajectory_cache_stats(self):
        """Get trajectory cache statistics"""
        if hasattr(self, '_trajectory_cache'):
            cache_stats = {
                'cache_size': len(self._trajectory_cache),
                'cached_wells': list(self._trajectory_cache.keys()),
                'total_memory_estimate': sum(
                    len(data[0]) * len(data[0].columns) * 8  # Rough estimate: 8 bytes per float
                    for data in self._trajectory_cache.values()
                ) if self._trajectory_cache else 0
            }
            return cache_stats
        else:
            return {'cache_size': 0, 'cached_wells': [], 'total_memory_estimate': 0}

    def get_preferred_tdr(self, well):
        """
        Get the preferred TDR for a well, returning raw data without sign manipulation.
        Sign standardization is now handled in construct_well_data() method.
        """
        # Check if tdr_mappings exists and contains the well
        if ('tdr_mappings' in self.loaded_data and 
            self.loaded_data['tdr_mappings'] is not None and 
            well in self.loaded_data['tdr_mappings']):
            mapping = self.loaded_data['tdr_mappings'][well]
            source_well = mapping['source_well']
            checkshot_id = mapping['checkshot_id']
            if ('checkshot' in self.loaded_data and 
                source_well in self.loaded_data['checkshot']):
                well_tdr = self.loaded_data['checkshot'][source_well][self.loaded_data['checkshot'][source_well]['CheckShotID'] == checkshot_id].copy()
                return well_tdr
        
        # If no mapping found, fall back to the most recent checkshot
        if ('checkshot_mapping' not in self.loaded_data or 
            self.loaded_data['checkshot_mapping'] is None):
            safe_print(f"Warning: No checkshot_mapping data available for well {well}")
            return pd.DataFrame()
            
        mapping = self.loaded_data['checkshot_mapping']
        well_mapping = mapping[mapping['Well'] == well].sort_values('ImportTime', ascending=False).iloc[0] if not mapping[mapping['Well'] == well].empty else None
        
        if well_mapping is not None:
            source_well = well_mapping['SourceWell']
            checkshot_id = well_mapping['CheckShotID']
            if ('checkshot' in self.loaded_data and 
                source_well in self.loaded_data['checkshot']):
                well_tdr = self.loaded_data['checkshot'][source_well][self.loaded_data['checkshot'][source_well]['CheckShotID'] == checkshot_id].copy()
                return well_tdr
            else:
                safe_print(f"Warning: Checkshot data not available for source well {source_well}")
        else:
            safe_print(f"Warning: No checkshot mapping found for well {well}")
        
        # Last resort: try to find any checkshot data for the well directly
        if 'checkshot' in self.loaded_data and self.loaded_data['checkshot']:
            # Look for checkshot data where the well name matches
            for source_well, checkshot_df in self.loaded_data['checkshot'].items():
                if (hasattr(checkshot_df, 'columns') and 'Well' in checkshot_df.columns and 
                    well in checkshot_df['Well'].values):
                    safe_print(f"Found direct checkshot data for {well} in source {source_well}")
                    return checkshot_df[checkshot_df['Well'] == well].copy()
                # If source well name matches the target well
                elif source_well == well:
                    safe_print(f"Found checkshot data using well name as source: {well}")
                    return checkshot_df.copy()
        
        safe_print(f"No TDR/checkshot data found for well {well}")
        return pd.DataFrame()

    def _standardize_tdr_columns(self, tdr_df):
        """
        Standardize TDR column names to ensure consistent 'MD' and 'TWT picked' columns.
        Handles various column naming conventions used in checkshot data.
        """
        if tdr_df.empty:
            return tdr_df
            
        tdr_standardized = tdr_df.copy()
        
        # Map depth columns to 'MD'
        depth_aliases = ['DEPTH', 'Z', 'TVD', 'MEASURED_DEPTH', 'MD_VALUE']
        if 'MD' not in tdr_standardized.columns:
            for alias in depth_aliases:
                if alias in tdr_standardized.columns:
                    tdr_standardized['MD'] = tdr_standardized[alias]
                    safe_print(f"Mapped TDR depth column '{alias}' to 'MD'")
                    break
                    
        # Map time columns to 'TWT picked'
        time_aliases = ['TWT', 'TIME', 'TWT_PICKED', 'TWTT', 'TWO_WAY_TIME', 'TWT_MS']
        if 'TWT picked' not in tdr_standardized.columns:
            for alias in time_aliases:
                if alias in tdr_standardized.columns:
                    tdr_standardized['TWT picked'] = tdr_standardized[alias]
                    safe_print(f"Mapped TDR time column '{alias}' to 'TWT picked'")
                    break
                    
        # Check if we have the required columns after mapping
        required_cols = ['MD', 'TWT picked']
        missing_cols = [col for col in required_cols if col not in tdr_standardized.columns]
        
        if missing_cols:
            safe_print(f"Warning: TDR data missing required columns after standardization: {missing_cols}")
            safe_print(f"Available columns: {list(tdr_standardized.columns)}")
            
        return tdr_standardized

    def get_standardized_tdr(self, well_name):
        """
        Get TDR data with standardized column names and sign conventions for consistent access.
        Returns TDR with 'MD' and 'TWT picked' columns with positive-down sign convention.
        """
        raw_tdr = self.get_preferred_tdr(well_name)
        if raw_tdr is None or raw_tdr.empty:
            return raw_tdr
            
        # First standardize column names
        tdr_with_standard_cols = self._standardize_tdr_columns(raw_tdr)
        
        # Then apply sign standardization if we have the required columns
        if 'MD' in tdr_with_standard_cols.columns and 'TWT picked' in tdr_with_standard_cols.columns:
            try:
                # Apply sign standardization to ensure positive-down convention
                standardized_tdr, z_mult, t_mult = self._standardise_tdr(tdr_with_standard_cols, 'MD', 'TWT picked')
                safe_print(f"Applied sign standardization to TDR for {well_name}: MD_mult={z_mult}, TWT_mult={t_mult}")
                return standardized_tdr
            except Exception as e:
                safe_print(f"Warning: Failed to apply sign standardization to TDR for {well_name}: {e}")
                return tdr_with_standard_cols
        else:
            return tdr_with_standard_cols

    def find_nearest_well(self, target_well):
        tdr_df = self.get_data('checkshot')
        wells_df = self.get_data('well_heads')
        
        # Check if target well exists in well_heads
        target_well_data = wells_df.loc[wells_df['Name'] == target_well]
        if target_well_data.empty:
            safe_print(f"Target well '{target_well}' not found in well heads")
            return None
            
        target_x = target_well_data['Surface X'].values[0]
        target_y = target_well_data['Surface Y'].values[0]
        
        if tdr_df is None or tdr_df.empty:
            safe_print("No TDR data available")
            return None
            
        wells_with_tdr = tdr_df['Well'].unique()
        distances = []
        
        for well in wells_with_tdr:
            well_data = wells_df.loc[wells_df['Name'] == well]
            if well_data.empty:
                continue  # Skip if well not found in well_heads
            well_x = well_data['Surface X'].values[0]
            well_y = well_data['Surface Y'].values[0]
            distance = np.sqrt((well_x - target_x)**2 + (well_y - target_y)**2)
            distances.append((well, distance))
        
        if not distances:
            safe_print("No wells with TDR data found")
            return None
            
        nearest_well = min(distances, key=lambda x: x[1])[0]
        return nearest_well

    
    
    def import_file(self, data_type, file_path, overwrite_deviation=False):
        if data_type == 'well_heads':
            self.loaded_data[data_type] = self.parse_well_head_file(file_path)
        elif data_type == 'well_tops':
            self.loaded_data[data_type] = self.parse_well_tops_file(file_path)
        elif data_type == 'deviation':
            if file_path.lower().endswith('.xlsx'):
                new_deviation_data = self.parse_deviation_file(file_path)
                safe_print(f'new_deviation_data import_file: {new_deviation_data}')
            else:
                new_deviation_data = self.parse_single_well_deviation_file(file_path)
                #  {well_info['name']: {'well_info': well_info, 'dev_data': df}}
                safe_print(f'new_deviation_data: {new_deviation_data}')
            
            warnings = self.merge_deviation_data(new_deviation_data, overwrite=overwrite_deviation)
            if warnings:
                safe_print("\n".join(warnings))
            return new_deviation_data
        elif data_type == 'checkshot':
            self.loaded_data[data_type] = self.parse_check_shot_file(file_path)
        return data_type, file_path


    def get_well_deviation(self, well_name):
        deviation_data = self.loaded_data.get('deviation', {})
        return deviation_data.get(well_name, {})
    def get_well_tops_for_well(self, well_name):
        well_tops = self.get_data('well_tops')
        if well_tops is None or well_tops.empty:
            # Return empty DataFrame with expected columns when no well tops data
            return pd.DataFrame(columns=['Z', 'MD', 'Surface'])
        
        well_specific_tops = well_tops[well_tops['Well'] == well_name]
        if well_specific_tops.empty:
            # Return empty DataFrame if no tops found for this specific well
            return pd.DataFrame(columns=['Z', 'MD', 'Surface'])
        
        # Select base columns
        well_tops_df = well_specific_tops[['Z', 'MD', 'Surface']].copy()
        
        # Add TWT calculation using unified deviation-aware method (same as ML data preparation)
        try:
            from geoagent.utils.unified_twt_calculator import calculate_twt_for_depths
            
            # Use the same deviation-aware method as ML data preparation
            twt_values = calculate_twt_for_depths(
                depths=well_tops_df['MD'],  # Use MD depths from well tops
                depth_type='MD',
                well_name=well_name,
                well_handler=self,
                use_deviation_aware=True  # This ensures consistency with ML data preparation
            )
            
            well_tops_df['TWT'] = twt_values
            
            valid_twt_count = np.sum(~np.isnan(twt_values))
            safe_print(f"Well handler: Added {valid_twt_count}/{len(twt_values)} TWT values for {well_name} using deviation-aware method")
            
        except Exception as e:
            safe_print(f"Well handler: Error calculating deviation-aware TWT for {well_name}: {e}")
            safe_print("Falling back to simple MD interpolation...")
            
            # Fallback to simple method if deviation-aware fails
            try:
                tdr_data = self.get_standardized_tdr(well_name)
                if tdr_data is not None and not tdr_data.empty and 'TWT picked' in tdr_data.columns:
                    from scipy.interpolate import interp1d
                    tdr_interp = interp1d(tdr_data['MD'], tdr_data['TWT picked'], 
                                        kind='linear', fill_value='extrapolate')
                    well_tops_df['TWT'] = tdr_interp(well_tops_df['MD'])
                    safe_print(f"Well handler: Fallback TWT calculation successful for {well_name}")
                else:
                    well_tops_df['TWT'] = None
            except Exception as e2:
                safe_print(f"Well handler: Both deviation-aware and fallback TWT calculations failed for {well_name}: {e2}")
                well_tops_df['TWT'] = None
        
        return well_tops_df
    def get_well_data_status(self):
        # Get all the necessary data
        well_heads = self.get_data('well_heads')
        deviation_data = self.get_data('deviation')
        checkshot_data = self.get_data('checkshot')
        well_tops_data = self.get_data('well_tops')

        # Prepare the status DataFrame
        status_data = []

        # Iterate through all wells
        for _, well in well_heads.iterrows():
            well_name = well['Name']
            kb = well['Well datum value']
            td = well['TD (MD)']

            # Check if deviation data is available
            deviation_available = 'Yes' if well_name in deviation_data else 'No'

            # Check if checkshot data is available
            checkshot_available = 'Yes' if checkshot_data is not None and not checkshot_data[checkshot_data['Well'] == well_name].empty else 'No'

            # Get well tops count
            well_tops_count = len(well_tops_data[well_tops_data['Well'] == well_name]) if well_tops_data is not None else 0

            # Append the data for this well
            status_data.append({
                'Well': well_name,
                'KB': kb,
                'TD': td,
                'Deviation': deviation_available,
                'Check Shot': checkshot_available,
                'Well Tops Count': well_tops_count
            })

        # Create the DataFrame
        status_df = pd.DataFrame(status_data)

        return status_df
    def attach_checkshot(self, target_well, source_well, checkshot_id):
        if source_well not in self.loaded_data['checkshot'] or checkshot_id not in self.loaded_data['checkshot'][source_well]['CheckShotID'].values:
            raise ValueError(f"Check shot {checkshot_id} not found for well {source_well}")
        
        # Update the mapping table
        existing_mapping = self.loaded_data['checkshot_mapping']
        checkshot_info = existing_mapping[(existing_mapping['Well'] == source_well) & (existing_mapping['CheckShotID'] == checkshot_id)].iloc[0]
        
        new_mapping = pd.DataFrame({
            'Well': [target_well], 
            'CheckShotID': [checkshot_id], 
            'SourceWell': [source_well],
            'FileName': [checkshot_info['FileName']],
            'ImportTime': [checkshot_info['ImportTime']]
        })
        self.loaded_data['checkshot_mapping'] = pd.concat([existing_mapping, new_mapping], ignore_index=True)
    def get_z_from_md(self, well_name, md_values):
        """
        Calculate Z (vertical depth) values from given MD (measured depth) values.
        Uses deviation data if available, otherwise assumes a vertical well.

        Args:
        well_name (str): Name of the well
        md_values (float or array-like): MD value(s) to convert

        Returns:
        float or np.array: Corresponding Z value(s)
        """
        well_head = self.get_well_data(well_name)
        if well_head.empty:
            raise ValueError(f"No well head data found for well {well_name}")

        kb = well_head['Well datum value'].values[0]
        deviation_data = self.get_well_deviation(well_name)

        if deviation_data and 'dev_data' in deviation_data:
            # Well has deviation data
            dev_df = deviation_data['dev_data']
            z_values = np.interp(md_values, dev_df['MD'], dev_df['TVD']) - kb
        else:
            # Assume vertical well
            z_values = np.array(md_values) - kb

        return z_values
    def get_well_names(self):
        return self.loaded_data['well_heads']['Name'].tolist()

    def get_surface_names(self):
        return self.loaded_data['well_tops']['Surface'].unique().tolist()

    def calculate_well_top_data(self, well_name, md):
        well_data = self.get_well_data(well_name)
        if well_data.empty:
            raise ValueError(f"No well data found for well {well_name}")

        # Extract well head information
        surface_x = well_data['Surface X'].values[0]
        surface_y = well_data['Surface Y'].values[0]
        kb = well_data['Well datum value'].values[0]

        deviation_data = self.get_deviation_data(well_name)
        
        if not deviation_data.empty:
            # Deviated well
            required_columns = ['MD', 'X', 'Y', 'TVD']
            missing_columns = [col for col in required_columns if col not in deviation_data.columns]
            if missing_columns:
                raise ValueError(f"Deviation data is missing required columns: {', '.join(missing_columns)}")

            try:
                x = np.interp(md, deviation_data['MD'], deviation_data['X'])
                y = np.interp(md, deviation_data['MD'], deviation_data['Y'])
                z = np.interp(md, deviation_data['MD'], deviation_data['TVD'])
            except ValueError as e:
                raise ValueError(f"Error interpolating deviation data for well {well_name}: {str(e)}")
        else:
            # Vertical well
            x = surface_x
            y = surface_y
            z = kb - md  # Assuming MD is measured from KB

        # Calculate TWT using unified deviation-aware method (same as ML data preparation)
        try:
            from geoagent.utils.unified_twt_calculator import calculate_twt_for_depths
            
            # Use unified deviation-aware TWT calculation for consistency with ML preparation
            twt_values = calculate_twt_for_depths(
                depths=[md],
                depth_type='MD', 
                well_name=well_name,
                well_handler=self,
                use_deviation_aware=True  # This ensures consistency with ML data preparation
            )
            
            twt = twt_values[0] if len(twt_values) > 0 and not np.isnan(twt_values[0]) else None
            
            if twt is not None:
                safe_print(f"calculate_well_top_data: TWT calculated using unified deviation-aware method for {well_name} at MD={md:.2f}m: {twt:.2f}ms")
            else:
                safe_print(f"Warning: Unified TWT calculation returned invalid result for {well_name} at MD={md:.2f}m")
                
        except Exception as e:
            safe_print(f"Warning: Error in unified TWT calculation for well {well_name} at MD={md:.2f}m: {str(e)}")
            safe_print("Falling back to simple TDR interpolation method")
            
            # Fallback to simple method if unified calculation fails
            try:
                tdr_data = self.get_standardized_tdr(well_name)
                if tdr_data is not None and not tdr_data.empty and 'TWT picked' in tdr_data.columns:
                    from scipy.interpolate import interp1d
                    tdr_interp = interp1d(tdr_data['MD'], tdr_data['TWT picked'], 
                                        kind='linear', fill_value='extrapolate')
                    twt = tdr_interp(md)
                    safe_print(f"calculate_well_top_data: Fallback TWT calculation successful for {well_name} at MD={md:.2f}m: {twt:.2f}ms")
                else:
                    safe_print(f"Warning: No TDR data available for fallback TWT calculation for {well_name}")
                    twt = None
            except Exception as e2:
                safe_print(f"Warning: Fallback TWT calculation also failed for well {well_name}: {str(e2)}")
                twt = None

        return {'X': x, 'Y': y, 'Z': z, 'TWT Auto': twt}

    def get_well_tdrs(self, well_name):
        """
        Retrieve all TDR mappings for the specified well.

        Args:
            well_name (str): The target well name.

        Returns:
            List[Dict]: List of TDR mappings, each as a dictionary with 'source_well', 'checkshot_id', 'is_active'.
        """
        tdr_mappings = self.loaded_data.get('tdr_mappings', {})
        return tdr_mappings.get(well_name, [])

    def activate_existing_tdr(self, target_well, source_well, checkshot_id):
        """
        Activate an existing TDR for the target well.

        Args:
            target_well (str): The well to activate TDR for.
            source_well (str): The source well of the TDR.
            checkshot_id (str): The CheckShot ID of the TDR.

        Raises:
            ValueError: If the specified TDR does not exist for the target well.
        """
        current_tdr = self.get_well_tdrs(target_well)
        if not current_tdr:
            raise ValueError(f"No active TDR found for well '{target_well}' to activate.")
        
        if current_tdr['source_well'] == source_well and current_tdr['checkshot_id'] == checkshot_id:
            # TDR is already active; nothing to do
            return

        # Update the active TDR
        self.loaded_data['tdr_mappings'][target_well] = {
            'source_well': source_well,
            'checkshot_id': checkshot_id
        }
        safe_print(f"Activated existing TDR from well '{source_well}' with CheckShot ID '{checkshot_id}' for well '{target_well}'.")
    def copy_checkshot_from_well(self, target_well, source_well, checkshot_id):
        """
        Copy a CheckShot from the source well to the target well and set it as the active TDR.

        Args:
            target_well (str): The well to assign the TDR to.
            source_well (str): The source well from which to copy the TDR.
            checkshot_id (str): The CheckShot ID to copy.

        Raises:
            ValueError: If copying fails due to invalid source well or CheckShot ID.
        """
        # Verify source well and CheckShot ID existence
        if source_well not in self.loaded_data['checkshot']:
            raise ValueError(f"Source well '{source_well}' does not have any checkshot data.")

        source_checkshots = self.loaded_data['checkshot'][source_well]
        matching_checkshot = source_checkshots[source_checkshots['CheckShotID'] == checkshot_id]
        if matching_checkshot.empty:
            raise ValueError(f"CheckShot ID '{checkshot_id}' not found for well '{source_well}'.")

        # Copy the CheckShot data
        copied_checkshot = matching_checkshot.copy()
        copied_checkshot['Well'] = target_well

        # Assign to target well's CheckShot data
        if target_well not in self.loaded_data['checkshot']:
            self.loaded_data['checkshot'][target_well] = copied_checkshot
        else:
            # Check if the CheckShot ID already exists
            if copied_checkshot['CheckShotID'].iloc[0] in self.loaded_data['checkshot'][target_well]['CheckShotID'].values:
                raise ValueError(f"CheckShot ID '{checkshot_id}' already exists for well '{target_well}'.")
            self.loaded_data['checkshot'][target_well] = pd.concat(
                [self.loaded_data['checkshot'][target_well], copied_checkshot],
                ignore_index=True
            )

        # Update tdr_mappings
        self.loaded_data['tdr_mappings'][target_well] = {
            'source_well': source_well,
            'checkshot_id': checkshot_id
        }
        safe_print(f"Copied TDR from well '{source_well}' with CheckShot ID '{checkshot_id}' to well '{target_well}' and set as active.")


    def get_checkshot_list(self, well_name):
        checkshotdata = self.loaded_data['checkshot']
        if well_name in checkshotdata.keys():
            checkshot_df = checkshotdata[well_name]
            # Defensive check for missing CheckShotID column
            if 'CheckShotID' in checkshot_df.columns:
                return list(checkshot_df['CheckShotID'].unique())
            else:
                safe_print(f"Warning: CheckShotID column missing for well {well_name}, returning empty list")
                return []
        else:
            return []
    
    def get_all_checkshot_data(self):
        """Get all checkshot data from all wells in a single DataFrame for display purposes"""
        all_checkshot_data = []
        
        # Use the same pattern as get_checkshot_list
        checkshotdata = self.loaded_data['checkshot']
        
        for well_name, checkshot_df in checkshotdata.items():
            if not checkshot_df.empty and 'CheckShotID' in checkshot_df.columns:
                # Create a copy and add well name
                df_copy = checkshot_df.copy()
                df_copy['Well'] = well_name
                
                # Select relevant columns for display
                display_columns = ['Well', 'CheckShotID']
                if 'FileName' in df_copy.columns:
                    display_columns.append('FileName')
                if 'ImportTime' in df_copy.columns:
                    display_columns.append('ImportTime')
                
                # Include only available columns
                available_columns = [col for col in display_columns if col in df_copy.columns]
                df_display = df_copy[available_columns]
                
                all_checkshot_data.append(df_display)
        
        if all_checkshot_data:
            return pd.concat(all_checkshot_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_checkshot_data_grouped_by_id(self):
        """Get checkshot data grouped by CheckShotID for deletion dialog"""
        all_checkshot_data = []
        
        checkshotdata = self.loaded_data.get('checkshot', {})
        
        for well_name, checkshot_df in checkshotdata.items():
            if not checkshot_df.empty and 'CheckShotID' in checkshot_df.columns:
                # Define aggregation dictionary dynamically
                agg_dict = {
                    'CheckShotID': 'first'
                }
                if 'MD' in checkshot_df.columns:
                    agg_dict['MD'] = 'count'
                if 'FileName' in checkshot_df.columns:
                    agg_dict['FileName'] = 'first'
                if 'ImportTime' in checkshot_df.columns:
                    agg_dict['ImportTime'] = 'first'

                # Group by CheckShotID and get summary information
                grouped = checkshot_df.groupby('CheckShotID').agg(agg_dict)
                
                if 'MD' in grouped.columns:
                    grouped = grouped.rename(columns={'MD': 'Data_Points'})
                else:
                    grouped['Data_Points'] = 0

                # Reset index to make CheckShotID a column
                grouped = grouped.reset_index(drop=True)
                
                # Add well name
                grouped['Well'] = well_name
                
                # Ensure optional columns exist before reordering
                if 'FileName' not in grouped.columns:
                    grouped['FileName'] = 'N/A'
                if 'ImportTime' not in grouped.columns:
                    grouped['ImportTime'] = 'N/A'

                # Reorder columns
                column_order = ['Well', 'CheckShotID', 'Data_Points', 'FileName', 'ImportTime']
                
                grouped = grouped[column_order]
                
                all_checkshot_data.append(grouped)
        
        if all_checkshot_data:
            return pd.concat(all_checkshot_data, ignore_index=True)
        else:
            return pd.DataFrame(columns=['Well', 'CheckShotID', 'Data_Points', 'FileName', 'ImportTime'])
    def delete_all_checkshots(self, well_name):
        # Delete all checkshot data for the well
        if 'checkshot' in self.loaded_data and well_name in self.loaded_data['checkshot']:
            del self.loaded_data['checkshot'][well_name]
        
        # Delete all related checkshot mappings
        if 'checkshot_mapping' in self.loaded_data:
            self.loaded_data['checkshot_mapping'] = self.loaded_data['checkshot_mapping'][
                self.loaded_data['checkshot_mapping']['Well'] != well_name
            ]
        if well_name in self.loaded_data['tdr_mappings']:
            del self.loaded_data['tdr_mappings'][well_name]

    def delete_checkshot(self, well_name, checkshot_id):
        if 'checkshot' in self.loaded_data and well_name in self.loaded_data['checkshot']:
            safe_print(f"In delete_check shot\nUpdated {self.loaded_data['checkshot'][well_name]['CheckShotID'].unique()} ")
            checkshot_df = self.loaded_data['checkshot'][well_name]
            self.loaded_data['checkshot'][well_name] = checkshot_df[checkshot_df['CheckShotID'] != checkshot_id]
            safe_print(f"Updated {self.loaded_data['checkshot'][well_name]['CheckShotID'].unique()} ")
            # Also update checkshot_mapping
            if 'checkshot_mapping' in self.loaded_data:
                self.loaded_data['checkshot_mapping'] = self.loaded_data['checkshot_mapping'][
                    ~((self.loaded_data['checkshot_mapping']['Well'] == well_name) & 
                    (self.loaded_data['checkshot_mapping']['CheckShotID'] == checkshot_id))
                ]
        else:
            raise ValueError(f"CheckShotID '{checkshot_id}' for well '{well_name}' not found.")
    def update_checkshot_data(self, well_name, checkshot_id, modified_df):
        if 'checkshot' in self.loaded_data and well_name in self.loaded_data['checkshot']:
            # Update the checkshot data
            existing_df = self.loaded_data['checkshot'][well_name]
            # Replace the data for the specific CheckShotID
            other_df = existing_df[existing_df['CheckShotID'] != checkshot_id]
            self.loaded_data['checkshot'][well_name] = pd.concat([other_df, modified_df], ignore_index=True)
        else:
            raise ValueError(f"CheckShotID '{checkshot_id}' for well '{well_name}' not found.")

    def save_deviation_data(self, well_name, deviation_info):
        """
        Save deviation data for a single well.
        
        Args:
            well_name (str): Name of the well
            deviation_info (dict): Dictionary containing 'well_info' and 'dev_data'
        """
        if 'deviation' not in self.loaded_data:
            self.loaded_data['deviation'] = {}
        
        self.loaded_data['deviation'][well_name] = deviation_info
        safe_print(f"Saved deviation data for well: {well_name}")

    def save_checkshot_data(self, well_name, checkshot_data):
        """
        Save checkshot data for a single well.
        
        Args:
            well_name (str): Name of the well
            checkshot_data (pd.DataFrame): DataFrame containing checkshot data
        """
        if 'checkshot' not in self.loaded_data:
            self.loaded_data['checkshot'] = {}
        
        # If well already has checkshot data, append to it
        if well_name in self.loaded_data['checkshot']:
            existing_data = self.loaded_data['checkshot'][well_name]
            self.loaded_data['checkshot'][well_name] = pd.concat([existing_data, checkshot_data], ignore_index=True)
        else:
            self.loaded_data['checkshot'][well_name] = checkshot_data
        
        safe_print(f"Saved checkshot data for well: {well_name}")

    # Custom Trace Selection Methods
    def save_custom_trace_selection(self, well_name, trace_selection):
        """
        Save custom trace selection for a specific well.
        
        Args:
            well_name (str): Name of the well
            trace_selection (dict): Dictionary containing trace selection information
                {
                    'trace_index': int,
                    'inline': int,
                    'crossline': int,
                    'x': float,
                    'y': float,
                    'distance_from_well': float,
                    'selection_date': str,
                    'survey': str,
                    'attribute': str,
                    'enabled': bool
                }
        """
        try:
            # Initialize custom_trace_selections if not exists
            if 'custom_trace_selections' not in self.loaded_data:
                self.loaded_data['custom_trace_selections'] = {}
            
            # Save trace selection for the well
            self.loaded_data['custom_trace_selections'][well_name] = trace_selection
            
            safe_print(f"WellHandler: Saved custom trace selection for well {well_name} - "
                  f"Trace {trace_selection.get('trace_index')} at distance {trace_selection.get('distance_from_well', 0):.1f}m")
            
        except Exception as e:
            # safe_print(f"WellHandler: Error saving custom trace selection for {well_name}: {str(e)}")
            raise
    
    def get_custom_trace_selection(self, well_name):
        """
        Get custom trace selection for a specific well.
        
        Args:
            well_name (str): Name of the well
            
        Returns:
            dict or None: Trace selection information if exists, None otherwise
        """
        try:
            custom_selections = self.loaded_data.get('custom_trace_selections', {})
            return custom_selections.get(well_name, None)
            
        except Exception as e:
            # safe_print(f"WellHandler: Error getting custom trace selection for {well_name}: {str(e)}")
            return None
    
    def has_custom_trace_selection(self, well_name):
        """
        Check if a well has a custom trace selection.
        
        Args:
            well_name (str): Name of the well
            
        Returns:
            bool: True if well has custom trace selection, False otherwise
        """
        try:
            selection = self.get_custom_trace_selection(well_name)
            return selection is not None and selection.get('enabled', False)
            
        except Exception as e:
            # safe_print(f"WellHandler: Error checking custom trace selection for {well_name}: {str(e)}")
            return False
    
    def remove_custom_trace_selection(self, well_name):
        """
        Remove custom trace selection for a specific well.
        
        Args:
            well_name (str): Name of the well
        """
        try:
            custom_selections = self.loaded_data.get('custom_trace_selections', {})
            if well_name in custom_selections:
                del custom_selections[well_name]
                # safe_print(f"WellHandler: Removed custom trace selection for well {well_name}")
            # else:
            #     safe_print(f"WellHandler: No custom trace selection found for well {well_name}")
                
        except Exception as e:
            safe_print(f"WellHandler: Error removing custom trace selection for {well_name}: {str(e)}")
    
    def get_all_custom_trace_selections(self):
        """
        Get all custom trace selections.
        
        Returns:
            dict: Dictionary mapping well names to their trace selections
        """
        return self.loaded_data.get('custom_trace_selections', {})
    
    def enable_custom_trace_selection(self, well_name, enabled=True):
        """
        Enable or disable custom trace selection for a specific well.
        
        Args:
            well_name (str): Name of the well
            enabled (bool): Whether to enable or disable custom trace selection
        """
        try:
            selection = self.get_custom_trace_selection(well_name)
            if selection:
                selection['enabled'] = enabled
                self.save_custom_trace_selection(well_name, selection)
            #     safe_print(f"WellHandler: {'Enabled' if enabled else 'Disabled'} custom trace selection for well {well_name}")
            # else:
            #     safe_print(f"WellHandler: No custom trace selection found for well {well_name}")
                
        except Exception as e:
            safe_print(f"WellHandler: Error updating custom trace selection status for {well_name}: {str(e)}")
    
    def get_trace_for_well(self, well_name):
        """
        Get the appropriate trace information for a well.
        Returns custom trace selection if available and enabled, None otherwise.
        
        Args:
            well_name (str): Name of the well
            
        Returns:
            dict or None: Trace information if custom selection is available and enabled
        """
        try:
            if self.has_custom_trace_selection(well_name):
                selection = self.get_custom_trace_selection(well_name)
                # safe_print(f"WellHandler: Using custom trace {selection.get('trace_index')} for well {well_name}")
                return selection
            return None
            
        except Exception as e:
            # safe_print(f"WellHandler: Error getting trace for well {well_name}: {str(e)}")
            return None
