# seismic_handler.py

import os
import segyio
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import pickle
import time  # Add this import
from geoagent.utils.trace_spatial_indexer import TraceSpatialIndexer
from geoagent.utils.safe_print import safe_print

# Computational efficiency constants
STANDARD_TRACE_WINDOW = 5  # ±5 crosslines = 11 traces total for consistent windowing

class SeismicHandler:
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.loaded_data = {}
        self.project_files = []
        self.wavelet_folder = ''

        # KD-tree spatial indices for fast coordinate lookups
        self.spatial_indices = {}  # {survey_name: TraceSpatialIndexer}

        # Trace data cache to prevent duplicate SEGY file opens
        self._trace_cache = {}  # {cache_key: (traces, time_array, headerdata, timestamp)}
        self._cache_timeout = 30  # seconds - covers project load period

        if 'wavelets' not in self.loaded_data:
            self.loaded_data['wavelets'] = []
    def import_wavelet(self, file_path, wavelet_name):
        """
        Import a wavelet from a file and store it in memory.
        """
        try:
            metadata = {}
            wavelet_data = []
            
            with open(file_path, 'r') as f:
                # Read metadata
                for line in f:
                    if line.strip() == 'EOH':
                        break
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                    elif '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip()
                    elif 'WAVELET-TFS' in line or 'SAMPLE-RATE' in line:
                        key, value = line.split()
                        metadata[key.strip()] = value.strip()
                        
                
                # Extract specific metadata
                metadata['WAVELET-TFS'] = float(metadata.get('WAVELET-TFS', 0))
                metadata['SAMPLE-RATE'] = float(metadata.get('SAMPLE-RATE', 0))
                
                # Extract phase information
                phase_info = metadata.get('Phase manipulation', '')
                metadata['initial_peak_phase'] = float(phase_info.split('Initial peak phase = [')[1].split(']')[0]) if 'Initial peak phase' in phase_info else None
                metadata['rotated_phase'] = float(phase_info.split('Rotated phase = [')[1].split(']')[0]) if 'Rotated phase' in phase_info else None
                metadata['converted_to_zero_phase'] = phase_info.split('Converted to zero phase = [')[1].split(']')[0] == 'True' if 'Converted to zero phase' in phase_info else False

                # Extract time shift and scale factor
                time_shift_info = metadata.get('Time shift', '')
                metadata['time_shift'] = float(time_shift_info.split('Modified time shift = [')[1].split(']')[0]) if 'Modified time shift' in time_shift_info else None
                
                scale_factor_info = metadata.get('Scale factor', '')
                metadata['scale_factor'] = float(scale_factor_info.split('=')[1].strip()) if '=' in scale_factor_info else 1.0

                # Read wavelet data
                for line in f:
                    if line.strip() == 'EOD' or line.strip() == 'EOD---':
                        break
                    values = line.split()
                    if len(values) >= 2:
                        # Take first two values (time, amplitude) and ignore extra columns
                        try:
                            time_val = float(values[0])
                            amp_val = float(values[1])
                            wavelet_data.append([time_val, amp_val])
                        except (ValueError, IndexError):
                            # Skip malformed lines
                            continue
                    elif len(values) == 1:
                        # Single value - assume it's amplitude, generate time
                        try:
                            amp_val = float(values[0])
                            time_val = len(wavelet_data) * metadata.get('SAMPLE-RATE', 2.0)
                            wavelet_data.append([time_val, amp_val])
                        except ValueError:
                            continue
            
            wavelet_data = np.array(wavelet_data)
            
            # Create a dictionary to store wavelet information
            wavelet_info = {
                'name': wavelet_name,
                'data': wavelet_data,
                'metadata': metadata
            }
            
            # Add the wavelet to the loaded_data
            self.loaded_data['wavelets'].append(wavelet_info)
            
            safe_print(f"Wavelet '{wavelet_name}' imported successfully.")
            return True
        except Exception as e:
            safe_print(f"Error importing wavelet: {str(e)}")
            return False

    def load_wavelet(self, wavelet_name):
        """
        Load a wavelet from the project by name.
        """
        for wavelet in self.loaded_data['wavelets']:
            if wavelet['name'] == wavelet_name:
                return wavelet['data'],wavelet['metadata']
        return None,None

    def get_wavelet_list(self):
        """
        Return a list of all available wavelets.
        """
        return [wavelet['name'] for wavelet in self.loaded_data['wavelets']]
    
    def save_wavelet(self, wavelet_name, wavelet_data, metadata=None):
        """
        Save a wavelet to the project.
        
        Args:
            wavelet_name (str): Name of the wavelet
            wavelet_data (np.ndarray): Wavelet data
            metadata (dict): Optional metadata
        """
        if metadata is None:
            metadata = {}
            
        # Remove existing wavelet with same name
        self.loaded_data['wavelets'] = [w for w in self.loaded_data['wavelets'] if w['name'] != wavelet_name]
        
        # Add new wavelet
        wavelet_entry = {
            'name': wavelet_name,
            'data': wavelet_data,
            'metadata': metadata
        }
        self.loaded_data['wavelets'].append(wavelet_entry)
        safe_print(f"Added wavelet '{wavelet_name}' to loaded_data. Total wavelets: {len(self.loaded_data['wavelets'])}")
        
        # Save to project folder if available
        if hasattr(self, 'project_folder') and self.project_folder:
            safe_print(f"Saving project to: {self.project_folder}")
            self.save_project(self.project_folder)
        
        # Also save wavelets specifically
        self.save_wavelets_to_project()
        safe_print(f"Wavelet '{wavelet_name}' successfully saved and stored")
            
        return True

    def save_wavelets_to_project(self):
        """Save wavelets to project folder."""
        if hasattr(self, 'project_folder') and self.project_folder:
            try:
                # Save wavelets
                wavelets_path = os.path.join(self.project_folder, 'wavelets.pkl')
                with open(wavelets_path, 'wb') as f:
                    pickle.dump(self.loaded_data['wavelets'], f)
                safe_print(f"Saved {len(self.loaded_data['wavelets'])} wavelets to project")
            except Exception as e:
                safe_print(f"Error saving wavelets: {str(e)}")
                raise e

    def get_start_time(self, survey_name, attribute_name):
        if survey_name in self.loaded_data and 'volumes' in self.loaded_data[survey_name]:
            if attribute_name in self.loaded_data[survey_name]['volumes']:
                return self.loaded_data[survey_name]['volumes'][attribute_name].get('start_time', 0)
        return 0  # Return 0 if start time is not found


    def load_project(self, project_folder):
        self.project_folder = project_folder
        project_structure_path = os.path.join(project_folder, 'seismic_project_structure.pkl')
        if os.path.exists(project_structure_path):
            with open(project_structure_path, 'rb') as f:
                self.project_files = pickle.load(f)

        for survey in self.project_files:
            survey_folder = os.path.join(project_folder, survey['sub_folder_name'])
            survey_data = {}

            # Load headerdata and endpoints
            for file_name in ['headerdata', 'endpoints']:
                file_path = os.path.join(survey_folder, survey[file_name])
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        survey_data[file_name] = pickle.load(f)
                else:
                    survey_data[file_name] = {}  # Initialize empty dictionary if file does not exist

            # Ensure endpoints are a dictionary
            if 'endpoints' in survey_data and not isinstance(survey_data['endpoints'], dict):
                survey_data['endpoints'] = {
                    'inlines': {},
                    'crosslines': {}
                }

            # Load or rebuild spatial index
            spatial_index = self._load_spatial_index(survey_folder)
            if spatial_index:
                survey_data['spatial_index'] = spatial_index
                self.spatial_indices[survey['survey_name']] = spatial_index

            # Load volume metadata (not the actual SEGY data)
            survey_data['volumes'] = {}
            for volume in survey['survey_volumes']:
                volume_data = {
                    'segy_path': os.path.join(survey_folder, volume['segy']),
                    'attribute_name': volume['attribute_name']
                }

                # Load seis_params
                seis_params_path = os.path.join(survey_folder, volume['seis_params'])
                if os.path.exists(seis_params_path):
                    with open(seis_params_path, 'rb') as f:
                        volume_data['seis_params'] = pickle.load(f)

                survey_data['volumes'][volume['attribute_name']] = volume_data

            # Add the survey data to loaded_data dictionary
            self.loaded_data[survey['survey_name']] = survey_data
            
        # Load wavelets
        wavelets_path = os.path.join(project_folder, 'wavelets.pkl')
        if os.path.exists(wavelets_path):
            try:
                with open(wavelets_path, 'rb') as f:
                    self.loaded_data['wavelets'] = pickle.load(f)
                safe_print(f"Loaded {len(self.loaded_data['wavelets'])} wavelets from project")
            except Exception as e:
                safe_print(f"Error loading wavelets: {str(e)}")
                self.loaded_data['wavelets'] = []
        else:
            self.loaded_data['wavelets'] = []


    def save_project(self, project_folder):
        for survey in self.project_files:
            survey_folder = os.path.join(project_folder, survey['sub_folder_name'])
            os.makedirs(survey_folder, exist_ok=True)
            
            survey_data = self.loaded_data[survey['survey_name']]
            
            # Save headerdata and endpoints
            for file_name in ['headerdata', 'endpoints']:
                file_path = os.path.join(survey_folder, survey[file_name])
                with open(file_path, 'wb') as f:
                    pickle.dump(survey_data[file_name], f)
            
            # Save volume metadata
            for volume in survey['survey_volumes']:
                volume_data = survey_data['volumes'][volume['attribute_name']]
                
                # Save seis_params
                seis_params_path = os.path.join(survey_folder, volume['seis_params'])
                with open(seis_params_path, 'wb') as f:
                    pickle.dump(volume_data['seis_params'], f)
                
                # Update the segy path in project_files if it has changed
                volume['segy'] = os.path.basename(volume_data['segy_path'])
            # Save wavelets
        # Save wavelets
        wavelets_path = os.path.join(project_folder, 'wavelets.pkl')
        with open(wavelets_path, 'wb') as f:
            pickle.dump(self.loaded_data['wavelets'], f)

        # Save the updated project structure
        project_structure_path = os.path.join(project_folder, 'seismic_project_structure.pkl')
        with open(project_structure_path, 'wb') as f:
            pickle.dump(self.project_files, f)

        safe_print(f"Project saved successfully to {project_folder}")
        # safe_print(f"Updated project structure: {self.project_files}")

    def remove_volume(self, survey_name, attribute_name):
        """Remove a seismic volume and its associated files"""
        try:
            safe_print(f"SeismicHandler: *** REMOVE VOLUME CALLED *** '{attribute_name}' from survey '{survey_name}'")
            
            # Find the survey in project_files
            survey_entry = None
            for survey in self.project_files:
                if survey['survey_name'] == survey_name:
                    survey_entry = survey
                    break
                    
            if not survey_entry:
                safe_print(f"SeismicHandler: Survey '{survey_name}' not found")
                return False
                
            # Find the volume in survey_volumes
            volume_to_remove = None
            for i, volume in enumerate(survey_entry['survey_volumes']):
                if volume['attribute_name'] == attribute_name:
                    volume_to_remove = volume
                    volume_index = i
                    break
                    
            if not volume_to_remove:
                safe_print(f"SeismicHandler: Volume '{attribute_name}' not found in survey '{survey_name}'")
                return False
                
            # Remove physical SEGY file
            normalized_survey_name = self._normalize_folder_name(survey_name)
            survey_folder = os.path.join(self.project_folder, normalized_survey_name)
            segy_path = os.path.join(survey_folder, volume_to_remove['segy'])
            
            if os.path.exists(segy_path):
                os.remove(segy_path)
                safe_print(f"SeismicHandler: Removed SEGY file: {segy_path}")
            else:
                safe_print(f"SeismicHandler: SEGY file not found: {segy_path}")
                
            # Remove seis_params file
            seis_params_path = os.path.join(survey_folder, volume_to_remove['seis_params'])
            if os.path.exists(seis_params_path):
                os.remove(seis_params_path)
                safe_print(f"SeismicHandler: Removed seis_params file: {seis_params_path}")
            else:
                safe_print(f"SeismicHandler: seis_params file not found: {seis_params_path}")
                
            # Remove volume from project_files
            survey_entry['survey_volumes'].pop(volume_index)
            
            # Remove volume from loaded_data
            if survey_name in self.loaded_data and 'volumes' in self.loaded_data[survey_name]:
                if attribute_name in self.loaded_data[survey_name]['volumes']:
                    del self.loaded_data[survey_name]['volumes'][attribute_name]
                    
            # If this was the last volume in the survey, remove the entire survey
            if not survey_entry['survey_volumes']:
                # Remove survey folder if empty (only if no other files remain)
                try:
                    if os.path.exists(survey_folder) and not os.listdir(survey_folder):
                        os.rmdir(survey_folder)
                        safe_print(f"SeismicHandler: Removed empty survey folder: {survey_folder}")
                except OSError:
                    safe_print(f"SeismicHandler: Survey folder not empty, keeping: {survey_folder}")
                    
                # Remove survey from project_files and loaded_data
                self.project_files.remove(survey_entry)
                if survey_name in self.loaded_data:
                    del self.loaded_data[survey_name]
                    
            # Save updated project structure
            if self.project_folder:
                project_structure_path = os.path.join(self.project_folder, 'seismic_project_structure.pkl')
                with open(project_structure_path, 'wb') as f:
                    pickle.dump(self.project_files, f)
                    
            safe_print(f"SeismicHandler: Successfully removed volume '{attribute_name}' from survey '{survey_name}'")
            return True
            
        except Exception as e:
            safe_print(f"SeismicHandler: Error removing volume: {str(e)}")
            return False

    def extract_default_values_from_headers(self, text_header, binary_header, trace_header):
        # Extract coordinate scalar from SEGY trace header
        coord_scalar = self._extract_coordinate_scalar(trace_header)
        
        default_values = {
            'CDP_X': segyio.TraceField.CDP_X,
            'CDP_Y': segyio.TraceField.CDP_Y,
            'Inline': segyio.TraceField.INLINE_3D,
            'Crossline': segyio.TraceField.CROSSLINE_3D,
            'Sampling Rate': binary_header[segyio.BinField.Interval],
            'Number of Samples': binary_header[segyio.BinField.Samples],
            'Format': 'IBM',  # Assuming default as IBM, modify based on actual format
            'Coord_Mult_Factor': coord_scalar,
            'Start Time': segyio.TraceField.DelayRecordingTime,
            'End Time': None  # We'll calculate this in the dialog
        }
        return default_values
    
    def _extract_coordinate_scalar(self, trace_header):
        """
        Extract coordinate scalar/multiplier from SEGY trace header.
        
        In SEGY format, the coordinate scalar is typically stored in bytes 71-72
        and represents the multiplier to apply to coordinate values.
        
        Args:
            trace_header: SEGY trace header dictionary
            
        Returns:
            float: Coordinate multiplication factor
        """
        try:
            # Try to get coordinate scalar from standard SEGY fields
            # Common field names for coordinate scalar in segyio
            possible_scalar_fields = [
                71,  # Standard byte position for coordinate scalar
                72,  # Alternative byte position
                'SourceCoordinateUnit',
                'CoordinateUnits',
                'EnergySourceCoordinateUnit'
            ]
            
            for field in possible_scalar_fields:
                try:
                    if field in trace_header:
                        scalar_value = trace_header[field]
                        if scalar_value != 0:
                            safe_print(f"SeismicHandler: Found coordinate scalar {scalar_value} in field {field}")
                            # SEGY scalar interpretation:
                            # If positive: multiply by this value
                            # If negative: divide by absolute value
                            if scalar_value > 0:
                                return float(scalar_value)
                            elif scalar_value < 0:
                                return 1.0 / abs(float(scalar_value))
                except (KeyError, ValueError, TypeError):
                    continue
            
            # Try accessing by byte position directly
            try:
                # Byte 71-72 in trace header (coordinate scalar)
                if 71 in trace_header:
                    scalar_raw = trace_header[71]
                    if scalar_raw != 0:
                        safe_print(f"SeismicHandler: Found coordinate scalar {scalar_raw} at byte 71")
                        if scalar_raw > 0:
                            return float(scalar_raw)
                        elif scalar_raw < 0:
                            return 1.0 / abs(float(scalar_raw))
            except (KeyError, ValueError, TypeError):
                pass
                
            safe_print("SeismicHandler: No coordinate scalar found in SEGY headers, using default 0.01")
            return 0.01  # Default fallback
            
        except Exception as e:
            safe_print(f"SeismicHandler: Error extracting coordinate scalar: {str(e)}, using default 0.01")
            return 0.01
    
    def setProjectPath(self, project_folder):
        self.project_folder=project_folder
    
    def read_segy_headers(self, file_path):
        with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
            text_header = segyfile.text[0]
            binary_header = segyfile.bin

            trace_header = segyfile.header[0]
            trace_field_info = {k: (v, segyio.TraceField.__dict__[k]) for k, v in vars(segyio.TraceField).items() if not k.startswith('__')}
            trace_field_names = {v: k for k, v in vars(segyio.TraceField).items() if not k.startswith('__')}

            trace_header_info = []
            for i, key in enumerate(trace_field_names):
                field_name = trace_field_names.get(key, f"Unknown field ({key})")
                value = trace_header[key]
                trace_header_info.append({"Field Name": field_name, "Byte Pos": key, "Value": value})

        df_trace_header = pd.DataFrame(trace_header_info)
        return text_header, binary_header, df_trace_header, trace_header, list(trace_field_names.keys())

   
    def _normalize_folder_name(self, name):
        """Normalize folder name to lowercase with underscores."""
        return name.lower().replace(' ', '_')

    def _get_normalized_path(self, survey_name, file_name):
        """Get the normalized path for a file within a survey folder."""
        normalized_survey_name = self._normalize_folder_name(survey_name)
        return os.path.join(self.project_folder, f"{normalized_survey_name}", file_name)


    def import_segy_file(self, file_path, byte_positions, survey_name, attribute_name, progress_callback=None):
        if progress_callback:
            progress_callback(('import', 0))
        safe_print(f"in  import_segy_file survey_name: {survey_name} attribute_name: {attribute_name}")

        # Normalize survey name for folder structure
        normalized_survey_name = self._normalize_folder_name(survey_name)
        safe_print(f"in  import_segy_file normalized_survey_name: {normalized_survey_name} attribute_name: {attribute_name}")
        survey_folder = os.path.join(self.project_folder, normalized_survey_name)
        os.makedirs(survey_folder, exist_ok=True)

        # Check if survey exists in project_files, create if not
        survey_entry = next((s for s in self.project_files if s['survey_name'] == survey_name), None)
        if survey_entry is None:
            survey_entry = {
                'survey_name': survey_name,
                'sub_folder_name': normalized_survey_name,
                'headerdata': 'headerdata.pkl',
                'endpoints': 'endpoints.pkl',
                'survey_volumes': []
            }
            self.project_files.append(survey_entry)

        headerdata_file = os.path.join(survey_folder, 'headerdata.pkl')
        endpoints_file = os.path.join(survey_folder, 'endpoints.pkl')

        # Check if headerdata and endpoints already exist for this survey
        if os.path.exists(headerdata_file) and os.path.exists(endpoints_file):
            with open(headerdata_file, 'rb') as f:
                headerdata = pickle.load(f)
            with open(endpoints_file, 'rb') as f:
                endpoints = pickle.load(f)
            # Load or rebuild spatial index
            spatial_index = self._load_spatial_index(survey_folder)
            if progress_callback:
                progress_callback(('import', 50))
        else:
            # Calculate headerdata, endpoints, and spatial index for the first attribute
            headerdata, endpoints, spatial_index = self._calculate_headerdata_and_endpoints(file_path, byte_positions, progress_callback)

            # Save headerdata and endpoints for the survey
            with open(headerdata_file, 'wb') as f:
                pickle.dump(headerdata, f)
            with open(endpoints_file, 'wb') as f:
                pickle.dump(endpoints, f)

            # Save spatial index
            self._save_spatial_index(survey_folder, spatial_index)

            if progress_callback:
                progress_callback(('import', 95))

        # Extract time-related information
        with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
            dt = segyfile.samples[1] - segyfile.samples[0]
            start_time = segyfile.header[0][segyio.TraceField.DelayRecordingTime]
            num_samples = len(segyfile.samples)
            end_time = (num_samples - 1) * dt + start_time
            time_array = np.arange(start_time, end_time + dt, dt)

        # Prepare result
        segy_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{attribute_name}.sgy"
        segy_project_path = os.path.join(survey_folder, segy_filename)

        # Update survey_volumes in project_files
        volume_entry = next((v for v in survey_entry['survey_volumes'] if v['attribute_name'] == attribute_name), None)
        if volume_entry is None:
            volume_entry = {
                'attribute_name': attribute_name,
                'segy': segy_filename,
                'seis_params': f'{attribute_name}_seis_params.pkl'
            }
            survey_entry['survey_volumes'].append(volume_entry)
        else:
            volume_entry.update({
                'segy': segy_filename,
                'seis_params': f'{attribute_name}_seis_params.pkl'
            })

        # Prepare result
        result = {
            'survey_name': survey_name,
            'attribute_name': attribute_name,
            'segy': segy_filename,
            'segy_path': segy_project_path,
            'headerdata': headerdata.tolist() if isinstance(headerdata, np.ndarray) else headerdata,
            'endpoints': endpoints,
            'seis_params': byte_positions,
            'start_time': start_time,
            'end_time': end_time,
            'dt': dt,
            'num_samples': num_samples,
            'time_array': time_array.tolist() if isinstance(time_array, np.ndarray) else time_array
        }

        # Save seis_params for this attribute
        seis_params_file = os.path.join(survey_folder, f'{attribute_name}_seis_params.pkl')
        with open(seis_params_file, 'wb') as f:
            pickle.dump(byte_positions, f)

        # Update loaded_data
        if survey_name not in self.loaded_data:
            self.loaded_data[survey_name] = {
                'volumes': {},
                'headerdata': headerdata,
                'endpoints': endpoints,
                'spatial_index': spatial_index
            }
            # Store in spatial_indices for fast access
            if spatial_index:
                self.spatial_indices[survey_name] = spatial_index
        # Before the problematic assignment, add debugging
        # safe_print(f"Type of self.loaded_data['{survey_name}']: {type(self.loaded_data[survey_name])}")
        # safe_print(f"Contents of self.loaded_data['{survey_name}']: {self.loaded_data[survey_name]}")

        self.loaded_data[survey_name]['volumes'][attribute_name] = result

        if progress_callback:
            progress_callback(('import', 100))
        self.save_project( self.project_folder)

        return result

    def _calculate_endpoints(self, headerdata):
        # Ensure headerdata is a numpy array
        headerdata = np.array(headerdata)

        # Extract inline, crossline, x, and y data
        inlines = headerdata[:, 1]
        crosslines = headerdata[:, 2]
        x_coords = headerdata[:, 3]
        y_coords = headerdata[:, 4]

        # Get unique inlines and crosslines
        unique_inlines = np.unique(inlines)
        unique_crosslines = np.unique(crosslines)

        inline_endpoints = {}
        crossline_endpoints = {}

        # Calculate inline endpoints
        for il in unique_inlines:
            il_mask = inlines == il
            il_x = x_coords[il_mask]
            il_y = y_coords[il_mask]
            
            # Find the indices of min and max crosslines for this inline
            min_xl_idx = np.argmin(crosslines[il_mask])
            max_xl_idx = np.argmax(crosslines[il_mask])
            
            inline_endpoints[int(il)] = [
                [float(il_x[min_xl_idx]), float(il_y[min_xl_idx])],
                [float(il_x[max_xl_idx]), float(il_y[max_xl_idx])]
            ]

        # Calculate crossline endpoints
        for xl in unique_crosslines:
            xl_mask = crosslines == xl
            xl_x = x_coords[xl_mask]
            xl_y = y_coords[xl_mask]
            
            # Find the indices of min and max inlines for this crossline
            min_il_idx = np.argmin(inlines[xl_mask])
            max_il_idx = np.argmax(inlines[xl_mask])
            
            crossline_endpoints[int(xl)] = [
                [float(xl_x[min_il_idx]), float(xl_y[min_il_idx])],
                [float(xl_x[max_il_idx]), float(xl_y[max_il_idx])]
            ]

        return {
            'inlines': inline_endpoints,
            'crosslines': crossline_endpoints
        }

    def _build_spatial_index(self, headerdata):
        """
        Build KD-tree spatial index from headerdata.

        Args:
            headerdata: numpy array [trace_idx, inline, crossline, cdp_x, cdp_y]

        Returns:
            TraceSpatialIndexer instance or None if build fails
        """
        try:
            # Extract coordinates (columns 3-4: cdp_x, cdp_y)
            trace_coordinates = [(row[3], row[4]) for row in headerdata]

            # Extract metadata for each trace
            trace_metadata = [
                {
                    'trace_index': int(row[0]),
                    'inline': int(row[1]),
                    'crossline': int(row[2]),
                }
                for row in headerdata
            ]

            # Build spatial index
            spatial_index = TraceSpatialIndexer(trace_coordinates, trace_metadata)

            safe_print(f"✅ Built KD-tree spatial index for {len(trace_coordinates)} traces")
            return spatial_index

        except Exception as e:
            safe_print(f"⚠️ Warning: Failed to build spatial index: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_spatial_index(self, survey_folder, spatial_index):
        """
        Save spatial index to disk for fast project loading.

        Args:
            survey_folder: Path to survey folder
            spatial_index: TraceSpatialIndexer instance to save
        """
        if spatial_index is None:
            return

        try:
            index_path = os.path.join(survey_folder, 'kdtree_index.pkl')
            with open(index_path, 'wb') as f:
                pickle.dump(spatial_index, f, protocol=4)
            safe_print(f"✅ Saved KD-tree index to {index_path}")
        except Exception as e:
            safe_print(f"⚠️ Warning: Failed to save spatial index: {e}")

    def _load_spatial_index(self, survey_folder):
        """
        Load spatial index from disk, or rebuild if missing.

        Args:
            survey_folder: Path to survey folder

        Returns:
            TraceSpatialIndexer instance or None if load/rebuild fails
        """
        index_path = os.path.join(survey_folder, 'kdtree_index.pkl')

        # Try to load cached index
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    spatial_index = pickle.load(f)
                safe_print(f"✅ Loaded KD-tree index from cache ({index_path})")
                return spatial_index
            except Exception as e:
                safe_print(f"⚠️ Warning: Failed to load cached index, rebuilding: {e}")

        # Rebuild from headerdata if cache load failed or missing
        headerdata_path = os.path.join(survey_folder, 'headerdata.pkl')
        if os.path.exists(headerdata_path):
            try:
                with open(headerdata_path, 'rb') as f:
                    headerdata = pickle.load(f)
                safe_print(f"📊 Rebuilding KD-tree index from headerdata ({len(headerdata)} traces)...")
                spatial_index = self._build_spatial_index(headerdata)

                # ✅ FIX: Save the rebuilt index for future loads
                if spatial_index:
                    self._save_spatial_index(survey_folder, spatial_index)

                return spatial_index
            except Exception as e:
                safe_print(f"⚠️ Warning: Failed to rebuild index from headerdata: {e}")
                import traceback
                traceback.print_exc()

        return None

    # def get_data(self, survey_name, data_type, attribute_name=None):
    #     if survey_name in self.loaded_data:
    #         if data_type in ['headerdata', 'endpoints']:
    #             return self.loaded_data[survey_name].get(data_type)
    #         elif data_type == 'volumes' and attribute_name:
    #             return self.loaded_data[survey_name]['volumes'].get(attribute_name)
    #     return None
    def get_data(self, survey_name, data_type, attribute_name=None):
        # Check if the survey exists in loaded_data
        if survey_name not in self.loaded_data:
            safe_print(f"Survey '{survey_name}' not found in loaded_data.")
            return None

        survey_data = self.loaded_data[survey_name]

        # Ensure the survey data is in dictionary format
        if not isinstance(survey_data, dict):
            safe_print(f"Survey data for '{survey_name}' is not in the correct format. Expected a dictionary.")
            return None

        # Fetch data based on the type requested
        if data_type in ['headerdata', 'endpoints']:
            return survey_data.get(data_type)
        elif data_type == 'volumes' and attribute_name:
            return survey_data['volumes'].get(attribute_name)
        
        return None

    def get_survey_data(self, survey_name):
        """
        Get the full survey data structure including volumes, headerdata, and endpoints.
        
        Args:
            survey_name (str): Name of the survey
            
        Returns:
            dict: Full survey data structure or None if survey not found
        """
        if survey_name not in self.loaded_data:
            safe_print(f"Survey '{survey_name}' not found in loaded_data.")
            return None
            
        return self.loaded_data[survey_name]

    def get_available_surveys(self):
        survey_list=list(self.loaded_data.keys())
        return [s for s in survey_list if s !='wavelets']

    def get_survey_key_values(self, survey_name, key):
        for survey in self.project_files:
            if survey.get('survey_name') == survey_name:
                return survey.get(key)
        return None

    def get_available_attributes(self, survey_name):
        if survey_name in self.loaded_data:
            if survey_name !='wavelets':
                return list(self.loaded_data[survey_name]['volumes'].keys())
        return []

    def get_available_datatypes(self):
        datatypes = []
        for survey_name, survey_data in self.loaded_data.items():
            datatypes.append(f"seismic_{survey_name}")
            safe_print(f"survey_data {survey_name}")
            if survey_name !='wavelets':
                for attribute_name in survey_data['volumes'].keys():
                    datatypes.append(f"seismic_{survey_name}_{attribute_name}")
        return datatypes


    def get_nearest_in_crosslines(self, survey_name, x, y):
        """
        Find nearest inline/crossline using KD-tree spatial index.
        Falls back to endpoints-based search if KD-tree unavailable.

        Args:
            survey_name: Name of the seismic survey
            x: X coordinate
            y: Y coordinate

        Returns:
            Tuple of (inline, crossline, trace_index) where trace_index is None if unavailable
        """
        # Try KD-tree lookup first (O(log N) performance)
        spatial_index = self.spatial_indices.get(survey_name)
        if spatial_index:
            result = spatial_index.find_nearest_trace(x, y, max_distance=5000.0)

            if result:
                trace_idx, distance = result
                metadata = spatial_index.trace_metadata[trace_idx]
                inline = metadata['inline']
                crossline = metadata['crossline']
                trace_index = metadata['trace_index']
                return inline, crossline, trace_index

        # Fallback to original endpoints-based method (O(N+M) performance)
        inline, crossline = self._get_nearest_in_crosslines_fallback(survey_name, x, y)
        return inline, crossline, None  # No trace_index available from fallback

    def _get_nearest_in_crosslines_fallback(self, survey_name, x, y):
        """
        Original brute-force implementation (kept as fallback).
        Uses endpoints to find nearest inline/crossline.

        Args:
            survey_name: Name of the seismic survey
            x: X coordinate
            y: Y coordinate

        Returns:
            Tuple of (inline, crossline) or (None, None) if not found
        """
        endpoints = self.get_data(survey_name, 'endpoints')
        if not endpoints or 'inlines' not in endpoints or 'crosslines' not in endpoints:
            safe_print(f"Endpoints data not found or incorrectly formatted for survey {survey_name}")
            return None, None

        in_lines = endpoints['inlines']
        crosslines = endpoints['crosslines']

        nearest_inline = min(in_lines.keys(), key=lambda k: min(
            ((x - in_lines[k][0][0])**2 + (y - in_lines[k][0][1])**2)**0.5,
            ((x - in_lines[k][1][0])**2 + (y - in_lines[k][1][1])**2)**0.5
        )) if in_lines else None

        nearest_crossline = min(crosslines.keys(), key=lambda k: min(
            ((x - crosslines[k][0][0])**2 + (y - crosslines[k][0][1])**2)**0.5,
            ((x - crosslines[k][1][0])**2 + (y - crosslines[k][1][1])**2)**0.5
        )) if crosslines else None

        # Ensure return values are integers
        if nearest_inline is not None:
            nearest_inline = int(nearest_inline)
        if nearest_crossline is not None:
            nearest_crossline = int(nearest_crossline)

        safe_print(f'get_nearest_in_crosslines result: inline={nearest_inline}, crossline={nearest_crossline} (fallback)')
        return nearest_inline, nearest_crossline


    def get_trace_indices_line(self, survey_name, attribute_name, line_number, is_inline):
        headerdata = self.get_data(survey_name, 'headerdata')
        # volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        if headerdata is None:
            return []
        if is_inline:
            return [row[0] for row in headerdata if row[1] == line_number ]
        else:
            return [row[0] for row in headerdata if row[2] == line_number ]


    def get_traces_for_line(self, survey_name, attribute_name, line_number, is_inline, number_of_traces=None):
        # Create cache key for this specific request
        cache_key = f"{survey_name}_{attribute_name}_{'IL' if is_inline else 'XL'}{line_number}_{number_of_traces}"

        # Check cache first (prevents duplicate SEGY file opens)
        if cache_key in self._trace_cache:
            cached_data, cached_time, cached_header, timestamp = self._trace_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                safe_print(f"SeismicHandler: ⚡ Using cached traces for {'inline' if is_inline else 'crossline'} {line_number} (saved disk I/O)")
                return cached_data, cached_time, cached_header

        headerdata = self.get_data(survey_name, 'headerdata')
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)

        if headerdata is None or volume_data is None:
            return None, None, None

        # Ensure line_number is integer
        line_number = int(line_number)

        filtered_headerdata = [row for row in headerdata if int(row[1 if is_inline else 2]) == line_number]

        # Sort traces by crossline/inline for proper ordering
        if is_inline:
            filtered_headerdata.sort(key=lambda x: int(x[2]))  # Sort by crossline
        else:
            filtered_headerdata.sort(key=lambda x: int(x[1]))  # Sort by inline

        # Limit the number of traces if specified, otherwise take all
        if number_of_traces is not None:
            filtered_headerdata = filtered_headerdata[:number_of_traces]

        safe_print(f"SeismicHandler: Extracting {len(filtered_headerdata)} traces for {'inline' if is_inline else 'crossline'} {line_number}")

        traces = []
        # Use the full segy_path directly
        file_path = volume_data['segy_path']

        safe_print(f"SeismicHandler: Opening SEGY file: {file_path}")

        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            f.mmap()
            for row in filtered_headerdata:
                # Ensure trace index is integer
                trace_index = int(row[0])
                traces.append(f.trace[trace_index])

        time_array = self.get_time_array(survey_name, attribute_name)

        # Cache the result for subsequent calls
        traces_array = np.array(traces)
        self._trace_cache[cache_key] = (traces_array, time_array, filtered_headerdata, time.time())

        return traces_array, time_array, filtered_headerdata
    
    def get_traces_by_indices(self, survey_name, attribute_name, trace_indices):
        """
        Extract traces from a SEGY file using pre-computed trace indices for optimization.
        
        This method is optimized for attribute changes on existing line selections.
        It bypasses the header data filtering and directly extracts traces using
        the provided trace indices.
        
        Args:
            survey_name (str): Name of the survey
            attribute_name (str): Name of the seismic attribute
            trace_indices (list): List of trace indices to extract
            
        Returns:
            tuple: (traces_array, time_array) or (None, None) if error
        """
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        
        if volume_data is None:
            safe_print(f"SeismicHandler: No volume data found for {survey_name}, {attribute_name}")
            return None, None
        
        if not trace_indices:
            safe_print(f"SeismicHandler: No trace indices provided for extraction")
            return None, None
            
        traces = []
        file_path = volume_data['segy_path']
        
        try:
            with segyio.open(file_path, "r", ignore_geometry=True) as f:
                f.mmap()
                for trace_index in trace_indices:
                    # Ensure trace index is integer and within file bounds
                    trace_index = int(trace_index)
                    if trace_index < len(f.trace):
                        traces.append(f.trace[trace_index])
                    else:
                        safe_print(f"SeismicHandler: Warning - trace index {trace_index} out of bounds")
                        
            time_array = self.get_time_array(survey_name, attribute_name)
            safe_print(f"SeismicHandler: Successfully extracted {len(traces)} traces using stored indices")
            return np.array(traces), time_array
            
        except Exception as e:
            safe_print(f"SeismicHandler: Error extracting traces by indices: {e}")
            return None, None
    
    def get_traces_for_line_range(self, survey_name, attribute_name, line_number, is_inline, line_range):
        headerdata = self.get_data(survey_name, 'headerdata')
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        
        if headerdata is None or volume_data is None:
            return None, None, None

        if is_inline:
            filtered_headerdata = [row for row in headerdata if row[1] == line_number and line_range[0] <= row[2] <= line_range[1]]
            filtered_headerdata.sort(key=lambda x: x[2])  # Sort by crossline
        else:
            filtered_headerdata = [row for row in headerdata if row[2] == line_number and line_range[0] <= row[1] <= line_range[1]]
            filtered_headerdata.sort(key=lambda x: x[1])  # Sort by inline
            
        safe_print(f"SeismicHandler: Found {len(filtered_headerdata)} traces for line {line_number}")
        if len(filtered_headerdata) > 0:
            safe_print(f"SeismicHandler: Sample trace indices: {[row[0] for row in filtered_headerdata[:3]]}")
            safe_print(f"SeismicHandler: Sample trace index types: {[type(row[0]) for row in filtered_headerdata[:3]]}")
        
        traces = []
        # Use the full segy_path directly
        file_path = volume_data['segy_path']
        
        safe_print(f"SeismicHandler: Opening SEGY file: {file_path}")
        
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            f.mmap()
            for row in filtered_headerdata:
                # Ensure trace index is integer
                trace_index = int(row[0])
                traces.append(f.trace[trace_index])
        
        time_array = self.get_time_array(survey_name, attribute_name)
        return np.array(traces), time_array, filtered_headerdata

    def get_num_samples(self,survey_name,attribute_name):
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        start_time = volume_data['seis_params']['Start Time']
        end_time = volume_data['seis_params']['End Time']
        dt = volume_data['seis_params']['Sampling Rate']/1000
        time_array = np.arange(start_time, end_time + dt, dt)
        return len(time_array)
    def get_time_array(self,survey_name,attribute_name):
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        start_time = volume_data['seis_params']['Start Time']
        end_time = volume_data['seis_params']['End Time']
        sampling_rate = volume_data['seis_params']['Sampling Rate']
        
        # Convert sampling rate to milliseconds
        # If sampling rate is in microseconds (e.g., 2000.0), convert to ms (2.0)
        # If sampling rate is already in milliseconds (e.g., 2.0), use as is
        if sampling_rate > 100:  # Assume values > 100 are in microseconds
            dt = sampling_rate / 1000.0
        else:  # Assume values <= 100 are already in milliseconds
            dt = sampling_rate
            
        time_array = np.arange(start_time, end_time + dt, dt)
        return time_array

    def get_crossline(self, survey_name, attribute_name, inline_number, crossline_number):
        headerdata = self.get_data(survey_name, 'headerdata')
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        if headerdata is None or volume_data is None:
            return None, None

        all_crosslines = sorted(set(row[2] for row in headerdata))
        actual_crossline = self._get_nearest_number(crossline_number, all_crosslines)
        
        inlines = sorted(set(row[1] for row in headerdata if row[2] == actual_crossline))
        actual_inline = self._get_nearest_number(inline_number, inlines)
        
        restricted_inlines = self._get_restricted_range(inlines, actual_inline)
        
        trace_indices = [row[0] for row in headerdata if row[2] == actual_crossline and row[1] in restricted_inlines]
        
        # file_path = os.path.join(self.project_folder, survey_name, volume_data['segy'])
        file_path = self._get_normalized_path(survey_name, volume_data['segy_path'])
        trace_data, samples = self._get_trace_data(file_path, trace_indices)
        return trace_data, samples

    def get_along_well_path(self, survey_name, well_path):
        headerdata = self.get_data(survey_name, 'headerdata')
        if headerdata is None:
            return []

        traces_along_path = []
        for well_point in well_path:
            well_x, well_y = well_point
            distances = np.sqrt((headerdata[:, 3] - well_x)**2 + (headerdata[:, 4] - well_y)**2)
            nearest_indices = np.argsort(distances)[:11]  # 5 traces left and right plus the nearest trace
            traces_along_path.extend(headerdata[nearest_indices])

        return traces_along_path

    # Helper methods
    def _get_nearest_number(self, target, number_list):
        return min(number_list, key=lambda x: abs(x - target))

    def _get_restricted_range(self, numbers, target_number, window=STANDARD_TRACE_WINDOW):
        """
        Get restricted range of numbers around target with standardized window size
        
        Args:
            numbers: List of numbers to restrict
            target_number: Target number to center window around
            window: Window size (default: STANDARD_TRACE_WINDOW)
            
        Returns:
            List of numbers in restricted range (±window from target)
        """
        index = numbers.index(target_number)
        start_index = max(0, index - window)
        end_index = min(len(numbers), index + window + 1)
        return numbers[start_index:end_index]


    def _get_trace_data(self, file_path, trace_indices):
        trace_data = []
        # safe_print(f"in _get_trace_data file path: {file_path}")
        try:
            with segyio.open(file_path, "r", ignore_geometry=True) as f:
                f.mmap()
                # Convert trace_indices to integers if they aren't already
                for trace_index in trace_indices:
                    try:
                        idx = int(trace_index)  # Explicitly convert to integer
                        trace_data.append(f.trace[idx])
                    except (ValueError, TypeError) as e:
                        safe_print(f"Invalid trace index {trace_index}: {str(e)}")
                        continue
                samples = f.samples
            return np.array(trace_data), np.array(samples)
        except IOError as e:
            safe_print(f"IOError when reading file {file_path}: {str(e)}")
            return None, None
        except Exception as e:
            safe_print(f"Unexpected error when reading file {file_path}: {str(e)}")
            return None, None

    def get_trace_data_by_index(self, survey_name, attribute_name, trace_index, inline=None, crossline=None):
        """
        Extract trace data using trace_index directly (O(1) - optimized).
        Skips the O(N) headerdata search by using trace_index from KD-tree.

        Args:
            survey_name: Name of the seismic survey
            attribute_name: Name of the seismic attribute
            trace_index: Direct trace index from KD-tree metadata
            inline: Inline number (for metadata only)
            crossline: Crossline number (for metadata only)

        Returns:
            Tuple of (traces, time_array, None)
        """
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)

        if volume_data is None:
            return None, None, None

        # Direct O(1) trace access using trace_index from KD-tree
        file_path = self._get_normalized_path(survey_name, volume_data['segy_path'])
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            f.mmap()
            traces = [f.trace[trace_index]]

        time_array = self.get_time_array(survey_name, attribute_name)
        return np.array(traces), time_array, None

    def get_trace_data(self, survey_name, attribute_name, inline, crossline):
        import time

        # ⏱️ TIMING: Get headerdata and volume_data
        get_data_start = time.time()
        headerdata = self.get_data(survey_name, 'headerdata')
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        get_data_time = (time.time() - get_data_start) * 1000
        safe_print(f"[TIMING]     get_data (headerdata + volume_data): {get_data_time:.2f} ms")

        if headerdata is None or volume_data is None:
            return None, None, None

        # try:
        # Convert inline and crossline to integers
        inline = int(inline)
        crossline = int(crossline)

        # ⏱️ TIMING: Search headerdata for trace indices (O(N) - BOTTLENECK?)
        search_start = time.time()
        trace_indices = [int(row[0]) for row in headerdata
                        if int(row[1]) == inline and int(row[2]) == crossline]
        search_time = (time.time() - search_start) * 1000
        safe_print(f"[TIMING]     headerdata search (O(N)={len(headerdata)} rows): {search_time:.2f} ms")

        if not trace_indices:
            return None, None, None

        # ⏱️ TIMING: SEGY file I/O
        segy_start = time.time()
        file_path = self._get_normalized_path(survey_name, volume_data['segy_path'])
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            f.mmap()
            traces = [f.trace[idx] for idx in trace_indices]
        segy_time = (time.time() - segy_start) * 1000
        safe_print(f"[TIMING]     SEGY file I/O ({len(trace_indices)} traces): {segy_time:.2f} ms")

        time_array = self.get_time_array(survey_name, attribute_name)
        return np.array(traces), time_array, None

        # except (ValueError, TypeError) as e:
        #     safe_print(f"Error processing trace indices: {str(e)}")
        #     return None, None, None
        # except Exception as e:
        #     safe_print(f"Unexpected error in get_trace_data: {str(e)}")
        #     return None, None, None

    def get_inline(self, survey_name, attribute_name, inline_number, crossline_number):
        try:
            headerdata = self.get_data(survey_name, 'headerdata')
            volume_data = self.get_data(survey_name, 'volumes', attribute_name)
            if headerdata is None or volume_data is None:
                return None, None

            # Convert inline_number and crossline_number to integers
            inline_number = int(inline_number)
            crossline_number = int(crossline_number)

            all_inlines = sorted(set(int(row[1]) for row in headerdata))
            actual_inline = self._get_nearest_number(inline_number, all_inlines)
            
            crosslines = sorted(set(int(row[2]) for row in headerdata if int(row[1]) == actual_inline))
            actual_crossline = self._get_nearest_number(crossline_number, crosslines)
            
            restricted_crosslines = self._get_restricted_range(crosslines, actual_crossline)
            
            trace_indices = [int(row[0]) for row in headerdata 
                            if int(row[1]) == actual_inline and int(row[2]) in restricted_crosslines]
            
            file_path = self._get_normalized_path(survey_name, volume_data['segy_path'])
            trace_data, samples = self._get_trace_data(file_path, trace_indices)
            
            if trace_data is None or samples is None:
                safe_print(f"Failed to read data from {file_path}")
                return None, None

            # Get the time array
            times = samples
            return trace_data, times

        except (ValueError, TypeError) as e:
            safe_print(f"Error processing inline/crossline numbers: {str(e)}")
            return None, None
        except Exception as e:
            safe_print(f"Unexpected error in get_inline: {str(e)}")
            return None, None

    def get_full_line(self, survey_name, attribute_name, line_number, is_inline=True):
        """
        Extract ALL traces for a complete inline or crossline (not just ±5 traces)
        
        Args:
            survey_name (str): Name of the survey
            attribute_name (str): Name of the seismic attribute
            line_number (int): Line number to extract
            is_inline (bool): True for inline, False for crossline
            
        Returns:
            tuple: (trace_data, time_array, header_data) or (None, None, None) if error
        """
        try:
            headerdata = self.get_data(survey_name, 'headerdata')
            volume_data = self.get_data(survey_name, 'volumes', attribute_name)
            if headerdata is None or volume_data is None:
                return None, None, None

            # Convert line_number to integer
            line_number = int(line_number)

            if is_inline:
                # Get ALL traces for this inline (no restriction)
                filtered_headerdata = [row for row in headerdata if int(row[1]) == line_number]
                # Sort by crossline for proper ordering
                filtered_headerdata.sort(key=lambda x: int(x[2]))
                line_type = "inline"
            else:
                # Get ALL traces for this crossline (no restriction)
                filtered_headerdata = [row for row in headerdata if int(row[2]) == line_number]
                # Sort by inline for proper ordering
                filtered_headerdata.sort(key=lambda x: int(x[1]))
                line_type = "crossline"
            
            if not filtered_headerdata:
                safe_print(f"No traces found for {line_type} {line_number}")
                return None, None, None
                
            safe_print(f"Found {len(filtered_headerdata)} traces for {line_type} {line_number}")
            
            # Extract trace indices
            trace_indices = [int(row[0]) for row in filtered_headerdata]
            
            # Get trace data
            file_path = volume_data['segy_path']
            safe_print(f"Opening SEGY file: {file_path}")
            
            traces = []
            with segyio.open(file_path, "r", ignore_geometry=True) as f:
                f.mmap()
                for trace_idx in trace_indices:
                    traces.append(f.trace[trace_idx])
            
            # Get time array
            time_array = self.get_time_array(survey_name, attribute_name)
            
            safe_print(f"Successfully extracted {len(traces)} traces for {line_type} {line_number}")
            return np.array(traces), time_array, filtered_headerdata

        except (ValueError, TypeError) as e:
            safe_print(f"Error processing line number: {str(e)}")
            return None, None, None
        except Exception as e:
            safe_print(f"Unexpected error in get_full_line: {str(e)}")
            return None, None, None

    def _calculate_headerdata_and_endpoints(self, file_path, byte_positions, progress_callback=None):
        try:
            with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
                if progress_callback:
                    progress_callback(('import', 10))

                total_traces = segyfile.tracecount
                headerdata = []
                
                # Header reading progress
                for i in range(total_traces):
                    if progress_callback and i % 1000 == 0:
                        progress_callback(('import', 10 + int((i / total_traces) * 60)))
                    header = segyfile.header[i]
                    
                    # Ensure all values are converted to integers
                    cdp_x = int(header[byte_positions['CDP_X']])
                    cdp_y = int(header[byte_positions['CDP_Y']])
                    inline = int(header[byte_positions['Inline']])
                    crossline = int(header[byte_positions['Crossline']])
                    headerdata.append([i, inline, crossline, cdp_x, cdp_y])
                
                headerdata = np.array(headerdata, dtype=np.float64)  # Use float64 for coordinates
                headerdata[:, 3:] = headerdata[:, 3:] * float(byte_positions['Coord_Mult_Factor'])

            if progress_callback:
                progress_callback(('import', 70))

            # Calculate endpoints
            endpoints = self._calculate_endpoints(headerdata)

            if progress_callback:
                progress_callback(('import', 80))

            # Build KD-tree spatial index
            spatial_index = self._build_spatial_index(headerdata)

            if progress_callback:
                progress_callback(('import', 90))

            return headerdata, endpoints, spatial_index

        except Exception as e:
            safe_print(f"Error in _calculate_headerdata_and_endpoints: {str(e)}")
            raise


    def verify_segy_integrity(self, survey_name, attribute_name):
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        if volume_data is None:
            return False

        file_path = self._get_normalized_path(survey_name, volume_data['segy_path'])
        try:
            with segyio.open(file_path, "r", ignore_geometry=True) as f:
                # Try to read the first and last trace
                f.trace[0]
                f.trace[-1]
            return True
        except Exception as e:
            safe_print(f"SEGY file integrity check failed for {file_path}: {str(e)}")
            return False

    def load_headerdata(self, survey_name):
        headerdata_file = os.path.join(self.project_folder, self._normalize_folder_name(survey_name), 'headerdata.pkl')
        if os.path.exists(headerdata_file):
            with open(headerdata_file, 'rb') as f:
                return pickle.load(f)
        return None

    def load_endpoints(self, survey_name):
        endpoints_file = os.path.join(self.project_folder, self._normalize_folder_name(survey_name), 'endpoints.pkl')
        if os.path.exists(endpoints_file):
            with open(endpoints_file, 'rb') as f:
                return pickle.load(f)
        return None

    def load_seis_params(self, survey_name, attribute_name):
        seis_params_file = os.path.join(self.project_folder, self._normalize_folder_name(survey_name), f'{attribute_name}_seis_params.pkl')
        if os.path.exists(seis_params_file):
            with open(seis_params_file, 'rb') as f:
                return pickle.load(f)
        return None

    def get_headerdata(self, survey_name):
        if survey_name in self.loaded_data and 'headerdata' in self.loaded_data[survey_name]:
            return self.loaded_data[survey_name]['headerdata']
        return self.load_headerdata(survey_name)

    def get_endpoints(self, survey_name):
        if survey_name in self.loaded_data and 'endpoints' in self.loaded_data[survey_name]:
            return self.loaded_data[survey_name]['endpoints']
        return self.load_endpoints(survey_name)

    def get_seis_params(self, survey_name, attribute_name):
        if survey_name in self.loaded_data and 'volumes' in self.loaded_data[survey_name]:
            if attribute_name in self.loaded_data[survey_name]['volumes']:
                return self.loaded_data[survey_name]['volumes'][attribute_name].get('seis_params')
        return self.load_seis_params(survey_name, attribute_name)

    def get_nearest_coordinates_batch(self, survey_name, xy_points):
        """
        Convert multiple (X,Y) points to (inline,crossline) coordinates using batch processing
        
        Args:
            survey_name (str): Name of the survey
            xy_points (list): List of (x, y) coordinate tuples
            
        Returns:
            list: List of (inline, crossline) tuples corresponding to input points
        """
        safe_print(f'get_nearest_coordinates_batch: Processing {len(xy_points)} coordinate points')
        
        endpoints = self.get_data(survey_name, 'endpoints')
        if not endpoints or 'inlines' not in endpoints or 'crosslines' not in endpoints:
            safe_print(f"Endpoints data not found or incorrectly formatted for survey {survey_name}")
            return [(None, None)] * len(xy_points)

        in_lines = endpoints['inlines']
        crosslines = endpoints['crosslines']
        
        coordinate_results = []
        
        for x, y in xy_points:
            # Find nearest inline using existing logic
            nearest_inline = min(in_lines.keys(), key=lambda k: min(
                ((x - in_lines[k][0][0])**2 + (y - in_lines[k][0][1])**2)**0.5,
                ((x - in_lines[k][1][0])**2 + (y - in_lines[k][1][1])**2)**0.5
            )) if in_lines else None

            # Find nearest crossline using existing logic
            nearest_crossline = min(crosslines.keys(), key=lambda k: min(
                ((x - crosslines[k][0][0])**2 + (y - crosslines[k][0][1])**2)**0.5,
                ((x - crosslines[k][1][0])**2 + (y - crosslines[k][1][1])**2)**0.5
            )) if crosslines else None
            
            # Ensure integer conversion
            if nearest_inline is not None:
                nearest_inline = int(nearest_inline)
            if nearest_crossline is not None:
                nearest_crossline = int(nearest_crossline)
            
            coordinate_results.append((nearest_inline, nearest_crossline))
            
        safe_print(f'get_nearest_coordinates_batch: Converted {len(coordinate_results)} coordinate pairs')
        return coordinate_results

    def get_traces_for_multiple_points(self, survey_name, attribute_name, xy_points):
        """
        Extract traces for multiple (X,Y) coordinate points using batch processing
        
        Args:
            survey_name (str): Name of the survey
            attribute_name (str): Name of the seismic attribute
            xy_points (list): List of (x, y) coordinate tuples
            
        Returns:
            tuple: (traces_array, times_array, coordinates_list, headerdata_list)
                - traces_array: numpy array of shape (num_points, num_samples)
                - times_array: time array for traces
                - coordinates_list: list of (x, y) coordinates
                - headerdata_list: list of (inline, crossline) for each trace
        """
        safe_print(f'get_traces_for_multiple_points: Extracting traces for {len(xy_points)} points')
        
        headerdata = self.get_data(survey_name, 'headerdata')
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        
        if headerdata is None or volume_data is None:
            safe_print(f"Required data not found for survey {survey_name}, attribute {attribute_name}")
            return None, None, None, None

        # Convert (X,Y) coordinates to (inline,crossline) using batch processing
        inline_crossline_pairs = self.get_nearest_coordinates_batch(survey_name, xy_points)
        
        # Extract traces for each coordinate pair
        extracted_traces = []
        valid_coordinates = []
        valid_headerdata = []
        
        # Use the full segy_path directly
        file_path = volume_data['segy_path']
        
        try:
            with segyio.open(file_path, "r", ignore_geometry=True) as f:
                f.mmap()
                
                for i, ((x, y), (inline, crossline)) in enumerate(zip(xy_points, inline_crossline_pairs)):
                    if inline is None or crossline is None:
                        safe_print(f"Skipping point {i}: Invalid coordinates ({x}, {y})")
                        continue
                    
                    # Convert to integers for comparison
                    inline = int(inline)
                    crossline = int(crossline)
                    
                    # Find trace indices for this inline/crossline
                    trace_indices = [int(row[0]) for row in headerdata 
                                   if int(row[1]) == inline and int(row[2]) == crossline]
                    
                    if trace_indices:
                        # Extract the first matching trace
                        trace_data = f.trace[trace_indices[0]]
                        extracted_traces.append(trace_data)
                        valid_coordinates.append((x, y))
                        valid_headerdata.append((inline, crossline))
                    else:
                        safe_print(f"No trace found for point {i}: inline {inline}, crossline {crossline}")
                        
        except Exception as e:
            safe_print(f"Error reading traces from {file_path}: {str(e)}")
            return None, None, None, None
        
        if not extracted_traces:
            safe_print("No valid traces extracted")
            return None, None, None, None
            
        # Convert to numpy array and get time array
        traces_array = np.array(extracted_traces)
        time_array = self.get_time_array(survey_name, attribute_name)
        
        safe_print(f'get_traces_for_multiple_points: Successfully extracted {len(extracted_traces)} traces')
        safe_print(f'Trace array shape: {traces_array.shape}')
        
        return traces_array, time_array, valid_coordinates, valid_headerdata

    def get_traces_for_line_centered(self, survey_name, attribute_name, line_number, center_crossline, is_inline, traces_each_side=100):
        """
        Extract traces for a line centered around a specific crossline/inline position
        
        Args:
            survey_name (str): Name of the survey
            attribute_name (str): Name of the seismic attribute  
            line_number (int): Line number to extract
            center_crossline (int): Crossline to center around (for inline) or inline to center around (for crossline)
            is_inline (bool): True for inline, False for crossline
            traces_each_side (int): Number of traces to extract on each side of center (default 100 = total 201)
            
        Returns:
            tuple: (trace_data, time_array, header_data) or (None, None, None) if error
        """
        headerdata = self.get_data(survey_name, 'headerdata')
        volume_data = self.get_data(survey_name, 'volumes', attribute_name)
        
        if headerdata is None or volume_data is None:
            return None, None, None

        # Ensure integers
        line_number = int(line_number)
        center_crossline = int(center_crossline)
        
        # Filter for the specific line
        if is_inline:
            line_traces = [row for row in headerdata if int(row[1]) == line_number]
            # Sort by crossline and find center position
            line_traces.sort(key=lambda x: int(x[2]))
            center_field = 2  # crossline field
        else:
            line_traces = [row for row in headerdata if int(row[2]) == line_number]
            # Sort by inline and find center position
            line_traces.sort(key=lambda x: int(x[1]))
            center_field = 1  # inline field
            
        if not line_traces:
            safe_print(f"SeismicHandler: No traces found for {'inline' if is_inline else 'crossline'} {line_number}")
            return None, None, None
            
        # Find the trace closest to the center position
        center_idx = 0
        min_distance = float('inf')
        
        for i, trace in enumerate(line_traces):
            distance = abs(int(trace[center_field]) - center_crossline)
            if distance < min_distance:
                min_distance = distance
                center_idx = i
                
        safe_print(f"SeismicHandler: Center trace found at index {center_idx}, crossline {int(line_traces[center_idx][center_field])}, distance from click: {min_distance}")
        
        # Extract traces around the center
        start_idx = max(0, center_idx - traces_each_side)
        end_idx = min(len(line_traces), center_idx + traces_each_side + 1)
        
        filtered_headerdata = line_traces[start_idx:end_idx]
        
        safe_print(f"SeismicHandler: Extracting {len(filtered_headerdata)} traces centered around position {center_crossline}")
        safe_print(f"SeismicHandler: Trace range: {int(filtered_headerdata[0][center_field])} to {int(filtered_headerdata[-1][center_field])}")
        
        # Extract trace data
        traces = []
        file_path = volume_data['segy_path']
        
        safe_print(f"SeismicHandler: Opening SEGY file: {file_path}")
        
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            f.mmap()
            for row in filtered_headerdata:
                trace_index = int(row[0])
                traces.append(f.trace[trace_index])
        
        time_array = self.get_time_array(survey_name, attribute_name)
        return np.array(traces), time_array, filtered_headerdata

    def get_traces_along_arbitrary_line(self, survey_name, attribute_name, line_coordinates, trace_spacing=25.0):
        """
        Extract traces along an arbitrary line with specified trace spacing
        
        Args:
            survey_name (str): Name of the survey
            attribute_name (str): Name of the seismic attribute
            line_coordinates (list): List of (x, y) coordinates defining the line
            trace_spacing (float): Distance between traces in meters
            
        Returns:
            dict: Dictionary containing extracted trace data with keys:
                - 'traces': numpy array of traces
                - 'times': time array
                - 'coordinates': list of (x, y) positions where traces were extracted
                - 'headerdata': list of (inline, crossline) for each trace
                - 'line_coordinates': original line definition
                - 'trace_spacing': spacing used
        """
        safe_print(f'get_traces_along_arbitrary_line: Processing line with {len(line_coordinates)} points, spacing {trace_spacing}m')
        
        if len(line_coordinates) < 2:
            safe_print("Need at least 2 points to define a line")
            return None
            
        # Generate trace positions along the line
        trace_positions = self._generate_trace_positions_along_line(line_coordinates, trace_spacing)
        
        if not trace_positions:
            safe_print("No trace positions generated")
            return None
            
        # Extract traces at generated positions
        traces_array, time_array, valid_coordinates, valid_headerdata = self.get_traces_for_multiple_points(
            survey_name, attribute_name, trace_positions)
        
        if traces_array is None:
            safe_print("Failed to extract traces along arbitrary line")
            return None
            
        result = {
            'traces': traces_array,
            'times': time_array,
            'coordinates': valid_coordinates,
            'headerdata': valid_headerdata,
            'line_coordinates': line_coordinates,
            'trace_spacing': trace_spacing,
            'num_traces': len(valid_coordinates)
        }
        
        safe_print(f'get_traces_along_arbitrary_line: Successfully extracted {len(valid_coordinates)} traces')
        return result
    
    def get_trace_coordinates_from_headerdata(self, headerdata):
        """Extract coordinates from headerdata for coordinate-based plotting
        
        Args:
            headerdata: Array with format [trace_index, inline, crossline, cdp_x, cdp_y]
            
        Returns:
            list: List of (x, y) coordinate tuples for each trace
        """
        if headerdata is None or len(headerdata) == 0:
            return []
        
        # Extract CDP_X and CDP_Y coordinates (columns 3 and 4)
        coordinates = []
        for row in headerdata:
            if len(row) >= 5:
                x_coord = row[3]  # CDP_X
                y_coord = row[4]  # CDP_Y
                coordinates.append((x_coord, y_coord))
            else:
                # Fallback if coordinates not available
                coordinates.append((0, 0))
        
        return coordinates

    def _generate_trace_positions_along_line(self, line_coordinates, trace_spacing):
        """
        Generate trace positions along an arbitrary line with specified spacing
        
        Args:
            line_coordinates (list): List of (x, y) coordinates defining the line
            trace_spacing (float): Distance between traces in meters
            
        Returns:
            list: List of (x, y) positions for trace extraction
        """
        positions = []
        
        # Calculate total line length and segment lengths
        total_length = 0
        segment_lengths = []
        
        for i in range(len(line_coordinates) - 1):
            x1, y1 = line_coordinates[i]
            x2, y2 = line_coordinates[i + 1]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            segment_lengths.append(length)
            total_length += length
            
        if total_length == 0:
            safe_print("Line has zero length")
            return positions
            
        # Generate positions along the line
        current_distance = 0
        
        while current_distance <= total_length:
            # Find which segment this distance falls in
            cumulative_length = 0
            
            for seg_idx, seg_length in enumerate(segment_lengths):
                if current_distance <= cumulative_length + seg_length:
                    # Calculate position within this segment
                    segment_distance = current_distance - cumulative_length
                    segment_ratio = segment_distance / seg_length if seg_length > 0 else 0
                    
                    x1, y1 = line_coordinates[seg_idx]
                    x2, y2 = line_coordinates[seg_idx + 1]
                    
                    x = x1 + (x2 - x1) * segment_ratio
                    y = y1 + (y2 - y1) * segment_ratio
                    
                    positions.append((x, y))
                    break
                    
                cumulative_length += seg_length
            
            current_distance += trace_spacing
            
        safe_print(f'_generate_trace_positions_along_line: Generated {len(positions)} positions along {total_length:.1f}m line')
        return positions

    # ===================================================================
    # NEW: GRID-SNAPPED TRACE EXTRACTION WITH EXACT SAMPLE OWNERSHIP
    # ===================================================================

    def extract_synthetic_trace_along_well_path(self, survey_name, attribute_name, well_trajectory_df):
        """
        Extracts synthetic trace along well path using grid-snapped exact sample ownership.
        
        Snaps each well TWT to the 2ms seismic grid, finds owning trace via nearest midpoint,
        groups consecutive points with same trace into segments, then extracts exact samples.

        Args:
            survey_name (str): The name of the seismic survey
            attribute_name (str): The name of the seismic attribute  
            well_trajectory_df (pd.DataFrame): DataFrame with columns ['X', 'Y', 'TWT']

        Returns:
            dict: Result with 'success', 'synthetic_trace', 'times', 'segment_info', 'quality_metrics'
        """
        safe_print("🎯 Starting grid-snapped trace extraction along well path...")
        try:
            # 1. Get seismic time array and parameters
            time_array = self.get_time_array(survey_name, attribute_name)
            if time_array is None or len(time_array) == 0:
                return {'success': False, 'error': "Could not retrieve seismic time array."}
            
            t0 = float(time_array[0])
            dt = float(time_array[1] - time_array[0]) if len(time_array) > 1 else 2.0
            
            safe_print(f"✓ Seismic grid: {len(time_array)} samples, t0={t0:.1f}ms, dt={dt:.1f}ms")

            # 2. Get or build survey cache (KDTree + header data)
            survey_cache_data = self._get_or_build_survey_cache(survey_name)
            if not survey_cache_data:
                return {'success': False, 'error': "Failed to initialize survey cache."}

            # Extract cached components
            kdtree = survey_cache_data['kdtree']
            coords_xy = survey_cache_data['coords_xy'] 
            trace_keys = survey_cache_data['trace_keys']
            safe_print(f"✓ Using survey cache: {len(trace_keys)} traces indexed")

            # 4. Snap well points to grid and assign trace ownership
            assignments = self._assign_trace_ownership_and_snap(
                well_trajectory_df, kdtree, coords_xy, trace_keys, t0, dt, time_array
            )
            if not assignments:
                return {'success': False, 'error': "No valid assignments within seismic range."}
            
            safe_print(f"✓ Assigned {len(assignments)} well points to traces (snapped to {dt:.1f}ms grid)")
            
            # 4.5. Ensure complete seismic range coverage (including boundaries)
            assignments = self._ensure_complete_seismic_coverage(
                assignments, well_trajectory_df, kdtree, coords_xy, trace_keys, t0, dt, time_array
            )
            safe_print(f"✓ Enhanced to {len(assignments)} assignments for complete seismic coverage")

            # 5. Create complete grid coverage with no gaps
            segments = self._create_complete_grid_segments(assignments, time_array)
            safe_print(f"✓ Created {len(segments)} segments with 100% grid coverage")

            # 6. Extract exact samples for each segment
            extracted_traces = {}
            final_times = []
            final_vals = []
            
            for i, seg in enumerate(segments):
                trace_key = seg['trace_key']
                il, xl = trace_key
                
                # Get trace data if not already cached
                if trace_key not in extracted_traces:
                    trace_data, _, _ = self.get_trace_data(survey_name, attribute_name, il, xl)
                    if trace_data is None or len(trace_data) == 0:
                        safe_print(f"⚠️ Could not extract trace IL{il}XL{xl}")
                        continue
                    extracted_traces[trace_key] = trace_data[0]  # Get 1D array

                trace_vec = extracted_traces[trace_key]
                start_i, end_i = seg['sample_idx_start'], seg['sample_idx_end']
                
                # Extract exact samples
                segment_samples = trace_vec[start_i:end_i+1]
                segment_times = time_array[start_i:end_i+1]
                
                final_vals.append(segment_samples)
                final_times.append(segment_times)
                
                # Store sample count in segment info
                seg['n_samples'] = len(segment_samples)
                
                safe_print(f"  Seg {i+1}: IL{il}XL{xl} → {seg['twt_start']:.1f}-{seg['twt_end']:.1f}ms "
                      f"({seg['n_samples']} samples, avg dist {seg['avg_distance']:.1f}m)")

            # 7. Concatenate all segments
            synthetic_trace = np.concatenate(final_vals) if final_vals else np.array([])
            times = np.concatenate(final_times) if final_times else np.array([])
            
            safe_print(f"✓ Concatenated {len(segments)} segments into {len(synthetic_trace)} total samples")

            # 8. Calculate quality metrics and export CSV
            quality_metrics = self._compute_quality_metrics_segments(segments, assignments, time_array)
            well_name = well_trajectory_df.iloc[0].get('Well', 'Unknown') if len(well_trajectory_df) > 0 else 'Unknown'
            csv_file = f"{well_name}_trace_segments.csv"
            self._export_segment_csv(segments, csv_file)

            safe_print(f"✅ Grid-snapped extraction complete: {len(synthetic_trace)} samples from {len(segments)} segments")
            safe_print(f"✅ Quality: {quality_metrics['coverage_percentage']:.1f}% coverage, avg distance {quality_metrics['avg_distance']:.1f}m")

            return {
                'success': True,
                'synthetic_trace': synthetic_trace,
                'times': times,
                'segment_info': segments,
                'quality_metrics': quality_metrics
            }

        except Exception as e:
            import traceback
            safe_print(f"❌ Error during grid-snapped trace extraction: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _get_or_build_survey_cache(self, survey_name):
        """Get or build survey-level cache using the singleton cache manager"""
        try:
            # Import here to avoid circular imports
            from geoagent.utils.survey_cache import get_survey_cache_instance
            
            survey_cache = get_survey_cache_instance()
            
            # Try to get cached data
            cached_data = survey_cache.get_survey_cache(survey_name)
            if cached_data:
                return cached_data
            
            # Need to build cache - get header data
            headerdata = self.get_data(survey_name, 'headerdata')
            if headerdata is None:
                safe_print(f"❌ No header data available for survey {survey_name}")
                return None
            
            # Build and return survey cache  
            return survey_cache.build_survey_cache(
                survey_name, headerdata,
                coordinate_system="default"
            )
            
        except ImportError:
            safe_print("⚠️ Survey cache not available, falling back to direct KDTree build")
            # Fallback to original method
            headerdata = self.get_data(survey_name, 'headerdata')
            if headerdata is None:
                return None
                
            kdtree, coords_xy, trace_keys = self._build_kdtree_from_header(headerdata)
            return {
                'kdtree': kdtree,
                'coords_xy': coords_xy,
                'trace_keys': trace_keys,
                'headerdata': headerdata,
                'trace_count': len(trace_keys)
            }
        except Exception as e:
            safe_print(f"❌ Failed to get/build survey cache: {e}")
            return None

    def _build_kdtree_from_header(self, headerdata):
        """Build KDTree from headerdata for trace ownership assignment."""
        from scipy.spatial import cKDTree
        
        # Extract trace coordinates (X, Y) and trace keys (inline, crossline)
        coords_xy = headerdata[:, 3:5]  # Columns 3, 4 are CDP_X, CDP_Y
        trace_keys = [(int(row[1]), int(row[2])) for row in headerdata]  # (inline, crossline)
        
        kdtree = cKDTree(coords_xy)
        return kdtree, coords_xy, trace_keys

    def _assign_trace_ownership_and_snap(self, well_df, kdtree, coords_xy, trace_keys, t0, dt, time_array):
        """Snap well TWTs to seismic grid and assign trace ownership via nearest midpoint."""
        assignments = []
        
        for idx, row in well_df.iterrows():
            twt_raw = row['TWT']
            
            # Snap to seismic grid
            sample_idx = int(round((twt_raw - t0) / dt))
            
            # Check if within seismic range
            if sample_idx < 0 or sample_idx >= len(time_array):
                continue  # Skip points outside seismic range
                
            twt_snapped = t0 + sample_idx * dt
            
            # Find nearest trace (k=1 for closest midpoint)
            dist, nn_idx = kdtree.query([row['X'], row['Y']], k=1)
            
            # Handle scalar vs array return
            if np.isscalar(nn_idx):
                owner_idx = int(nn_idx)
                distance = float(dist)
            else:
                owner_idx = int(nn_idx[0])
                distance = float(dist[0])
                
            trace_key = trace_keys[owner_idx]
            
            assignments.append({
                "well_point_idx": int(idx),
                "twt_raw": twt_raw,
                "twt_snapped": twt_snapped,
                "sample_idx": sample_idx,
                "trace_key": trace_key,
                "distance_to_trace": distance,
                "x": row['X'],
                "y": row['Y']
            })
            
        safe_print(f"  Snapped {len(assignments)} points to 2ms grid (from {len(well_df)} total well points)")
        return assignments

    def _create_complete_grid_segments(self, assignments, time_array):
        """Create segments that cover every sample in the FULL seismic range with no gaps."""
        segments = []
        if not assignments:
            return segments

        # Sort assignments by sample_idx
        assignments = sorted(assignments, key=lambda x: x['sample_idx'])
        
        # Create a mapping from sample_idx to assignment
        sample_to_assignment = {a['sample_idx']: a for a in assignments}
        
        # Use the range covered by assignments (which now includes boundary points)
        min_sample = assignments[0]['sample_idx']
        max_sample = assignments[-1]['sample_idx']
        
        # Ensure we stay within seismic bounds
        min_sample = max(0, min_sample)
        max_sample = min(len(time_array) - 1, max_sample)
        
        safe_print(f"  Filling complete grid from sample {min_sample} to {max_sample} ({max_sample - min_sample + 1} samples)")
        safe_print(f"  Assignment range: samples {min_sample}-{max_sample}, Seismic range: 0-{len(time_array)-1}")
        
        # Fill every sample index in the range
        current_sample = min_sample
        while current_sample <= max_sample:
            # Find the trace ownership for this sample
            if current_sample in sample_to_assignment:
                # Use exact assignment
                assignment = sample_to_assignment[current_sample]
                trace_key = assignment['trace_key']
                distance = assignment['distance_to_trace']
            else:
                # Fill gap by using the nearest assignment
                # Fill gap by using the nearest assignment (now includes boundary points)
                nearest_assignment = self._find_nearest_assignment(current_sample, assignments)
                
                trace_key = nearest_assignment['trace_key']
                distance = nearest_assignment['distance_to_trace']
            
            # Find the end of this segment (consecutive samples with same trace)
            segment_end = current_sample
            while (segment_end + 1 <= max_sample and 
                   self._get_trace_for_sample(segment_end + 1, sample_to_assignment, assignments) == trace_key):
                segment_end += 1
            
            # Create segment
            t0 = time_array[0]
            dt = time_array[1] - time_array[0] if len(time_array) > 1 else 2.0
            
            # Count well points in this segment for CSV export compatibility
            segment_well_points = 0
            for sample_idx in range(current_sample, segment_end + 1):
                if sample_idx in sample_to_assignment:
                    segment_well_points += 1
            
            segments.append({
                "trace_key": trace_key,
                "twt_start": t0 + current_sample * dt,
                "twt_end": t0 + segment_end * dt,
                "sample_idx_start": current_sample,
                "sample_idx_end": segment_end,
                "avg_distance": distance,
                "n_samples": segment_end - current_sample + 1,
                "n_well_points": segment_well_points  # Fix CSV export KeyError
            })
            
            # Move to next segment
            current_sample = segment_end + 1
        
        total_samples_covered = sum(seg['n_samples'] for seg in segments)
        safe_print(f"  Created {len(segments)} segments covering all {total_samples_covered} samples")
        safe_print(f"  Seismic range: {time_array[min_sample]:.1f}ms to {time_array[max_sample]:.1f}ms")
        return segments
    
    def _find_nearest_assignment(self, sample_idx, assignments):
        """Find the nearest assignment for gap filling."""
        min_distance = float('inf')
        nearest = assignments[0]
        
        for assignment in assignments:
            distance = abs(assignment['sample_idx'] - sample_idx)
            if distance < min_distance:
                min_distance = distance
                nearest = assignment
        
        return nearest
    
    def _get_trace_for_sample(self, sample_idx, sample_to_assignment, assignments):
        """Get trace ownership for a sample index (with gap filling)."""
        if sample_idx in sample_to_assignment:
            return sample_to_assignment[sample_idx]['trace_key']
        else:
            # Fill gap by finding nearest assignment
            nearest = self._find_nearest_assignment(sample_idx, assignments)
            return nearest['trace_key']
    
    def _ensure_complete_seismic_coverage(self, assignments, well_df, kdtree, coords_xy, trace_keys, t0, dt, time_array):
        """Ensure we have assignments that cover the complete seismic range including boundaries."""
        if not assignments:
            return assignments
            
        # Sort assignments and find coverage gaps
        assignments = sorted(assignments, key=lambda x: x['sample_idx'])
        enhanced_assignments = assignments.copy()
        
        # Get the time range of the well trajectory
        well_t_min = well_df['TWT'].min()
        well_t_max = well_df['TWT'].max()
        
        # Convert to sample indices
        seismic_start_idx = 0
        seismic_end_idx = len(time_array) - 1
        well_start_idx = int(round((well_t_min - t0) / dt))
        well_end_idx = int(round((well_t_max - t0) / dt))
        
        # Clamp to seismic range
        well_start_idx = max(seismic_start_idx, well_start_idx)
        well_end_idx = min(seismic_end_idx, well_end_idx)
        
        safe_print(f"  Seismic range: samples {seismic_start_idx}-{seismic_end_idx} ({t0:.1f}-{t0 + (len(time_array)-1)*dt:.1f}ms)")
        safe_print(f"  Well range: samples {well_start_idx}-{well_end_idx} ({well_t_min:.1f}-{well_t_max:.1f}ms)")
        
        # Add assignment at seismic start if missing
        first_assignment_idx = assignments[0]['sample_idx']
        if first_assignment_idx > well_start_idx:
            # Interpolate well position at start boundary
            start_well_point = self._interpolate_well_position(well_df, t0 + well_start_idx * dt)
            if start_well_point is not None:
                # Find nearest trace for this position
                dist, nn_idx = kdtree.query([start_well_point['X'], start_well_point['Y']], k=1)
                owner_idx = int(nn_idx) if np.isscalar(nn_idx) else int(nn_idx[0])
                distance = float(dist) if np.isscalar(dist) else float(dist[0])
                trace_key = trace_keys[owner_idx]
                
                boundary_assignment = {
                    "well_point_idx": -1,  # Synthetic boundary point
                    "twt_raw": t0 + well_start_idx * dt,
                    "twt_snapped": t0 + well_start_idx * dt,
                    "sample_idx": well_start_idx,
                    "trace_key": trace_key,
                    "distance_to_trace": distance,
                    "x": start_well_point['X'],
                    "y": start_well_point['Y']
                }
                enhanced_assignments.insert(0, boundary_assignment)
                safe_print(f"  Added boundary assignment at {t0 + well_start_idx * dt:.1f}ms (sample {well_start_idx})")
        
        # Add assignment at seismic end if missing
        last_assignment_idx = assignments[-1]['sample_idx']
        if last_assignment_idx < well_end_idx:
            # Interpolate well position at end boundary
            end_well_point = self._interpolate_well_position(well_df, t0 + well_end_idx * dt)
            if end_well_point is not None:
                # Find nearest trace for this position
                dist, nn_idx = kdtree.query([end_well_point['X'], end_well_point['Y']], k=1)
                owner_idx = int(nn_idx) if np.isscalar(nn_idx) else int(nn_idx[0])
                distance = float(dist) if np.isscalar(dist) else float(dist[0])
                trace_key = trace_keys[owner_idx]
                
                boundary_assignment = {
                    "well_point_idx": -1,  # Synthetic boundary point
                    "twt_raw": t0 + well_end_idx * dt,
                    "twt_snapped": t0 + well_end_idx * dt,
                    "sample_idx": well_end_idx,
                    "trace_key": trace_key,
                    "distance_to_trace": distance,
                    "x": end_well_point['X'],
                    "y": end_well_point['Y']
                }
                enhanced_assignments.append(boundary_assignment)
                safe_print(f"  Added boundary assignment at {t0 + well_end_idx * dt:.1f}ms (sample {well_end_idx})")
        
        return sorted(enhanced_assignments, key=lambda x: x['sample_idx'])
    
    def _interpolate_well_position(self, well_df, target_twt):
        """Interpolate well X,Y position at a given TWT."""
        try:
            from scipy.interpolate import interp1d
            
            # Create interpolation functions
            f_x = interp1d(well_df['TWT'], well_df['X'], kind='linear', bounds_error=False, fill_value='extrapolate')
            f_y = interp1d(well_df['TWT'], well_df['Y'], kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Interpolate position
            x_interp = float(f_x(target_twt))
            y_interp = float(f_y(target_twt))
            
            return {'X': x_interp, 'Y': y_interp, 'TWT': target_twt}
            
        except Exception as e:
            safe_print(f"  Warning: Could not interpolate well position at {target_twt:.1f}ms: {e}")
            return None
    
    def _group_segments(self, assignments):
        """DEPRECATED: Old method that creates gaps. Use _create_complete_grid_segments instead."""
        segments = []
        if not assignments:
            return segments

        # Sort by sample_idx to ensure time order
        assignments = sorted(assignments, key=lambda x: x['sample_idx'])
        
        start = 0
        for i in range(1, len(assignments) + 1):
            # Check if we've reached end or trace ownership changed
            if i == len(assignments) or assignments[i]['trace_key'] != assignments[i-1]['trace_key']:
                seg_points = assignments[start:i]
                
                # Calculate segment properties
                trace_key = seg_points[0]['trace_key']
                sample_indices = [p['sample_idx'] for p in seg_points]
                distances = [p['distance_to_trace'] for p in seg_points]
                
                segments.append({
                    "trace_key": trace_key,
                    "twt_start": seg_points[0]['twt_snapped'],
                    "twt_end": seg_points[-1]['twt_snapped'],
                    "sample_idx_start": min(sample_indices),
                    "sample_idx_end": max(sample_indices),
                    "avg_distance": float(np.mean(distances)),
                    "start_well_idx": seg_points[0]['well_point_idx'],
                    "end_well_idx": seg_points[-1]['well_point_idx'],
                    "n_well_points": len(seg_points)
                })
                start = i
                
        return segments

    def _compute_quality_metrics_segments(self, segments, assignments, time_array):
        """Compute quality metrics for segment-based extraction."""
        if not segments or not assignments:
            return {
                'coverage_percentage': 0.0,
                'avg_distance': 0.0,
                'total_segments': 0,
                'total_samples': 0,
                'unique_traces': 0
            }
        
        total_samples = sum(seg.get('n_samples', 0) for seg in segments)
        total_possible_samples = len(time_array)
        coverage_percentage = (total_samples / total_possible_samples) * 100.0 if total_possible_samples > 0 else 0.0
        
        avg_distance = np.mean([seg['avg_distance'] for seg in segments])
        unique_traces = len(set(seg['trace_key'] for seg in segments))
        
        return {
            'coverage_percentage': coverage_percentage,
            'avg_distance': float(avg_distance),
            'total_segments': len(segments),
            'total_samples': total_samples,
            'unique_traces': unique_traces
        }

    def _export_segment_csv(self, segments, output_file):
        """Export segment information to CSV file."""
        try:
            import csv
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'segment_id', 'inline', 'crossline', 'twt_start_ms', 'twt_end_ms',
                    'sample_idx_start', 'sample_idx_end', 'n_samples', 'avg_distance_m', 'n_well_points'
                ])
                
                # Write segment data
                for i, seg in enumerate(segments, 1):
                    il, xl = seg['trace_key']
                    writer.writerow([
                        i, il, xl, seg['twt_start'], seg['twt_end'],
                        seg['sample_idx_start'], seg['sample_idx_end'],
                        seg.get('n_samples', 0), seg['avg_distance'], seg['n_well_points']
                    ])
            
            safe_print(f"✓ Segment CSV exported: {output_file}")
            return True
            
        except Exception as e:
            safe_print(f"⚠️ Failed to export CSV: {e}")
            return False

    # Legacy patch-based methods (kept for compatibility)
    def _calculate_mutually_exclusive_patches(self, well_trace_mappings, times, patch_size_ms=None, gap_size_ms=2):
        """
        Calculates sequential, non-overlapping patch boundaries with adaptive sizing.
        
        Args:
            well_trace_mappings: List of well point to trace mappings
            times: Seismic time array
            patch_size_ms: Patch size in milliseconds (calculated adaptively if None)
            gap_size_ms: Gap size between patches in milliseconds
        """
        # Sort well points by TWT to ensure patches follow the well's time progression
        sorted_mappings = sorted(well_trace_mappings, key=lambda x: x['twt'])
        
        # Calculate adaptive patch size if not provided
        if patch_size_ms is None:
            patch_size_ms = self._calculate_adaptive_patch_size(sorted_mappings, times)
            
        safe_print(f"📊 Patch sizing analysis:")
        safe_print(f"  - Seismic time range: {times[0]:.0f}-{times[-1]:.0f}ms ({len(times)} samples)")
        safe_print(f"  - Well points: {len(sorted_mappings)} trajectory points")
        safe_print(f"  - Adaptive patch size: {patch_size_ms}ms")
        safe_print(f"  - Gap between patches: {gap_size_ms}ms")
        
        # Analyze trace proximity patterns
        trace_usage_analysis = self._analyze_trace_proximity_patterns(sorted_mappings)
        
        patches = []
        current_time = times[0]
        
        safe_print(f"\n🔍 Detailed patch-to-trace mapping:")

        for i, mapping in enumerate(sorted_mappings):
            start_time = current_time
            end_time = start_time + patch_size_ms

            # Ensure the patch does not exceed the seismic time range
            if start_time >= times[-1]:
                break
            end_time = min(end_time, times[-1])

            start_idx = np.argmin(np.abs(times - start_time))
            end_idx = np.argmin(np.abs(times - end_time))

            if start_idx >= end_idx:
                continue

            trace_key = mapping['trace_key']
            distance = mapping['distance_to_trace']
            
            patch = {
                'well_point_idx': mapping['well_point_idx'],
                'source_trace_key': trace_key,
                'time_range': (start_time, end_time),
                'sample_range': (start_idx, end_idx),
                'well_twt': mapping['twt'],
                'distance_to_trace': distance
            }
            patches.append(patch)
            
            # Enhanced logging with trace proximity details
            safe_print(f"  Patch {i+1:2d}: {start_time:7.1f}-{end_time:7.1f}ms → IL{trace_key[0]}XL{trace_key[1]} (dist: {distance:.1f}m, well TWT: {mapping['twt']:.1f}ms)")

            # CRITICAL STEP: Advance the time for the next patch, ensuring a gap
            current_time = end_time + gap_size_ms

        # Print trace usage summary
        self._print_trace_usage_summary(patches, trace_usage_analysis)
        
        return patches
        
    def _calculate_adaptive_patch_size(self, sorted_mappings, times):
        """Calculate adaptive patch size based on well trajectory and seismic characteristics."""
        total_time_range = times[-1] - times[0]
        num_well_points = len(sorted_mappings)
        
        # Base patch size calculation
        if num_well_points <= 10:
            base_patch_size = 30  # Longer patches for sparse wells
        elif num_well_points <= 25:
            base_patch_size = 25  # Standard patches for typical wells
        elif num_well_points <= 50:
            base_patch_size = 20  # Shorter patches for dense wells
        else:
            base_patch_size = 15  # Very short patches for very dense wells
            
        # Adjust based on total time range
        time_adjustment = min(1.5, total_time_range / 400.0)  # Scale based on 400ms reference
        adaptive_size = int(base_patch_size * time_adjustment)
        
        # Ensure reasonable bounds
        adaptive_size = max(10, min(50, adaptive_size))  # 10-50ms range
        
        safe_print(f"📊 Adaptive patch sizing rationale:")
        safe_print(f"  - Well density: {num_well_points} points → base size {base_patch_size}ms")
        safe_print(f"  - Time range adjustment: {time_adjustment:.2f}x → final size {adaptive_size}ms")
        safe_print(f"  - Reasoning: {'Dense trajectory needs shorter patches' if num_well_points > 25 else 'Standard trajectory uses optimal patches' if num_well_points > 10 else 'Sparse trajectory needs longer patches'}")
        
        return adaptive_size
        
    def _analyze_trace_proximity_patterns(self, sorted_mappings):
        """Analyze which traces are used across different time ranges."""
        trace_usage = {}
        
        for mapping in sorted_mappings:
            trace_key = mapping['trace_key']
            trace_name = f"IL{trace_key[0]}XL{trace_key[1]}"
            
            if trace_name not in trace_usage:
                trace_usage[trace_name] = {
                    'time_ranges': [],
                    'distances': [],
                    'count': 0
                }
            
            trace_usage[trace_name]['time_ranges'].append(mapping['twt'])
            trace_usage[trace_name]['distances'].append(mapping['distance_to_trace'])
            trace_usage[trace_name]['count'] += 1
            
        return trace_usage
        
    def _print_trace_usage_summary(self, patches, trace_usage_analysis):
        """Print summary of which traces are used for which time ranges."""
        safe_print(f"\n📋 Trace usage summary across {len(patches)} patches:")
        
        # Group patches by trace
        trace_patch_groups = {}
        for i, patch in enumerate(patches):
            trace_key = patch['source_trace_key']
            trace_name = f"IL{trace_key[0]}XL{trace_key[1]}"
            
            if trace_name not in trace_patch_groups:
                trace_patch_groups[trace_name] = []
            
            trace_patch_groups[trace_name].append({
                'patch_num': i + 1,
                'time_range': patch['time_range'],
                'distance': patch['distance_to_trace']
            })
        
        # Print usage for each trace
        for trace_name, patch_group in trace_patch_groups.items():
            patch_nums = [p['patch_num'] for p in patch_group]
            time_start = min(p['time_range'][0] for p in patch_group)
            time_end = max(p['time_range'][1] for p in patch_group)
            avg_distance = np.mean([p['distance'] for p in patch_group])
            
            safe_print(f"  {trace_name}: Patches {patch_nums} → {time_start:.0f}-{time_end:.0f}ms (avg dist: {avg_distance:.1f}m)")
            
        safe_print(f"✓ Total unique traces used: {len(trace_patch_groups)}")
        safe_print(f"✓ Patch distribution: {len(patches)} patches across {len(trace_patch_groups)} traces")

    def _splice_patches(self, patches, extracted_traces, times):
        """
        Splices the calculated patches into a single synthetic trace.
        """
        # Initialize the synthetic trace with zeros
        synthetic_trace = np.zeros_like(times, dtype=np.float32)
        patch_info_detailed = []

        for patch in patches:
            start_idx, end_idx = patch['sample_range']
            source_key = patch['source_trace_key']

            if source_key in extracted_traces:
                source_trace_data = extracted_traces[source_key]
                # Copy the data from the source trace patch to the synthetic trace
                patch_len = end_idx - start_idx
                synthetic_trace[start_idx:end_idx] = source_trace_data[start_idx:end_idx]

                patch_info_detailed.append({
                    **patch,
                    'source_trace': f"IL{source_key[0]}_XL{source_key[1]}",
                })
        return synthetic_trace, patch_info_detailed

    def _calculate_quality_metrics(self, synthetic_trace, patches, times):
        """Calculates quality metrics for the generated synthetic trace."""
        if not patches:
            return {
                'coverage_percentage': 0.0,
                'gap_regions': 0,
                'total_gap_time_ms': 0
            }

        total_samples = len(times)
        covered_samples = np.count_nonzero(synthetic_trace)
        coverage_percentage = (covered_samples / total_samples) * 100

        gap_count = 0
        total_gap_time = 0
        sorted_patches = sorted(patches, key=lambda x: x['time_range'][0])
        for i in range(len(sorted_patches) - 1):
            gap = sorted_patches[i+1]['time_range'][0] - sorted_patches[i]['time_range'][1]
            if gap > 0:
                gap_count += 1
                total_gap_time += gap

        return {
            'coverage_percentage': coverage_percentage,
            'gap_regions': gap_count,
            'total_gap_time_ms': total_gap_time
        }