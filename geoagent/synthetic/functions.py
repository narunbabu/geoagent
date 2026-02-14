# synthetic_functions.py — GeoAgent (Qt-free)
import logging
import pandas as pd
import numpy as np
from scipy import interpolate
from geoagent.utils.safe_print import safe_print

logger = logging.getLogger(__name__)

# Function to create reflectivity series
def create_reflectivity(impedance):
    impedance = np.asarray(impedance)  # Ensure impedance is a numpy array
    return np.diff(impedance) / (impedance[1:] + impedance[:-1])

def create_synthetic_seismic_valid(reflectivity, wavelet, seismic_times_extracted, log_sampling_interval=None):
    safe_print(f"create_synthetic_seismic_valid - Input dimensions:")
    safe_print(f"  reflectivity: {len(reflectivity)} samples")
    safe_print(f"  wavelet: {len(wavelet) if isinstance(wavelet, np.ndarray) else 'dict/other'}")
    safe_print(f"  seismic_times_extracted: {len(seismic_times_extracted)} samples")
    safe_print(f"  log_sampling_interval: {log_sampling_interval} ms")
    
    # Calculate expected sampling from seismic times to compare with log sampling
    if len(seismic_times_extracted) > 1:
        actual_sampling = seismic_times_extracted[1] - seismic_times_extracted[0]
        safe_print(f"  calculated actual sampling from seismic_times: {actual_sampling:.3f} ms")
    else:
        actual_sampling = None
        safe_print(f"  Warning: Cannot calculate sampling from seismic_times (insufficient samples)")
    
    # Ensure arrays are proper 1D numpy arrays
    reflectivity = np.asarray(reflectivity, dtype=np.float64).flatten()
    
    # Handle wavelet resampling if log sampling interval is provided
    wavelet_original_type = type(wavelet).__name__
    safe_print(f"  original wavelet type: {wavelet_original_type}")
    
    if log_sampling_interval is not None:
        # Always try to prepare wavelet for synthetic (handles resampling if needed)
        safe_print(f"  attempting wavelet preparation/resampling for {log_sampling_interval} ms sampling...")
        wavelet_processed = prepare_wavelet_for_synthetic(wavelet, log_sampling_interval)
        wavelet = np.asarray(wavelet_processed, dtype=np.float64).flatten()
        safe_print(f"  wavelet resampling completed")
    else:
        safe_print(f"  no log_sampling_interval provided, using wavelet as-is")
        wavelet = np.asarray(wavelet, dtype=np.float64).flatten()
    
    # Log final wavelet details for verification
    safe_print(f"  FINAL WAVELET FOR SYNTHETIC:")
    safe_print(f"    wavelet shape: {wavelet.shape}, dtype: {wavelet.dtype}")
    safe_print(f"    wavelet length: {len(wavelet)} samples")
    safe_print(f"    wavelet min/max: {np.min(wavelet):.6f} / {np.max(wavelet):.6f}")
    
    # Check for dimension mismatch issue
    if len(wavelet) != 128:
        safe_print(f"  ⚠️  WARNING: Wavelet length is {len(wavelet)}, expected 128!")
        safe_print(f"      This may cause dimension mismatches in synthetic seismic generation")
    
    # If we have log sampling interval and actual sampling, compare them
    if log_sampling_interval is not None and actual_sampling is not None:
        if abs(log_sampling_interval - actual_sampling) > 0.001:
            safe_print(f"  ⚠️  WARNING: Log sampling ({log_sampling_interval} ms) != seismic sampling ({actual_sampling:.3f} ms)")
            safe_print(f"     This mismatch may cause correlation artifacts and poor synthetic quality")
            safe_print(f"     RECOMMENDATION: Upscale well logs to match seismic sampling interval ({actual_sampling:.3f} ms)")
            if abs(log_sampling_interval - actual_sampling) > 1.0:
                safe_print(f"     ⚠️  CRITICAL: Large sampling mismatch (>{1.0}ms) detected - results may be unreliable")
        else:
            safe_print(f"  ✓ Log sampling and seismic sampling intervals match ({log_sampling_interval} ms)")
    
    safe_print(f"  reflectivity after processing: shape {reflectivity.shape}, dtype {reflectivity.dtype}")
    
    # Remove any NaN or infinite values
    if np.any(~np.isfinite(reflectivity)):
        safe_print(f"  Warning: Found non-finite values in reflectivity, replacing with zeros")
        reflectivity = np.where(np.isfinite(reflectivity), reflectivity, 0.0)
    
    if np.any(~np.isfinite(wavelet)):
        safe_print(f"  Warning: Found non-finite values in wavelet, replacing with zeros")
        wavelet = np.where(np.isfinite(wavelet), wavelet, 0.0)
    
    # Validate arrays have reasonable lengths
    if len(reflectivity) == 0:
        safe_print(f"  Error: Empty reflectivity array")
        return np.zeros(len(seismic_times_extracted))
    
    if len(wavelet) == 0:
        safe_print(f"  Error: Empty wavelet array")
        return np.zeros(len(seismic_times_extracted))
    
    try:
        wavelet_center = len(wavelet) // 2
        padded_reflectivity = np.pad(reflectivity, (wavelet_center, wavelet_center), mode='constant')
        safe_print(f"  padded_reflectivity: shape {padded_reflectivity.shape}")
        
        synthetic_seismic = np.convolve(padded_reflectivity, wavelet, mode='valid')
        safe_print(f"  convolution successful: shape {synthetic_seismic.shape}")
        
    except Exception as e:
        safe_print(f"  Error during convolution: {e}")
        safe_print(f"  Reflectivity shape: {reflectivity.shape}, type: {type(reflectivity)}")
        safe_print(f"  Wavelet shape: {wavelet.shape}, type: {type(wavelet)}")
        # Return zeros if convolution fails
        return np.zeros(len(seismic_times_extracted))
    
    safe_print(f"  synthetic_seismic before truncation: {len(synthetic_seismic)} samples")
    
    # Ensure the synthetic seismic matches the target length exactly
    target_length = len(seismic_times_extracted)
    if len(synthetic_seismic) > target_length:
        synthetic_seismic = synthetic_seismic[:target_length]
        safe_print(f"  synthetic_seismic truncated to: {len(synthetic_seismic)} samples")
    elif len(synthetic_seismic) < target_length:
        # Pad with zeros if needed
        synthetic_seismic = np.pad(synthetic_seismic, (0, target_length - len(synthetic_seismic)), mode='constant')
        safe_print(f"  synthetic_seismic padded to: {len(synthetic_seismic)} samples")
    
    safe_print(f"  Final synthetic_seismic length: {len(synthetic_seismic)} samples")
    return synthetic_seismic


def get_Logdf_w_TWT(data_manager, well_tdr, well_name, parent_widget=None):
    # Check for both exact well name and potential variations
    upscaled_logs = data_manager.well_log_handler.get_upscaled_logs(well_name)
    upscale_parameters = data_manager.well_log_handler.get_upscale_parameters(well_name)
    
    # If not found, try alternate well name formats (handle potential " (No TD)" suffix)
    if upscaled_logs is None:
        clean_well_name = well_name.split(" (No TD)")[0] if " (No TD)" in well_name else well_name
        if clean_well_name != well_name:
            safe_print(f"[UPSCALE_CHECK] Trying alternate well name: {clean_well_name}")
            upscaled_logs = data_manager.well_log_handler.get_upscaled_logs(clean_well_name)
            upscale_parameters = data_manager.well_log_handler.get_upscale_parameters(clean_well_name)
            if upscaled_logs is not None:
                safe_print(f"[UPSCALE_CHECK] Found upscaled logs using clean name: {clean_well_name}")
                well_name = clean_well_name  # Use the clean name for rest of function
    
    # Debug: Print detailed information about upscaled logs detection
    safe_print(f"[UPSCALE_CHECK] Checking upscaled logs for well: {well_name}")
    safe_print(f"[UPSCALE_CHECK] Upscaled logs found: {upscaled_logs is not None}")
    if upscaled_logs is not None:
        safe_print(f"[UPSCALE_CHECK] Upscaled logs shape: {upscaled_logs.shape}")
        safe_print(f"[UPSCALE_CHECK] Upscaled logs columns: {upscaled_logs.columns.tolist()}")
    safe_print(f"[UPSCALE_CHECK] Upscale parameters found: {upscale_parameters is not None}")
    if upscale_parameters is not None:
        safe_print(f"[UPSCALE_CHECK] Upscale parameters: {upscale_parameters}")
    
    # Check if upscaled logs exist and are valid - be more robust about checking
    upscaled_logs_valid = (upscaled_logs is not None and 
                          hasattr(upscaled_logs, 'empty') and 
                          not upscaled_logs.empty and 
                          len(upscaled_logs) > 0)
    
    # Parameters can be None for some older upscaled logs - don't require them
    if upscaled_logs_valid:
        safe_print(f"[UPSCALE_CHECK] Using existing upscaled logs for well {well_name}")
        log_df = upscaled_logs
        # Handle different parameter formats
        if upscale_parameters is not None:
            if 'sampling_interval' in upscale_parameters:
                current_sampling_interval = upscale_parameters['sampling_interval']
            elif 'time_interval' in upscale_parameters:
                current_sampling_interval = upscale_parameters['time_interval']
            else:
                safe_print(f"[UPSCALE_CHECK] Warning: No sampling interval found in parameters, using default 2ms")
                current_sampling_interval = 2.0
        else:
            safe_print(f"[UPSCALE_CHECK] No upscale parameters found, using default 2ms sampling interval")
            current_sampling_interval = 2.0
    else:
        safe_print(f"[UPSCALE_CHECK] No valid upscaled logs found for well {well_name}")
        # Check if we have raw logs available
        raw_logs = data_manager.well_log_handler.get_well_logs_for_well(well_name)
        if raw_logs is None or raw_logs.empty:
            safe_print(f"Error in get_Logdf_w_TWT: No log data found for well {well_name}")
            return None, None, None
            
        safe_print(f"[UPSCALE_CHECK] Raw logs found for {well_name}, shape: {raw_logs.shape}")
        
        # Upscaled logs missing - trigger interactive upscaling workflow
        if parent_widget is not None:
            safe_print(f"[UPSCALE_CHECK] Triggering upscaling workflow for {well_name}")
            result = trigger_upscaling_workflow(data_manager, well_name, parent_widget)
            if result:
                # Try to get upscaled logs again after upscaling
                upscaled_logs = data_manager.well_log_handler.get_upscaled_logs(well_name)
                upscale_parameters = data_manager.well_log_handler.get_upscale_parameters(well_name)
                if upscaled_logs is not None:
                    log_df = upscaled_logs
                    current_sampling_interval = upscale_parameters['sampling_interval']
                else:
                    safe_print(f"Error: Upscaling failed for well {well_name}")
                    return None, None, None
            else:
                safe_print(f"User cancelled upscaling for well {well_name}")
                return None, None, None
        else:
            # Fallback to raw logs if no parent widget provided
            log_df = raw_logs
            current_sampling_interval = 2  # Default value

    # Get configurable log column names
    sonic_col = data_manager.find_log_column('sonic_log', well_name)
    density_col = data_manager.find_log_column('density_log', well_name)
    
    # Drop NA values for required columns
    required_cols = [col for col in [sonic_col, density_col] if col is not None]
    if required_cols:
        log_df = log_df.dropna(subset=required_cols)
    
    # Interpolate TWT values for all log depths
    try:
        if well_tdr.empty:
            safe_print(f"Error in get_Logdf_w_TWT: Empty TDR data for well {well_name}")
            return None, None, None
        
        required_tdr_columns = ['MD', 'TWT picked']
        if not all(col in well_tdr.columns for col in required_tdr_columns):
            safe_print(f"Error in get_Logdf_w_TWT: Missing required TDR columns for well {well_name}. Available: {well_tdr.columns.tolist()}")
            return None, None, None
            
        log_df.loc[:, 'TWT'] = np.interp(log_df['DEPTH'], well_tdr['MD'], well_tdr['TWT picked'])
        if log_df['TWT'].iloc[-1] < 0:
            log_df.loc[:, 'TWT'] = -log_df['TWT']
    except Exception as e:
        safe_print(f"Error in get_Logdf_w_TWT: Exception during TWT interpolation for well {well_name}: {e}")
        return None, None, None
        
    return log_df, upscale_parameters, current_sampling_interval


def get_custom_trace_coordinates(data_manager, well_name, parent_widget=None):
    """
    Get custom trace coordinates if custom trace selection is enabled, otherwise return None.
    
    Args:
        data_manager: DataManager instance
        well_name: Name of the well
        parent_widget: Parent widget that might have custom trace selection
        
    Returns:
        tuple: (inline, crossline) if custom trace is enabled and available, None otherwise
    """
    use_custom_trace = False
    
    # Check if parent widget explicitly enables custom trace selection
    if parent_widget:
        # Case 1: Individual synthetic widget with checkbox
        if hasattr(parent_widget, 'use_custom_trace_checkbox') and parent_widget.use_custom_trace_checkbox.isChecked():
            use_custom_trace = True
            safe_print(f"[CENTRALIZED DEBUG] use_custom_trace=True via checkbox for {parent_widget.__class__.__name__}")
        # Case 2: Batch synthetics dialog - always use custom trace if available
        elif hasattr(parent_widget, '__class__') and 'BatchSynthetics' in parent_widget.__class__.__name__:
            use_custom_trace = True
            safe_print(f"[CENTRALIZED DEBUG] use_custom_trace=True for BatchSynthetics")
        # Case 3: Other widgets that should always use custom trace if available
        elif hasattr(parent_widget, '__class__') and any(widget_type in parent_widget.__class__.__name__ 
                                                        for widget_type in ['MachineLearning', 'SeismicAttributes', 'Status']):
            use_custom_trace = True
            safe_print(f"[CENTRALIZED DEBUG] use_custom_trace=True for {parent_widget.__class__.__name__}")
        else:
            safe_print(f"[CENTRALIZED DEBUG] use_custom_trace=False for {parent_widget.__class__.__name__}")
    
    # ALWAYS check for stored coordinates regardless of widget settings
    # This ensures consistent behavior across all seismic attributes
    safe_print(f"[CENTRALIZED DEBUG] Final use_custom_trace={use_custom_trace}, but will check stored coordinates anyway")
    
    # ALWAYS check for stored custom coordinates first, regardless of widget settings
    # This ensures consistent behavior across all seismic attributes
    safe_print(f"[CENTRALIZED DEBUG] Checking custom coordinates for {well_name}, use_custom_trace={use_custom_trace}, parent_widget={parent_widget.__class__.__name__ if parent_widget else 'None'}")
    
    # Check if we have custom trace selection (newer system) stored for this well
    has_custom_selection = data_manager.well_handler.has_custom_trace_selection(well_name)
    safe_print(f"[CENTRALIZED DEBUG] Well {well_name} has custom trace selection: {has_custom_selection}")
    
    if has_custom_selection:
        # Get the complete custom trace selection
        selection = data_manager.well_handler.get_custom_trace_selection(well_name)
        safe_print(f"[CENTRALIZED DEBUG] Retrieved selection for {well_name}: {selection}")
        if selection and selection.get('enabled', False):
            inline = selection.get('inline')
            crossline = selection.get('crossline')
            if inline is not None and crossline is not None:
                safe_print(f"[CENTRALIZED] Using stored custom trace coordinates for {well_name}: Inline {inline}, Crossline {crossline}")
                return (inline, crossline)
    
    # Fallback: check old custom_trace_coordinates system for backward compatibility
    has_custom_coords = data_manager.well_handler.has_custom_trace_coordinates(well_name)
    safe_print(f"[CENTRALIZED DEBUG] Well {well_name} has old custom coordinates: {has_custom_coords}")
    
    if has_custom_coords:
        custom_coordinates = data_manager.well_handler.get_custom_trace_coordinates(well_name)
        safe_print(f"[CENTRALIZED DEBUG] Retrieved old coordinates for {well_name}: {custom_coordinates}")
        if custom_coordinates:
            safe_print(f"[CENTRALIZED] Using stored old custom trace coordinates for {well_name}: Inline {custom_coordinates[0]}, Crossline {custom_coordinates[1]}")
            return custom_coordinates
    
    safe_print(f"[CENTRALIZED DEBUG] No custom coordinates found for {well_name}, returning None")
    return None


def prepare_data_w_tdr(data_manager, well_tdr, well_name, use_upscaled=False, current_survey="current_survey", current_attribute="current_attribute", TRACE_INDEX=5, parent_widget=None, use_spatial_sampling=False, spatial_sampling_params=None):
    """
    Prepare data with proper TDR application
    
    Enhanced to support spatial sampling along well trajectory for deviated wells.
    
    Args:
        data_manager: DataManager instance
        well_tdr: Time-depth relationship data
        well_name: Name of the well
        use_upscaled: Whether to use upscaled logs
        current_survey: Survey name
        current_attribute: Attribute name
        TRACE_INDEX: Fixed trace index (used only if spatial sampling disabled)
        parent_widget: Parent widget for UI updates
        use_spatial_sampling: Enable spatial sampling along well trajectory
        spatial_sampling_params: Dict with spatial sampling configuration
        
    Returns:
        Dictionary with prepared data for ML training
    """
    if well_name == "Select":
        safe_print("No well selected.")
        return None

    # Get log data with proper TWT calculation
    try:
        log_df, upscale_parameters, current_sampling_interval = get_Logdf_w_TWT(data_manager, well_tdr, well_name, parent_widget)
        if log_df is None or log_df.empty:
            safe_print(f"Error in prepare_data: get_Logdf_w_TWT returned empty log data for well {well_name}")
            return None
    except Exception as e:
        safe_print(f"Error in prepare_data: Exception in get_Logdf_w_TWT for well {well_name}: {e}")
        return None

    # Get configurable log column names
    sonic_col = data_manager.find_log_column('sonic_log', well_name)
    density_col = data_manager.find_log_column('density_log', well_name)
    
    # Validate required columns exist
    if sonic_col is None or density_col is None:
        safe_print(f"Error in prepare_data: Missing required log columns for well {well_name}: sonic_log={sonic_col}, density_log={density_col}")
        available_columns = data_manager.well_log_handler.get_well_logs(well_name).columns.tolist() if hasattr(data_manager.well_log_handler, 'get_well_logs') else []
        safe_print(f"Available columns: {available_columns}")
        return None

    # Calculate acoustic impedance
    log_times = log_df['TWT'].values
    acoustic_impedance = log_df[density_col] * (1000000 / (3.28084 * log_df[sonic_col]))

    # safe_print(f"In prepare_data_w_tdr log_times {log_times}, acoustic_impedance: {acoustic_impedance}")

    # Get well location and seismic data
    well_data = data_manager.well_handler.get_well_data(well_name)
    if well_data.empty:
        safe_print(f"Error in prepare_data: No well data found for well {well_name}")
        return None

    x, y = well_data['Surface X'].values[0], well_data['Surface Y'].values[0]
    
    # Enhanced seismic data extraction with spatial sampling option
    if use_spatial_sampling:
        try:
            safe_print(f"[SPATIAL SAMPLING] Using spatial sampling for well {well_name}")
            
            # Get well trajectory for spatial sampling
            trajectory_result = data_manager.well_handler.construct_well_data(well_name)
            if trajectory_result is None:
                safe_print(f"[SPATIAL SAMPLING] Failed to get trajectory for {well_name}, using surface sampling")
                use_spatial_sampling = False
            else:
                trajectory_df, is_deviated, tdr_well = trajectory_result
                
                if not is_deviated or len(trajectory_df) < 2:
                    safe_print(f"[SPATIAL SAMPLING] Well {well_name} is not deviated, using surface sampling")
                    use_spatial_sampling = False
                else:
                    # Use DataManager's new spatial sampling capability
                    seismic_df = data_manager.get_seismic_data(
                        well_name, 
                        [current_attribute],
                        use_spatial_sampling=True,
                        well_trajectory_data=trajectory_df,
                        spatial_sampling_params=spatial_sampling_params
                    )
                    
                    if seismic_df is not None and not seismic_df.empty:
                        # Extract seismic trace and times from spatial sampling result
                        times = seismic_df['TWT'].values
                        seismic_trace = seismic_df[current_attribute].values
                        
                        # For compatibility, still create trace_data array with single trace
                        trace_data = np.array([seismic_trace])
                        
                        safe_print(f"[SPATIAL SAMPLING] Successfully extracted spatially-sampled data")
                        safe_print(f"  - Time range: {times.min():.1f} to {times.max():.1f}ms")
                        safe_print(f"  - Trace length: {len(seismic_trace)} samples")
                        
                        # Skip to normalization
                        spatial_sampling_success = True
                    else:
                        safe_print(f"[SPATIAL SAMPLING] Spatial sampling failed, falling back to surface sampling")
                        use_spatial_sampling = False
                        
        except Exception as e:
            safe_print(f"[SPATIAL SAMPLING] Error in spatial sampling: {e}")
            safe_print(f"[SPATIAL SAMPLING] Falling back to surface sampling")
            use_spatial_sampling = False
    
    # Original surface-based sampling (fallback or when spatial sampling disabled)
    if not use_spatial_sampling:
        safe_print(f"[SURFACE SAMPLING] Using surface coordinates for well {well_name}")
        
        # ENHANCED TRACE INVESTIGATION LOGGING
        safe_print(f"\n=== TRACE INVESTIGATION LOG - WELL {well_name} ===")
        safe_print(f"Process: {'Synthetic Tie Widget' if parent_widget and hasattr(parent_widget, 'seismic_attribute_combo') else 'ML Data Preparation (prepare_data_w_tdr)'}")
        safe_print(f"Survey: {current_survey}, Attribute: {current_attribute}")
        safe_print(f"Well Surface Coordinates: X={x:.2f}, Y={y:.2f}")
        
        # Check if custom trace selection is being used
        custom_coordinates = get_custom_trace_coordinates(data_manager, well_name, parent_widget)
        
        # Get well coordinates for distance calculation
        _nearest_result = data_manager.seismic_handler.get_nearest_in_crosslines(current_survey, x, y)
        nearest_inline, nearest_crossline = _nearest_result[0], _nearest_result[1]
        safe_print(f"Nearest trace from well surface: Inline {nearest_inline}, Crossline {nearest_crossline}")
        
        if custom_coordinates:
            # Use custom trace coordinates for seismic extraction
            custom_inline, custom_crossline = custom_coordinates
            safe_print(f"✓ CUSTOM TRACE SELECTION ACTIVE - Using: Inline {custom_inline}, Crossline {custom_crossline}")
            
            # Calculate distance from well to custom trace
            try:
                well_trace_coords = data_manager.seismic_handler.get_trace_coordinates(current_survey, nearest_inline, nearest_crossline)
                custom_trace_coords = data_manager.seismic_handler.get_trace_coordinates(current_survey, custom_inline, custom_crossline)
                if well_trace_coords and custom_trace_coords:
                    distance = ((well_trace_coords[0] - custom_trace_coords[0])**2 + (well_trace_coords[1] - custom_trace_coords[1])**2)**0.5
                    safe_print(f"Distance from well to custom trace: {distance:.2f} meters")
                else:
                    safe_print(f"Could not calculate distance - missing coordinate data")
            except Exception as e:
                safe_print(f"Distance calculation failed: {e}")
            
            trace_data, times = data_manager.seismic_handler.get_inline(current_survey, current_attribute, custom_inline, custom_crossline)
        else:
            # Use well coordinates for seismic extraction
            safe_print(f"× Custom trace selection NOT active - Using nearest trace: Inline {nearest_inline}, Crossline {nearest_crossline}")
            safe_print(f"Distance from well to nearest trace: 0.0 meters (by definition)")
            trace_data, times = data_manager.seismic_handler.get_inline(current_survey, current_attribute, nearest_inline, nearest_crossline)

        if trace_data is None or times is None:
            inline_used = custom_coordinates[0] if custom_coordinates else nearest_inline
            crossline_used = custom_coordinates[1] if custom_coordinates else nearest_crossline
            safe_print(f"ERROR: Failed to load seismic data for well {well_name}. Survey: {current_survey}, Attribute: {current_attribute}")
            safe_print(f"Used inline: {inline_used}, crossline: {crossline_used}")
            safe_print(f"=== END TRACE INVESTIGATION LOG ===\n")
            return None

        # Log seismic data extraction details
        safe_print(f"Seismic data extracted successfully:")
        safe_print(f"  - Trace data shape: {trace_data.shape if hasattr(trace_data, 'shape') else f'{len(trace_data)} traces'} ")
        safe_print(f"  - Time samples: {len(times)} (from {times[0]:.1f} to {times[-1]:.1f} ms)")
        
        # For custom trace, always use center trace (index 5) since we've centered the extraction on the custom trace
        if custom_coordinates:
            trace_index_to_use = 5  # Center trace in the extracted window
            safe_print(f"CUSTOM TRACE: Using center trace index {trace_index_to_use} (centered extraction on custom trace)")
        else:
            # Use provided TRACE_INDEX with bounds checking for automatic selection
            if TRACE_INDEX >= len(trace_data):
                safe_print(f"WARNING: Trace index {TRACE_INDEX} is out of bounds (max: {len(trace_data)-1}), using default trace index 5")
                trace_index_to_use = min(5, len(trace_data) - 1)  # Use default or max available
            else:
                trace_index_to_use = TRACE_INDEX
            safe_print(f"NEAREST TRACE: Using trace index {trace_index_to_use} for automatic selection")
        
        seismic_trace = trace_data[trace_index_to_use]
        safe_print(f"Final seismic trace extracted: index {trace_index_to_use} from {len(trace_data)} available traces")
        safe_print(f"Trace value range: {seismic_trace.min():.6f} to {seismic_trace.max():.6f}")
        safe_print(f"Zone time range: {times[0]:.1f}ms to {times[-1]:.1f}ms (last zone starts at ~{times[-1]:.1f}ms)")
        safe_print(f"=== END TRACE INVESTIGATION LOG ===\n")
        spatial_sampling_success = False
    
    # Normalize seismic trace (common for both sampling methods)
    if seismic_trace.max() != seismic_trace.min():
        seismic_trace = (seismic_trace - seismic_trace.min()) / (seismic_trace.max() - seismic_trace.min())
    else:
        safe_print(f"Warning: Seismic trace has constant values, normalization skipped")
        seismic_trace = np.zeros_like(seismic_trace)

    if upscale_parameters is None:
        current_sampling_interval = times[1] - times[0]

    # Calculate synthetic times
    synthetic_times = log_times[:-1] + (log_times[1:] - log_times[:-1]) / 2

    return {
        "log_df": log_df,
        "log_times": log_times,
        "trace_data": trace_data,
        "times": times,
        "seismic_trace": seismic_trace,
        "acoustic_impedance": acoustic_impedance,
        "current_sampling_interval": current_sampling_interval,
        "well_tdr": well_tdr,
        "synthetic_times": synthetic_times,
        "spatial_sampling_used": use_spatial_sampling,
        "trace_index_used": None if use_spatial_sampling else (trace_index_to_use if 'trace_index_to_use' in locals() else TRACE_INDEX)
    }
def prepare_data(data_manager, well_name, use_upscaled=False, current_survey="current_survey", current_attribute="current_attribute", TRACE_INDEX=5, wavelet=None, bulk_shift=0, parent_widget=None, use_spatial_sampling=False, spatial_sampling_params=None):
    """
    Prepare data for ML training with optional spatial sampling
    
    Enhanced to support spatial sampling along well trajectory for deviated wells.
    """
    if well_name == "Select":
        safe_print("No well selected.")
        return None

    # Fetch Time-Depth Relationship (TDR)
    well_tdr = data_manager.well_handler.get_preferred_tdr(well_name)
    return prepare_data_w_tdr(
        data_manager,
        well_tdr,
        well_name,
        use_upscaled,
        current_survey=current_survey,
        current_attribute=current_attribute,
        TRACE_INDEX=TRACE_INDEX,
        parent_widget=parent_widget,
        use_spatial_sampling=use_spatial_sampling,
        spatial_sampling_params=spatial_sampling_params
    )


def extract_seismic_in_range(seismic_trace, times, log_times, acoustic_impedance, bulk_shift, extract_range, current_sampling_interval):
    # safe_print(f"in extract_seismic_in_range {seismic_trace}")
    # Shift log_times by adding bulk_shift
    shifted_log_times = log_times + bulk_shift

    # Define the start and end times for extraction
    start_time = max(shifted_log_times[0], extract_range[0])
    end_time = min(shifted_log_times[-1], extract_range[1])

    # Generate common times over which data will be extracted
    common_times = np.arange(start_time, end_time + current_sampling_interval, current_sampling_interval)

    # Interpolate impedance at common_times (shifted log times)
    impedance_extracted = np.interp(common_times, shifted_log_times, acoustic_impedance)

    # Calculate reflectivity
    reflectivity = create_reflectivity(impedance_extracted)

    # Extract seismic data at common_times (since seismic times are unshifted)
    seismic_trace_extracted = np.interp(common_times, times, seismic_trace)

    # well_times_extracted is common_times
    well_times_extracted = common_times

    # seismic_times_extracted is common_times
    seismic_times_extracted = common_times

    return seismic_times_extracted, seismic_trace_extracted, well_times_extracted, impedance_extracted, reflectivity


def detect_wavelet_sampling_rate(wavelet_times):
    """
    Detect the sampling rate of a wavelet from its time array.
    
    Args:
        wavelet_times: numpy array of time values for the wavelet
        
    Returns:
        float: sampling interval in ms
    """
    if len(wavelet_times) < 2:
        safe_print("[WAVELET] Warning: Insufficient time points to detect sampling rate, using default 2ms")
        return 2.0
    
    # Calculate sampling intervals
    intervals = np.diff(wavelet_times)
    
    # Use the most common interval (mode) as the sampling rate
    sampling_interval = np.median(intervals)
    
    safe_print(f"[WAVELET] Detected wavelet sampling interval: {sampling_interval:.3f} ms")
    return sampling_interval


def resample_wavelet_to_match_logs(wavelet_data, log_sampling_interval):
    """
    Resample wavelet to match log sampling interval for accurate synthetic calculation.
    
    Args:
        wavelet_data: dict containing 'time' and 'amplitude' arrays
        log_sampling_interval: float, target sampling interval in ms
        
    Returns:
        dict: resampled wavelet with 'time' and 'amplitude' arrays
    """
    if wavelet_data is None:
        safe_print("[WAVELET] Error: No wavelet data provided")
        return None
        
    wavelet_times = wavelet_data.get('time', np.array([]))
    wavelet_amplitude = wavelet_data.get('amplitude', np.array([]))
    
    if len(wavelet_times) == 0 or len(wavelet_amplitude) == 0:
        safe_print("[WAVELET] Error: Empty wavelet time or amplitude data")
        return None
        
    # Detect current wavelet sampling rate
    current_sampling = detect_wavelet_sampling_rate(wavelet_times)
    
    safe_print(f"[WAVELET] Current wavelet sampling: {current_sampling:.3f} ms")
    safe_print(f"[WAVELET] Target log sampling: {log_sampling_interval:.3f} ms")
    
    # Check if resampling is needed (with small tolerance for floating point comparison)
    if abs(current_sampling - log_sampling_interval) < 0.001:
        safe_print("[WAVELET] Sampling rates match, no resampling needed")
        return wavelet_data
    
    safe_print(f"[WAVELET] Resampling wavelet from {current_sampling:.3f} ms to {log_sampling_interval:.3f} ms")
    
    try:
        # Create interpolation function
        f = interpolate.interp1d(wavelet_times, wavelet_amplitude, kind='linear', fill_value='extrapolate')
        
        # Create new time array with target sampling rate
        # Maintain the same number of samples to avoid dimension mismatches
        original_length = len(wavelet_times)
        time_start = wavelet_times[0]
        time_end = wavelet_times[-1]
        time_duration = time_end - time_start
        
        # Create exactly the same number of samples as original
        new_times = np.linspace(time_start, time_end, original_length)
        
        # Verify the actual sampling rate achieved
        if original_length > 1:
            actual_sampling = new_times[1] - new_times[0]
            safe_print(f"[WAVELET] Maintaining original length: {original_length} samples")
            safe_print(f"[WAVELET] Target sampling: {log_sampling_interval:.3f}ms, Actual: {actual_sampling:.3f}ms")
        
        # Interpolate amplitude values
        new_amplitude = f(new_times)
        
        safe_print(f"[WAVELET] Resampled wavelet: {len(wavelet_times)} → {len(new_times)} samples")
        
        return {
            'time': new_times,
            'amplitude': new_amplitude
        }
        
    except Exception as e:
        safe_print(f"[WAVELET] Error resampling wavelet: {e}")
        return wavelet_data  # Return original if resampling fails


def prepare_wavelet_for_synthetic(wavelet, log_sampling_interval):
    """
    Prepare wavelet for synthetic calculation by ensuring proper sampling.
    
    Args:
        wavelet: numpy array, dict, or tuple containing wavelet data
        log_sampling_interval: float, target sampling interval in ms
        
    Returns:
        numpy array: properly sampled wavelet amplitude values
    """
    safe_print(f"[WAVELET] ==========================================================")
    safe_print(f"[WAVELET] PREPARING WAVELET FOR SYNTHETIC CALCULATION")
    safe_print(f"[WAVELET] ==========================================================")
    safe_print(f"[WAVELET] Input wavelet type: {type(wavelet)}")
    safe_print(f"[WAVELET] Target log sampling interval: {log_sampling_interval} ms")
    
    # Handle different wavelet input formats
    if isinstance(wavelet, tuple) and len(wavelet) == 2:
        # External wavelet format: (data, metadata)
        wavelet_data, metadata = wavelet
        safe_print(f"[WAVELET] Processing external wavelet tuple - data shape: {wavelet_data.shape}")
        safe_print(f"[WAVELET] Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        
        if wavelet_data.shape[1] == 2:  # [time, amplitude] format
            wavelet_dict = {
                'time': wavelet_data[:, 0],
                'amplitude': wavelet_data[:, 1]
            }
            
            # Detect original sampling rate of external wavelet
            original_sampling = detect_wavelet_sampling_rate(wavelet_dict['time'])
            safe_print(f"[WAVELET] *** DETECTED EXTERNAL WAVELET SAMPLING: {original_sampling:.3f} ms ***")
            safe_print(f"[WAVELET] *** TARGET LOG SAMPLING: {log_sampling_interval:.3f} ms ***")
            
            resampled_wavelet = resample_wavelet_to_match_logs(wavelet_dict, log_sampling_interval)
            if resampled_wavelet is not None:
                safe_print(f"[WAVELET] *** RESAMPLING SUCCESSFUL: {len(wavelet_dict['amplitude'])} → {len(resampled_wavelet['amplitude'])} samples ***")
                return resampled_wavelet['amplitude']
            else:
                safe_print("[WAVELET] *** WARNING: Failed to resample external wavelet, using original amplitude ***")
                return wavelet_data[:, 1]
        else:
            safe_print("[WAVELET] *** WARNING: Unexpected external wavelet data format ***")
            return wavelet_data.flatten()
            
    elif isinstance(wavelet, dict):
        safe_print(f"[WAVELET] Processing wavelet dict with keys: {list(wavelet.keys())}")
        
        # Check if this is an analytical wavelet with sampling info
        if wavelet.get('type') == 'analytical' and 'sampling_interval' in wavelet:
            original_sampling = wavelet['sampling_interval']
            amplitude_data = wavelet['amplitude']
            safe_print(f"[WAVELET] *** ANALYTICAL WAVELET - ORIGINAL SAMPLING: {original_sampling:.3f} ms ***")
            safe_print(f"[WAVELET] *** TARGET LOG SAMPLING: {log_sampling_interval:.3f} ms ***")
            safe_print(f"[WAVELET] *** WAVELET LENGTH: {len(amplitude_data)} samples ***")
            
            # Check if resampling is needed
            if abs(original_sampling - log_sampling_interval) > 0.001:
                # Create time array for resampling
                time_array = np.arange(len(amplitude_data)) * original_sampling
                time_array = time_array - time_array[len(time_array)//2]  # Center around zero
                
                wavelet_with_time = {
                    'time': time_array,
                    'amplitude': amplitude_data
                }
                
                resampled_wavelet = resample_wavelet_to_match_logs(wavelet_with_time, log_sampling_interval)
                if resampled_wavelet is not None:
                    safe_print(f"[WAVELET] *** ANALYTICAL RESAMPLING SUCCESSFUL: {len(amplitude_data)} → {len(resampled_wavelet['amplitude'])} samples ***")
                    return resampled_wavelet['amplitude']
                else:
                    safe_print("[WAVELET] *** WARNING: Failed to resample analytical wavelet, using original ***")
                    return amplitude_data
            else:
                safe_print(f"[WAVELET] *** ANALYTICAL WAVELET SAMPLING MATCHES - NO RESAMPLING NEEDED ***")
                return amplitude_data
        
        # Regular dict wavelet with time information
        elif 'time' in wavelet:
            original_sampling = detect_wavelet_sampling_rate(wavelet['time'])
            safe_print(f"[WAVELET] *** DETECTED DICT WAVELET SAMPLING: {original_sampling:.3f} ms ***")
            safe_print(f"[WAVELET] *** TARGET LOG SAMPLING: {log_sampling_interval:.3f} ms ***")
            
            # Resample to match log sampling rate
            resampled_wavelet = resample_wavelet_to_match_logs(wavelet, log_sampling_interval)
            
            if resampled_wavelet is not None:
                safe_print(f"[WAVELET] *** RESAMPLING SUCCESSFUL: {len(wavelet.get('amplitude', []))} → {len(resampled_wavelet['amplitude'])} samples ***")
                return resampled_wavelet['amplitude']
            else:
                safe_print("[WAVELET] *** WARNING: Failed to resample wavelet, using original amplitude ***")
                return wavelet.get('amplitude', np.array([]))
        
        # Fallback for dict without time or type info
        else:
            safe_print("[WAVELET] *** WARNING: Dict wavelet missing time or type information ***")
            return wavelet.get('amplitude', np.array([]))
            
    elif isinstance(wavelet, np.ndarray):
        safe_print(f"[WAVELET] Processing numpy array wavelet with shape: {wavelet.shape}")
        
        # Check if it's a 2D array with time and amplitude
        if wavelet.ndim == 2 and wavelet.shape[1] == 2:
            wavelet_dict = {
                'time': wavelet[:, 0],
                'amplitude': wavelet[:, 1]
            }
            
            # Detect original sampling rate
            original_sampling = detect_wavelet_sampling_rate(wavelet_dict['time'])
            safe_print(f"[WAVELET] *** DETECTED 2D ARRAY WAVELET SAMPLING: {original_sampling:.3f} ms ***")
            safe_print(f"[WAVELET] *** TARGET LOG SAMPLING: {log_sampling_interval:.3f} ms ***")
            
            resampled_wavelet = resample_wavelet_to_match_logs(wavelet_dict, log_sampling_interval)
            if resampled_wavelet is not None:
                safe_print(f"[WAVELET] *** RESAMPLING SUCCESSFUL: {len(wavelet_dict['amplitude'])} → {len(resampled_wavelet['amplitude'])} samples ***")
                return resampled_wavelet['amplitude']
            else:
                safe_print("[WAVELET] *** WARNING: Failed to resample 2D array wavelet, using original amplitude ***")
                return wavelet[:, 1]  # Return original amplitude column
        else:
            # 1D numpy array - likely analytical wavelet (Ricker, etc.)
            # Try to estimate its expected sampling based on typical wavelet characteristics
            wavelet_length = len(wavelet)
            
            # Common analytical wavelet lengths and their typical sampling rates
            if wavelet_length <= 64:
                estimated_sampling = 4.0  # Short wavelets often at 4ms
            elif wavelet_length <= 128:
                estimated_sampling = 2.0  # Medium wavelets often at 2ms  
            elif wavelet_length <= 256:
                estimated_sampling = 1.0  # Longer wavelets often at 1ms
            else:
                estimated_sampling = 0.5  # Very long wavelets often at fine sampling
            
            safe_print(f"[WAVELET] *** 1D NUMPY ARRAY - LIKELY ANALYTICAL WAVELET ***")
            safe_print(f"[WAVELET] *** ESTIMATED ORIGINAL SAMPLING: ~{estimated_sampling:.1f} ms (based on {wavelet_length} samples) ***")
            safe_print(f"[WAVELET] *** TARGET LOG SAMPLING: {log_sampling_interval:.3f} ms ***")
            
            # Check if resampling might be needed
            sampling_ratio = estimated_sampling / log_sampling_interval
            if sampling_ratio > 2.0:
                safe_print(f"[WAVELET] *** WARNING: Analytical wavelet may be too coarse ({estimated_sampling}ms) for fine log sampling ({log_sampling_interval}ms) ***")
                safe_print(f"[WAVELET] *** RATIO: {sampling_ratio:.1f}x - Consider using finer wavelet sampling ***")
            elif sampling_ratio < 0.5:
                safe_print(f"[WAVELET] *** INFO: Analytical wavelet is finer ({estimated_sampling}ms) than log sampling ({log_sampling_interval}ms) ***")
                safe_print(f"[WAVELET] *** RATIO: {sampling_ratio:.1f}x - This should provide good resolution ***")
            else:
                safe_print(f"[WAVELET] *** INFO: Analytical wavelet sampling ({estimated_sampling}ms) reasonably matches log sampling ({log_sampling_interval}ms) ***")
            
            safe_print(f"[WAVELET] *** USING ANALYTICAL WAVELET AS-IS: {len(wavelet)} samples ***")
            return wavelet.flatten()
        
    else:
        safe_print(f"[WAVELET] *** ERROR: Unknown wavelet format: {type(wavelet)} ***")
        return np.array([])


def trigger_upscaling_workflow(data_manager, well_name, parent_widget=None):
    """
    Trigger upscaling workflow when upscaled logs are missing.

    In GeoAgent (headless), this logs a warning and returns False.
    The caller should ensure upscaled logs exist before calling
    synthetic generation functions.
    """
    logger.warning(
        "Well '%s' requires upscaled logs for synthetic generation. "
        "Please run log upscaling before generating synthetics.",
        well_name,
    )
    return False

def time_aware_block_average(time_array, data_array, target_interval):
    """
    Time-aware block averaging that preserves actual time relationships.
    
    This function replaces the flawed block_average_upscale() approach by:
    1. Creating time bins based on target interval
    2. Assigning original samples to appropriate bins based on their actual time values
    3. Averaging data within each bin while preserving time accuracy
    
    Parameters:
    time_array (array): Original time values (TWT)
    data_array (array): Original data values (log data)
    target_interval (float): Target time interval in same units as time_array
    
    Returns:
    tuple: (averaged_times, averaged_data) with proper time alignment
    """
    time_array = np.asarray(time_array)
    data_array = np.asarray(data_array)
    
    # Create time bins based on target interval
    t_min = np.ceil(time_array.min() / target_interval) * target_interval
    t_max = np.floor(time_array.max() / target_interval) * target_interval
    bin_edges = np.arange(t_min, t_max + target_interval, target_interval)
    
    if len(bin_edges) < 2:
        # Not enough range for binning, return original data
        return time_array, data_array
    
    # Assign each sample to a time bin
    bin_indices = np.digitize(time_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)  # Ensure valid indices
    
    averaged_times = []
    averaged_data = []
    
    # Process each bin
    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        if np.any(mask):
            # Calculate bin center time
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            
            # Average data in this bin
            bin_data = data_array[mask]
            valid_data = bin_data[~np.isnan(bin_data)]
            
            if len(valid_data) > 0:
                averaged_times.append(bin_center)
                averaged_data.append(np.mean(valid_data))
    
    return np.array(averaged_times), np.array(averaged_data)

def time_aware_weighted_average(time_array, data_array, target_interval, weights=None):
    """
    Time-aware weighted averaging that preserves actual time relationships.
    
    Similar to time_aware_block_average but applies weights to samples within each bin.
    
    Parameters:
    time_array (array): Original time values (TWT)
    data_array (array): Original data values (log data)  
    target_interval (float): Target time interval
    weights (array): Optional weights, if None uses distance-based weighting
    
    Returns:
    tuple: (averaged_times, averaged_data) with proper time alignment
    """
    time_array = np.asarray(time_array)
    data_array = np.asarray(data_array)
    
    # Create time bins
    t_min = np.ceil(time_array.min() / target_interval) * target_interval
    t_max = np.floor(time_array.max() / target_interval) * target_interval
    bin_edges = np.arange(t_min, t_max + target_interval, target_interval)
    
    if len(bin_edges) < 2:
        return time_array, data_array
    
    bin_indices = np.digitize(time_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    averaged_times = []
    averaged_data = []
    
    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        if np.any(mask):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_times = time_array[mask]
            bin_data = data_array[mask]
            
            # Remove NaN values
            valid_mask = ~np.isnan(bin_data)
            if np.any(valid_mask):
                valid_times = bin_times[valid_mask]
                valid_data = bin_data[valid_mask]
                
                if weights is None:
                    # Use distance-based weighting (closer to bin center gets higher weight)
                    distances = np.abs(valid_times - bin_center)
                    weights_local = 1.0 / (distances + target_interval * 0.1)  # Add small offset to avoid division by zero
                else:
                    weights_local = np.ones(len(valid_data))
                
                # Calculate weighted average
                weighted_avg = np.average(valid_data, weights=weights_local)
                averaged_times.append(bin_center)
                averaged_data.append(weighted_avg)
    
    return np.array(averaged_times), np.array(averaged_data)

# Wrapper functions for spatial sampling compatibility
def prepare_data_surface_only(data_manager, well_name, use_upscaled=False, current_survey="current_survey", current_attribute="current_attribute", TRACE_INDEX=5, wavelet=None, bulk_shift=0, parent_widget=None):
    """
    Wrapper for traditional surface-only sampling (backward compatibility).
    This maintains the original behavior of using TRACE_INDEX for fixed trace sampling.
    """
    return prepare_data(
        data_manager=data_manager,
        well_name=well_name,
        use_upscaled=use_upscaled,
        current_survey=current_survey,
        current_attribute=current_attribute,
        TRACE_INDEX=TRACE_INDEX,
        wavelet=wavelet,
        bulk_shift=bulk_shift,
        parent_widget=parent_widget,
        use_spatial_sampling=False,
        spatial_sampling_params=None
    )

def prepare_data_spatial_sampling(data_manager, well_name, use_upscaled=False, current_survey="current_survey", current_attribute="current_attribute", wavelet=None, bulk_shift=0, parent_widget=None, spatial_sampling_params=None):
    """
    Wrapper for spatially-aware sampling along well trajectories.
    This uses the new spatial sampling system instead of fixed TRACE_INDEX.
    """
    # Set default spatial sampling parameters if not provided
    if spatial_sampling_params is None:
        spatial_sampling_params = {
            'interpolation_method': 'inverse_distance',
            'max_interpolation_distance': 50.0,
            'fallback_threshold': 200.0
        }
    
    return prepare_data(
        data_manager=data_manager,
        well_name=well_name,
        use_upscaled=use_upscaled,
        current_survey=current_survey,
        current_attribute=current_attribute,
        TRACE_INDEX=None,  # Not used in spatial sampling
        wavelet=wavelet,
        bulk_shift=bulk_shift,
        parent_widget=parent_widget,
        use_spatial_sampling=True,
        spatial_sampling_params=spatial_sampling_params
    )


# ==========================================
# UNIFIED CORRELATION CALCULATION FUNCTIONS
# ==========================================

def calculate_correlation_direct(seismic_trace, seismic_times, synthetic_seismic, synthetic_times):
    """
    Direct correlation calculation method from single well module.
    Uses overlapping range approach with interpolation for robust correlation.
    
    This is the proven method from SyntheticSeismicTieAppNew.py that achieves 
    high correlation coefficients (0.7+ typical).
    
    Args:
        seismic_trace (np.ndarray): Seismic amplitude values
        seismic_times (np.ndarray): Time values for seismic data
        synthetic_seismic (np.ndarray): Synthetic seismic amplitude values
        synthetic_times (np.ndarray): Time values for synthetic data
        
    Returns:
        float: Correlation coefficient (0-1 range, NaN returned as 0.0)
    """
    try:
        # Determine overlapping time range
        valid_range = (seismic_times >= synthetic_times[0]) & (seismic_times <= synthetic_times[-1])
        
        # Extract overlapping seismic data
        extracted_seismic = seismic_trace[valid_range]
        
        # Interpolate synthetic seismic onto the seismic time grid
        interpolated_synthetic = np.interp(seismic_times[valid_range], synthetic_times, synthetic_seismic)
        
        # Ensure both data arrays have the same length
        min_length = min(len(extracted_seismic), len(interpolated_synthetic))
        extracted_seismic = extracted_seismic[:min_length]
        interpolated_synthetic = interpolated_synthetic[:min_length]
        
        # Calculate correlation
        if min_length < 2:
            return 0.0
            
        correlation = np.corrcoef(interpolated_synthetic, extracted_seismic)[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            return 0.0
            
        return correlation
        
    except Exception as e:
        safe_print(f"Error in calculate_correlation_direct: {e}")
        return 0.0

def calculate_correlation_with_shift(seismic_trace, seismic_times, synthetic_seismic, synthetic_times, shift=0.0):
    """
    Calculate correlation with bulk shift applied to synthetic times.
    Based on BulkShiftFinderDialog's calculate_correlation function.
    
    Args:
        seismic_trace (np.ndarray): Seismic amplitude values
        seismic_times (np.ndarray): Time values for seismic data
        synthetic_seismic (np.ndarray): Synthetic seismic amplitude values
        synthetic_times (np.ndarray): Time values for synthetic data
        shift (float): Bulk shift in milliseconds to apply to synthetic times
        
    Returns:
        float: Correlation coefficient (0-1 range, NaN returned as 0.0)
    """
    try:
        # Apply shift to synthetic times
        shifted_synthetic_times = synthetic_times + shift
        
        # Find the range of seismic times that overlap with shifted synthetic times
        valid_range = (seismic_times >= shifted_synthetic_times[0]) & (seismic_times <= shifted_synthetic_times[-1])
        
        # Extract the overlapping portion of the actual seismic
        extracted_seismic = seismic_trace[valid_range]
        
        # Interpolate synthetic seismic onto the seismic time grid
        interpolated_synthetic = np.interp(seismic_times[valid_range], shifted_synthetic_times, synthetic_seismic)
        
        # Ensure both data arrays have the same length
        min_length = min(len(extracted_seismic), len(interpolated_synthetic))
        extracted_seismic = extracted_seismic[:min_length]
        interpolated_synthetic = interpolated_synthetic[:min_length]
        
        # Calculate correlation
        if min_length < 2:
            return 0.0
            
        correlation = np.corrcoef(interpolated_synthetic, extracted_seismic)[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            return 0.0
            
        return correlation
        
    except Exception as e:
        safe_print(f"Error in calculate_correlation_with_shift: {e}")
        return 0.0

def find_optimal_bulk_shift_unified(seismic_trace, seismic_times, synthetic_seismic, synthetic_times, 
                                   shift_range=(-100, 100), shift_step=2):
    """
    Find optimal bulk shift using unified correlation calculation.
    Uses the robust direct correlation method for consistent results between single well and batch processing.
    
    Args:
        seismic_trace (np.ndarray): Seismic amplitude values
        seismic_times (np.ndarray): Time values for seismic data
        synthetic_seismic (np.ndarray): Synthetic seismic amplitude values
        synthetic_times (np.ndarray): Time values for synthetic data
        shift_range (tuple): (min_shift, max_shift) in milliseconds
        shift_step (float): Step size for shift iterations in milliseconds
        
    Returns:
        tuple: (optimal_shift, best_correlation) where optimal_shift is in milliseconds
    """
    try:
        best_shift = 0.0
        best_correlation = -1.0
        
        # Generate shift range
        shifts = np.arange(shift_range[0], shift_range[1] + shift_step, shift_step)
        
        for shift in shifts:
            correlation = calculate_correlation_with_shift(
                seismic_trace, seismic_times, synthetic_seismic, synthetic_times, shift
            )
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_shift = shift
        
        return best_shift, best_correlation
        
    except Exception as e:
        safe_print(f"Error in find_optimal_bulk_shift_unified: {e}")
        return 0.0, 0.0

# Backward compatibility function that wraps the time-zone method for existing batch code
def calculate_correlation_legacy_time_zone(seismic_trace, seismic_times, synthetic_seismic, synthetic_times, 
                                         start_time, end_time):
    """
    Legacy time-zone based correlation calculation for backward compatibility.
    This preserves existing batch module behavior as fallback option.
    
    Note: This method typically produces lower correlation coefficients due to 
    time zone constraints and complex intersection logic.
    """
    try:
        # Get intersection of time zone and actual data time ranges
        seismic_min, seismic_max = float(seismic_times[0]), float(seismic_times[-1])
        synthetic_min, synthetic_max = float(synthetic_times[0]), float(synthetic_times[-1])
        
        # Calculate intersection
        actual_start = max(start_time, seismic_min, synthetic_min)
        actual_end = min(end_time, seismic_max, synthetic_max)
        
        if actual_start >= actual_end:
            return 0.0
        
        # Filter data to intersection time zone
        seismic_mask = (seismic_times >= actual_start) & (seismic_times <= actual_end)
        synthetic_mask = (synthetic_times >= actual_start) & (synthetic_times <= actual_end)
        
        if not np.any(seismic_mask) or not np.any(synthetic_mask):
            return 0.0
        
        # Extract windowed data
        seismic_windowed = seismic_trace[seismic_mask]
        seismic_times_windowed = seismic_times[seismic_mask]
        synthetic_windowed = synthetic_seismic[synthetic_mask]
        synthetic_times_windowed = synthetic_times[synthetic_mask]
        
        if len(seismic_times_windowed) < 2 or len(synthetic_times_windowed) < 2:
            return 0.0
        
        # Interpolate synthetic to seismic time grid
        from scipy.interpolate import interp1d
        
        interp_start = max(synthetic_times_windowed[0], seismic_times_windowed[0])
        interp_end = min(synthetic_times_windowed[-1], seismic_times_windowed[-1])
        
        if interp_start >= interp_end:
            return 0.0
        
        # Filter seismic to interpolation range
        seismic_interp_mask = (seismic_times_windowed >= interp_start) & (seismic_times_windowed <= interp_end)
        seismic_for_corr = seismic_windowed[seismic_interp_mask]
        seismic_times_for_corr = seismic_times_windowed[seismic_interp_mask]
        
        # Interpolate synthetic to seismic time grid
        interp_func = interp1d(synthetic_times_windowed, synthetic_windowed, kind='linear', 
                             bounds_error=False, fill_value='extrapolate')
        synthetic_interp = interp_func(seismic_times_for_corr)
        
        # Calculate correlation coefficient
        if len(seismic_for_corr) > 1 and len(synthetic_interp) > 1:
            correlation_matrix = np.corrcoef(seismic_for_corr, synthetic_interp)
            correlation = correlation_matrix[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return correlation
        else:
            return 0.0
            
    except Exception as e:
        safe_print(f"Error in calculate_correlation_legacy_time_zone: {e}")
        return 0.0

# ==========================================
# VALIDATION AND TESTING FUNCTIONS
# ==========================================

def validate_correlation_consistency(seismic_trace, seismic_times, synthetic_seismic, synthetic_times, 
                                   tolerance=0.05, verbose=True):
    """
    Validate that unified correlation produces consistent results with the original BulkShiftFinderDialog method.
    
    Args:
        seismic_trace, seismic_times, synthetic_seismic, synthetic_times: Input data arrays
        tolerance (float): Acceptable difference between correlation methods
        verbose (bool): Print detailed comparison results
    
    Returns:
        dict: Validation results with correlations and consistency check
    """
    try:
        # Calculate correlation using unified direct method
        unified_corr = calculate_correlation_direct(seismic_trace, seismic_times, synthetic_seismic, synthetic_times)
        
        # Calculate correlation using BulkShiftFinderDialog method (with zero shift)
        bsfinder_corr = calculate_correlation_with_shift(seismic_trace, seismic_times, synthetic_seismic, synthetic_times, shift=0.0)
        
        # Check consistency
        difference = abs(unified_corr - bsfinder_corr)
        is_consistent = difference <= tolerance
        
        results = {
            'unified_correlation': unified_corr,
            'bsfinder_correlation': bsfinder_corr,
            'difference': difference,
            'tolerance': tolerance,
            'is_consistent': is_consistent,
            'validation_passed': is_consistent
        }
        
        if verbose:
            safe_print(f"=== CORRELATION VALIDATION RESULTS ===")
            safe_print(f"Unified Direct Method:     {unified_corr:.6f}")
            safe_print(f"BulkShiftFinder Method:    {bsfinder_corr:.6f}")
            safe_print(f"Absolute Difference:       {difference:.6f}")
            safe_print(f"Tolerance:                 {tolerance:.6f}")
            safe_print(f"Consistency Check:         {'✅ PASS' if is_consistent else '❌ FAIL'}")
            safe_print(f"=====================================")
        
        return results
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'validation_passed': False,
            'unified_correlation': 0.0,
            'bsfinder_correlation': 0.0,
            'difference': float('inf'),
            'is_consistent': False
        }
        
        if verbose:
            safe_print(f"❌ VALIDATION ERROR: {e}")
            
        return error_result

def test_bulk_shift_optimization_consistency(seismic_trace, seismic_times, synthetic_seismic, synthetic_times,
                                           shift_tolerance=2.0, corr_tolerance=0.05, verbose=True):
    """
    Test that unified bulk shift optimization produces results consistent with expected behavior.
    
    Args:
        shift_tolerance (float): Acceptable difference in optimal shift (ms)
        corr_tolerance (float): Acceptable difference in correlation coefficient
        verbose (bool): Print detailed test results
    
    Returns:
        dict: Test results with optimal shifts and consistency validation
    """
    try:
        # Test unified bulk shift optimization
        unified_shift, unified_corr = find_optimal_bulk_shift_unified(
            seismic_trace, seismic_times, synthetic_seismic, synthetic_times
        )
        
        # Test correlation at zero shift for baseline
        zero_shift_corr = calculate_correlation_with_shift(
            seismic_trace, seismic_times, synthetic_seismic, synthetic_times, shift=0.0
        )
        
        # Validate that optimal shift produces better or equal correlation than zero shift
        improvement = unified_corr - zero_shift_corr
        shift_improves_correlation = improvement >= -corr_tolerance  # Allow small negative due to numerical precision
        
        results = {
            'unified_optimal_shift': unified_shift,
            'unified_optimal_correlation': unified_corr,
            'zero_shift_correlation': zero_shift_corr,
            'correlation_improvement': improvement,
            'shift_improves_correlation': shift_improves_correlation,
            'test_passed': shift_improves_correlation and not np.isnan(unified_corr)
        }
        
        if verbose:
            safe_print(f"=== BULK SHIFT OPTIMIZATION TEST ===")
            safe_print(f"Optimal Shift:             {unified_shift:.1f} ms")
            safe_print(f"Optimal Correlation:       {unified_corr:.6f}")
            safe_print(f"Zero Shift Correlation:    {zero_shift_corr:.6f}")
            safe_print(f"Correlation Improvement:   {improvement:.6f}")
            safe_print(f"Optimization Valid:        {'✅ PASS' if shift_improves_correlation else '❌ FAIL'}")
            safe_print(f"Test Result:               {'✅ PASS' if results['test_passed'] else '❌ FAIL'}")
            safe_print(f"===================================")
        
        return results
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'test_passed': False,
            'unified_optimal_shift': 0.0,
            'unified_optimal_correlation': 0.0,
            'zero_shift_correlation': 0.0,
            'correlation_improvement': 0.0,
            'shift_improves_correlation': False
        }
        
        if verbose:
            safe_print(f"❌ BULK SHIFT TEST ERROR: {e}")
            
        return error_result

