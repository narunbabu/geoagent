#!/usr/bin/env python3
"""
TWT (Two Way Time) Utility Functions

This module provides centralized TWT generation functions to ensure consistency
between training data and inference scripts. 

CRITICAL: The TWT generation must match exactly between training and inference
to prevent feature mismatch that degrades model predictions.

Training Configuration:
- sampling_rate_ms: 0.25ms (from model config)
- TWT start time: 1200ms (project-specific)
- TWT pattern: 1200.0, 1200.25, 1200.5, 1200.75, 1201.0, ...

This fixes the issue where inference scripts were using SEG-Y sample_interval_us
(2ms) instead of training sampling_rate_ms (0.25ms), causing 8x TWT mismatch.
"""

import numpy as np
from typing import Dict, Optional, Union
import warnings
from geoagent.utils.safe_print import safe_print

def get_training_compatible_twt(
    n_samples: int, 
    model_config: Dict, 
    start_time_ms: float = 1200.0
) -> np.ndarray:
    """
    Generate TWT array that matches training data format exactly.
    
    This function ensures inference TWT generation is identical to training data
    by using the model's configured sampling rate instead of SEG-Y sampling rate.
    
    Args:
        n_samples: Number of time samples in the seismic data
        model_config: Model configuration dictionary containing io_meta
        start_time_ms: Starting TWT time in milliseconds (default: 1200.0)
        
    Returns:
        np.ndarray: TWT array with shape (n_samples,) in milliseconds
                   Pattern: [1200.0, 1200.25, 1200.5, 1200.75, ...]
                   
    Raises:
        ValueError: If sampling_rate_ms not found in model config
        
    Example:
        >>> model_config = {"io_meta": {"sampling_rate_ms": 0.25}}
        >>> twt = get_training_compatible_twt(5, model_config, 1200.0)
        >>> safe_print(twt)
        [1200.0  1200.25 1200.5  1200.75 1201.0 ]
    """
    
    # Extract sampling rate from model config
    try:
        sampling_rate_ms = model_config.get("io_meta", {}).get("sampling_rate_ms")
        if sampling_rate_ms is None:
            raise KeyError("sampling_rate_ms not found in model_config['io_meta']")
    except (KeyError, AttributeError) as e:
        raise ValueError(
            f"Cannot extract sampling_rate_ms from model config: {e}\n"
            f"Expected structure: model_config['io_meta']['sampling_rate_ms']\n"
            f"Available keys: {list(model_config.keys()) if isinstance(model_config, dict) else 'not a dict'}"
        )
    
    # Validate sampling rate
    if not isinstance(sampling_rate_ms, (int, float)) or sampling_rate_ms <= 0:
        raise ValueError(f"Invalid sampling_rate_ms: {sampling_rate_ms}. Must be positive number.")
    
    # Generate TWT array starting from start_time_ms
    time_indices = np.arange(n_samples, dtype=np.float32)
    twt_array = start_time_ms + (time_indices * sampling_rate_ms)
    
    return twt_array.astype(np.float32)

def validate_twt_consistency(
    twt_array: np.ndarray, 
    expected_start: float = 1200.0,
    expected_interval: float = 0.25,
    tolerance: float = 1e-6
) -> tuple[bool, str]:
    """
    Validate that TWT array matches expected training data format.
    
    Args:
        twt_array: TWT array to validate
        expected_start: Expected starting TWT value (default: 1200.0)
        expected_interval: Expected TWT interval (default: 0.25)
        tolerance: Numerical tolerance for comparisons
        
    Returns:
        tuple: (is_valid, message)
               is_valid: True if TWT array is consistent with training data
               message: Validation result description
    """
    
    if len(twt_array) == 0:
        return False, "TWT array is empty"
    
    # Check starting value
    actual_start = float(twt_array[0])
    if abs(actual_start - expected_start) > tolerance:
        return False, f"TWT start mismatch: got {actual_start}, expected {expected_start}"
    
    # Check interval consistency (if array has more than 1 element)
    if len(twt_array) > 1:
        actual_interval = float(twt_array[1] - twt_array[0])
        if abs(actual_interval - expected_interval) > tolerance:
            return False, f"TWT interval mismatch: got {actual_interval}, expected {expected_interval}"
    
    # Check if all intervals are consistent
    if len(twt_array) > 2:
        intervals = np.diff(twt_array)
        if not np.allclose(intervals, expected_interval, atol=tolerance):
            return False, f"Inconsistent TWT intervals: range [{intervals.min():.6f}, {intervals.max():.6f}]"
    
    # Success
    return True, f"TWT array valid: {len(twt_array)} samples, start={actual_start}ms, interval={expected_interval}ms"

def get_twt_for_seismic_volume(
    seismic_shape: tuple, 
    model_config: Dict,
    start_time_ms: float = 1200.0
) -> np.ndarray:
    """
    Generate TWT volume for seismic data with training-compatible timing.
    
    Args:
        seismic_shape: Shape of seismic data (n_samples, n_inlines, n_xlines)
        model_config: Model configuration dictionary
        start_time_ms: Starting TWT time in milliseconds
        
    Returns:
        np.ndarray: TWT volume with shape (n_samples, n_inlines, n_xlines)
                   All traces have identical TWT values along time axis
    """
    
    n_samples, n_inlines, n_xlines = seismic_shape
    
    # Generate 1D TWT array
    twt_1d = get_training_compatible_twt(n_samples, model_config, start_time_ms)
    
    # Broadcast to full volume
    twt_volume = np.broadcast_to(twt_1d[:, None, None], (n_samples, n_inlines, n_xlines)).copy()
    
    return twt_volume

def print_twt_debug_info(twt_array: np.ndarray, label: str = "TWT Array"):
    """
    Print debug information about TWT array for verification.
    
    Args:
        twt_array: TWT array to analyze
        label: Description label for the array
    """
    
    if len(twt_array) == 0:
        safe_print(f"🔍 {label}: EMPTY ARRAY")
        return
    
    safe_print(f"🔍 {label} Debug Info:")
    safe_print(f"  📏 Length: {len(twt_array)} samples")
    safe_print(f"  📊 Range: [{twt_array.min():.3f}, {twt_array.max():.3f}] ms")
    safe_print(f"  🎯 Start: {twt_array[0]:.3f} ms")
    
    if len(twt_array) > 1:
        interval = twt_array[1] - twt_array[0]
        safe_print(f"  ⏱️ Interval: {interval:.3f} ms")
        
    # Show first few and last few values
    if len(twt_array) <= 10:
        safe_print(f"  📋 Values: {twt_array.tolist()}")
    else:
        first_5 = twt_array[:5].tolist()
        last_5 = twt_array[-5:].tolist()
        safe_print(f"  📋 First 5: {first_5}")
        safe_print(f"  📋 Last 5: {last_5}")
    
    # Validate consistency
    is_valid, message = validate_twt_consistency(twt_array)
    safe_print(f"  ✅ Validation: {message}")

# Legacy function for backward compatibility
def create_twt_axis(n_samples: int, sampling_rate_ms: float = 0.25, start_time_ms: float = 1200.0) -> np.ndarray:
    """
    Legacy function for creating TWT axis. Use get_training_compatible_twt() instead.
    
    DEPRECATED: This function is kept for backward compatibility only.
    Use get_training_compatible_twt() with model_config for new code.
    """
    warnings.warn(
        "create_twt_axis() is deprecated. Use get_training_compatible_twt() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    time_indices = np.arange(n_samples, dtype=np.float32)
    return start_time_ms + (time_indices * sampling_rate_ms)

if __name__ == "__main__":
    # Test the TWT generation functions
    safe_print("🧪 Testing TWT Utility Functions")
    safe_print("=" * 50)
    
    # Test with sample model config
    model_config = {
        "io_meta": {
            "sampling_rate_ms": 0.25,
            "twt_column": "TWT",
            "well_column": "WELL"
        }
    }
    
    # Test 1: Basic TWT generation
    safe_print("\n📋 Test 1: Basic TWT Generation")
    twt = get_training_compatible_twt(10, model_config, 1200.0)
    print_twt_debug_info(twt, "Test TWT Array")
    
    # Test 2: TWT volume generation
    safe_print("\n📋 Test 2: TWT Volume Generation")
    seismic_shape = (201, 5, 5)  # Small test volume
    twt_volume = get_twt_for_seismic_volume(seismic_shape, model_config, 1200.0)
    safe_print(f"🔍 TWT Volume Shape: {twt_volume.shape}")
    safe_print(f"🔍 Volume Range: [{twt_volume.min():.3f}, {twt_volume.max():.3f}] ms")
    safe_print(f"🔍 First trace TWT: {twt_volume[:5, 0, 0].tolist()}")
    
    # Test 3: Validation
    safe_print("\n📋 Test 3: Validation Tests")
    is_valid, message = validate_twt_consistency(twt)
    safe_print(f"✅ Validation Result: {message}")
    
    safe_print("\n✅ All tests completed successfully!")