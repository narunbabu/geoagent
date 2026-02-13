#!/usr/bin/env python3
"""
Trace Spatial Indexer - Efficient spatial indexing for seismic trace coordinates

This module provides efficient spatial indexing and nearest neighbor search
for seismic trace coordinates, optimized for deviated well sampling.

Author: Claude Code
Date: 2025-07-23
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

@dataclass
class SpatialLookupResult:
    """Result of spatial lookup operation"""
    trace_index: int
    distance: float
    x_coord: float
    y_coord: float
    metadata: Optional[Dict] = None

# Try to import scipy for KD-tree, fallback to manual implementation
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, using fallback distance calculations for spatial indexing")

class TraceSpatialIndexer:
    """
    Efficient spatial indexing for seismic trace coordinates
    
    Provides fast nearest neighbor search for well coordinate to trace mapping
    with fallback implementations when scipy is not available.
    """
    
    def __init__(self, trace_coordinates: List[Tuple[float, float]], trace_metadata: Optional[List[Dict]] = None):
        """
        Initialize spatial indexer with trace coordinates
        
        Args:
            trace_coordinates: List of (x, y) coordinate tuples
            trace_metadata: Optional metadata for each trace (inline, crossline, etc.)
        """
        self.coordinates = np.array(trace_coordinates)
        self.trace_metadata = trace_metadata or []
        self.num_traces = len(trace_coordinates)
        
        # Build spatial index
        if SCIPY_AVAILABLE:
            self._build_kdtree_index()
        else:
            self._build_fallback_index()
        
        # Create coordinate to trace index mapping
        self.coordinate_to_trace = {}
        for i, (x, y) in enumerate(trace_coordinates):
            # Use rounded coordinates as keys to handle floating point precision
            key = (round(x, 1), round(y, 1))
            self.coordinate_to_trace[key] = i
    
    def _build_kdtree_index(self):
        """Build KD-tree spatial index using scipy"""
        try:
            self.kdtree = cKDTree(self.coordinates)
            self.index_type = "kdtree"
            logging.info(f"Built KD-tree spatial index for {self.num_traces} traces")
        except Exception as e:
            logging.warning(f"Failed to build KD-tree index: {e}, falling back to manual search")
            self._build_fallback_index()
    
    def _build_fallback_index(self):
        """Build fallback index for manual distance calculations"""
        self.kdtree = None
        self.index_type = "fallback"
        logging.info(f"Built fallback spatial index for {self.num_traces} traces")
    
    def find_nearest_trace(self, x: float, y: float, max_distance: float = 1000.0) -> Optional[Tuple[int, float]]:
        """
        Find nearest trace to given coordinates
        
        Args:
            x, y: Target coordinates
            max_distance: Maximum search distance in meters
            
        Returns:
            Tuple of (trace_index, distance) or None if no trace within max_distance
        """
        if self.num_traces == 0:
            return None
        
        target_point = np.array([x, y])
        
        if self.index_type == "kdtree":
            return self._find_nearest_kdtree(target_point, max_distance)
        else:
            return self._find_nearest_fallback(target_point, max_distance)
    
    def _find_nearest_kdtree(self, target_point: np.ndarray, max_distance: float) -> Optional[Tuple[int, float]]:
        """Find nearest trace using KD-tree"""
        try:
            distance, index = self.kdtree.query(target_point, distance_upper_bound=max_distance)
            
            if np.isinf(distance):
                return None
            
            return int(index), float(distance)
        
        except Exception as e:
            logging.warning(f"KD-tree query failed: {e}, falling back to manual search")
            return self._find_nearest_fallback(target_point, max_distance)
    
    def _find_nearest_fallback(self, target_point: np.ndarray, max_distance: float) -> Optional[Tuple[int, float]]:
        """Find nearest trace using manual distance calculation"""
        if self.num_traces == 0:
            return None
        
        # Calculate distances to all traces
        distances = np.sqrt(np.sum((self.coordinates - target_point)**2, axis=1))
        
        # Find minimum distance
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        
        if min_distance <= max_distance:
            return int(min_index), float(min_distance)
        else:
            return None
    
    def find_nearest_traces_batch(self, coordinates_list: List[Tuple[float, float]], 
                                 max_distance: float = 1000.0) -> List[Optional[Tuple[int, float]]]:
        """
        Find nearest traces for multiple coordinates in batch
        
        Args:
            coordinates_list: List of (x, y) coordinate tuples
            max_distance: Maximum search distance in meters
            
        Returns:
            List of (trace_index, distance) tuples or None for each input coordinate
        """
        results = []
        
        for x, y in coordinates_list:
            result = self.find_nearest_trace(x, y, max_distance)
            results.append(result)
        
        return results
    
    def get_traces_within_radius(self, x: float, y: float, radius: float) -> List[Tuple[int, float]]:
        """
        Get all traces within specified radius of target coordinates
        
        Args:
            x, y: Target coordinates
            radius: Search radius in meters
            
        Returns:
            List of (trace_index, distance) tuples sorted by distance
        """
        target_point = np.array([x, y])
        
        if self.index_type == "kdtree":
            return self._get_traces_within_radius_kdtree(target_point, radius)
        else:
            return self._get_traces_within_radius_fallback(target_point, radius)
    
    def _get_traces_within_radius_kdtree(self, target_point: np.ndarray, radius: float) -> List[Tuple[int, float]]:
        """Get traces within radius using KD-tree"""
        try:
            indices = self.kdtree.query_ball_point(target_point, radius)
            
            if not indices:
                return []
            
            # Calculate distances and sort
            results = []
            for idx in indices:
                distance = np.linalg.norm(self.coordinates[idx] - target_point)
                results.append((idx, distance))
            
            results.sort(key=lambda x: x[1])  # Sort by distance
            return results
        
        except Exception as e:
            logging.warning(f"KD-tree radius query failed: {e}, falling back to manual search")
            return self._get_traces_within_radius_fallback(target_point, radius)
    
    def _get_traces_within_radius_fallback(self, target_point: np.ndarray, radius: float) -> List[Tuple[int, float]]:
        """Get traces within radius using manual calculation"""
        distances = np.sqrt(np.sum((self.coordinates - target_point)**2, axis=1))
        
        # Find indices within radius
        within_radius = np.where(distances <= radius)[0]
        
        # Create results with distances
        results = [(int(idx), float(distances[idx])) for idx in within_radius]
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        
        return results
    
    def get_trace_coordinate(self, trace_index: int) -> Optional[Tuple[float, float]]:
        """
        Get coordinate for specified trace index
        
        Args:
            trace_index: Index of trace
            
        Returns:
            (x, y) coordinate tuple or None if index invalid
        """
        if 0 <= trace_index < self.num_traces:
            coord = self.coordinates[trace_index]
            return float(coord[0]), float(coord[1])
        return None
    
    def get_trace_metadata(self, trace_index: int) -> Optional[Dict]:
        """
        Get metadata for specified trace index
        
        Args:
            trace_index: Index of trace
            
        Returns:
            Metadata dictionary or None if not available
        """
        if self.trace_metadata and 0 <= trace_index < len(self.trace_metadata):
            return self.trace_metadata[trace_index]
        return None
    
    def get_coverage_info(self) -> Dict[str, Any]:
        """
        Get information about spatial coverage of traces
        
        Returns:
            Dictionary with coverage statistics
        """
        if self.num_traces == 0:
            return {"num_traces": 0}
        
        x_coords = self.coordinates[:, 0]
        y_coords = self.coordinates[:, 1]
        
        return {
            "num_traces": self.num_traces,
            "x_range": [float(x_coords.min()), float(x_coords.max())],
            "y_range": [float(y_coords.min()), float(y_coords.max())],
            "x_extent": float(x_coords.max() - x_coords.min()),
            "y_extent": float(y_coords.max() - y_coords.min()),
            "index_type": self.index_type
        }
    
    def validate_coordinates(self, coordinates_list: List[Tuple[float, float]], 
                           max_distance: float = 100.0) -> Dict[str, Any]:
        """
        Validate that coordinates can be mapped to traces within specified distance
        
        Args:
            coordinates_list: List of coordinates to validate
            max_distance: Maximum acceptable distance to nearest trace
            
        Returns:
            Validation results dictionary
        """
        results = {
            "total_coordinates": len(coordinates_list),
            "mapped_coordinates": 0,
            "unmapped_coordinates": 0,
            "max_distance_found": 0.0,
            "avg_distance": 0.0,
            "unmapped_list": []
        }
        
        if not coordinates_list:
            return results
        
        distances = []
        
        for i, (x, y) in enumerate(coordinates_list):
            nearest = self.find_nearest_trace(x, y, max_distance)
            
            if nearest is not None:
                trace_idx, distance = nearest
                results["mapped_coordinates"] += 1
                distances.append(distance)
                results["max_distance_found"] = max(results["max_distance_found"], distance)
            else:
                results["unmapped_coordinates"] += 1
                results["unmapped_list"].append({"index": i, "x": x, "y": y})
        
        if distances:
            results["avg_distance"] = sum(distances) / len(distances)
        
        return results


def create_spatial_indexer_from_headerdata(headerdata: List, coordinate_extractor_func=None) -> TraceSpatialIndexer:
    """
    Create spatial indexer from seismic header data
    
    Args:
        headerdata: List of header data entries
        coordinate_extractor_func: Optional function to extract coordinates from header entries
                                 Default assumes headerdata contains coordinate information
    
    Returns:
        TraceSpatialIndexer instance
    """
    if coordinate_extractor_func is None:
        # Default extraction - assumes headerdata has coordinate information
        # This would need to be adapted based on actual headerdata structure
        coordinate_extractor_func = lambda entry: (entry.get('x', 0), entry.get('y', 0))
    
    trace_coordinates = []
    trace_metadata = []
    
    for i, entry in enumerate(headerdata):
        try:
            x, y = coordinate_extractor_func(entry)
            trace_coordinates.append((float(x), float(y)))
            
            # Store metadata if available
            metadata = {
                "trace_index": i,
                "inline": entry.get("inline"),
                "crossline": entry.get("crossline"),
                "original_entry": entry
            }
            trace_metadata.append(metadata)
            
        except (ValueError, TypeError, KeyError) as e:
            logging.warning(f"Failed to extract coordinates from header entry {i}: {e}")
            continue
    
    return TraceSpatialIndexer(trace_coordinates, trace_metadata)


def test_spatial_indexer():
    """Test function for spatial indexer"""
    print("Testing TraceSpatialIndexer...")
    
    # Create test coordinates (grid pattern)
    test_coords = []
    for x in range(270000, 270500, 50):  # 500m x 500m grid
        for y in range(2541000, 2541500, 50):
            test_coords.append((x, y))
    
    print(f"Created {len(test_coords)} test coordinates")
    
    # Create indexer
    indexer = TraceSpatialIndexer(test_coords)
    
    # Test coverage info
    coverage = indexer.get_coverage_info()
    print(f"Coverage info: {coverage}")
    
    # Test nearest neighbor search
    test_x, test_y = 270125, 2541125  # Point between grid points
    nearest = indexer.find_nearest_trace(test_x, test_y)
    
    if nearest:
        trace_idx, distance = nearest
        coord = indexer.get_trace_coordinate(trace_idx)
        print(f"Nearest trace to ({test_x}, {test_y}): index {trace_idx} at {coord}, distance {distance:.1f}m")
    
    # Test radius search
    within_radius = indexer.get_traces_within_radius(test_x, test_y, 100)
    print(f"Found {len(within_radius)} traces within 100m")
    
    # Test batch search
    batch_coords = [(270075, 2541075), (270175, 2541175), (270275, 2541275)]
    batch_results = indexer.find_nearest_traces_batch(batch_coords)
    print(f"Batch search results: {len([r for r in batch_results if r is not None])} successful mappings")
    
    print("TraceSpatialIndexer test completed successfully!")


if __name__ == "__main__":
    test_spatial_indexer()