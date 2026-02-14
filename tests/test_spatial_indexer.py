"""Tests for TraceSpatialIndexer."""
import numpy as np
import pytest
from geoagent.utils.trace_spatial_indexer import TraceSpatialIndexer


class TestTraceSpatialIndexer:

    @pytest.fixture
    def indexer(self):
        """Create indexer with a 10x10 grid of trace coordinates."""
        coords = []
        for x in range(270000, 270500, 50):
            for y in range(2541000, 2541500, 50):
                coords.append((x, y))
        return TraceSpatialIndexer(coords)

    def test_creation(self, indexer):
        assert indexer.num_traces == 100  # 10 x 10 grid
        info = indexer.get_coverage_info()
        assert info['num_traces'] == 100

    def test_nearest_trace(self, indexer):
        result = indexer.find_nearest_trace(270025, 2541025)
        assert result is not None
        idx, dist = result
        assert dist < 50  # Should be within one grid cell

    def test_nearest_trace_out_of_range(self, indexer):
        result = indexer.find_nearest_trace(999999, 999999, max_distance=100)
        assert result is None

    def test_traces_within_radius(self, indexer):
        results = indexer.get_traces_within_radius(270100, 2541100, 100)
        assert len(results) > 0
        # Should be sorted by distance
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_batch_search(self, indexer):
        coords = [(270050, 2541050), (270150, 2541150), (270250, 2541250)]
        results = indexer.find_nearest_traces_batch(coords)
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_get_coordinate(self, indexer):
        coord = indexer.get_trace_coordinate(0)
        assert coord is not None
        assert len(coord) == 2

    def test_invalid_index(self, indexer):
        assert indexer.get_trace_coordinate(-1) is None
        assert indexer.get_trace_coordinate(99999) is None
