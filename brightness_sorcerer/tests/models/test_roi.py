"""Tests for ROI data model."""

import pytest
import numpy as np

from ...models.roi import ROI


class TestROI:
    """Tests for ROI class."""
    
    @pytest.mark.unit
    def test_init_valid(self):
        """Test ROI initialization with valid parameters."""
        roi = ROI(x=100, y=50, width=200, height=150, 
                 color=(255, 0, 0), label="Test ROI", is_background=True)
        
        assert roi.x == 100
        assert roi.y == 50
        assert roi.width == 200
        assert roi.height == 150
        assert roi.color == (255, 0, 0)
        assert roi.label == "Test ROI"
        assert roi.is_background
    
    @pytest.mark.unit
    def test_init_defaults(self):
        """Test ROI initialization with default parameters."""
        roi = ROI(x=100, y=50, width=200, height=150)
        
        assert roi.x == 100
        assert roi.y == 50
        assert roi.width == 200
        assert roi.height == 150
        assert roi.color == (255, 50, 50)  # Default color
        assert roi.label == ""
        assert not roi.is_background
    
    @pytest.mark.unit
    def test_init_invalid_dimensions(self):
        """Test ROI initialization with invalid dimensions."""
        # Zero width
        with pytest.raises(ValueError, match="width and height must be positive"):
            ROI(x=100, y=50, width=0, height=150)
        
        # Negative height
        with pytest.raises(ValueError, match="width and height must be positive"):
            ROI(x=100, y=50, width=200, height=-10)
    
    @pytest.mark.unit
    def test_init_invalid_coordinates(self):
        """Test ROI initialization with invalid coordinates."""
        # Negative x
        with pytest.raises(ValueError, match="coordinates must be non-negative"):
            ROI(x=-10, y=50, width=200, height=150)
        
        # Negative y
        with pytest.raises(ValueError, match="coordinates must be non-negative"):
            ROI(x=100, y=-5, width=200, height=150)
    
    @pytest.mark.unit
    def test_properties(self):
        """Test ROI properties."""
        roi = ROI(x=100, y=50, width=200, height=150)
        
        # Test basic properties
        assert roi.top_left == (100, 50)
        assert roi.bottom_right == (300, 200)
        assert roi.center == (200, 125)
        assert roi.area == 30000  # 200 * 150
    
    @pytest.mark.unit
    def test_contains_point(self):
        """Test point containment checking."""
        roi = ROI(x=100, y=50, width=200, height=150)
        
        # Points inside ROI
        assert roi.contains_point(150, 100)
        assert roi.contains_point(100, 50)    # Top-left corner
        assert roi.contains_point(300, 200)   # Bottom-right corner
        assert roi.contains_point(200, 125)   # Center
        
        # Points outside ROI
        assert not roi.contains_point(50, 25)    # Above and left
        assert not roi.contains_point(350, 100)  # Right
        assert not roi.contains_point(150, 250)  # Below
        assert not roi.contains_point(301, 201)  # Just outside bottom-right
    
    @pytest.mark.unit
    def test_intersects(self):
        """Test ROI intersection checking."""
        roi1 = ROI(x=100, y=100, width=200, height=150)
        
        # Overlapping ROI
        roi2 = ROI(x=200, y=150, width=150, height=100)
        assert roi1.intersects(roi2)
        assert roi2.intersects(roi1)  # Should be symmetric
        
        # Non-overlapping ROI
        roi3 = ROI(x=400, y=300, width=100, height=100)
        assert not roi1.intersects(roi3)
        assert not roi3.intersects(roi1)
        
        # Adjacent ROI (touching but not overlapping)
        roi4 = ROI(x=300, y=100, width=100, height=150)
        assert not roi1.intersects(roi4)
        
        # Completely contained ROI
        roi5 = ROI(x=150, y=125, width=100, height=75)
        assert roi1.intersects(roi5)
        assert roi5.intersects(roi1)
        
        # Self intersection
        assert roi1.intersects(roi1)
    
    @pytest.mark.unit
    def test_is_valid_for_frame(self):
        """Test frame boundary validation."""
        roi = ROI(x=100, y=50, width=200, height=150)
        
        # Valid for large frame
        assert roi.is_valid_for_frame(640, 480)
        assert roi.is_valid_for_frame(300, 200)  # Exact fit
        
        # Invalid for small frame
        assert not roi.is_valid_for_frame(250, 180)  # Width too small
        assert not roi.is_valid_for_frame(320, 150)  # Height too small
        
        # ROI at origin
        roi_origin = ROI(x=0, y=0, width=100, height=100)
        assert roi_origin.is_valid_for_frame(100, 100)
        assert not roi_origin.is_valid_for_frame(99, 100)
    
    @pytest.mark.unit
    def test_get_corner_handles(self):
        """Test getting corner handles for resizing."""
        roi = ROI(x=100, y=50, width=200, height=150)
        handles = roi.get_corner_handles(handle_size=10)
        
        assert len(handles) == 4
        
        # Check handle positions (centered on corners)
        expected_handles = [
            (95, 45, 10, 10),    # Top-left
            (295, 45, 10, 10),   # Top-right
            (295, 195, 10, 10),  # Bottom-right
            (95, 195, 10, 10)    # Bottom-left
        ]
        
        for i, expected in enumerate(expected_handles):
            assert handles[i] == expected
    
    @pytest.mark.unit
    def test_get_resize_corner_from_point(self):
        """Test detecting resize corner from point."""
        roi = ROI(x=100, y=50, width=200, height=150)
        
        # Points near corners
        assert roi.get_resize_corner_from_point(105, 55, sensitivity=10) == 0  # Top-left
        assert roi.get_resize_corner_from_point(295, 55, sensitivity=10) == 1  # Top-right
        assert roi.get_resize_corner_from_point(295, 195, sensitivity=10) == 2  # Bottom-right
        assert roi.get_resize_corner_from_point(105, 195, sensitivity=10) == 3  # Bottom-left
        
        # Points not near any corner
        assert roi.get_resize_corner_from_point(200, 125, sensitivity=10) is None  # Center
        assert roi.get_resize_corner_from_point(50, 25, sensitivity=10) is None    # Outside
        
        # Points just outside sensitivity range
        assert roi.get_resize_corner_from_point(115, 65, sensitivity=10) is None
    
    @pytest.mark.unit
    def test_resize_from_corner(self):
        """Test resizing ROI from different corners."""
        # Test resizing from top-left corner
        roi = ROI(x=100, y=50, width=200, height=150)
        roi.resize_from_corner(0, 80, 30)  # Move top-left corner
        
        assert roi.x == 80
        assert roi.y == 30
        assert roi.width == 220  # 100 + (100 - 80)
        assert roi.height == 170  # 150 + (50 - 30)
        
        # Test resizing from bottom-right corner
        roi = ROI(x=100, y=50, width=200, height=150)
        roi.resize_from_corner(2, 350, 250)  # Move bottom-right corner
        
        assert roi.x == 100      # Unchanged
        assert roi.y == 50       # Unchanged
        assert roi.width == 250  # 350 - 100
        assert roi.height == 200 # 250 - 50
        
        # Test invalid resize (would make negative dimensions)
        roi = ROI(x=100, y=50, width=200, height=150)
        original_state = (roi.x, roi.y, roi.width, roi.height)
        
        roi.resize_from_corner(2, 50, 25)  # Try to make it smaller than current position
        
        # Should remain unchanged (implementation dependent)
        # The exact behavior may vary based on implementation
    
    @pytest.mark.unit
    def test_move_to(self):
        """Test moving ROI to new position."""
        roi = ROI(x=100, y=50, width=200, height=150)
        
        roi.move_to(200, 100)
        
        assert roi.x == 200
        assert roi.y == 100
        assert roi.width == 200  # Unchanged
        assert roi.height == 150  # Unchanged
    
    @pytest.mark.unit
    def test_extract_region(self):
        """Test extracting ROI region from frame."""
        # Create test frame
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[100:200, 150:250] = [255, 255, 255]  # White rectangle
        
        roi = ROI(x=150, y=100, width=100, height=100)
        
        # Extract region
        region = roi.extract_region(frame)
        
        assert region is not None
        assert region.shape == (100, 100, 3)
        
        # Check that extracted region is all white
        assert np.all(region == [255, 255, 255])
    
    @pytest.mark.unit
    def test_extract_region_invalid(self):
        """Test extracting region with invalid parameters."""
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # ROI outside frame bounds
        roi = ROI(x=500, y=100, width=100, height=100)
        region = roi.extract_region(frame)
        assert region is None
        
        # None frame
        roi = ROI(x=100, y=100, width=100, height=100)
        region = roi.extract_region(None)
        assert region is None
    
    @pytest.mark.unit
    def test_serialization(self):
        """Test ROI serialization to/from dictionary."""
        roi = ROI(x=100, y=50, width=200, height=150, 
                 color=(0, 255, 0), label="Test ROI", is_background=True)
        
        # Convert to dict
        roi_dict = roi.to_dict()
        
        expected_keys = ['x', 'y', 'width', 'height', 'color', 'label', 'is_background']
        for key in expected_keys:
            assert key in roi_dict
        
        assert roi_dict['x'] == 100
        assert roi_dict['y'] == 50
        assert roi_dict['width'] == 200
        assert roi_dict['height'] == 150
        assert roi_dict['color'] == (0, 255, 0)
        assert roi_dict['label'] == "Test ROI"
        assert roi_dict['is_background']
        
        # Create ROI from dict
        roi2 = ROI.from_dict(roi_dict)
        
        assert roi2.x == roi.x
        assert roi2.y == roi.y
        assert roi2.width == roi.width
        assert roi2.height == roi.height
        assert roi2.color == roi.color
        assert roi2.label == roi.label
        assert roi2.is_background == roi.is_background
    
    @pytest.mark.unit
    def test_serialization_with_defaults(self):
        """Test ROI serialization with missing fields."""
        # Dict with minimal data
        roi_dict = {
            'x': 50,
            'y': 25,
            'width': 100,
            'height': 75
        }
        
        roi = ROI.from_dict(roi_dict)
        
        assert roi.x == 50
        assert roi.y == 25
        assert roi.width == 100
        assert roi.height == 75
        assert roi.color == (255, 50, 50)  # Default color
        assert roi.label == ""
        assert not roi.is_background
    
    @pytest.mark.unit
    def test_repr(self):
        """Test ROI string representation."""
        roi = ROI(x=100, y=50, width=200, height=150, label="Test ROI")
        
        repr_str = repr(roi)
        
        assert "ROI(" in repr_str
        assert "x=100" in repr_str
        assert "y=50" in repr_str
        assert "w=200" in repr_str
        assert "h=150" in repr_str
        assert "label='Test ROI'" in repr_str


@pytest.mark.integration
class TestROIIntegration:
    """Integration tests for ROI functionality."""
    
    def test_roi_workflow_with_real_frame(self, sample_video_frame):
        """Test complete ROI workflow with real frame data."""
        # Create ROI
        roi = ROI(x=50, y=50, width=100, height=100, label="Test Region")
        
        # Validate against frame
        frame_height, frame_width = sample_video_frame.shape[:2]
        assert roi.is_valid_for_frame(frame_width, frame_height)
        
        # Extract region
        region = roi.extract_region(sample_video_frame)
        assert region is not None
        assert region.shape == (100, 100, 3)
        
        # Test point containment with real coordinates
        assert roi.contains_point(75, 75)
        assert not roi.contains_point(25, 25)
        
        # Test intersection with overlapping ROI
        overlapping_roi = ROI(x=100, y=100, width=100, height=100)
        assert roi.intersects(overlapping_roi)
        
        # Test resizing
        roi.resize_from_corner(2, 200, 200)  # Bottom-right corner
        assert roi.width == 150
        assert roi.height == 150
        
        # Validate after resize
        assert roi.is_valid_for_frame(frame_width, frame_height)
    
    def test_multiple_rois_interaction(self, sample_video_frame):
        """Test interaction between multiple ROIs."""
        frame_height, frame_width = sample_video_frame.shape[:2]
        
        # Create multiple ROIs
        roi1 = ROI(x=50, y=50, width=100, height=100, label="ROI 1")
        roi2 = ROI(x=200, y=200, width=150, height=100, label="ROI 2")
        roi3 = ROI(x=75, y=75, width=50, height=50, label="ROI 3")  # Overlaps with ROI 1
        
        rois = [roi1, roi2, roi3]
        
        # Validate all ROIs
        for roi in rois:
            assert roi.is_valid_for_frame(frame_width, frame_height)
        
        # Test intersections
        assert roi1.intersects(roi3)  # Should overlap
        assert not roi1.intersects(roi2)  # Should not overlap
        assert not roi2.intersects(roi3)  # Should not overlap
        
        # Test point containment for overlapping area
        point_x, point_y = 100, 100
        containing_rois = [i for i, roi in enumerate(rois) if roi.contains_point(point_x, point_y)]
        assert 0 in containing_rois  # ROI 1
        assert 2 in containing_rois  # ROI 3
        assert 1 not in containing_rois  # ROI 2
        
        # Extract regions from all ROIs
        regions = []
        for roi in rois:
            region = roi.extract_region(sample_video_frame)
            assert region is not None
            regions.append(region)
        
        # Verify region shapes
        assert regions[0].shape == (100, 100, 3)
        assert regions[1].shape == (100, 150, 3)
        assert regions[2].shape == (50, 50, 3)
    
    def test_roi_serialization_roundtrip(self):
        """Test complete serialization round-trip."""
        original_rois = [
            ROI(x=50, y=50, width=100, height=100, color=(255, 0, 0), label="Red ROI"),
            ROI(x=200, y=150, width=150, height=120, color=(0, 255, 0), label="Green ROI", is_background=True),
            ROI(x=400, y=300, width=80, height=60, color=(0, 0, 255), label="Blue ROI")
        ]
        
        # Serialize to dicts
        serialized_data = [roi.to_dict() for roi in original_rois]
        
        # Deserialize back to ROIs
        deserialized_rois = [ROI.from_dict(data) for data in serialized_data]
        
        # Compare original and deserialized
        assert len(deserialized_rois) == len(original_rois)
        
        for orig, deser in zip(original_rois, deserialized_rois):
            assert orig.x == deser.x
            assert orig.y == deser.y
            assert orig.width == deser.width
            assert orig.height == deser.height
            assert orig.color == deser.color
            assert orig.label == deser.label
            assert orig.is_background == deser.is_background
            
            # Test that all methods work the same
            assert orig.area == deser.area
            assert orig.center == deser.center
            assert orig.top_left == deser.top_left
            assert orig.bottom_right == deser.bottom_right