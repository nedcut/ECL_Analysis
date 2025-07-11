"""Tests for ROIManager class."""

import pytest
import numpy as np
from unittest.mock import Mock

from ...core.roi_manager import ROIManager
from ...models.roi import ROI


class TestROIManager:
    """Tests for ROIManager class."""
    
    @pytest.mark.unit
    def test_init(self):
        """Test ROIManager initialization."""
        manager = ROIManager()
        
        assert manager.rois == []
        assert manager.selected_roi_index is None
        assert manager.background_roi_index is None
        assert not manager.drawing_mode
        assert not manager.moving_mode
        assert not manager.resizing_mode
        assert manager.resize_corner is None
        assert manager.draw_start_point is None
        assert manager.draw_current_point is None
        assert manager.move_offset is None
        assert manager.on_roi_changed is None
        assert manager.on_selection_changed is None
    
    @pytest.mark.unit
    def test_add_roi(self, roi_manager):
        """Test adding ROIs."""
        # Add first ROI
        roi_index = roi_manager.add_roi(100, 100, 200, 150, "Test ROI 1")
        
        assert roi_index == 0
        assert len(roi_manager.rois) == 1
        assert roi_manager.selected_roi_index == 0
        
        roi = roi_manager.rois[0]
        assert roi.x == 100
        assert roi.y == 100
        assert roi.width == 200
        assert roi.height == 150
        assert roi.label == "Test ROI 1"
        assert not roi.is_background
        
        # Add second ROI
        roi_index = roi_manager.add_roi(300, 200, 100, 100, "Test ROI 2")
        
        assert roi_index == 1
        assert len(roi_manager.rois) == 2
        assert roi_manager.selected_roi_index == 1  # Should select newly added ROI
    
    @pytest.mark.unit
    def test_add_roi_with_background(self, roi_manager):
        """Test adding ROI as background."""
        roi_index = roi_manager.add_roi(100, 100, 200, 150, "Background", is_background=True)
        
        assert roi_index == 0
        assert roi_manager.background_roi_index == 0
        assert roi_manager.rois[0].is_background
    
    @pytest.mark.unit
    def test_add_roi_with_invalid_dimensions(self, roi_manager):
        """Test adding ROI with invalid dimensions."""
        # Should handle gracefully and return -1
        roi_index = roi_manager.add_roi(100, 100, 0, 150, "Invalid ROI")
        assert roi_index == -1
        assert len(roi_manager.rois) == 0
    
    @pytest.mark.unit
    def test_delete_roi(self, roi_manager, sample_rois):
        """Test deleting ROIs."""
        # Add ROIs
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 1
        roi_manager.background_roi_index = 2
        
        # Delete middle ROI
        assert roi_manager.delete_roi(1)
        
        assert len(roi_manager.rois) == 2
        assert roi_manager.selected_roi_index is None  # Selection cleared
        assert roi_manager.background_roi_index == 1   # Background index adjusted
    
    @pytest.mark.unit
    def test_delete_selected_roi(self, roi_manager, sample_rois):
        """Test deleting the currently selected ROI."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 0
        
        assert roi_manager.delete_roi(0)
        assert roi_manager.selected_roi_index is None
        assert len(roi_manager.rois) == 2
    
    @pytest.mark.unit
    def test_delete_background_roi(self, roi_manager, sample_rois):
        """Test deleting the background ROI."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.background_roi_index = 2
        
        assert roi_manager.delete_roi(2)
        assert roi_manager.background_roi_index is None
        assert len(roi_manager.rois) == 2
    
    @pytest.mark.unit
    def test_delete_invalid_roi(self, roi_manager, sample_rois):
        """Test deleting ROI with invalid index."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # Invalid indices
        assert not roi_manager.delete_roi(-1)
        assert not roi_manager.delete_roi(10)
        assert len(roi_manager.rois) == 3  # No ROIs deleted
    
    @pytest.mark.unit
    def test_select_roi(self, roi_manager, sample_rois):
        """Test selecting ROIs."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # Select valid ROI
        assert roi_manager.select_roi(1)
        assert roi_manager.selected_roi_index == 1
        
        # Select None
        assert roi_manager.select_roi(None)
        assert roi_manager.selected_roi_index is None
        
        # Select invalid index
        assert not roi_manager.select_roi(10)
        assert roi_manager.selected_roi_index is None
    
    @pytest.mark.unit
    def test_get_roi_at_point(self, roi_manager, sample_rois):
        """Test finding ROI at specific point."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # Point inside first ROI (50, 50, 100, 100)
        roi_index = roi_manager.get_roi_at_point(75, 75)
        assert roi_index == 0
        
        # Point inside second ROI (200, 200, 150, 100)
        roi_index = roi_manager.get_roi_at_point(250, 220)
        assert roi_index == 1
        
        # Point outside all ROIs
        roi_index = roi_manager.get_roi_at_point(10, 10)
        assert roi_index is None
        
        # Point in overlapping region (should return last ROI - topmost)
        # Add overlapping ROI
        roi_manager.add_roi(40, 40, 50, 50, "Overlapping")
        roi_index = roi_manager.get_roi_at_point(60, 60)
        assert roi_index == 3  # Should return the newest (topmost) ROI
    
    @pytest.mark.unit
    def test_move_roi(self, roi_manager, sample_rois):
        """Test moving ROIs."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # Move ROI within bounds
        assert roi_manager.move_roi(0, 150, 150, 640, 480)
        
        roi = roi_manager.rois[0]
        assert roi.x == 150
        assert roi.y == 150
        
        # Move ROI to boundary (should clamp)
        assert roi_manager.move_roi(0, 600, 400, 640, 480)
        roi = roi_manager.rois[0]
        assert roi.x == 540  # 640 - 100 (width)
        assert roi.y == 380  # 480 - 100 (height)
        
        # Move ROI beyond boundary (should clamp to 0)
        assert roi_manager.move_roi(0, -50, -50, 640, 480)
        roi = roi_manager.rois[0]
        assert roi.x == 0
        assert roi.y == 0
    
    @pytest.mark.unit
    def test_move_invalid_roi(self, roi_manager):
        """Test moving ROI with invalid index."""
        assert not roi_manager.move_roi(0, 100, 100, 640, 480)
        assert not roi_manager.move_roi(-1, 100, 100, 640, 480)
    
    @pytest.mark.unit
    def test_resize_roi(self, roi_manager, sample_rois):
        """Test resizing ROIs."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        original_roi = roi_manager.rois[0]
        original_x, original_y = original_roi.x, original_roi.y
        original_width, original_height = original_roi.width, original_roi.height
        
        # Resize from bottom-right corner
        assert roi_manager.resize_roi(0, 2, original_x + 150, original_y + 120, 640, 480)
        
        roi = roi_manager.rois[0]
        assert roi.width == 150
        assert roi.height == 120
    
    @pytest.mark.unit
    def test_resize_roi_invalid_size(self, roi_manager, sample_rois):
        """Test resizing ROI to invalid size."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        original_roi = roi_manager.rois[0]
        original_width = original_roi.width
        
        # Try to resize to very small size (should fail and restore)
        assert not roi_manager.resize_roi(0, 2, original_roi.x + 5, original_roi.y + 5, 640, 480)
        
        # ROI should be unchanged
        roi = roi_manager.rois[0]
        assert roi.width == original_width
    
    @pytest.mark.unit
    def test_set_background_roi(self, roi_manager, sample_rois):
        """Test setting background ROI."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # Set ROI 1 as background
        assert roi_manager.set_background_roi(1)
        assert roi_manager.background_roi_index == 1
        assert roi_manager.rois[1].is_background
        
        # Previously set background ROI should be unset
        assert not roi_manager.rois[2].is_background
        
        # Unset background ROI
        assert roi_manager.set_background_roi(None)
        assert roi_manager.background_roi_index is None
        assert not roi_manager.rois[1].is_background
    
    @pytest.mark.unit
    def test_get_background_roi(self, roi_manager, sample_rois):
        """Test getting background ROI."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # No background ROI set
        assert roi_manager.get_background_roi() is None
        
        # Set background ROI
        roi_manager.set_background_roi(1)
        bg_roi = roi_manager.get_background_roi()
        assert bg_roi is roi_manager.rois[1]
    
    @pytest.mark.unit
    def test_render_rois(self, roi_manager, sample_rois, sample_video_frame):
        """Test rendering ROIs on frame."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 0
        
        # Render ROIs
        rendered_frame = roi_manager.render_rois(sample_video_frame)
        
        assert rendered_frame is not None
        assert rendered_frame.shape == sample_video_frame.shape
        assert not np.array_equal(rendered_frame, sample_video_frame)  # Should be different
    
    @pytest.mark.unit
    def test_render_rois_none_frame(self, roi_manager):
        """Test rendering ROIs on None frame."""
        rendered_frame = roi_manager.render_rois(None)
        assert rendered_frame is None
    
    @pytest.mark.unit
    def test_drawing_interaction(self, roi_manager):
        """Test drawing interaction workflow."""
        # Start drawing
        roi_manager.start_drawing(100, 100)
        assert roi_manager.drawing_mode
        assert roi_manager.draw_start_point == (100, 100)
        assert roi_manager.draw_current_point == (100, 100)
        
        # Update drawing
        roi_manager.update_drawing(200, 150)
        assert roi_manager.draw_current_point == (200, 150)
        
        # Finish drawing
        roi_index = roi_manager.finish_drawing(200, 150, 640, 480)
        assert roi_index >= 0
        assert not roi_manager.drawing_mode
        assert roi_manager.draw_start_point is None
        assert roi_manager.draw_current_point is None
        assert len(roi_manager.rois) == 1
    
    @pytest.mark.unit
    def test_drawing_too_small(self, roi_manager):
        """Test drawing ROI that's too small."""
        roi_manager.start_drawing(100, 100)
        
        # Finish with very small size
        roi_index = roi_manager.finish_drawing(105, 105, 640, 480)
        assert roi_index == -1  # Should fail
        assert len(roi_manager.rois) == 0
    
    @pytest.mark.unit
    def test_cancel_drawing(self, roi_manager):
        """Test canceling drawing operation."""
        roi_manager.start_drawing(100, 100)
        roi_manager.update_drawing(200, 150)
        
        roi_manager.cancel_drawing()
        assert not roi_manager.drawing_mode
        assert roi_manager.draw_start_point is None
        assert roi_manager.draw_current_point is None
    
    @pytest.mark.unit
    def test_moving_interaction(self, roi_manager, sample_rois):
        """Test moving interaction workflow."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 0
        
        # Start moving
        assert roi_manager.start_moving(0, 75, 75)  # Point inside ROI
        assert roi_manager.moving_mode
        assert roi_manager.move_offset == (25, 25)  # 75 - 50 = 25
        
        # Update moving
        assert roi_manager.update_moving(125, 125, 640, 480)
        roi = roi_manager.rois[0]
        assert roi.x == 100  # 125 - 25 = 100
        assert roi.y == 100  # 125 - 25 = 100
        
        # Finish moving
        roi_manager.finish_moving()
        assert not roi_manager.moving_mode
        assert roi_manager.move_offset is None
    
    @pytest.mark.unit
    def test_resizing_interaction(self, roi_manager, sample_rois):
        """Test resizing interaction workflow."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 0
        
        # Start resizing
        assert roi_manager.start_resizing(0, 2)  # Bottom-right corner
        assert roi_manager.resizing_mode
        assert roi_manager.resize_corner == 2
        
        # Update resizing
        assert roi_manager.update_resizing(180, 180, 640, 480)
        
        # Finish resizing
        roi_manager.finish_resizing()
        assert not roi_manager.resizing_mode
        assert roi_manager.resize_corner is None
    
    @pytest.mark.unit
    def test_get_resize_corner_at_point(self, roi_manager, sample_rois):
        """Test getting resize corner at point."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        roi = roi_manager.rois[0]  # 50, 50, 100, 100
        
        # Test corners
        # Top-left corner (50, 50)
        corner = roi_manager.get_resize_corner_at_point(0, 55, 55)
        assert corner == 0
        
        # Top-right corner (150, 50)
        corner = roi_manager.get_resize_corner_at_point(0, 145, 55)
        assert corner == 1
        
        # Bottom-right corner (150, 150)
        corner = roi_manager.get_resize_corner_at_point(0, 145, 145)
        assert corner == 2
        
        # Bottom-left corner (50, 150)
        corner = roi_manager.get_resize_corner_at_point(0, 55, 145)
        assert corner == 3
        
        # Point not near any corner
        corner = roi_manager.get_resize_corner_at_point(0, 100, 100)
        assert corner is None
    
    @pytest.mark.unit
    def test_clear_all(self, roi_manager, sample_rois):
        """Test clearing all ROIs."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 1
        roi_manager.background_roi_index = 2
        
        roi_manager.clear_all()
        
        assert len(roi_manager.rois) == 0
        assert roi_manager.selected_roi_index is None
        assert roi_manager.background_roi_index is None
        assert not roi_manager.drawing_mode
        assert not roi_manager.moving_mode
        assert not roi_manager.resizing_mode
    
    @pytest.mark.unit
    def test_cancel_all_interactions(self, roi_manager):
        """Test canceling all interactions."""
        # Set up various interaction states
        roi_manager.drawing_mode = True
        roi_manager.moving_mode = True
        roi_manager.resizing_mode = True
        roi_manager.resize_corner = 1
        roi_manager.draw_start_point = (100, 100)
        roi_manager.move_offset = (10, 10)
        
        roi_manager.cancel_all_interactions()
        
        assert not roi_manager.drawing_mode
        assert not roi_manager.moving_mode
        assert not roi_manager.resizing_mode
        assert roi_manager.resize_corner is None
        assert roi_manager.draw_start_point is None
        assert roi_manager.move_offset is None
    
    @pytest.mark.unit
    def test_get_roi_count(self, roi_manager, sample_rois):
        """Test getting ROI count."""
        assert roi_manager.get_roi_count() == 0
        
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        assert roi_manager.get_roi_count() == 3
    
    @pytest.mark.unit
    def test_get_roi(self, roi_manager, sample_rois):
        """Test getting ROI by index."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # Valid indices
        roi = roi_manager.get_roi(0)
        assert roi is sample_rois[0]
        
        roi = roi_manager.get_roi(1)
        assert roi is sample_rois[1]
        
        # Invalid indices
        assert roi_manager.get_roi(-1) is None
        assert roi_manager.get_roi(10) is None
    
    @pytest.mark.unit
    def test_get_selected_roi(self, roi_manager, sample_rois):
        """Test getting selected ROI."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        
        # No selection
        assert roi_manager.get_selected_roi() is None
        
        # With selection
        roi_manager.selected_roi_index = 1
        selected_roi = roi_manager.get_selected_roi()
        assert selected_roi is sample_rois[1]
    
    @pytest.mark.unit
    def test_serialization(self, roi_manager, sample_rois):
        """Test ROI manager serialization."""
        for roi in sample_rois:
            roi_manager.rois.append(roi)
        roi_manager.selected_roi_index = 1
        roi_manager.background_roi_index = 2
        
        # Export to dict
        data = roi_manager.to_dict()
        
        assert 'rois' in data
        assert 'selected_roi_index' in data
        assert 'background_roi_index' in data
        assert len(data['rois']) == 3
        assert data['selected_roi_index'] == 1
        assert data['background_roi_index'] == 2
        
        # Create new manager and import
        new_manager = ROIManager()
        new_manager.from_dict(data)
        
        assert len(new_manager.rois) == 3
        assert new_manager.selected_roi_index == 1
        assert new_manager.background_roi_index == 2
        
        # Compare ROIs
        for i, roi in enumerate(new_manager.rois):
            original_roi = sample_rois[i]
            assert roi.x == original_roi.x
            assert roi.y == original_roi.y
            assert roi.width == original_roi.width
            assert roi.height == original_roi.height
            assert roi.color == original_roi.color
            assert roi.label == original_roi.label
            assert roi.is_background == original_roi.is_background
    
    @pytest.mark.unit
    def test_callbacks(self, roi_manager):
        """Test ROI change callbacks."""
        roi_changed_called = False
        selection_changed_called = False
        
        def on_roi_changed():
            nonlocal roi_changed_called
            roi_changed_called = True
        
        def on_selection_changed():
            nonlocal selection_changed_called
            selection_changed_called = True
        
        roi_manager.on_roi_changed = on_roi_changed
        roi_manager.on_selection_changed = on_selection_changed
        
        # Add ROI (should trigger both callbacks)
        roi_manager.add_roi(100, 100, 200, 150)
        assert roi_changed_called
        assert selection_changed_called
        
        # Reset flags
        roi_changed_called = False
        selection_changed_called = False
        
        # Select different ROI (should trigger selection callback)
        roi_manager.add_roi(200, 200, 100, 100)
        roi_manager.select_roi(0)
        assert selection_changed_called
        
        # Reset flags
        roi_changed_called = False
        selection_changed_called = False
        
        # Delete ROI (should trigger both callbacks)
        roi_manager.delete_roi(0)
        assert roi_changed_called
        assert selection_changed_called


@pytest.mark.integration
class TestROIManagerIntegration:
    """Integration tests for ROIManager."""
    
    def test_complete_roi_workflow(self, roi_manager, sample_video_frame):
        """Test complete ROI management workflow."""
        # Start with empty manager
        assert roi_manager.get_roi_count() == 0
        
        # Draw first ROI
        roi_manager.start_drawing(50, 50)
        roi_manager.update_drawing(150, 150)
        roi1_index = roi_manager.finish_drawing(150, 150, 640, 480)
        assert roi1_index == 0
        
        # Draw second ROI
        roi_manager.start_drawing(200, 200)
        roi2_index = roi_manager.finish_drawing(350, 300, 640, 480)
        assert roi2_index == 1
        
        # Set first as background
        roi_manager.set_background_roi(0)
        assert roi_manager.background_roi_index == 0
        
        # Move second ROI
        roi_manager.start_moving(1, 275, 250)  # Point inside ROI
        roi_manager.update_moving(300, 275, 640, 480)
        roi_manager.finish_moving()
        
        # Verify final state
        assert roi_manager.get_roi_count() == 2
        assert roi_manager.get_background_roi() is not None
        
        # Render ROIs
        rendered_frame = roi_manager.render_rois(sample_video_frame)
        assert rendered_frame is not None
        assert not np.array_equal(rendered_frame, sample_video_frame)
        
        # Clean up
        roi_manager.clear_all()
        assert roi_manager.get_roi_count() == 0