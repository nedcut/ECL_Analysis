"""ROI (Region of Interest) data model."""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ROI:
    """Represents a region of interest for brightness analysis."""
    
    x: int
    y: int
    width: int
    height: int
    color: Tuple[int, int, int] = (255, 50, 50)  # BGR color
    label: str = ""
    is_background: bool = False
    
    def __post_init__(self):
        """Validate ROI parameters after initialization."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI coordinates must be non-negative")
    
    @property
    def top_left(self) -> Tuple[int, int]:
        """Get top-left corner coordinates."""
        return (self.x, self.y)
    
    @property 
    def bottom_right(self) -> Tuple[int, int]:
        """Get bottom-right corner coordinates."""
        return (self.x + self.width, self.y + self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get ROI area in pixels."""
        return self.width * self.height
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this ROI."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def intersects(self, other: 'ROI') -> bool:
        """Check if this ROI intersects with another ROI."""
        return not (self.x + self.width <= other.x or 
                   other.x + other.width <= self.x or
                   self.y + self.height <= other.y or 
                   other.y + other.height <= self.y)
    
    def is_valid_for_frame(self, frame_width: int, frame_height: int) -> bool:
        """Check if ROI is within frame boundaries."""
        return (self.x >= 0 and self.y >= 0 and 
                self.x + self.width <= frame_width and 
                self.y + self.height <= frame_height)
    
    def get_corner_handles(self, handle_size: int = 10) -> list:
        """Get corner handle rectangles for resizing."""
        handles = []
        corners = [
            (self.x, self.y),  # Top-left
            (self.x + self.width, self.y),  # Top-right
            (self.x + self.width, self.y + self.height),  # Bottom-right
            (self.x, self.y + self.height)  # Bottom-left
        ]
        
        for corner_x, corner_y in corners:
            handle_x = corner_x - handle_size // 2
            handle_y = corner_y - handle_size // 2
            handles.append((handle_x, handle_y, handle_size, handle_size))
        
        return handles
    
    def get_resize_corner_from_point(self, x: int, y: int, sensitivity: int = 10) -> Optional[int]:
        """Determine which corner handle a point is closest to."""
        corners = [
            (self.x, self.y),  # 0: Top-left
            (self.x + self.width, self.y),  # 1: Top-right  
            (self.x + self.width, self.y + self.height),  # 2: Bottom-right
            (self.x, self.y + self.height)  # 3: Bottom-left
        ]
        
        for i, (corner_x, corner_y) in enumerate(corners):
            if abs(x - corner_x) <= sensitivity and abs(y - corner_y) <= sensitivity:
                return i
        
        return None
    
    def resize_from_corner(self, corner_index: int, new_x: int, new_y: int):
        """Resize ROI by dragging a specific corner."""
        if corner_index == 0:  # Top-left
            new_width = self.width + (self.x - new_x)
            new_height = self.height + (self.y - new_y)
            if new_width > 0 and new_height > 0:
                self.width = new_width
                self.height = new_height
                self.x = new_x
                self.y = new_y
        elif corner_index == 1:  # Top-right
            new_width = new_x - self.x
            new_height = self.height + (self.y - new_y)
            if new_width > 0 and new_height > 0:
                self.width = new_width
                self.height = new_height
                self.y = new_y
        elif corner_index == 2:  # Bottom-right
            new_width = new_x - self.x
            new_height = new_y - self.y
            if new_width > 0 and new_height > 0:
                self.width = new_width
                self.height = new_height
        elif corner_index == 3:  # Bottom-left
            new_width = self.width + (self.x - new_x)
            new_height = new_y - self.y
            if new_width > 0 and new_height > 0:
                self.width = new_width
                self.height = new_height
                self.x = new_x
    
    def move_to(self, new_x: int, new_y: int):
        """Move ROI to new position."""
        self.x = new_x
        self.y = new_y
    
    def extract_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract the ROI region from a frame."""
        if frame is None:
            return None
        
        frame_height, frame_width = frame.shape[:2]
        if not self.is_valid_for_frame(frame_width, frame_height):
            return None
        
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]
    
    def to_dict(self) -> dict:
        """Convert ROI to dictionary for serialization."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'color': self.color,
            'label': self.label,
            'is_background': self.is_background
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ROI':
        """Create ROI from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'], 
            width=data['width'],
            height=data['height'],
            color=tuple(data.get('color', (255, 50, 50))),
            label=data.get('label', ''),
            is_background=data.get('is_background', False)
        )
    
    def __repr__(self) -> str:
        return f"ROI(x={self.x}, y={self.y}, w={self.width}, h={self.height}, label='{self.label}')"