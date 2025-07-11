"""Video data model for Brightness Sorcerer."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import cv2
from datetime import datetime


@dataclass
class VideoMetadata:
    """Metadata for video files."""
    file_path: str
    file_size_mb: float
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    
    @classmethod
    def from_file_path(cls, file_path: str) -> 'VideoMetadata':
        """Create metadata from file path."""
        file_size = 0.0
        creation_date = None
        modification_date = None
        
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                file_size = stat.st_size / (1024 * 1024)  # Convert to MB
                creation_date = datetime.fromtimestamp(stat.st_ctime)
                modification_date = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            pass
        
        return cls(
            file_path=file_path,
            file_size_mb=file_size,
            creation_date=creation_date,
            modification_date=modification_date
        )


@dataclass
class VideoData:
    """Complete video data including technical properties and metadata."""
    
    # File information
    file_path: str
    filename: str
    file_extension: str
    metadata: VideoMetadata
    
    # Video properties
    width: int
    height: int
    total_frames: int
    fps: float
    duration_seconds: float
    
    # Codec information
    fourcc: Optional[str] = None
    codec_name: Optional[str] = None
    bitrate: Optional[float] = None
    
    # Additional properties
    color_space: Optional[str] = None
    pixel_format: Optional[str] = None
    
    @classmethod
    def from_video_file(cls, file_path: str) -> Optional['VideoData']:
        """Create VideoData from video file analysis."""
        if not os.path.exists(file_path):
            return None
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None
            
            # Get basic properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Get codec information
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            # File information
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(filename)[1].lower()
            metadata = VideoMetadata.from_file_path(file_path)
            
            return cls(
                file_path=file_path,
                filename=filename,
                file_extension=file_extension,
                metadata=metadata,
                width=width,
                height=height,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration,
                fourcc=fourcc,
                codec_name=cls._get_codec_name(fourcc)
            )
            
        except Exception as e:
            print(f"Error analyzing video file {file_path}: {e}")
            return None
    
    @staticmethod
    def _get_codec_name(fourcc: str) -> str:
        """Get human-readable codec name from FourCC."""
        codec_map = {
            'H264': 'H.264/AVC',
            'h264': 'H.264/AVC', 
            'avc1': 'H.264/AVC',
            'H265': 'H.265/HEVC',
            'hev1': 'H.265/HEVC',
            'hvc1': 'H.265/HEVC',
            'mp4v': 'MPEG-4 Visual',
            'XVID': 'Xvid',
            'DIVX': 'DivX',
            'VP80': 'VP8',
            'VP90': 'VP9',
            'AV01': 'AV1',
            'MJPG': 'Motion JPEG'
        }
        
        return codec_map.get(fourcc, fourcc)
    
    def get_aspect_ratio(self) -> float:
        """Get video aspect ratio."""
        if self.height > 0:
            return self.width / self.height
        return 0.0
    
    def get_resolution_string(self) -> str:
        """Get formatted resolution string."""
        return f"{self.width}×{self.height}"
    
    def get_duration_string(self) -> str:
        """Get formatted duration string."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_fps_string(self) -> str:
        """Get formatted FPS string."""
        return f"{self.fps:.2f} fps"
    
    def get_file_size_string(self) -> str:
        """Get formatted file size string."""
        if self.metadata.file_size_mb < 1:
            return f"{self.metadata.file_size_mb * 1024:.1f} KB"
        elif self.metadata.file_size_mb < 1024:
            return f"{self.metadata.file_size_mb:.1f} MB"
        else:
            return f"{self.metadata.file_size_mb / 1024:.1f} GB"
    
    def is_high_resolution(self) -> bool:
        """Check if video is high resolution (>= 1080p)."""
        return self.height >= 1080
    
    def is_high_framerate(self) -> bool:
        """Check if video is high framerate (>= 60 fps)."""
        return self.fps >= 60
    
    def estimate_analysis_time(self, frame_range: tuple = None) -> float:
        """Estimate analysis time in seconds based on video properties."""
        # Base time per frame (empirical estimate)
        base_time_per_frame = 0.01  # 10ms per frame baseline
        
        # Calculate frames to analyze
        if frame_range:
            start_frame, end_frame = frame_range
            frames_to_analyze = end_frame - start_frame + 1
        else:
            frames_to_analyze = self.total_frames
        
        # Resolution factor (higher resolution takes longer)
        pixel_count = self.width * self.height
        resolution_factor = pixel_count / (1920 * 1080)  # Normalized to 1080p
        
        # Estimate total time
        estimated_time = frames_to_analyze * base_time_per_frame * resolution_factor
        
        return max(estimated_time, 1.0)  # Minimum 1 second
    
    def get_technical_summary(self) -> Dict[str, Any]:
        """Get technical summary of video properties."""
        return {
            'resolution': self.get_resolution_string(),
            'aspect_ratio': f"{self.get_aspect_ratio():.2f}",
            'duration': self.get_duration_string(),
            'framerate': self.get_fps_string(),
            'total_frames': f"{self.total_frames:,}",
            'file_size': self.get_file_size_string(),
            'codec': self.codec_name or 'Unknown',
            'fourcc': self.fourcc or 'Unknown'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'filename': self.filename,
            'file_extension': self.file_extension,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.duration_seconds,
            'fourcc': self.fourcc,
            'codec_name': self.codec_name,
            'file_size_mb': self.metadata.file_size_mb,
            'creation_date': self.metadata.creation_date.isoformat() if self.metadata.creation_date else None,
            'modification_date': self.metadata.modification_date.isoformat() if self.metadata.modification_date else None
        }
    
    def __repr__(self) -> str:
        return (f"VideoData('{self.filename}', "
                f"{self.get_resolution_string()}, "
                f"{self.get_duration_string()}, "
                f"{self.get_fps_string()})")