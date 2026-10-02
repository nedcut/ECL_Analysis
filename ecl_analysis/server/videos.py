"""Open-video registry with cached, lock-guarded frame access.

The browser cannot decode arbitrary lab video codecs (MJPEG AVIs, exotic MOVs),
so frames are decoded server-side with OpenCV and served as JPEG. One
``cv2.VideoCapture`` is kept open per video; sequential reads avoid a seek so
forward scrubbing stays fast.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}


class VideoOpenError(RuntimeError):
    """Raised when a video path cannot be opened for reading."""


@dataclass
class VideoSession:
    """An opened video and its cached capture handle."""

    video_id: str
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    _cap: cv2.VideoCapture = field(repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _next_read_index: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def metadata(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "path": self.path,
            "name": Path(self.path).name,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
        }

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """Read frame ``index`` (BGR) or None when it cannot be decoded."""
        if index < 0 or (self.frame_count > 0 and index >= self.frame_count):
            return None
        with self._lock:
            if index != self._next_read_index:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # A failed read leaves the decoder position unknown; force a
                # seek on the next request instead of trusting the counter.
                self._next_read_index = -1
                return None
            self._next_read_index = index + 1
            return frame

    def close(self) -> None:
        with self._lock:
            self._cap.release()


class VideoRegistry:
    """Thread-safe registry of opened videos keyed by opaque id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, VideoSession] = {}
        self._by_path: Dict[str, str] = {}
        self._lock = threading.Lock()

    def open(self, path: str) -> VideoSession:
        resolved = str(Path(path).expanduser().resolve())
        with self._lock:
            existing_id = self._by_path.get(resolved)
            if existing_id is not None:
                return self._sessions[existing_id]

        if not Path(resolved).is_file():
            raise VideoOpenError(f"File not found: {resolved}")

        cap = cv2.VideoCapture(resolved)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(f"Could not open video file: {resolved}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        session = VideoSession(
            video_id=uuid.uuid4().hex[:12],
            path=resolved,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            _cap=cap,
        )
        with self._lock:
            self._sessions[session.video_id] = session
            self._by_path[resolved] = session.video_id
        return session

    def get(self, video_id: str) -> Optional[VideoSession]:
        with self._lock:
            return self._sessions.get(video_id)

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._by_path.clear()
        for session in sessions:
            session.close()
