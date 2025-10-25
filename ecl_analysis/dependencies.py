"""Handle optional third-party dependencies used throughout the app."""

import logging

PYGAME_AVAILABLE = False
PLOTLY_AVAILABLE = False
LIBROSA_AVAILABLE = False

pygame = None
librosa = None
sf = None
go = None
make_subplots = None

try:
    import pygame as _pygame

    pygame = _pygame
    PYGAME_AVAILABLE = True
except ImportError:
    logging.warning("pygame not available - audio features disabled")

try:
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    go = _go
    make_subplots = _make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logging.warning("plotly not available - interactive plots disabled")

try:
    import librosa as _librosa
    import soundfile as _sf

    librosa = _librosa
    sf = _sf
    LIBROSA_AVAILABLE = True
except ImportError:
    logging.warning("librosa not available - audio analysis disabled")
