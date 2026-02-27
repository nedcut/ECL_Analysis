"""Handle optional third-party dependencies used throughout the app."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Callable, Optional, Tuple

PYGAME_AVAILABLE = False
PLOTLY_AVAILABLE = False
LIBROSA_AVAILABLE = False

pygame: Optional[ModuleType] = None
librosa: Optional[ModuleType] = None
sf: Optional[ModuleType] = None
go: Optional[ModuleType] = None
make_subplots: Optional[Callable[..., object]] = None

_pygame_load_attempted = False
_plotly_load_attempted = False
_librosa_load_attempted = False


def get_pygame() -> Optional[ModuleType]:
    """Return pygame module if available, importing lazily."""
    global PYGAME_AVAILABLE, pygame, _pygame_load_attempted

    if _pygame_load_attempted:
        return pygame

    _pygame_load_attempted = True
    try:
        pygame = importlib.import_module("pygame")
        PYGAME_AVAILABLE = True
    except ImportError:
        logging.info("pygame not available - audio features disabled")

    return pygame


def has_pygame() -> bool:
    """Check whether pygame is available."""
    return get_pygame() is not None


def get_plotly() -> Tuple[Optional[ModuleType], Optional[Callable[..., object]]]:
    """Return plotly graph_objects and make_subplots callable, importing lazily."""
    global PLOTLY_AVAILABLE, go, make_subplots, _plotly_load_attempted

    if _plotly_load_attempted:
        return go, make_subplots

    _plotly_load_attempted = True
    try:
        go = importlib.import_module("plotly.graph_objects")
        subplots_module = importlib.import_module("plotly.subplots")
        make_subplots = getattr(subplots_module, "make_subplots", None)
        PLOTLY_AVAILABLE = go is not None and make_subplots is not None
    except ImportError:
        logging.info("plotly not available - interactive plots disabled")

    return go, make_subplots


def has_plotly() -> bool:
    """Check whether plotly interactive plotting is available."""
    loaded_go, loaded_make_subplots = get_plotly()
    return loaded_go is not None and loaded_make_subplots is not None


def get_librosa() -> Tuple[Optional[ModuleType], Optional[ModuleType]]:
    """Return librosa and soundfile modules if available, importing lazily."""
    global LIBROSA_AVAILABLE, librosa, sf, _librosa_load_attempted

    if _librosa_load_attempted:
        return librosa, sf

    _librosa_load_attempted = True
    try:
        librosa = importlib.import_module("librosa")
        sf = importlib.import_module("soundfile")
        LIBROSA_AVAILABLE = librosa is not None and sf is not None
    except ImportError:
        logging.info("librosa not available - audio analysis disabled")

    return librosa, sf


def has_librosa() -> bool:
    """Check whether librosa-based audio analysis is available."""
    loaded_librosa, loaded_sf = get_librosa()
    return loaded_librosa is not None and loaded_sf is not None
