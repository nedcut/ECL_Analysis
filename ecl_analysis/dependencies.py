"""Handle optional third-party dependencies used throughout the app."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Callable, Optional, Tuple

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
    global pygame, _pygame_load_attempted

    if _pygame_load_attempted:
        return pygame

    _pygame_load_attempted = True
    try:
        pygame = importlib.import_module("pygame")
    except ImportError:
        logging.info("pygame not available - audio features disabled")

    return pygame


def get_plotly() -> Tuple[Optional[ModuleType], Optional[Callable[..., object]]]:
    """Return plotly graph_objects and make_subplots callable, importing lazily."""
    global go, make_subplots, _plotly_load_attempted

    if _plotly_load_attempted:
        return go, make_subplots

    _plotly_load_attempted = True
    try:
        go = importlib.import_module("plotly.graph_objects")
        subplots_module = importlib.import_module("plotly.subplots")
        make_subplots = getattr(subplots_module, "make_subplots", None)
    except ImportError:
        logging.info("plotly not available - interactive plots disabled")

    return go, make_subplots


def get_librosa() -> Tuple[Optional[ModuleType], Optional[ModuleType]]:
    """Return librosa and soundfile modules if available, importing lazily."""
    global librosa, sf, _librosa_load_attempted

    if _librosa_load_attempted:
        return librosa, sf

    _librosa_load_attempted = True
    try:
        librosa = importlib.import_module("librosa")
        sf = importlib.import_module("soundfile")
    except ImportError:
        logging.info("librosa not available - audio analysis disabled")

    return librosa, sf
