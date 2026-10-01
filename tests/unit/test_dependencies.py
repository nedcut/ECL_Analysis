"""Optional-dependency loaders must degrade gracefully on any import failure."""

from __future__ import annotations

import logging

import pytest

from ecl_analysis import dependencies


@pytest.fixture
def fresh_dependencies(monkeypatch):
    """Reset the lazy-load caches so each test re-attempts the imports."""
    for flag in ("_pygame_load_attempted", "_plotly_load_attempted", "_librosa_load_attempted"):
        monkeypatch.setattr(dependencies, flag, False)
    for name in ("pygame", "librosa", "sf", "go", "make_subplots"):
        monkeypatch.setattr(dependencies, name, None)
    return dependencies


def _failing_import(exc: Exception):
    def _import(name, *args, **kwargs):
        raise exc

    return _import


@pytest.mark.parametrize(
    "loader, expected",
    [
        ("get_librosa", (None, None)),
        ("get_plotly", (None, None)),
        ("get_pygame", None),
    ],
)
def test_non_import_errors_disable_the_feature(fresh_dependencies, monkeypatch, caplog, loader, expected):
    monkeypatch.setattr(
        fresh_dependencies.importlib,
        "import_module",
        _failing_import(RuntimeError("numba cache is corrupt")),
    )

    with caplog.at_level(logging.WARNING):
        assert getattr(fresh_dependencies, loader)() == expected

    assert "failed to import" in caplog.text


def test_import_error_still_logs_at_info(fresh_dependencies, monkeypatch, caplog):
    monkeypatch.setattr(
        fresh_dependencies.importlib,
        "import_module",
        _failing_import(ImportError("No module named 'librosa'")),
    )

    with caplog.at_level(logging.INFO):
        assert fresh_dependencies.get_librosa() == (None, None)

    assert "librosa not available" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
