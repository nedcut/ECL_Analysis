import json
import os

import pytest

from ecl_analysis.video_analyzer import _atomic_write_json


def test_atomic_write_json_writes_data(tmp_path):
    target = tmp_path / "settings.json"
    _atomic_write_json(str(target), {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_atomic_write_json_leaves_original_intact_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"original": True}))

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(json, "dump", boom)

    with pytest.raises(RuntimeError):
        _atomic_write_json(str(target), {"new": "data"})

    # Original file must be untouched.
    assert json.loads(target.read_text()) == {"original": True}

    # No leftover temp files should remain in the directory.
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp_settings_")]
    assert leftovers == []
