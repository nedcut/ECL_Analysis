# Repository Guidelines

## Project Structure & Module Organization
The UI entry point is `main.py`, which wires Qt events to the analysis modules under `ecl_analysis/`. `app.py` hosts the PyQt widgets, `video_analyzer.py` coordinates frame processing, `audio.py` and `cache.py` provide optional helpers, and `constants.py` centralizes shared values. Runtime preferences persist in `brightness_analyzer_settings.json`. Keep datasets, screenshots, and large binaries out of the repo; stash ad-hoc experiments under `ecl_analysis/local/` (git-ignored) if you need scratch space.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` — create a clean environment for GUI dependencies.
- `pip install -r requirements.txt` — install PyQt, OpenCV, and analysis libraries.
- `python main.py` — launch the Brightness Sorcerer UI; pass a video path to auto-load (`python main.py sample.mp4`).
- `python -m compileall ecl_analysis` — optional import check that catches syntax errors before packaging.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and `snake_case` functions (`analyze_roi`), `CamelCase` widgets (`RoiTableModel`), and ALL_CAPS constants (see `constants.py`). Type hints are expected on public functions, and docstrings should summarize side effects. Keep UI text in `constants.py` to support future localization. Favor small, single-purpose methods; when adding Qt signals, mirror the naming already used in `app.py`.

## Testing Guidelines
No automated suite exists yet, so every change must be exercised manually. Launch `python main.py`, load a short MP4, and verify ROI drawing, auto-detection, and CSV/plot exports that your change touches. Capture console warnings, and keep observations in the PR body. If you add algorithmic logic, include a self-check function (e.g., in `video_analyzer.py`) guarded by `if __name__ == "__main__"` so others can rerun it quickly.

## Commit & Pull Request Guidelines
Commits are short, imperative summaries (`Refactor code structure for improved readability and maintainability`). Prefix a scope when helpful (`feature: pixel count`). Group related file edits and avoid reformats mixed with logic changes. PRs should describe motivation, outline manual test steps, attach before/after screenshots for UI tweaks, and link tracking issues. Request a review once linting and the manual checklist pass.

## Security & Configuration Tips
User-specific settings live in `brightness_analyzer_settings.json`; never hardcode paths or credentials there. Treat sample videos as sensitive client data—store them outside the repo and reference relative paths in docs only. When sharing debug logs, scrub file paths and machine identifiers. For reproducible releases, freeze dependency versions in `requirements.txt` and document any external codec installations in the PR.
