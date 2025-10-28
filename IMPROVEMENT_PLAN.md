# Brightness Sorcerer Improvement Plan

This document tracks the optimization work inside `ecl_analysis`. Each task links back to the earlier review and will land in its own commit.

## Task Table

| Status | Task | Notes |
| --- | --- | --- |
| ✅ | Harden Qt startup (set HiDPI attributes, guard `VideoAnalyzer` creation, prefer `app.exec()`). | Implemented in `ecl_analysis/app.py`. |
| ✅ | Decouple heavy analysis/audio operations from UI thread (worker objects, signal wiring). | Async workers in `ecl_analysis/workers.py` with signal-driven UI updates in `video_analyzer.py`. |
| 🔄 | Reduce redundant per-frame conversions in analysis loop (cache LAB/background data). | `video_analyzer.py` now caches L* frames for analysis, masking, and background calculations. |
| ⏳ | Break out plotting/IO helpers and move stylesheet/QSS to dedicated module/file. | `video_analyzer.py` + new helper modules. |
| ⏳ | Improve frame cache configurability (expose size cap, avoid unnecessary copies). | `cache.py`, settings persistence. |
| ⏳ | Add comprehensive testing suite (unit tests for analysis, integration, tests for UI, performance, end-to-end). | Use `pytest`, `unittest`. |

Legend: 🔄 = in progress, ✅ = complete, ⏳ = queued, ⚠️ = blocked.
