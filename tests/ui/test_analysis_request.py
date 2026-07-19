from PyQt5 import QtCore, QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


def test_analyze_video_passes_manual_threshold(qt_application: QtWidgets.QApplication, monkeypatch, tmp_path):
    window = VideoAnalyzer()
    try:
        window.video_path = "dummy.mp4"
        window.rects = [((0, 0), (10, 10))]
        window.start_frame = 0
        window.end_frame = 1
        window.threshold_spin.setValue(12.5)

        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: str(tmp_path)),
        )
        monkeypatch.setattr(QtCore.QThread, "start", lambda self, *args, **kwargs: None)

        window.analyze_video()

        assert window._analysis_worker is not None
        request = window._analysis_worker._request
        assert request.manual_threshold == 12.5
    finally:
        if window._analysis_progress is not None:
            window._analysis_progress.close()
        window.close()
