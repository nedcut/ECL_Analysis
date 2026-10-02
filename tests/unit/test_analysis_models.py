from ecl_analysis.analysis.models import AnalysisRequest, has_analyzable_rois


def test_has_analyzable_rois_true_when_no_background_designated():
    rects = [((0, 0), (10, 10)), ((10, 10), (20, 20))]
    assert has_analyzable_rois(rects, None) is True


def test_has_analyzable_rois_true_when_extra_rois_besides_background():
    rects = [((0, 0), (10, 10)), ((10, 10), (20, 20))]
    assert has_analyzable_rois(rects, 0) is True


def test_has_analyzable_rois_false_when_only_roi_is_background():
    rects = [((0, 0), (10, 10))]
    assert has_analyzable_rois(rects, 0) is False


def test_has_analyzable_rois_false_when_no_rois_defined():
    assert has_analyzable_rois([], None) is False


def _request(**overrides) -> AnalysisRequest:
    kwargs = dict(
        video_path="v.mp4",
        rects=[((0, 0), (4, 4)), ((4, 4), (8, 8))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=1,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )
    kwargs.update(overrides)
    return AnalysisRequest(**kwargs)


def test_threshold_mode_background_roi_takes_precedence():
    assert _request(background_roi_idx=1, manual_threshold=5.0).threshold_mode == "background_roi"


def test_threshold_mode_manual_when_positive_threshold():
    assert _request(manual_threshold=5.0).threshold_mode == "manual_threshold"


def test_threshold_mode_none_when_no_background_and_zero_threshold():
    assert _request(manual_threshold=0.0).threshold_mode == "none"
