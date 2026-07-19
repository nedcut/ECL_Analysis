from ecl_analysis.analysis.models import has_analyzable_rois


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
