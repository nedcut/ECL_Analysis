import numpy as np

from ecl_analysis.analysis.masking import (
    build_consensus_mask,
    build_signal_mask,
    evaluate_mask_candidate,
)


def test_build_signal_mask_removes_isolated_hot_pixel():
    roi_l_star = np.zeros((8, 8), dtype=np.float32)
    roi_l_star[2, 2] = 80.0

    mask, threshold_value, min_area = build_signal_mask(
        roi_l_star=roi_l_star,
        background_brightness=5.0,
        noise_floor_threshold=5.0,
        morphological_kernel_size=3,
    )

    assert threshold_value == 5.0
    assert min_area >= 4
    assert not np.any(mask)


def test_evaluate_mask_candidate_scores_small_bright_region():
    roi_l_star = np.full((10, 10), 2.0, dtype=np.float32)
    roi_l_star[4:7, 4:7] = 25.0

    candidate = evaluate_mask_candidate(
        roi_l_star=roi_l_star,
        background_brightness=2.0,
        noise_floor_threshold=5.0,
        morphological_kernel_size=3,
        frame_idx=12,
    )

    assert candidate is not None
    assert candidate.frame_idx == 12
    assert candidate.score > 0.0
    assert candidate.pixel_count >= 4


def test_build_consensus_mask_marks_unstable_capture():
    base_mask = np.zeros((8, 8), dtype=bool)
    base_mask[2:5, 2:5] = True

    shifted_mask = np.zeros((8, 8), dtype=bool)
    shifted_mask[3:6, 3:6] = True

    candidates = [
        type("Candidate", (), {
            "frame_idx": 10,
            "score": 12.0,
            "background_brightness": 1.0,
            "mask": base_mask,
            "pixel_count": int(np.count_nonzero(base_mask)),
            "signal_peak": 10.0,
            "threshold_value": 5.0,
            "min_component_area": 4,
        })(),
        type("Candidate", (), {
            "frame_idx": 12,
            "score": 11.0,
            "background_brightness": 1.0,
            "mask": shifted_mask,
            "pixel_count": int(np.count_nonzero(shifted_mask)),
            "signal_peak": 9.0,
            "threshold_value": 5.0,
            "min_component_area": 4,
        })(),
    ]

    mask, metadata = build_consensus_mask(
        candidates=candidates,
        capture_mode="per_roi_auto",
        noise_floor_threshold=5.0,
        morphological_kernel_size=3,
    )

    assert mask is not None
    assert metadata.capture_mode == "per_roi_auto"
    assert metadata.primary_source_frame == 10
    assert metadata.consensus_ratio < 1.0
    assert "low_consensus" in metadata.warnings
