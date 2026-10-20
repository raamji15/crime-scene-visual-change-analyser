import cv2
import numpy as np
import pytest

from crime_scene_analyser.analysis import AnalysisError, analyse_pair


def synthetic_pair():
    before = np.full((220, 300, 3), 230, dtype=np.uint8)
    cv2.rectangle(before, (30, 40), (150, 170), (120, 120, 120), -1)
    after = before.copy()
    cv2.rectangle(after, (200, 80), (250, 140), (20, 20, 20), -1)
    return before, after


def test_detects_added_object():
    before, after = synthetic_pair()
    result = analyse_pair(before, after, align=False, min_region_area=100)

    assert result.region_count >= 1
    assert result.changed_pixels > 2000
    assert result.changed_percentage > 2
    assert result.ssim_score < 0.99
    assert any(x <= 205 <= x + width and y <= 85 <= y + height for x, y, width, height, _ in result.regions)


def test_identical_images_have_no_change_regions():
    before, _ = synthetic_pair()
    result = analyse_pair(before, before.copy(), align=False, min_region_area=20)

    assert result.region_count == 0
    assert result.changed_pixels == 0
    assert result.ssim_score == pytest.approx(1.0)


def test_resizes_candidate_to_reference_dimensions():
    before, after = synthetic_pair()
    smaller_after = cv2.resize(after, (150, 110))
    result = analyse_pair(before, smaller_after, align=False, min_region_area=100)

    assert result.aligned_after.shape == before.shape


def test_rejects_tiny_image():
    with pytest.raises(AnalysisError, match="at least 32"):
        analyse_pair(np.zeros((20, 20, 3), dtype=np.uint8), np.zeros((20, 20, 3), dtype=np.uint8))
