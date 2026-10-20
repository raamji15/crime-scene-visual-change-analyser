"""Image alignment and visual-difference analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity


class AnalysisError(ValueError):
    """Raised when a submitted image pair cannot be analysed safely."""


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    applied: bool
    message: str
    inliers: int = 0
    matches: int = 0


@dataclass(frozen=True)
class AnalysisResult:
    aligned_after: np.ndarray
    difference_map: np.ndarray
    mask: np.ndarray
    annotated: np.ndarray
    overlay: np.ndarray
    ssim_score: float
    changed_pixels: int
    changed_percentage: float
    region_count: int
    regions: tuple[tuple[int, int, int, int, float], ...]
    alignment: AlignmentResult


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise AnalysisError(f"{name} must be a valid grayscale or colour image.")
    if image.size == 0:
        raise AnalysisError(f"{name} is empty.")
    if min(image.shape[:2]) < 32:
        raise AnalysisError(f"{name} must be at least 32 pixels wide and high.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise AnalysisError(f"{name} must have one, three, or four channels.")
    return image.astype(np.uint8, copy=False)


def _resize_to_reference(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    height, width = reference.shape[:2]
    if candidate.shape[:2] == (height, width):
        return candidate
    return cv2.resize(candidate, (width, height), interpolation=cv2.INTER_AREA)


def align_image(reference: np.ndarray, candidate: np.ndarray) -> AlignmentResult:
    """Align candidate to reference using ORB features and RANSAC homography.

    If the scene lacks enough reliable features, the resized candidate is returned
    unchanged. This makes alignment an optional enhancement, not a failure mode.
    """

    reference = _validate_image(reference, "Before image")
    candidate = _resize_to_reference(reference, _validate_image(candidate, "After image"))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
    ref_keypoints, ref_descriptors = detector.detectAndCompute(ref_gray, None)
    cand_keypoints, cand_descriptors = detector.detectAndCompute(candidate_gray, None)

    if ref_descriptors is None or cand_descriptors is None:
        return AlignmentResult(candidate, False, "Alignment skipped: insufficient scene features.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(ref_descriptors, cand_descriptors), key=lambda match: match.distance)
    good_matches = matches[: min(len(matches), 300)]
    if len(good_matches) < 8:
        return AlignmentResult(candidate, False, "Alignment skipped: too few feature matches.", matches=len(good_matches))

    reference_points = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    candidate_points = np.float32([cand_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(candidate_points, reference_points, cv2.RANSAC, 4.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if homography is None or inliers < 6:
        return AlignmentResult(
            candidate,
            False,
            "Alignment skipped: no reliable geometric transform was found.",
            inliers=inliers,
            matches=len(good_matches),
        )

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(
        candidate,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return AlignmentResult(aligned, True, f"Aligned using {inliers}/{len(good_matches)} feature inliers.", inliers, len(good_matches))


def analyse_pair(
    before: np.ndarray,
    after: np.ndarray,
    *,
    align: bool = True,
    min_region_area: int = 100,
    sensitivity: int = 0,
) -> AnalysisResult:
    """Compare a before/after pair and return masks, regions, and quality metrics."""

    if min_region_area < 1:
        raise AnalysisError("Minimum region area must be at least one pixel.")
    if not -40 <= sensitivity <= 40:
        raise AnalysisError("Sensitivity must be between -40 and 40.")

    before = _validate_image(before, "Before image")
    after = _resize_to_reference(before, _validate_image(after, "After image"))
    alignment = align_image(before, after) if align else AlignmentResult(after, False, "Alignment disabled.")
    aligned_after = alignment.image

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2GRAY)
    ssim_score, similarity_map = structural_similarity(before_gray, after_gray, full=True, data_range=255)
    difference_map = cv2.normalize(1.0 - similarity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    difference_map = cv2.GaussianBlur(difference_map, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(difference_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = int(np.clip(otsu_threshold + sensitivity, 0, 255))
    _, mask = cv2.threshold(difference_map, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[tuple[int, int, int, int, float]] = []
    cleaned_mask = np.zeros_like(mask)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_region_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append((x, y, width, height, area))
        cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

    annotated = aligned_after.copy()
    for index, (x, y, width, height, area) in enumerate(sorted(regions, key=lambda item: item[4], reverse=True), start=1):
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"Change {index}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    tint = np.zeros_like(aligned_after)
    tint[:, :, 2] = 255
    overlay = np.where(cleaned_mask[:, :, None] > 0, cv2.addWeighted(aligned_after, 0.55, tint, 0.45, 0), aligned_after)
    changed_pixels = int(cv2.countNonZero(cleaned_mask))
    total_pixels = cleaned_mask.size
    return AnalysisResult(
        aligned_after=aligned_after,
        difference_map=difference_map,
        mask=cleaned_mask,
        annotated=annotated,
        overlay=overlay,
        ssim_score=float(ssim_score),
        changed_pixels=changed_pixels,
        changed_percentage=(changed_pixels / total_pixels) * 100,
        region_count=len(regions),
        regions=tuple(sorted(regions, key=lambda item: item[4], reverse=True)),
        alignment=alignment,
    )
