# Crime Scene Visual Change Analyser — Presentation Outline

## 1. Title page

- **Crime Scene Visual Change Analyser**
- Student project: computer-vision-assisted before/after scene comparison
- Demonstration scope: synthetic, non-graphic imagery only

## 2. Introduction and motivation

Manual comparison of repeated scene photographs is time-consuming and can overlook small changes. A visual comparison tool can focus an investigator's review on candidate regions while keeping the human expert responsible for interpretation.

## 3. Literature survey

- Wang et al. (2004) introduced **Structural Similarity (SSIM)**, which compares local structure in addition to pixel intensity.
- Rublee et al. (2011) presented **ORB**, an efficient feature detector/descriptor suitable for image matching.
- Fischler and Bolles (1981) introduced **RANSAC**, widely used to robustly estimate geometric transformations despite incorrect feature matches.
- Szeliski (2022) surveys image alignment, feature matching, and computer-vision foundations.

## 4. Challenges identified from literature

- Camera position, scale, and viewpoint differences misalign otherwise unchanged structures.
- Lighting, shadow, compression, and occlusion can look like scene changes.
- Feature-poor scenes may not support dependable registration.
- Threshold selection trades false positives against missed subtle differences.
- Forensic use requires privacy protection, validation, explainability, and chain-of-custody processes beyond this prototype.

## 5. Objectives

1. Accept paired before/after images in an approachable web interface.
2. Align the candidate image when reliable visual features are available.
3. Detect and localise meaningful visual differences.
4. Provide interpretable images and quantitative indicators for a demonstration.
5. Use only synthetic, non-sensitive sample material.

## 6. Proposed methodology

1. Load and validate uploaded RGB images.
2. Resize the candidate image to the reference dimensions.
3. Optionally detect ORB features, match them, estimate a RANSAC homography, and warp the candidate.
4. Compute grayscale SSIM and invert its full similarity map to form a difference intensity image.
5. Blur, Otsu-threshold, and morphologically open/close the difference image.
6. Find external contours, discard regions below a user-selected area, draw boxes, and calculate metrics.

## 7. Concise dataset description

The demo set consists of two 900 × 600 computer-generated interior-scene illustrations. The after image adds three benign visual changes: a coloured object, a circle, and a line. `generate_demo_assets.py` deterministically regenerates the pair; no person, victim, real location, or graphic content is present.

## 8. Working principles

- **ORB + RANSAC:** alignment is attempted only when enough inlier feature matches exist; otherwise the app reports that it was skipped.
- **SSIM:** a score near 1 means locally similar images; the application also uses the spatial SSIM map to localise dissimilar areas.
- **Morphology:** removes isolated noise and joins nearby changed pixels.
- **Contours:** turn mask components into inspectable candidate regions.

## 9. Performance metrics

| Metric | Meaning | Interpretation |
| --- | --- | --- |
| SSIM | Global structural similarity, from -1 to 1 | Higher indicates greater visual similarity |
| Changed pixels | Mask pixels within retained regions | Absolute estimated changed area |
| Changed area (%) | Changed pixels divided by image pixels | Normalised extent for comparisons |
| Detected regions | Contours above selected minimum area | Candidate locations for review |
| Alignment inliers | RANSAC-consistent ORB matches | Alignment reliability indicator, not accuracy proof |

## 10. Results and discussion

**Live demo:** select *Load synthetic demo pair*, leave alignment enabled, and press through the visual comparison. The application marks the three deliberate changes, reports a lower-than-perfect SSIM score, and exposes the intermediate intensity map and cleaned binary mask.

The result is interpretable because the red overlay, labelled boxes, and numeric indicators agree. However, changing the threshold or minimum-area slider demonstrates the false-positive/false-negative trade-off. The result must not be asserted as forensic proof.

## 11. Conclusion

The prototype provides an end-to-end, explainable visual-change workflow: input validation, optional geometric alignment, SSIM comparison, spatial filtering, region localisation, and measurable outputs. Future work includes controlled benchmark evaluation, illumination-invariant methods, audit logs, access controls, and expert-led forensic validation.

## 12. References

1. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
2. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *ICCV*, 2564–2571. https://doi.org/10.1109/ICCV.2011.6126544
3. Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus. *Communications of the ACM, 24*(6), 381–395. https://doi.org/10.1145/358669.358692
4. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.
