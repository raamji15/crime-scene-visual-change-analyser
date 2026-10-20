# Crime Scene Visual Change Analyser

A presentation-ready **Streamlit** application that compares authorised before/after scene images and highlights visual changes. It is an academic demonstration project; it is not a replacement for forensic validation, chain-of-custody controls, or expert interpretation.

The repository includes only a synthetic, non-graphic demo scene. Do not upload sensitive imagery, personal data, or real evidence to public deployments.

## Features

- Paired PNG/JPEG upload or a one-click synthetic demo
- Optional ORB + RANSAC homography alignment to compensate for modest camera movement
- SSIM-based visual difference map with Otsu thresholding and morphological noise removal
- Change mask, contours, labelled bounding boxes, red overlay, and PNG export
- Quantitative SSIM, changed-pixel, changed-area, and detected-region metrics
- Friendly errors for unreadable, empty, undersized, or incompatible images

## Setup and run

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_demo_assets.py
streamlit run app.py
```

Open the local URL shown by Streamlit, select **Load synthetic demo pair**, and present the four metrics plus the annotated regions, overlay, intensity map, and binary mask.

## Testing

```bash
pytest -q
```

## Method and limitations

The app resizes the candidate image to the reference dimensions, optionally aligns it with ORB keypoints and a RANSAC homography, then calculates grayscale SSIM. The inverted similarity map is blurred, Otsu-thresholded, and morphologically filtered. External contours over the selected minimum area become reported regions and bounding boxes.

Visual differences can be caused by illumination, shadows, perspective, occlusion, compression, or registration failures. Treat results as triage-oriented decision support and retain original files for any forensic workflow.

## Repository map

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit application |
| `crime_scene_analyser/analysis.py` | Testable alignment and change-analysis pipeline |
| `generate_demo_assets.py` | Reproducible non-graphic synthetic demo generator |
| `assets/` | Generated demo before/after pair |
| `tests/` | Core analysis tests |
| `PRESENTATION_OUTLINE.md` | Academic presentation content |
