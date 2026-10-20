"""Streamlit interface for the Crime Scene Visual Change Analyser."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from crime_scene_analyser.analysis import AnalysisError, analyse_pair


st.set_page_config(page_title="Crime Scene Visual Change Analyser", page_icon="🔍", layout="wide")


def bgr_from_upload(uploaded_file) -> np.ndarray:
    rgb = np.asarray(Image.open(uploaded_file).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_download(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("The result image could not be encoded.")
    return encoded.tobytes()


st.title("Crime Scene Visual Change Analyser")
st.caption("A privacy-conscious academic demonstration of registered before/after image comparison. Use only authorised, non-sensitive imagery.")

with st.sidebar:
    st.header("Analysis controls")
    align = st.toggle("Align after image", value=True, help="Uses ORB features and RANSAC to reduce camera-position differences.")
    min_region_area = st.slider("Minimum change region area (pixels)", 10, 5000, 120, 10)
    sensitivity = st.slider("Threshold adjustment", -30, 30, 0, 1, help="Lower values detect more subtle differences.")
    st.divider()
    st.info("No uploaded image is written to disk by this app.")

asset_path = Path(__file__).parent / "assets"
with st.expander("Demo mode: use the included synthetic images", expanded=True):
    st.write("The bundled scene is computer-generated and contains no real or graphic crime-scene content.")
    use_demo = st.button("Load synthetic demo pair", type="secondary", use_container_width=True)
    if use_demo:
        st.session_state["before_image"] = cv2.imread(str(asset_path / "demo_before.png"))
        st.session_state["after_image"] = cv2.imread(str(asset_path / "demo_after.png"))

left, right = st.columns(2)
with left:
    before_upload = st.file_uploader("Before image", type=["png", "jpg", "jpeg"], key="before_upload")
with right:
    after_upload = st.file_uploader("After image", type=["png", "jpg", "jpeg"], key="after_upload")

if before_upload is not None:
    try:
        st.session_state["before_image"] = bgr_from_upload(before_upload)
    except Exception as exc:
        st.error(f"Could not read the before image: {exc}")
if after_upload is not None:
    try:
        st.session_state["after_image"] = bgr_from_upload(after_upload)
    except Exception as exc:
        st.error(f"Could not read the after image: {exc}")

before = st.session_state.get("before_image")
after = st.session_state.get("after_image")
if before is None or after is None:
    st.warning("Upload both images or load the synthetic demo pair to begin.")
    st.stop()

try:
    result = analyse_pair(before, after, align=align, min_region_area=min_region_area, sensitivity=sensitivity)
except AnalysisError as exc:
    st.error(f"Analysis could not run: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected processing error: {exc}")
    st.stop()

metrics = st.columns(4)
metrics[0].metric("Structural similarity", f"{result.ssim_score:.3f}")
metrics[1].metric("Changed area", f"{result.changed_percentage:.2f}%")
metrics[2].metric("Changed pixels", f"{result.changed_pixels:,}")
metrics[3].metric("Detected regions", result.region_count)
if result.alignment.applied:
    st.success(result.alignment.message)
else:
    st.info(result.alignment.message)

st.subheader("Visual comparison")
compare_left, compare_right = st.columns(2)
compare_left.image(rgb(before), caption="Before / reference", use_container_width=True)
compare_right.image(rgb(result.aligned_after), caption="After / aligned candidate", use_container_width=True)

result_left, result_right = st.columns(2)
result_left.image(rgb(result.annotated), caption="Annotated change regions", use_container_width=True)
result_right.image(rgb(result.overlay), caption="Change-mask overlay", use_container_width=True)

diagnostic_left, diagnostic_right = st.columns(2)
diagnostic_left.image(result.difference_map, caption="SSIM-derived difference intensity", use_container_width=True, clamp=True)
diagnostic_right.image(result.mask, caption="Filtered binary change mask", use_container_width=True, clamp=True)

st.download_button(
    "Download annotated result (PNG)",
    data=image_download(result.annotated),
    file_name="annotated_change_analysis.png",
    mime="image/png",
)

with st.expander("How to interpret this demonstration"):
    st.markdown(
        """**Structural similarity (SSIM)** compares local luminance, contrast, and structure. The application
        thresholds the inverted SSIM map, filters noise morphologically, and reports connected change regions.
        Results are decision support—not forensic proof. Lighting, shadows, occlusion, and poor registration can
        create false positives; preserve original evidence and use validated laboratory procedures in real cases."""
    )
