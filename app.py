import os
import time

import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="MRI Image Enhancement & Tumor Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Session state defaults ---
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0


def enhance_mri(img):
    if img is None:
        return None, None, None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, enhanced, mask


def analyze_tumor(mask, area_ratio_threshold=0.01):
    cleaned = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    if num_labels <= 1:
        empty = np.zeros_like(mask)
        return False, 0.0, 0, empty

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + int(np.argmax(areas))
    largest_area = int(stats[max_idx, cv2.CC_STAT_AREA])
    tumor_mask = (labels == max_idx).astype(np.uint8) * 255
    area_ratio = largest_area / mask.size
    detected = area_ratio >= area_ratio_threshold
    return detected, area_ratio, largest_area, tumor_mask


# --- Page styling: white inputs, black result boxes, grey nav ---
st.markdown(
    """
<style>
    /* Prevent title from sitting under Streamlit header / top clip */
    .block-container {
        padding-top: 2.25rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100%;
    }
    /* Title: black + white strip (matches B/W app theme) */
    .title-bw-wrap {
        margin-top: 0.25rem;
        margin-bottom: 1.25rem;
        border-radius: 10px;
        overflow: visible;
        border: 1px solid #333;
    }
    .title-bw-black {
        background: #0d0d0d !important;
        color: #ffffff !important;
        padding: 20px 22px 18px;
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        line-height: 1.3;
    }
    .title-bw-white {
        background: #ffffff !important;
        color: #111111 !important;
        padding: 12px 22px 16px;
        margin: 0;
        font-size: 0.98rem;
        font-weight: 500;
        border-top: 1px solid #333;
    }
    .box-white {
        background: #ffffff !important;
        color: #111 !important;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .box-black {
        background: #0d0d0d !important;
        color: #f0f0f0 !important;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 72px;
    }
    .box-black h4, .box-black p, .box-black span {
        color: #f0f0f0 !important;
        margin: 0.35em 0;
    }
    .box-heading {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #aaa !important;
        margin-bottom: 8px !important;
    }
    /* Sidebar: grey navigation buttons */
    div[data-testid="stSidebar"] button {
        background-color: #808080 !important;
        color: #ffffff !important;
        border: 1px solid #666 !important;
        width: 100%;
        justify-content: center;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #6a6a6a !important;
        color: #ffffff !important;
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #eee;
    }
    /* Bordered containers in main = black result boxes */
    section.main div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0d0d0d !important;
        border-color: #444 !important;
        padding: 12px !important;
    }
    section.main div[data-testid="stVerticalBlockBorderWrapper"] p,
    section.main div[data-testid="stVerticalBlockBorderWrapper"] span,
    section.main div[data-testid="stVerticalBlockBorderWrapper"] label {
        color: #e8e8e8 !important;
    }
    /* Grey buttons in main (Reset / Exit row) */
    section.main div[data-testid="stHorizontalBlock"] button {
        background-color: #808080 !important;
        color: #ffffff !important;
        border: 1px solid #666 !important;
    }
    section.main div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #6a6a6a !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Main title (black / white combo) ---
st.markdown(
    """
<div class="title-bw-wrap">
  <div class="title-bw-black">MRI Image Enhancement &amp; Tumor Detection</div>
  <div class="title-bw-white">Upload an MRI image below — preprocessing, segmentation, and metrics run automatically.</div>
</div>
""",
    unsafe_allow_html=True,
)

# --- Reset & Exit only (grey) ---
wr, we = st.columns(2, gap="small")
with wr:
    if st.button("Reset", key="wf_reset", width="stretch"):
        st.session_state.reset_counter += 1
        st.rerun()
with we:
    if st.button("Exit project", key="wf_exit", width="stretch"):
        os._exit(0)

# --- White box: file input ---
st.markdown('<div class="box-white">', unsafe_allow_html=True)
st.markdown('<p class="box-heading" style="color:#333;">Select input image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose PNG / JPG / JPEG",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state.reset_counter}",
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    t0 = time.perf_counter()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        orig, enhanced, mask = enhance_mri(img)
        tumor_detected, tumor_area_ratio, tumor_area_pixels, _ = analyze_tumor(mask)
        elapsed_s = time.perf_counter() - t0

        st.markdown("### Pipeline output")
        c1, c2, c3 = st.columns(3, gap="small")

        with c1:
            with st.container(border=True):
                st.markdown("**Uploaded image**")
                st.image(orig, clamp=True, width="stretch")

        with c2:
            with st.container(border=True):
                st.markdown("**Filtering / pre-processing**")
                st.image(enhanced, clamp=True, width="stretch")

        with c3:
            with st.container(border=True):
                st.markdown("**Segmentation**")
                st.image(mask, clamp=True, width="stretch")

        st.markdown("### Detection & metrics")
        m1, m2, m3 = st.columns(3, gap="small")

        with m1:
            with st.container(border=True):
                st.markdown("**Detection result**")
                if tumor_detected:
                    st.markdown(
                        "**Status:** Tumor **detected** (heuristic on largest region)"
                    )
                else:
                    st.markdown("**Status:** No tumor detected")

        with m2:
            with st.container(border=True):
                st.markdown("**Tumor area**")
                st.markdown(f"**Pixels:** {tumor_area_pixels:,}")
                st.markdown(
                    f"**Fraction of image:** {tumor_area_ratio * 100:.4f}%"
                )

        with m3:
            with st.container(border=True):
                st.markdown("**Execution time**")
                st.markdown(f"**Seconds:** {elapsed_s:.4f} s")
    else:
        st.error("Could not read the image file. Please upload a valid MRI image.")
else:
    st.info("Upload an image using the white box above to run the pipeline.")
