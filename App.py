"""Simple frontend: upload a photograph, restore automatically, download PNG."""
import hashlib
import io
import logging
from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from model import ROOT, read_rgb, png_bytes, load_restorer, restore_automatic
from learned_repair import available as learned_available

st.set_page_config(page_title="Reframe | Photo Restoration", page_icon="\U0001F4F7",
                   layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {max-width:1000px;padding-top:3rem}
h1 {letter-spacing:-1.4px;font-weight:650!important}
div[data-testid="stFileUploader"] {border-radius:14px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cached_model(path, modified, device):
    return load_restorer(path, device)[0]


def prepare_photo(photo):
    """Keep aspect ratio and cap processing size to keep the app responsive."""
    height, width = photo.shape[:2]
    if max(height, width) > 1536:
        scale = 1536 / max(height, width)
        photo = cv2.resize(photo, (max(1, round(width*scale)), max(1, round(height*scale))),
                           interpolation=cv2.INTER_AREA)
    return photo


def restore_photo(photo, checkpoint):
    """Run the GAN and conservatively clean automatically detected scratches."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cached_model(str(checkpoint), checkpoint.stat().st_mtime_ns, device) if checkpoint.is_file() else None
    return restore_automatic(model, photo, device)


st.title("Restore your photographs")
st.write("Upload an old photo. Get a restored version to keep and share.")

uploaded = st.file_uploader("Choose a photograph", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"])
checkpoint = ROOT / "runs/pix2pix/best.pt"

if uploaded is not None:
    try:
        original = prepare_photo(read_rgb(io.BytesIO(uploaded.getvalue())))
        if not checkpoint.is_file() and not learned_available():
            st.error("Restoration is unavailable right now. Please try again later.")
        else:
            sources = [ROOT/"model.py",ROOT/"learned_repair.py",ROOT/"weights/learned_pipeline.json"]
            revision = ":".join(str(path.stat().st_mtime_ns) for path in sources if path.exists())
            weight_stamp = str(checkpoint.stat().st_mtime_ns) if checkpoint.is_file() else "pretrained-only"
            key = hashlib.sha256(revision.encode() + original.tobytes() + weight_stamp.encode()).hexdigest()
            result = st.session_state.get("restored_result")
            if result is None or result["key"] != key:
                with st.spinner("Restoring your photograph..."):
                    restored = restore_photo(original, checkpoint)
                result = {"key": key, "image": restored, "png": png_bytes(restored)}
                st.session_state["restored_result"] = result

            left, right = st.columns(2)
            with left:
                st.image(original, caption="Original", width="stretch")
            with right:
                st.image(result["image"], caption="Restored", width="stretch")
            filename = "restored_" + Path(getattr(uploaded, "name", "photo")).stem + ".png"
            st.download_button("Download restored photo", result["png"], file_name=filename,
                               mime="image/png", type="primary", width="stretch")
    except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError, RuntimeError, cv2.error):
        logging.exception("Photo restoration failed")
        st.error("This photo could not be restored. Please try another JPG or PNG image.")

