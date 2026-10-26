"""Streamlit frontend. Run: streamlit run App.py"""
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from model import (ROOT, read_rgb, png_bytes, detect_scratches, classical_restore,
                   load_restorer, neural_restore, colorize, damage)

st.set_page_config(page_title="Reframe · Photo Restoration", page_icon="📷", layout="wide")
st.markdown("""<style>
.block-container{max-width:1250px;padding-top:2.5rem}
h1{font-weight:650!important;letter-spacing:-1.8px;font-size:3.4rem!important}
.eyebrow{color:#d5b778;letter-spacing:3px;font-size:12px;text-transform:uppercase}
.intro{color:#b5bec8;max-width:650px;font-size:18px;line-height:1.7}
div[data-testid="stMetric"]{background:#1b222b;border:1px solid #303945;border-radius:14px;padding:15px}
div.stButton>button[kind="primary"]{border-radius:9px;min-height:48px;font-weight:600}
section[data-testid="stSidebar"]{border-right:1px solid #303945}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_model(path, mtime, device):
    return load_restorer(path, device)


with st.sidebar:
    st.markdown("### 📷 REFRAME")
    st.caption("OLD PHOTOGRAPHS. NEW LIFE.")
    st.divider()
    checkpoint = ROOT / "runs/pix2pix/best.pt"
    available = checkpoint.is_file()
    choices = ["OpenCV baseline"] + (["Trained Pix2Pix GAN"] if available else [])
    engine = st.selectbox("Restoration engine", choices, index=len(choices)-1)
    if available:
        st.success("Restoration checkpoint available")
    else:
        st.info("Pix2Pix GAN weights are not trained yet. Baseline uses OpenCV; it is not deep learning.")
    job_file = ROOT / "runs/pix2pix/job_status.json"
    if job_file.is_file():
        job = json.loads(job_file.read_text(encoding="utf-8"))
        if job["state"] == "running":
            st.info("GAN training/evaluation is still running. Available checkpoints are intermediate results.")
        elif job["state"] == "failed":
            st.warning("The training job stopped. Check the training log before using this checkpoint.")
    st.caption(f"Hardware: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    strength = st.slider("Restoration strength", 0., 1., .65, .05)
    mask_mode = st.radio("Damage mask", ["Automatic proposal", "Upload a mask", "No scratch repair"])
    sensitivity = st.slider("Detection threshold", 10, 80, 28,
                            help="Higher values select fewer structures.") if mask_mode == "Automatic proposal" else 28
    use_color = st.checkbox("Colorize black-and-white photo")
    color_dir = ROOT / "weights/colorization"
    color_ready = all((color_dir/name).is_file() for name in
        ["colorization_deploy_v2.prototxt", "colorization_release_v2.caffemodel", "pts_in_hull.npy"])
    if use_color:
        st.caption("Predicted colors are plausible guesses; original colors cannot be recovered reliably.")
        if not color_ready:
            st.warning("Download colorization assets with prepare_data.py --colorization.")
    max_side = st.select_slider("Processing resolution", [512, 768, 1024, 1536, 2048], value=1024)
    st.divider()
    st.caption("DL CA2 · PyTorch + Streamlit\n\nPhotos are processed locally. Uploads are not saved by the app.")

st.markdown('<div class="eyebrow">A second life for your memories</div>', unsafe_allow_html=True)
st.title("Bring the past into focus.")
st.markdown('<p class="intro">Repair scratches, lift faded details and rediscover your photographs. '
            'Restore a memory, one frame at a time.</p>', unsafe_allow_html=True)
studio, training, about = st.tabs(["Restoration studio", "Training & evidence", "How it works"])

with studio:
    st.subheader("01 / Choose a photograph")
    uploaded = st.file_uploader("Upload an old photograph", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"])
    demo = st.toggle("Presentation demo: add synthetic damage to your uploaded image",
                     help="Use a clean photo. This creates a known clean reference, not a real-world quality score.")
    if uploaded:
        try:
            image = read_rgb(uploaded)
            h, w = image.shape[:2]
            if max(h, w) > max_side:
                image = cv2.resize(image, (max(1,round(w*max_side/max(h,w))), max(1,round(h*max_side/max(h,w)))),
                                   interpolation=cv2.INTER_AREA)
            reference = image.copy() if demo else None
            if demo:
                image, true_mask = damage(image, 2026)
            else:
                true_mask = None
            mask = np.zeros(image.shape[:2], np.uint8)
            if mask_mode == "Automatic proposal":
                mask = detect_scratches(image, sensitivity)
                st.caption("Automatic detection may select hair, edges or texture. Inspect the red overlay before restoring.")
            elif mask_mode == "Upload a mask":
                mask_file = st.file_uploader("Damage mask: white = repair, black = preserve", type=["png", "jpg"])
                if mask_file:
                    with Image.open(mask_file) as mask_image:
                        mask_image = mask_image.convert("L").resize((image.shape[1],image.shape[0]), Image.Resampling.NEAREST)
                        mask = np.uint8(np.asarray(mask_image) > 127)*255
                else:
                    st.info("Upload a mask to enable scratch repair.")
            if demo and st.checkbox("Use the known synthetic damage mask", value=True):
                mask = true_mask
            coverage = float(np.mean(mask > 0)*100)
            left, right = st.columns([1.5, 1])
            with left:
                st.image(image, caption="Original / input", use_container_width=True)
            with right:
                overlay = image.copy()
                overlay[mask > 0] = (overlay[mask > 0]*.35 + np.array([235,80,90])*.65).astype(np.uint8)
                st.image(overlay, caption=f"Damage mask preview · {coverage:.2f}% selected", use_container_width=True)
            if coverage > 20:
                st.warning("More than 20% of the photo is selected. Try a higher threshold or a precise uploaded mask.")
            run_key = hashlib.sha256(image.tobytes()+mask.tobytes()+
                json.dumps([engine,strength,use_color,checkpoint.stat().st_mtime_ns if available else 0]).encode()).hexdigest()
            if st.button("✦ Restore photograph", type="primary", use_container_width=True,
                         disabled=use_color and not color_ready):
                start = time.perf_counter()
                with st.spinner("Restoring your photograph…"):
                    if engine == "Trained Pix2Pix GAN":
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        net, metadata = cached_model(str(checkpoint), checkpoint.stat().st_mtime_ns, device)
                        restored = neural_restore(net, image, mask, device)
                        restored = cv2.addWeighted(restored, strength, image, 1-strength, 0)
                    else:
                        restored = classical_restore(image, mask, strength)
                    if use_color:
                        restored = colorize(restored, color_dir)
                st.session_state["result"] = dict(key=run_key, image=restored, seconds=time.perf_counter()-start)
            result = st.session_state.get("result")
            if result and result["key"] == run_key:
                restored = result["image"]
                st.subheader("02 / Rediscover the details")
                divider = st.slider("Before / after comparison", 0, 100, 50,
                                    help="Left is input; right is restored.")
                split = round(image.shape[1]*divider/100)
                comparison = restored.copy()
                comparison[:, :split] = image[:, :split]
                if 0 < split < image.shape[1]:
                    comparison[:, split:split+2] = [213,183,120]
                st.image(comparison, caption="BEFORE ← | → AFTER", use_container_width=True)
                a,b,c = st.columns(3)
                a.metric("Processing time", f'{result["seconds"]:.2f} s')
                b.metric("Output size", f"{image.shape[1]} × {image.shape[0]}")
                c.metric("Engine", "Pix2Pix GAN" if engine == "Trained Pix2Pix GAN" else "OpenCV")
                if reference is not None:
                    from model import metric_pair
                    before_psnr, before_ssim = metric_pair(image, reference)
                    after_psnr, after_ssim = metric_pair(restored, reference)
                    st.caption(f"Synthetic demo reference only · PSNR {before_psnr:.2f} → {after_psnr:.2f} dB · "
                               f"SSIM {before_ssim:.3f} → {after_ssim:.3f}. Higher is better.")
                st.download_button("Download restored PNG", png_bytes(restored),
                                   file_name="restored_photograph.png", mime="image/png", use_container_width=True)
                st.download_button("Download damage mask", png_bytes(mask),
                                   file_name="damage_mask.png", mime="image/png")
        except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError, RuntimeError, cv2.error) as error:
            st.error(f"Could not process this photograph: {error}")
    else:
        st.info("Upload a photograph to open your restoration workspace. JPG, PNG, WebP and TIFF supported.")

with training:
    st.subheader("Measured progress, visible results")
    history = ROOT / "runs/pix2pix/history.csv"
    if history.is_file():
        frame = pd.read_csv(history)
        if not frame.empty:
            st.line_chart(frame.set_index("epoch")[["g_loss", "d_loss"]])
            st.line_chart(frame.set_index("epoch")[["psnr", "input_psnr"]])
            st.line_chart(frame.set_index("epoch")[["ssim", "input_ssim"]])
            st.dataframe(frame, use_container_width=True)
            st.caption("Metrics use fixed synthetic corruption on held-out source photos. They do not prove performance on real historical damage.")
    else:
        st.info("No completed training epochs found. No training metrics or quality claims are available yet.")
    st.code("python prepare_data.py --dataset\npython model.py --epochs 60 --batch-size 4\npython evaluate.py", language="bash")
    st.write("Pix2Pix GAN training uses a U-Net Generator and a conditional PatchGAN Discriminator. "
             "G loss = adversarial loss + 100 x L1 loss. D learns real versus generated image patches. "
             "Both networks use ordinary float32 PyTorch and Adam; only the Generator runs during restoration.")

with about:
    st.markdown("""### A clear restoration workflow
1. **Prepare** — correct EXIF orientation and resize within your processing limit.
2. **Locate damage** — inspect an automatic scratch proposal or supply a white-on-black mask.
3. **Restore** — run your trained Pix2Pix GAN Generator, or the clearly labeled OpenCV baseline.
4. **Colorize optionally** — use the pretrained ECCV16 colorization model by Zhang et al.
5. **Compare and download** — inspect the result and export a PNG.

The network learns from clean DIV2K images with synthetic fading, noise, JPEG compression and scratches.
The 800 training and 100 validation source images are kept separate. This project trains a Pix2Pix-style
conditional GAN: a compact residual U-Net Generator competes with a 70x70 PatchGAN Discriminator.
Large missing faces, severe tears and unseen damage
remain difficult. Colorization predicts plausible colors and may be inaccurate.
""")
