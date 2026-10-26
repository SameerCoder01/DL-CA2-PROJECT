"""Simple frontend: upload a photograph, restore automatically, download PNG."""
import hashlib
import io
import logging
import html
from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from model import ROOT, read_rgb, png_bytes, load_restorer, restore_automatic
from learned_repair import available as learned_available

st.set_page_config(page_title="Reframe | Photo Scratch Repair", page_icon="✦",
                   layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root {--ink:#f4f3ff;--muted:#a5a4bb;--accent:#c7b5ff;--line:rgba(204,193,255,.13)}
.stApp {background:radial-gradient(ellipse at 85% 0%,#28213d 0%,transparent 48%),#111017;color:var(--ink)}
[data-testid="stHeader"] {background:transparent}
.stAppDeployButton {display:none}
.block-container {max-width:1080px;padding:2.5rem 2.8rem 2rem}
[data-testid="stVerticalBlock"] {gap:1rem}
.brandbar {display:flex;justify-content:space-between;align-items:center;padding-bottom:23px;border-bottom:1px solid var(--line);margin-bottom:26px}
.brand {display:flex;gap:11px;align-items:center;font-size:17px;font-weight:750;letter-spacing:3px}
.brand-icon {display:grid;place-items:center;width:34px;height:34px;border:1px solid #a18dc7;border-radius:10px;color:#dccdff;font-size:25px;letter-spacing:0;background:#282136}
.private {font-size:11px;color:#c5bed8;border:1px solid var(--line);border-radius:30px;padding:7px 11px;white-space:nowrap;letter-spacing:.3px}
.private i {display:inline-block;width:6px;height:6px;background:#b0dbbd;border-radius:50%;margin-right:6px;box-shadow:0 0 9px #b0dbbd45}
.eyebrow {color:var(--accent);font-size:10px;font-weight:650;letter-spacing:2.5px;margin:4px 0 0}
h1 {font-size:clamp(32px,5.2vw,57px)!important;line-height:1.09!important;letter-spacing:-2.5px!important;font-weight:650!important;max-width:780px;padding:.3rem 0 .4rem!important;text-wrap:balance;color:var(--ink)!important}
.intro {color:#aaa7be;font-size:16px;line-height:1.7;max-width:590px;margin:0 0 12px}
.capabilities {display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.capabilities span {font-size:11px;color:#c8c4d8;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#ffffff03}
.capabilities b {color:#c7b5ff;margin-right:5px;font-weight:400}
.journey {display:flex;align-items:center;gap:12px;margin:12px 0 4px;color:#827e98;font-size:11px}
.journey .step {display:flex;align-items:center;gap:7px;white-space:nowrap}
.journey .number {display:grid;place-items:center;width:22px;height:22px;border:1px solid #3e374e;border-radius:50%;font-size:10px}
.journey .active {color:#e5dcff}
.journey .active .number {background:#c7b5ff;color:#241b37;border-color:#c7b5ff}
.journey .connector {height:1px;background:#3a3349;flex:1;max-width:90px}
.st-key-upload_panel {border:1px solid #443953!important;border-radius:20px!important;background:linear-gradient(130deg,#252030aa,#1b1822dd);padding:23px!important;box-shadow:0 16px 50px #00000018}
.panel-heading {display:flex;align-items:center;gap:11px;margin-bottom:2px}
.panel-symbol {font-size:22px;color:#cab9ef;background:#ffffff06;border:1px solid var(--line);border-radius:11px;padding:5px 12px}
.panel-title {font-size:16px;letter-spacing:-.3px;color:#f0ebfa;font-weight:600}
.panel-subtitle {font-size:12px;color:#a29bb4;margin-top:3px}
[data-testid="stFileUploader"] {padding:0}
[data-testid="stFileUploader"] section {background:#15121c99;border:1px dashed #6c5c88;border-radius:12px;transition:border-color .2s,background .2s;padding:22px 18px}
[data-testid="stFileUploader"] section:hover {border-color:#c7b5ff;background:#251e33}
[data-testid="stFileUploaderDropzoneInstructions"] {color:#c6c0d4}
[data-testid="stFileUploader"] button {background:#c7b5ff;color:#231a35;border:0;border-radius:8px;font-size:12px;font-weight:600;padding:8px 17px;box-shadow:0 3px 14px #00000024}
[data-testid="stFileUploader"] button:hover {background:#dfd2ff;color:#1c132c}
.upload-note {font-size:11px;color:#898399;display:flex;justify-content:space-between;gap:10px}
.empty-note {text-align:center;border:1px solid var(--line);background:#ffffff02;border-radius:16px;padding:24px 20px;margin-top:4px}
.empty-note .spark {color:#baabd6;font-size:23px;margin-bottom:8px}
.empty-note strong {display:block;font-size:14px;font-weight:500;color:#d4cedf;margin-bottom:5px}
.empty-note p {font-size:12px;line-height:1.7;color:#898398;margin:0}
.result-heading {display:flex;justify-content:space-between;align-items:center;gap:12px;padding-top:18px;margin-bottom:0}
.result-heading h2 {font-size:22px!important;letter-spacing:-.6px!important;padding:0!important;margin:0}
.ready {font-size:10px;letter-spacing:.6px;text-transform:uppercase;color:#b6dfc5;border:1px solid #b6dfc529;border-radius:30px;padding:6px 10px;white-space:nowrap}
.filename {font-size:11px;color:#898398;overflow-wrap:anywhere;margin-bottom:4px}
.st-key-before_panel,.st-key-after_panel {border:1px solid var(--line);border-radius:15px;background:#1b1723;padding:12px!important;height:100%}
.st-key-after_panel {border-color:#a48bc252;background:linear-gradient(160deg,#2b2238,#1b1723)}
.image-label {display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;font-size:10px;letter-spacing:1.5px;font-weight:650;color:#b7b0c5}
.image-label .tag {font-size:9px;letter-spacing:0;font-weight:400;color:#898398}
.image-label.after {color:#dcc9ff}
[data-testid="stImage"] img {border-radius:8px}
[data-testid="stImageCaption"] {font-size:11px!important;color:#a69db5!important}
[data-testid="stDownloadButton"] button {background:#c7b5ff;border:1px solid #dfcfff;color:#231a35;border-radius:11px;font-weight:650;min-height:48px;box-shadow:0 5px 24px #9877dd16;transition:background .2s,box-shadow .2s}
[data-testid="stDownloadButton"] button:hover {background:#deceff;color:#21162f;border-color:#e9dfff;box-shadow:0 5px 28px #9877dd33}
.download-note {text-align:center;color:#8e879d;font-size:11px;margin-top:-5px}
.footer {display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--line);padding-top:19px;margin-top:26px;color:#777084;font-size:10px;letter-spacing:.2px}
.footer .signature {color:#aaa0b9}
@media(min-width:480px){
 [data-testid="stHorizontalBlock"] {flex-wrap:nowrap!important}
 [data-testid="stColumn"] {min-width:0!important;flex:1 1 0!important}
}
@media(max-width:600px){
 .block-container {padding:1.2rem 1.1rem 2rem}
 .brandbar {margin-bottom:16px;padding-bottom:17px}
 .brand {font-size:14px;letter-spacing:2px}
 .private {font-size:9px;padding:6px 8px}
 h1 {font-size:36px!important;letter-spacing:-1.6px!important}
 .intro {font-size:13px}
 .st-key-upload_panel {padding:16px!important;border-radius:15px!important}
 .panel-title {font-size:14px}.panel-subtitle {font-size:11px}
 .journey {gap:6px;font-size:10px}.journey .connector {min-width:8px}
 .upload-note {font-size:9px}.capabilities {gap:5px}.capabilities span {font-size:10px;padding:5px 7px}
 .footer {font-size:9px;flex-wrap:wrap}
}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
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


st.markdown('''
<div class="brandbar"><div class="brand"><span class="brand-icon">✦</span> REFRAME</div>
<div class="private"><i></i> Private by design</div></div>
<div class="eyebrow">OLD PHOTOS. FEWER SCRATCHES.</div>
''', unsafe_allow_html=True)
st.title("Restore scratched photographs")
st.markdown('''
<p class="intro">Upload a scratched photograph. Automatically repair detected scratches, compare the before and after, and save your restored photo.</p>
<div class="capabilities"><span><b>✧</b> Automatic scratch repair</span><span><b>◐</b> Before &amp; after</span><span><b>↧</b> PNG download</span></div>
<div class="journey"><div class="step active"><span class="number">1</span> Upload</div><span class="connector"></span>
<div class="step"><span class="number">2</span> Repair scratches</div><span class="connector"></span>
<div class="step"><span class="number">3</span> Keep forever</div></div>
''', unsafe_allow_html=True)
with st.container(key="upload_panel"):
    st.markdown('''<div class="panel-heading"><span class="panel-symbol">↥</span><div>
    <div class="panel-title">Give your scratched photo a fresh start</div>
    <div class="panel-subtitle">Choose a scratched photo. Repair starts automatically.</div></div></div>''', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a photograph", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"], label_visibility="collapsed")
    st.markdown('<div class="upload-note"><span>JPG · PNG · WEBP · TIFF &nbsp; / &nbsp; Up to 20 MB</span><span>Processed on your device</span></div>', unsafe_allow_html=True)
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
                with st.spinner("Detecting and repairing scratches..."):
                    restored = restore_photo(original, checkpoint)
                result = {"key": key, "image": restored, "png": png_bytes(restored)}
                st.session_state["restored_result"] = result

            st.markdown('<div class="result-heading"><h2>Your scratch-repair result.</h2><span class="ready">✓ Repair complete</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="filename">{html.escape(Path(uploaded.name).name)} · {original.shape[1]} × {original.shape[0]} px</div>', unsafe_allow_html=True)
            left, right = st.columns(2, gap="small")
            with left:
                with st.container(key="before_panel"):
                    st.markdown('<div class="image-label">BEFORE <span class="tag">The original</span></div>', unsafe_allow_html=True)
                    st.image(original, caption="Original", width="stretch")
            with right:
                with st.container(key="after_panel"):
                    st.markdown('<div class="image-label after">AFTER <span class="tag">✦ Reframed</span></div>', unsafe_allow_html=True)
                    st.image(result["image"], caption="Restored", width="stretch")
            filename = "restored_" + Path(getattr(uploaded, "name", "photo")).stem + ".png"
            st.download_button("Download restored photo", result["png"], file_name=filename,
                               mime="image/png", type="primary", width="stretch")
            st.markdown('<div class="download-note">Save your restored memory as a PNG. Yours to keep.</div>', unsafe_allow_html=True)
    except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError, RuntimeError, cv2.error):
        logging.exception("Photo restoration failed")
        st.error("This photo could not be restored. Please try another JPG or PNG image.")

else:
    st.markdown('''<div class="empty-note"><div class="spark">✧</div><strong>Scratches fade. Memories stay.</strong>
    <p>Your original and repaired photographs will appear here, side by side.<br>Upload a scratched photo to get started.</p></div>''', unsafe_allow_html=True)

st.markdown('''<div class="footer"><span class="signature">REFRAME &nbsp; / &nbsp; Photo scratch repair.</span>
<span>Local processing &nbsp;·&nbsp; Your original stays yours.</span></div>''', unsafe_allow_html=True)

