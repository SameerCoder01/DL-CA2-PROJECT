# Reframe — Old Photograph Restoration

**DL CA2 PROJECT · Streamlit frontend in `App.py` · PyTorch training in `model.py`**

[![Python checks](https://github.com/SameerCoder01/DL-CA2-PROJECT/actions/workflows/checks.yml/badge.svg)](https://github.com/SameerCoder01/DL-CA2-PROJECT/actions/workflows/checks.yml)

Upload a photo, inspect the damage mask, repair scratches and fading, compare before/after and download a PNG. Optional pretrained colorization is provided separately.

## Current status

The application and training/evaluation pipeline are implemented. **No trained restoration checkpoint, downloaded dataset or measured training results are included.** Until training completes, the frontend exposes a clearly labeled **OpenCV baseline**. It does not represent that baseline as deep learning. Consult GitHub Actions for the actual test outcome; writing a test does not mean it passed.

Initial validation: [5 functional tests passed](https://github.com/SameerCoder01/DL-CA2-PROJECT/actions/runs/40312392538), including a one-epoch training smoke test. Python 3.12 syntax checks also passed locally. This is execution validation, not restoration quality evaluation.

If the ChatGPT Work session cannot access Downloads or the internet, run `bootstrap_downloads.ps1` from your own PowerShell. It clones this repository into Downloads, installs a signed official Python 3.12 installer if necessary, creates the environment, verifies CUDA and tests, starts the app at `http://localhost:8501`, and runs dataset download, training and evaluation with console output. The script does not overwrite an unrelated existing Downloads folder. Keep the terminal open during training.

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_downloads.ps1
```

## Windows setup (RTX 4050 / 6 GB)

Install **Python 3.12 x64** from [python.org](https://www.python.org/downloads/windows/). Then, in PowerShell:

```powershell
cd "$env:USERPROFILE\Downloads"
git clone https://github.com/SameerCoder01/DL-CA2-PROJECT.git
cd DL-CA2-PROJECT
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

`setup.ps1` creates a project virtual environment and installs CUDA 12.8 PyTorch wheels. It checks `torch.cuda.is_available()`. The PyTorch wheel supplies the runtime; a compatible NVIDIA driver is required. CPU mode works with a CPU wheel but trains much more slowly. Keep Python 3.15 separate from this environment.

## Download and train

```powershell
.\.venv\Scripts\python.exe prepare_data.py --dataset --colorization
.\.venv\Scripts\python.exe model.py --epochs 60 --samples 4000 --batch-size 4
.\.venv\Scripts\python.exe evaluate.py
```

Alternatively run `train.ps1`. Allow several GB of free space for archives, extracted data, dependencies and checkpoints. Training time must be measured on the actual machine. If CUDA runs out of memory, reduce `--batch-size` to 2 or 1. Leave `--workers 0` on Windows initially.

First verify the full pipeline with a short run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe model.py --epochs 1 --samples 32 --size 128 --batch-size 2 --output runs/smoke
```

A smoke run only verifies execution; it is **not** a quality-trained model. The app only discovers `runs/restoration/best.pt`. Start/reload Streamlit after the training checkpoint is available.

Resume an interrupted run with the **same training settings**:

```powershell
.\.venv\Scripts\python.exe model.py --epochs 60 --samples 4000 --batch-size 4 --resume runs/restoration/last.pt
```

Resumption restores optimizer, scheduler and scaler; random data augmentations are not bitwise reproducible across interruptions. Configuration and source split lists are recorded in each run directory.

## Model and supervision

- Mask-conditioned residual U-Net: 4 input channels (RGB + damage mask), encoder/decoder skip connections, GroupNorm and SiLU, 3-channel RGB output.
- Clean DIV2K photographs are synthetically degraded with fading, color casts, Gaussian noise, blur, JPEG artifacts and bright/dark scratches.
- Training uses the known synthetic damage mask. Real inference uses an uploaded mask or a heuristic automatic proposal; inspect it because hair/texture may be mistaken for scratches.
- Loss: mean L1 reconstruction + 2 × masked L1 + 0.1 × gradient consistency. AdamW, cosine learning rate, gradient clipping and CUDA mixed precision.
- 800 training source images and 100 validation source images, split before cropping/corruption. Validation crops and degradation seeds are fixed; best checkpoint uses validation PSNR. Early stopping after 12 epochs without improvement.
- Overlapping 256-pixel tiles bound inference memory and preserve the selected processing resolution.

This project implements a U-Net, **not a GAN/Pix2Pix**. Starting from a supervised baseline avoids claiming a GAN that was never trained. The optional Zhang et al. model is pretrained; it is not trained by `model.py`.

## Evaluation and presentation

`evaluate.py` saves actual PSNR/SSIM values for the damaged input, OpenCV and U-Net, plus up to six comparison images in `outputs/evaluation`. The images are ordered **input | OpenCV | U-Net | clean reference**. Evaluation uses different degradation seeds and center crops, but the same held-out validation source set; do not call it an independent test set.

The app's Training tab reads real `history.csv`. For an uploaded genuine old photo, there is no clean reference and no PSNR/SSIM claim. The optional synthetic presentation demo has a known reference and labels its scores accordingly. Only report a quality improvement if actual results support it. Keep a separate folder of unseen real old photographs for a qualitative demonstration, preferably with precise masks.

Suggested viva demonstration:

1. Explain the clean → synthetic damage → U-Net → clean training pairs.
2. Show recorded training/validation curves and comparison with the input baseline.
3. Upload an unseen old photo, inspect the mask and select the trained U-Net.
4. Compare before/after, download PNG and explain optional colorization uncertainty.
5. Discuss failure cases: severe tears, missing facial features, false scratch detections and the synthetic-to-real domain gap.

## Files

| File | Responsibility |
|---|---|
| `App.py` | Streamlit interface, upload, mask preview, comparison, download, real training charts |
| `model.py` | U-Net, paired degradation, losses, training, checkpointing and inference |
| `prepare_data.py` | Official dataset and optional colorization downloads, provenance hashes |
| `evaluate.py` | Held-out synthetic metrics and image comparisons |
| `tests/test_pipeline.py` | Forward/backward, checkpoint, tiling, archive safety, frontend startup |
| `notebooks/Train_on_Colab.ipynb` | GPU training alternative with persistent checkpoints |

## Dataset, pretrained assets and references

- **DIV2K**, Agustsson & Timofte, NTIRE 2017: [official dataset and terms](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Academic research only; image copyrights remain with the original owners. Downloads use the original source, so a Kaggle token is not required. Dataset and archives are excluded from Git.
- **Colorful Image Colorization**, Zhang, Isola & Efros, ECCV 2016: [official project/code](https://github.com/richzhang/colorization/tree/caffe). Optional Caffe assets are downloaded from the authors' repository/site. Their original license and terms apply; no weights are bundled.
- **U-Net**, Ronneberger et al., 2015: [paper](https://arxiv.org/abs/1505.04597). The compact residual adaptation here is implemented in `model.py`.
- [PyTorch installation](https://pytorch.org/get-started/locally/) and [OpenCV inpainting](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html).

Download hashes are recorded for provenance, not compared with a publisher-signed checksum. If the authors move a colorization URL, obtain the asset from their official project and put it in `weights/colorization/`.

## Practical limits

Synthetic training does not guarantee successful restoration of every historical photo. Colorization predicts plausible colors and cannot establish original colors. Automatic damage masks can remove real detail. Uploads remain in memory; the application does not write them to disk. Inference is local. Large upload dimensions and processing resolution are limited to bound memory use.
