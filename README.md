# Reframe - Old Photograph Restoration with Pix2Pix GAN

Streamlit frontend: `App.py`. PyTorch Generator, Discriminator and training: `model.py`.

The user screen is deliberately simple: **upload -> automatic restoration -> original/restored photos -> one PNG download button**. There is no sidebar, technical settings, demo switch, mask upload or training dashboard in the customer-facing app. Scratch detection and processing size are chosen internally. Black-and-white photos keep their original color style.

## Model and training
- A compact residual U-Net **Generator** restores damaged RGB + a damage mask.
- A conditional **70x70 PatchGAN Discriminator** judges real clean and generated photos.
- D loss = average binary cross-entropy for real=1 and fake=0.
- G loss = adversarial binary cross-entropy + 100 x (scratch-weighted L1 + 0.15 x edge L1). Scratch pixels receive 9x weight, normalized by total weight.
- Training alternates one D update and one G update using Adam, ordinary backward() and step().
- Float32 keeps the code explainable. No mixed precision, perceptual network or learning-rate scheduler.
- Only G is used at inference. The compact Generator is a Pix2Pix-style adaptation, not the original eight-level architecture.
- App output combines GAN tone/detail restoration with narrow-mask OpenCV interpolation. Fills use original local context plus GAN tone correction estimated only from intact pixels. The damaged input is never blended back into a selected scratch.

Read [VIVA_GUIDE.md](VIVA_GUIDE.md) for the code explanation.

## Status
Training reached epoch 10 by resuming the existing epoch-6 GAN for four additional epochs. Verified weights are active at `runs/pix2pix/best.pt`; resumable G/D and optimizer states are in `last.pt`. The epoch-6 run is backed up in `runs/pix2pix-backup-6-quality`. Automatic-mask evaluation on 100 validation crops gives PSNR 21.12 -> 21.14 dB, SSIM 0.801 -> 0.813, and scratch MAE 45.52 -> 36.26. Automatic GAN strength protects intact texture in photos with healthy contrast. Separate faded, scratched, damaged and B&W checks improved both average PSNR and SSIM in every category. These are synthetic validation results, not independent historical-photo evidence. See [VALIDATION.md](VALIDATION.md).

Validation: 10 local functional tests passed, plus actual side-panel upload, before/after and PNG download checks. Tests establish execution; exported photo comparisons establish the limited quality evidence.

## Run on Windows
Keep the existing environment: GAN uses the same installed libraries. Only a fresh environment needs `setup.ps1`.

```powershell
cd "C:\Users\samee\Documents\Codex\2026-10-26\5-old-photograph-restoration-application-restore-4\DL-CA2-PROJECT"
& ..\work\python312\python.exe serve_here.py
# Optional continuation (existing data is already prepared):
& ..\work\python312\python.exe train_here.py --resume runs/pix2pix/last.pt --epochs 14 --samples 800 --batch-size 4 --train-dir data/fast/train --val-dir data/fast/valid --output runs/pix2pix-next --lr 0.0001
```

If RTX 4050 6GB runs out of memory, use batch size 2 or 1. The 3-epoch run uses smaller clean copies for a quick classroom demo; it is not a fully converged model. A tiny execution smoke test can use `--epochs 1 --samples 32 --size 128 --batch-size 2 --output runs/gan-smoke`.

Resume with `--resume runs/pix2pix/last.pt`; `--epochs` is the total target epoch, not the number of extra epochs. Both optimizers resume, then the requested learning rate is applied. Use a new output branch for changed training settings. Launchers reuse installed packages from the old Downloads environment, but import project code from this Documents folder. Old `runs/restoration` output is never used as GAN weights.

## Data, evaluation and limitations
DIV2K supplies 800 training and 100 validation source images. Source splitting happens before cropping/corruption. Synthetic fading, noise, blur, JPEG artifacts and scratches create damaged/clean pairs. Data and weights are excluded from Git.

Training CSV records G loss, D loss, adversarial loss, reconstruction, PSNR and SSIM. Evaluation compares input, OpenCV and GAN with a known clean reference using new synthetic corruption seeds on the validation sources. It is not an independent real-world test set.

Automatic masks can mistake texture for damage. Severe tears and missing faces remain difficult; GANs can invent plausible but incorrect details. Uploads are processed locally in memory. Optional ECCV16 colorization (`python prepare_data.py --colorization`) is a separate pretrained stage; original colors are not guaranteed.

## Files and references
`App.py`: UI; `model.py`: GAN; `prepare_data.py`: downloads; `evaluate.py`: measured comparisons; `tests/test_pipeline.py`: functional checks; `notebooks/Train_on_Colab.ipynb`: optional GPU training.

- [Pix2Pix paper](https://arxiv.org/abs/1611.07004) and [authors' implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- [U-Net paper](https://arxiv.org/abs/1505.04597).
- [DIV2K official dataset and terms](https://data.vision.ee.ethz.ch/cvl/DIV2K/): academic research only; original image copyrights remain with owners.
- [ECCV16 colorization](https://github.com/richzhang/colorization/tree/caffe): optional pretrained assets retain authors' terms.
