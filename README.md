# Reframe - Old Photograph Restoration with Pix2Pix GAN

Streamlit frontend: `App.py`. PyTorch Generator, Discriminator and training: `model.py`.

## Final selected repair backend

The accepted portrait result uses **pretrained learned scratch detection + LaMa GAN inpainting**. The locally trained epoch-40 Pix2Pix remains available for low-contrast tone correction and as the previous baseline. No CelebA-trained checkpoint is required or deployed, and no face-beautification GAN is enabled. See [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md) for roles, licences and the distinction between downloaded and locally trained weights.

Install/reinstall the accepted model files with `..\work\python312\python.exe setup_repair_models.py`, then run `serve_here.py` as below. Setup uses standard Python only; the existing app environment supplies PyTorch/OpenCV. Source archives/model weights are excluded from Git. The app also supports the pretrained path when a local Pix2Pix checkpoint is absent; optional local tone correction then remains disabled. If the accepted pretrained files are absent, the existing local Pix2Pix + OpenCV baseline remains available.

The user screen is deliberately simple: **upload -> automatic restoration -> original/restored photos -> one PNG download button**. There is no sidebar, technical settings, demo switch, mask upload or training dashboard in the customer-facing app. Scratch detection and processing size are chosen internally. Black-and-white photos keep their original color style.

The interface uses a responsive dark plum and lavender design, a dedicated upload card, framed before/after photographs and a clear PNG download action. The visual refresh does not change the restoration models or processing.

## Model and training
- A compact residual U-Net **Generator** restores damaged RGB + a damage mask.
- A conditional **70x70 PatchGAN Discriminator** judges real clean and generated photos.
- D loss = average binary cross-entropy for real=1 and fake=0.
- G loss = adversarial binary cross-entropy + 100 x (scratch-weighted L1 + 0.15 x edge L1). Scratch pixels receive 9x weight, normalized by total weight.
- Training alternates one D update and one G update using Adam , ordinary backward() and step().
- Float32 keeps the code explainable. No mixed precision, perceptual network or learning-rate scheduler.
- Only G is used at inference. The compact Generator is a Pix2Pix-style adaptation, not the original eight-level architecture.
- Selected app output uses learned damage segmentation and LaMa inpainting; local Pix2Pix tone correction is conditional on low contrast. The earlier Pix2Pix + OpenCV interpolation is retained as an explicit legacy backend for comparisons.

Read [VIVA_GUIDE.md](VIVA_GUIDE.md) for the code explanation.

## Status
Training reached **epoch 40** after 30 extra epochs on all 800 original HR training sources using the RTX 4050. Each source contributes a random 256×256 crop per epoch; 100 validation sources remain outside training. Epoch-40 best and resumable G/D state are active in `runs/pix2pix`; epoch 10 is backed up in `runs/pix2pix-backup-10-long`. Matched automatic-mask HR validation improved G PSNR 22.67 -> 25.81 dB and SSIM 0.8615 -> 0.8944. New grey-crease evaluation reduced scratch MAE 19%; separate category scores improved. Legacy mixed-damage PSNR is unchanged and SSIM slightly lower. See [TRAINING_REPORT.md](TRAINING_REPORT.md) for exact comparisons and limits.

Validation: 13 local functional tests pass, plus actual side-panel before/after and PNG download checks. The selected backend reproduces the user's accepted portrait (including additional facial mark removal); repeated CUDA runs differ by at most one intensity level in the final PNG. Tests establish execution; exported photo comparisons establish limited quality evidence, not perfect reconstruction.

## Run on Windows
Keep the existing environment: GAN uses the same installed libraries. Only a fresh environment needs `setup.ps1`.

```powershell
cd "C:\Users\samee\Documents\Codex\2026-10-26\5-old-photograph-restoration-application-restore-4\DL-CA2-PROJECT"
& ..\work\python312\python.exe serve_here.py
# Optional continuation (existing data is already prepared):
& ..\work\python312\python.exe training_job.py --resume runs/pix2pix/last.pt --epochs 60 --samples 800 --batch-size 4 --workers 2 --train-dir data/DIV2K_train_HR --val-dir data/DIV2K_valid_HR --output runs/pix2pix-next --lr 0.0001 --realistic-damage --automatic-mask-prob 0.75 --automatic-validation
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
