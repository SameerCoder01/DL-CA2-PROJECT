# Reframe - Old Photograph Restoration with Pix2Pix GAN

Streamlit frontend: `App.py`. PyTorch Generator, Discriminator and training: `model.py`.

## Model and training
- A compact residual U-Net **Generator** restores damaged RGB + a damage mask.
- A conditional **70x70 PatchGAN Discriminator** judges real clean and generated photos.
- D loss = average binary cross-entropy for real=1 and fake=0.
- G loss = adversarial binary cross-entropy + 100 x L1 reconstruction.
- Training alternates one D update and one G update using Adam, ordinary backward() and step().
- Float32 keeps the code explainable. No mixed precision, perceptual network or learning-rate scheduler.
- Only G is used at inference. The compact Generator is a Pix2Pix-style adaptation, not the original eight-level architecture.

Read [VIVA_GUIDE.md](VIVA_GUIDE.md) for the code explanation.

## Status
GAN implementation is available. Full dataset GAN training and quality evaluation remain pending until an actual run completes. Old supervised U-Net weights are rejected. The app uses the labeled OpenCV baseline until `runs/pix2pix/best.pt` exists.

Validation: eight local functional tests passed, including adversarial-only updates of both networks, checkpoint resume and Streamlit startup. These tests establish execution, not restoration quality.

## Run on Windows
Keep the existing environment: GAN uses the same installed libraries. Only a fresh environment needs `setup.ps1`.

```powershell
cd "$env:USERPROFILE\Downloads\DL-CA2-PROJECT"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe prepare_data.py --dataset
.\.venv\Scripts\python.exe model.py --epochs 60 --samples 4000 --batch-size 4
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m streamlit run App.py
```

If RTX 4050 6GB runs out of memory, use batch size 2 or 1. Runtime must be measured; 60 epochs is a starting configuration, not a quality guarantee. A short smoke run can use `--epochs 1 --samples 32 --size 128 --batch-size 2 --output runs/gan-smoke`.

Resume with `--resume runs/pix2pix/last.pt` and unchanged settings. Last checkpoint includes both networks and both optimizers; best checkpoint contains G selected by validation PSNR. Old `runs/restoration` output is preserved and is never used as GAN weights.

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
