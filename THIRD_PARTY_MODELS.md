# Pretrained components and locally trained GAN

The selected app uses genuine neural models; downloaded weights are not claimed as locally trained.

| Component | Role | Source and licence |
|---|---|---|
| Microsoft old-photo U-Net | Predicts damaged/scratched pixels from grayscale context | [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life), MIT. Adapted inference architecture is in `restoration_backends`; original notice is retained. Single-GPU BatchNorm replaces the unused DataParallel wrapper. |
| Pretrained LaMa generator | Context-aware filling of selected damage; GAN/adversarial training, with perceptual losses | [LaMa authors](https://github.com/advimman/lama), [paper](https://arxiv.org/abs/2109.07161). TorchScript export distributed by [IOPaint model releases](https://github.com/Sanster/models/releases/tag/add_big_lama); pinned MD5 `e3aa4aaa15225a33ec84f9f4bc47e500`. Original model Apache-2.0; training data terms remain applicable. |
| Local Pix2Pix-style GAN | Optional tone correction when repaired tonal range is low; previous restoration baseline is retained | This project's `model.py`, actually trained through epoch 40 on original DIV2K training sources. Both G/D and optimizer checkpoints are preserved. |

The Microsoft generator/detector architecture port comes from the reviewed `Global/detection_models/networks.py` and `antialiasing.py`; only detection classes are retained. No torchvision, dlib, CUDA extension build, or external service is required for the selected backend. PyTorch and OpenCV use the existing local environment. LaMa inference is local and uploads no photographs.

`setup_repair_models.py` reproduces installation, validates detector SHA256 and LaMa MD5, then writes the accepted selection config in `weights/learned_pipeline.json`. Model files, source archives and user photos remain outside Git. Primary model SHA256 values:

- Detector: `b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95`.
- LaMa TorchScript: `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`.

The user accepted the learned detector + LaMa portrait result. That result is retained for normal-contrast monochrome photographs; the local Pix2Pix model runs only for low-contrast tone correction. Color photographs additionally use conservative geometric proposals for straight synthetic scratches. LaMa changes only selected pixels before optional tone correction; B&W outputs remain achromatic.

CelebA was researched/downloaded, but **no CelebA-trained checkpoint was saved or deployed**. A 64-image timing probe updated a temporary in-memory model and was discarded; the planned full training job did not start. The user canceled it; the downloader/preparation workflow was removed and downloaded dataset files moved out of the project to Recycle Bin because permanent deletion was blocked by the execution approval review. No GFPGAN face enhancement is enabled. Remaining defects or missing original detail are not guaranteed to be reconstructed faithfully; a 100% scratch-removal guarantee is not made.
