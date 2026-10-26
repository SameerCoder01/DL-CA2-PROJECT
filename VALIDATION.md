# Validation - 26 October 2026

- Python 3.12 syntax checks passed.
- Eight local functional tests passed in 9.62 seconds on CPU.
- Tests cover adversarial-only updates of both networks, PatchGAN output dimensions, rejection of old supervised weights, GAN save/load/resume, tiled inference, safe extraction and Streamlit startup.
- One actual CUDA GAN training step passed on NVIDIA GeForce RTX 4050 Laptop GPU using the full base-32 Generator and Discriminator with random 128x128 tensors.
- CUDA step losses were finite: G 35.4401, D 0.7311; execution time after setup was approximately 2.31 seconds.

The random-tensor smoke test above is not quality training evidence.

## Completed short dataset training

- Completed 3 epochs, 800 training samples per epoch, batch size 4, 256-pixel crops.
- All 800 training source photos and 100 validation source photos were retained in separate splits. To shorten runtime, clean copies were resized to a maximum side of 512 pixels before cropping.
- Best checkpoint: epoch 3. Validation PSNR improved from 18.23 dB (damaged input) to 20.99 dB (Generator).
- This is a short classroom demonstration run, not a fully converged or production-quality model.

## Additional synthetic evaluation: 20 validation source photos

| Method | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Damaged input | 19.04 | 0.768 |
| OpenCV baseline | 21.23 | 0.820 |
| GAN Generator only | 21.97 | 0.800 |
| GAN + OpenCV selected-mask cleanup | 22.59 | 0.816 |

The combined method improved average PSNR, but OpenCV had the highest average SSIM on this subset. Improvement is not uniform across images. These results use synthetic damage, known masks and clean references on the validation sources; they are not independent real-world test results. The app uses the GAN with OpenCV mask cleanup because the short GAN run leaves some thin scratches. Real user-uploaded photos with automatic masks may perform differently.

## Three additional epochs and selected model

Continued from the saved epoch-3 G/D networks and optimizers, with 800 patches per epoch and a new training augmentation seed. The best checkpoint was epoch 6. Fixed validation PSNR improved from 20.99 to 22.16 dB and SSIM from 0.7834 to 0.7968.

On the same 20 synthetic evaluation photos:

| Method | Previous PSNR | New PSNR | Previous SSIM | New SSIM |
|---|---:|---:|---:|---:|
| GAN only | 21.97 | 22.54 | 0.8000 | 0.8046 |
| GAN + known-mask cleanup | 22.59 | 23.15 | 0.8161 | 0.8181 |

Both selected measures improved, so the epoch-6 checkpoint was promoted. The epoch-3 checkpoint and evaluation are retained in `runs/pix2pix-backup-3`. These scores use known synthetic masks; automatic-mask performance on real photos can differ.

The simplified frontend passed startup, simulated upload, automatic model restoration, one valid PNG download output and cached-rerun checks. It exposes no settings selectors, sliders, toggles or radio controls.

All 9 pipeline tests passed after the automatic-mask update. The detector proposes high-contrast straight lines and small white spots; the frontend limits mask coverage to reduce false positives. Visual inspection of the simulated upload confirmed removal of many white marks, with visible smearing around some repaired areas and a few missed scratches. It does not reconstruct severe missing detail reliably. Browser interaction was not automated; frontend verification used Streamlit AppTest.
