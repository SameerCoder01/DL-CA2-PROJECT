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
