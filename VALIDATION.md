# Validation - 26 October 2026

## Real uploaded portrait: faded-grey scratch detection fix

The user's `images.jpg` portrait exposed a real failure: the automatic mask covered 0% of the image, leaving the prominent right-side diagonal and vertical creases untouched. Absolute bright/dark thresholds did not cover faded-grey damage in this scan. A monochrome-only fallback now selects elongated local-contrast ridges when the original detector selects less than 0.5% of the image. Short interior highlights are excluded to avoid selecting features such as the mouth. Border-connected damage can use a shorter length threshold.

The final mask covers 7.66% of this portrait. Visual inspection confirmed removal of the large right-side cracks and some border marks, with facial features retained; small facial spots, faint scratches and damaged border remnants remain. No clean reference exists for this portrait, so no PSNR/SSIM improvement is claimed. The exact downloaded app PNG matches the locally inspected result. Local evidence is in `outputs/portrait_review` (input, old output, new output, masks, comparison and app screenshot); the portrait is excluded from Git.

Eleven functional tests passed, including a faded-grey crease below the previous white threshold and a short highlight that must remain untouched. Rechecking the same 100-photo mixed benchmark gave exactly unchanged PSNR 21.1437, SSIM 0.813252 and scratch MAE 36.2566. Category checks remain improved over their inputs; faded B&W now gives PSNR 25.40 and SSIM 0.9547 (earlier values below predate this fallback). Model weights remain the verified epoch-10 GAN; this change repairs detection, without claiming extra training. App result caching now also depends on the pipeline source modification time so updated code does not display the previous cached result.

## Current active model: epoch 10

Terminal access and CUDA RTX 4050 were verified in the Documents project. Resumed the existing epoch-6 G/D and optimizers for four additional epochs (800 samples/epoch, batch 4, 256 crops). Scratch pixels get 9x normalized L1 weight plus 0.15 edge L1; 30% paired grayscale and 25% estimated-mask inputs improve task coverage. Actual optimizer LR was 0.0002 inherited from the checkpoint; the initially requested 0.0001 did not take effect because the previous resume code restored optimizer LR. Metadata now records both values, and resume code applies requested LR for future runs.

Training-branch best is epoch 10. Its fixed validation PSNR is 23.19 dB, SSIM 0.8198. Validation augmentation changed, so that score must not be directly compared with the old epoch-6 selection score.

The fair comparison below uses identical 100 validation center crops, seeds 191000-191099, automatic masks, including 34 grayscale cases:

| Metric | Old app + epoch 6 | Current app + epoch 10 |
|---|---:|---:|
| PSNR (dB) | 21.1196 | 21.1437 |
| SSIM | 0.800816 | 0.813252 |
| Scratch MAE (0-255; lower better) | 45.5213 | 36.2566 |
| Mask recall | 0.7927 | 0.8696 |
| Mask precision | 0.2930 | 0.4720 |

Scratch MAE decreased 20.35%; overall PSNR is essentially unchanged. A weights-only change with the old pipeline gave PSNR 21.3335, SSIM 0.807727, scratch MAE 43.2653. Fixed 0.9 GAN strength gave mixed-damage PSNR 22.0740, SSIM 0.824957 and scratch MAE 34.4355, but reduced SSIM on scratch-only and damage-only photos. Final adaptive strength trades that mixed-fade score for preservation of intact texture across all required categories. Broader brightness thresholds initially regressed on the remaining 80 images and were rejected. Final masks retain conservative brightness thresholds, narrower contrast-supported Hough masks and a short isolated-line pass.

Separate category checks on 13 validation sources each (`category_review.py`, `outputs/category_review/metrics.json`), input -> restored:

| Category | PSNR (dB) | SSIM |
|---|---|---|
| Faded | 21.13 -> 24.48 | 0.8925 -> 0.9535 |
| Scratched | 21.52 -> 28.84 | 0.9388 -> 0.9597 |
| Damaged spots + scratch | 23.06 -> 29.69 | 0.9651 -> 0.9682 |
| Faded B&W | 21.74 -> 25.74 | 0.9038 -> 0.9631 |

All 13 B&W outputs have identical RGB channels. Category settings are distinct from the mixed synthetic corruption; these remain small validation checks, not independent testing. Colorization is optional and is not enabled in the simple app.

Evidence: `outputs/quality_baseline`, `outputs/quality_confirm_old`, `outputs/quality_weights_only`, and `outputs/quality_final/metrics.json` + `promotion.json`. Actual comparison images include grayscale penguin `0801_comparison.png` and the existing `local_demo_comparison.png`; both were visually inspected. Many scratches disappear, but some fine marks and smooth/incorrect large-area fills remain. This is synthetic damage on validation sources also used for checkpoint selection and detector tuning, not an independent test or real historical-photo benchmark.

Epoch-6 active files are backed up at `runs/pix2pix-backup-6-quality`. Epoch-10 best and resumable last are promoted at `runs/pix2pix`. Ten functional tests pass, including an identity-generator regression that checks white damage is not blended back into repaired pixels. The app's actual side-panel upload, before/after and download flow was exercised; the downloaded PNG exactly matches the evaluated local-demo result. Earlier sections below are historical results.

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
