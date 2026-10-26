# Full-dataset GAN continuation — 26 October 2026

The active model is a genuinely trained Pix2Pix-style GAN at **epoch 40**, continued from the existing epoch-10 Generator, Discriminator and Adam optimizers. Thirty additional complete epochs used the RTX 4050 Laptop GPU.

## Dataset coverage

- Every completed epoch uses all **800 original `DIV2K_train_HR` source photos once**, shuffled, with one random 256×256 crop per source. This is full source coverage, not every pixel of every HR image in one epoch.
- All **100 `DIV2K_valid_HR` sources** remain outside training. Their fixed crops/corruptions support checkpoint selection; they are validation, not an independent test set.
- `data/fast` contains resized clean duplicates, not defective LR/HR pairs. It is used for consistent old/new evaluation, not this continuation's training. ZIP archives are not extra training examples. No separate defective LR or test split was found in the available folders.
- Thirty completed extra epochs give 24,000 additional source crops. Partial batches interrupted during loader/phase transitions are not counted as completed epochs.

## Training phases

| Epochs | Changes | Best checkpoint |
|---|---|---|
| 11–20 | Original HR sources; diverse grey/transparent bent creases, peeling spots, fade-only, scratch-only, mixed and intact examples; 30% paired grayscale | 19 (known-mask validation) |
| 21–40 | 75% estimated training masks; true synthetic damage masks retained separately for weighted loss; checkpoint selection uses predicted masks | 40 (automatic-mask validation) |

G remains a residual U-Net and D a conditional PatchGAN. G loss is adversarial BCE + 100 × (normalized scratch-weighted L1 + 0.15 edge L1). Actual damaged pixels receive 9× weight even when the estimated condition mask misses them. Both networks train; inference uses G plus separately described automatic OpenCV repair. Learning rate was 0.0001 in both phases. Batch size 4, 200 batches/epoch, float32, two persistent Windows data-loader workers. Target 40 was reached without triggering early stopping.

Source snapshots, configs, splits, starting metrics, epoch CSVs, both-network checkpoints and logs are preserved in `runs/pix2pix-long-realistic` and `runs/pix2pix-long-auto`. The Windows launcher now imports model classes under a main guard so worker processes do not restart the training job.

## Matched old/new results

| Check | Epoch 10 | Epoch 40 |
|---|---:|---:|
| Same 100 HR validation crops, predicted masks: G PSNR | 22.6726 | 25.8097 |
| Same HR validation: G SSIM | 0.861454 | 0.894448 |
| New crease curriculum, 88 damaged evaluation crops: app PSNR | 22.3235 | 22.6212 |
| Same damaged crops: app SSIM | 0.881775 | 0.883969 |
| Crease-region MAE, 0–255 (lower better) | 69.4480 | 56.2307 |
| Legacy mixed-damage 100-crop app PSNR | 21.1437 | 21.1409 |
| Legacy mixed-damage app SSIM | 0.813252 | 0.809188 |
| Legacy scratch-region MAE | 36.2566 | 34.7241 |

Crease-region error decreased **19.0%**. The legacy benchmark has essentially unchanged PSNR and a small SSIM regression; improvements are not uniform. On 12 intact curriculum examples, output PSNR improved 33.66 → 34.43 and SSIM 0.97842 → 0.97888. The identity inputs themselves have perfect-reference PSNR, so overall input averages mixing intact/damaged examples are misleading; damaged-only rows above are used for restoration claims.

Separate 13-source category checks, epoch-10 → epoch-40 restored output:

| Category | PSNR | SSIM |
|---|---|---|
| Faded | 24.48 → 25.77 | 0.9533 → 0.9580 |
| Scratched | 28.84 → 29.01 | 0.9597 → 0.9602 |
| Damaged spots + scratch | 29.69 → 29.85 | 0.9682 → 0.9687 |
| Faded B&W | 25.40 → 25.86 | 0.9547 → 0.9575 |

These are synthetic validation checks. Sources were also used for checkpoint selection and earlier detector tuning; they are not independent historical-photo accuracy evidence. DIV2K provides no paired historical portraits here.

## Actual portrait and deployment

The user's uploaded portrait was compared at original, epoch 10 and epoch 40. The large right-side cracks remain repaired, with less aggressive tonal correction than epoch 10. Small facial spots, fine marks, smooth fills and damaged border remnants remain. There is no clean reference for this portrait, so no numerical historical-fidelity claim is made.

Epoch-40 best and resumable last are active at `runs/pix2pix/best.pt` and `last.pt`. Epoch-10 files are backed up in `runs/pix2pix-backup-10-long`. The current last checkpoint contains both networks and both optimizer states. The side-panel app was rerun with the existing upload, and the downloaded PNG exactly matched the evaluated epoch-40 portrait. UI remains upload → automatic restore → before/after → download; colorization remains optional and disabled.

Evidence: `outputs/dataset_audit.json`, `outputs/long_training_report`, `outputs/long_final_legacy`, `outputs/long_final_realistic`, and `outputs/long_final_categories`. Private images, datasets and weights remain excluded from Git. `training_report.py` regenerates a phase summary; `quality_review.py --realistic` reports damaged/intact cases separately.
