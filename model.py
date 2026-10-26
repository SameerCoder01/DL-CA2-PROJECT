"""Train and run a mask-conditioned residual U-Net for photograph restoration.

No pretrained restoration weights are bundled. OpenCV mode is a separate baseline.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_rgb(source) -> np.ndarray:
    with Image.open(source) as image:
        if image.width * image.height > 36_000_000:
            raise ValueError("Image exceeds 36 megapixels. Resize it before uploading.")
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def damage(clean: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic paired supervision: fade, noise, blur, JPEG and masked scratches.

    Source photos are split BEFORE degradation. A seed fixes validation corruption.
    Grayscale conversion is deliberately excluded; colorization is a separate task.
    """
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    x = clean.astype(np.float32) / 255
    x = x * rng.uniform(.6, 1.) + rng.uniform(.0, .18)
    x *= rng.uniform(.91, 1.09, size=(1, 1, 3))
    if rng.random() < .6:
        x = cv2.GaussianBlur(x, (3, 3), float(rng.uniform(.2, .9)))
    x += rng.normal(0, rng.uniform(.005, .035), x.shape)
    x = np.uint8(np.clip(x, 0, 1) * 255)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 96))])
    if ok:
        x = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(2, 14))):
        start = (int(rng.integers(w)), int(rng.integers(h)))
        end = (int(np.clip(start[0] + rng.integers(-w//3, w//3+1), 0, w-1)),
               int(np.clip(start[1] + rng.integers(-h//2, h//2+1), 0, h-1)))
        cv2.line(mask, start, end, 255, int(rng.integers(1, 4)))
    if rng.random() < .3:
        p = (int(rng.integers(w)), int(rng.integers(h)))
        cv2.circle(mask, p, int(rng.integers(2, max(3, min(h,w)//16))), 255, -1)
    x[mask > 0] = rng.choice([15, 235, 255])
    return x, mask


def detect_scratches(image: np.ndarray, sensitivity: int = 22) -> np.ndarray:
    """Heuristic proposal only: thin bright/dark structures can also be real detail."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    candidate = np.uint8(np.maximum(bright, dark) > sensitivity) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    mask = np.zeros_like(gray)
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        if area >= 10 and max(w, h) / max(1, min(w, h)) > 3:
            mask[labels == index] = 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def classical_restore(image: np.ndarray, mask: np.ndarray, strength: float = .65) -> np.ndarray:
    repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) if mask.any() else image.copy()
    bgr = cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR)
    denoised = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, 4, 4, 7, 21), cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return cv2.addWeighted(enhanced, strength, repaired, 1-strength, 0)


class Block(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(inp, out, 3, padding=1),
            nn.GroupNorm(8, out), nn.SiLU(), nn.Conv2d(out, out, 3, padding=1),
            nn.GroupNorm(8, out), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class RestorationUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1, self.e2 = Block(4, base), Block(base, base*2)
        self.e3, self.mid = Block(base*2, base*4), Block(base*4, base*8)
        self.d3, self.d2, self.d1 = Block(base*12, base*4), Block(base*6, base*2), Block(base*3, base)
        self.head = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        z = self.mid(F.avg_pool2d(c, 2))
        def up(z, skip, block):
            return block(torch.cat([F.interpolate(z, size=skip.shape[-2:],
                                                  mode="bilinear", align_corners=False), skip], 1))
        z = up(up(up(z, c, self.d3), b, self.d2), a, self.d1)
        return (x[:, :3] + self.head(z)).clamp(0, 1)


def tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255


class PhotoPairs(Dataset):
    def __init__(self, paths, size=256, samples=None, validation=False):
        self.paths, self.size = paths, size
        self.samples, self.validation = samples or len(paths), validation

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = random.Random(71000 + index) if self.validation else random
        path = self.paths[index % len(self.paths)]
        image = read_rgb(path)
        h, w = image.shape[:2]
        scale = max(1., self.size / min(h, w))
        if scale > 1:
            image = cv2.resize(image, (math.ceil(w*scale), math.ceil(h*scale)))
        h, w = image.shape[:2]
        y, x = rng.randrange(h-self.size+1), rng.randrange(w-self.size+1)
        clean = image[y:y+self.size, x:x+self.size].copy()
        if not self.validation:
            if rng.random() < .5:
                clean = np.fliplr(clean).copy()
            clean = np.rot90(clean, rng.randrange(4)).copy()
        degraded, mask = damage(clean, rng.randrange(2**31))
        return torch.cat([tensor(degraded), torch.from_numpy(mask.copy()).unsqueeze(0).float()/255]), tensor(clean)


def restoration_loss(pred, target, mask):
    error = (pred-target).abs()
    reconstruction = error.mean()
    masked = (error*mask).sum() / (3*mask.sum()).clamp_min(1)
    edge = ((pred[:, :, 1:, :] - pred[:, :, :-1, :]) -
            (target[:, :, 1:, :] - target[:, :, :-1, :])).abs().mean()
    edge += ((pred[:, :, :, 1:] - pred[:, :, :, :-1]) -
             (target[:, :, :, 1:] - target[:, :, :, :-1])).abs().mean()
    return reconstruction + 2*masked + .1*edge


def metric_pair(pred, target):
    mse = np.mean((pred.astype(np.float64)/255-target.astype(np.float64)/255)**2)
    psnr = float(-10*np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(target, pred, channel_axis=2, data_range=255))
    return psnr, ssim


@torch.inference_mode()
def validate(net, loader, device):
    net.eval()
    rows = []
    for inp, target in loader:
        pred = net(inp.to(device)).cpu()
        for p, t, source in zip(pred, target, inp):
            arrays = [np.uint8(v.permute(1, 2, 0).numpy().clip(0, 1)*255) for v in [p, t, source[:3]]]
            rows.append((*metric_pair(arrays[0], arrays[1]), *metric_pair(arrays[2], arrays[1])))
    return dict(zip(["psnr", "ssim", "input_psnr", "input_ssim"], np.mean(rows, axis=0).tolist()))


def image_paths(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in EXTENSIONS)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files, val_files = image_paths(args.train_dir), image_paths(args.val_dir)
    if not train_files or not val_files:
        raise ValueError("Training and validation images are required. Run prepare_data.py first.")
    if set(p.resolve() for p in train_files) & set(p.resolve() for p in val_files):
        raise ValueError("Training and validation sets must be disjoint.")
    if len({p.name for p in train_files} & {p.name for p in val_files}) > 0:
        raise ValueError("Duplicate filenames across splits: check for leakage.")
    train_set = PhotoPairs(train_files, args.size, args.samples)
    val_set = PhotoPairs(val_files, args.size, validation=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.workers)
    net = RestorationUNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start, best, stale = 0, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        print("Resume restores weights and optimizer; use the original data and schedule settings.", flush=True)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"], checkpoint["best_psnr"]
    (out/"config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out/"split.json").write_text(json.dumps({"train": [str(p) for p in train_files],
        "validation": [str(p) for p in val_files]}, indent=2), encoding="utf-8")
    print(f"Device: {device}; {len(train_files)} train / {len(val_files)} validation images", flush=True)
    for epoch in range(start, args.epochs):
        net.train()
        total, seen = 0., 0
        for step, (inp, target) in enumerate(train_loader):
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = net(inp)
                loss = restoration_loss(pred, target, inp[:, 3:4])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; checkpoint was not overwritten.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()*len(inp)
            seen += len(inp)
            if step % 25 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, batch {step+1}/{len(train_loader)}, loss {loss.item():.4f}", flush=True)
        schedule.step()
        metrics = validate(net, val_loader, device)
        improved = metrics["psnr"] > best
        best, stale = max(best, metrics["psnr"]), 0 if improved else stale+1
        row = {"epoch": epoch+1, "loss": total/seen, **metrics}
        log = out/"history.csv"
        with log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {"format_version": 1, "state_dict": net.state_dict(), "base": 32,
            "epoch": epoch+1, "best_psnr": best, "metrics": metrics, "config": vars(args),
            "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(), "scaler": scaler.state_dict()}
        temporary = out/"pending.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(out/"last.pt")
        if improved:
            torch.save({k: v for k, v in checkpoint.items() if k not in {"optimizer", "scheduler", "scaler"}}, out/"best.pt")
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            print("Early stopping.", flush=True)
            break
    print(f"Best validation PSNR: {best:.3f} dB. Check improvement over the input baseline before presenting.", flush=True)


def load_restorer(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    net = RestorationUNet(checkpoint.get("base", 32)).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint


@torch.inference_mode()
def neural_restore(net, image, mask, device="cpu", tile=256, overlap=48):
    """Overlap-add inference preserves full resolution and bounds VRAM use."""
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, tile-h), max(0, tile-w)
    padded = np.pad(image, ((0,pad_h),(0,pad_w),(0,0)), mode="edge")
    padded_mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode="constant")
    ph, pw = padded.shape[:2]
    def starts(length):
        return sorted(set(list(range(0, length-tile+1, tile-overlap)) + [length-tile]))
    result, weight = np.zeros_like(padded, dtype=np.float32), np.zeros((ph,pw,1), np.float32)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), .01)[:, :, None].astype(np.float32)
    for y in starts(ph):
        for x in starts(pw):
            inp = torch.cat([tensor(padded[y:y+tile,x:x+tile]),
                torch.from_numpy(padded_mask[y:y+tile,x:x+tile].copy()).unsqueeze(0).float()/255])
            pred = net(inp.unsqueeze(0).to(device))[0].cpu().permute(1,2,0).numpy()
            result[y:y+tile,x:x+tile] += pred*window
            weight[y:y+tile,x:x+tile] += window
    return np.uint8(np.clip(result[:h,:w]/weight[:h,:w], 0, 1)*255)


def colorize(image, directory):
    """Zhang et al. ECCV16 pretrained Caffe network, run with OpenCV DNN."""
    directory = Path(directory)
    net = cv2.dnn.readNetFromCaffe(str(directory/"colorization_deploy_v2.prototxt"),
                                 str(directory/"colorization_release_v2.caffemodel"))
    points = np.load(directory/"pts_in_hull.npy", allow_pickle=False)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.T.reshape(2,313,1,1).astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1,313), 2.606, np.float32)]
    lab = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    small = cv2.resize(lab[:,:,0], (224,224)) - 50
    net.setInput(cv2.dnn.blobFromImage(small))
    ab = net.forward()[0].transpose(1,2,0)
    ab = cv2.resize(ab, (image.shape[1],image.shape[0]))
    out = cv2.cvtColor(np.concatenate([lab[:,:,:1],ab], 2), cv2.COLOR_LAB2RGB)
    return np.uint8(np.clip(out,0,1)*255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(ROOT/"data/DIV2K_train_HR"))
    parser.add_argument("--val-dir", default=str(ROOT/"data/DIV2K_valid_HR"))
    parser.add_argument("--output", default=str(ROOT/"runs/restoration"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.size < 32 or args.size % 8 or min(args.epochs,args.samples,args.batch_size,args.patience) < 1:
        parser.error("size must be >=32 and divisible by 8; training counts must be positive")
    train(args)
