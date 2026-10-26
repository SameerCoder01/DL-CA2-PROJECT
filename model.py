"""Train and run a Pix2Pix conditional GAN for photograph restoration.

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
    """Conservative automatic proposal: thin straight scratches and bright spots.

    Hough lines handle diagonal scratches as well as vertical/horizontal ones.
    This remains a heuristic; real photographic detail can resemble damage.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    extreme = (gray >= 225) | (gray <= 25)
    candidate = np.uint8((np.maximum(bright, dark) > sensitivity) & extreme) * 255
    mask = np.zeros_like(gray)
    minimum = max(18, min(gray.shape)//9)
    lines = cv2.HoughLinesP(candidate, 1, np.pi/180, threshold=max(12,minimum//2),
                            minLineLength=minimum, maxLineGap=5)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(mask,(int(x1),int(y1)),(int(x2),int(y2)),255,3)
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8))
    # Small bright flat spots can be peeling or missing paper. Thin lines have
    # no thick core, so they do not interfere with this spot check.
    white = np.uint8(gray >= 230)*255
    core = np.uint8(cv2.distanceTransform(white,cv2.DIST_L2,3) > 3)*255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    spots = np.zeros_like(gray)
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        if 20 <= area <= gray.size*.008 and .6 <= w/max(h,1) <= 1.7 and area/(w*h) > .5:
            spots[labels == index] = 255
    spots = cv2.dilate(spots,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    return cv2.bitwise_or(mask,cv2.bitwise_and(spots,white))


def repair_scratches(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Optional OpenCV cleanup for selected scratches; separate from the GAN."""
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) if mask.any() else image.copy()


def classical_restore(image: np.ndarray, mask: np.ndarray, strength: float = .65) -> np.ndarray:
    repaired = repair_scratches(image, mask)
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


class Pix2PixGenerator(nn.Module):
    """Compact residual U-Net generator conditioned on RGB plus damage mask."""
    def __init__(self, base=32):
        super().__init__()
        self.e1, self.e2 = Block(4, base), Block(base, base*2)
        self.e3, self.mid = Block(base*2, base*4), Block(base*4, base*8)
        self.d3, self.d2, self.d1 = Block(base*12, base*4), Block(base*6, base*2), Block(base*3, base)
        self.head = nn.Conv2d(base, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        z = self.mid(F.avg_pool2d(c, 2))
        def up(z, skip, block):
            return block(torch.cat([F.interpolate(z, size=skip.shape[-2:],
                                                  mode="bilinear", align_corners=False), skip], 1))
        z = up(z, c, self.d3)
        z = up(z, b, self.d2)
        z = up(z, a, self.d1)
        return (x[:, :3] + self.head(z)).clamp(0, 1)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN: judges (damaged RGB, mask, candidate RGB) jointly.

    Outputs logits, not sigmoid probabilities, for BCEWithLogitsLoss.
    GroupNorm makes training with small laptop GPU batches practical.
    """
    def __init__(self, base=32):
        super().__init__()
        layers = [nn.Conv2d(7, base, 4, 2, 1), nn.LeakyReLU(.2)]
        inp = base
        for out, stride in [(base*2, 2), (base*4, 2), (base*8, 1)]:
            layers += [nn.Conv2d(inp, out, 4, stride, 1, bias=False),
                       nn.GroupNorm(8, out), nn.LeakyReLU(.2)]
            inp = out
        self.layers = nn.Sequential(*layers, nn.Conv2d(inp, 1, 4, 1, 1))

    def forward(self, condition, candidate):
        return self.layers(torch.cat([condition, candidate], dim=1))


def gan_train_step(generator, discriminator, optimizer_g, optimizer_d, inp, target,
                   lambda_l1=100.):
    """Viva flow: create fake -> train D -> train G. Ordinary float32 PyTorch."""
    # 1. Generator tries to turn the damaged photograph into a clean photograph.
    fake = generator(inp)

    # 2. Discriminator learns: clean target = real (1), generated image = fake (0).
    # detach() prevents this D update from changing the Generator.
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    real_logits = discriminator(inp, target)
    fake_logits = discriminator(inp, fake.detach())
    d_loss = .5 * (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
                   F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    if not torch.isfinite(d_loss):
        raise RuntimeError("Non-finite discriminator loss")
    d_loss.backward()
    optimizer_d.step()

    # 3. Generator learns to fool D, while matching the actual clean photograph.
    # Freeze D's weights, but keep the gradient flowing through D to the Generator.
    optimizer_d.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    try:
        logits = discriminator(inp, fake)
        adversarial = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        reconstruction = F.l1_loss(fake, target)
        g_loss = adversarial + lambda_l1 * reconstruction
        if not torch.isfinite(g_loss):
            raise RuntimeError("Non-finite generator loss")
        g_loss.backward()
        optimizer_g.step()
    finally:
        discriminator.requires_grad_(True)
    return {"g_loss": g_loss.item(), "d_loss": d_loss.item(),
            "g_adversarial": adversarial.item(), "reconstruction": reconstruction.item()}


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
    net = Pix2PixGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(.5, .999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(.5, .999))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start, best, stale = 0, -float("inf"), 0
    if (out/"history.csv").exists() and not args.resume:
        raise ValueError("Output already contains training history. Use --resume with a GAN last.pt or choose a new --output.")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("model_type") != "pix2pix_gan":
            raise ValueError("This is not a Pix2Pix GAN checkpoint. Start a fresh GAN run.")
        print("Resume restores both GAN networks and optimizers; keep original training settings.", flush=True)
        net.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start, best = checkpoint["epoch"], checkpoint["best_psnr"]
    (out/"config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out/"split.json").write_text(json.dumps({"train": [str(p) for p in train_files],
        "validation": [str(p) for p in val_files]}, indent=2), encoding="utf-8")
    print(f"Device: {device}; {len(train_files)} train / {len(val_files)} validation images", flush=True)
    for epoch in range(start, args.epochs):
        net.train()
        discriminator.train()
        total = dict.fromkeys(["g_loss", "d_loss", "g_adversarial", "reconstruction"], 0.)
        seen = 0
        for step, (inp, target) in enumerate(train_loader):
            inp, target = inp.to(device), target.to(device)
            losses = gan_train_step(net, discriminator, optimizer, optimizer_d, inp, target,
                                    args.lambda_l1)
            for name, value in losses.items():
                total[name] += value * len(inp)
            seen += len(inp)
            if step % 25 == 0:
                print(f"GAN epoch {epoch+1}/{args.epochs}, batch {step+1}/{len(train_loader)}, "
                      f"G {losses['g_loss']:.4f}, D {losses['d_loss']:.4f}", flush=True)
        metrics = validate(net, val_loader, device)
        improved = metrics["psnr"] > best
        best, stale = max(best, metrics["psnr"]), 0 if improved else stale+1
        row = {"epoch": epoch+1, **{k:v/seen for k,v in total.items()}, **metrics}
        log = out/"history.csv"
        with log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {"format_version": 2, "model_type": "pix2pix_gan", "state_dict": net.state_dict(), "base": 32,
            "epoch": epoch+1, "best_psnr": best, "metrics": metrics, "config": vars(args),
            "discriminator": discriminator.state_dict(), "optimizer": optimizer.state_dict(),
            "optimizer_d": optimizer_d.state_dict()}
        temporary = out/"pending.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(out/"last.pt")
        if improved:
            torch.save({k: v for k, v in checkpoint.items() if k not in
                        {"discriminator", "optimizer", "optimizer_d"}}, out/"best.pt")
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            print("Early stopping.", flush=True)
            break
    print(f"Best validation PSNR: {best:.3f} dB. Check improvement over the input baseline before presenting.", flush=True)


def load_restorer(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "pix2pix_gan":
        raise ValueError("Expected a Pix2Pix GAN checkpoint; supervised U-Net weights are not accepted as GAN weights.")
    net = Pix2PixGenerator(checkpoint.get("base", 32)).to(device)
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
    parser.add_argument("--output", default=str(ROOT/"runs/pix2pix"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100., help="Weight of reconstruction relative to the GAN objective")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.lambda_l1 < 0 or args.size < 32 or args.size % 8 or min(args.epochs,args.samples,args.batch_size,args.patience) < 1:
        parser.error("size must be >=32 and divisible by 8; training counts must be positive")
    train(args)
