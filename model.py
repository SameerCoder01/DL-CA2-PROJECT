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


def detect_faded_scratches(image: np.ndarray) -> np.ndarray:
    """Find pale creases in monochrome prints by local contrast, not pure white.

    Keep elongated connected ridges; broad facial shading is not a thin ridge.
    Absolute white thresholds fail on scanned prints whose damage has faded grey.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ridge = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    weak = np.uint8((ridge >= 12) & (gray >= 100))*255
    linked = cv2.morphologyEx(weak,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(linked)
    selected = np.zeros_like(gray)
    for index in range(1,count):
        x,y,w,h,area = stats[index]
        near_edge = x < gray.shape[1]*.08 or x+w > gray.shape[1]*.92 or y < gray.shape[0]*.08 or y+h > gray.shape[0]*.92
        minimum_length = 14 if near_edge else max(22,min(gray.shape)//8)
        if area < 10 or max(w,h) < minimum_length:
            continue
        region = labels[y:y+h,x:x+w] == index
        points = cv2.findNonZero(region.astype(np.uint8))
        _,(a,b),_ = cv2.minAreaRect(points)
        elongation = max(a,b)/max(min(a,b),1)
        strong_fraction = np.mean(ridge[y:y+h,x:x+w][region] >= 22)
        if elongation >= (1.7 if near_edge else 3.5) and strong_fraction >= .15:
            selected[y:y+h,x:x+w][region] = 255
    selected = cv2.bitwise_and(selected,weak)
    return cv2.dilate(selected,np.ones((3,3),np.uint8))


def realistic_damage(clean: np.ndarray, seed: int) -> tuple[np.ndarray,np.ndarray]:
    """Mixed training curriculum: intact, fade-only, scratches-only, and combined.

    Creases are bent, partly transparent and grey as well as white/dark. This
    reduces the gap between pure-white synthetic lines and faded scanned prints.
    The clean reference is never changed, including for monochrome pairs.
    """
    rng = np.random.default_rng(seed)
    h,w = clean.shape[:2]
    profile = rng.random()
    mask = np.zeros((h,w),np.uint8)
    if profile < .15:
        return clean.copy(),mask
    x = clean.astype(np.float32)
    if profile >= .35:
        x = x*rng.uniform(.55,.95)+rng.uniform(8,48)
        if rng.random() < .4:
            x = cv2.GaussianBlur(x,(3,3),float(rng.uniform(.2,.7)))
        x += rng.normal(0,rng.uniform(.5,4),x.shape)
    if profile < .35 or profile >= .55:
        for _ in range(int(rng.integers(2,12))):
            line = np.zeros((h,w),np.uint8)
            start = rng.integers([0,0],[w,h])
            end = np.clip(start+rng.integers([-w//3,-h//2],[w//3+1,h//2+1]),[0,0],[w-1,h-1])
            middle = np.clip((start+end)//2+rng.integers(-8,9,2),[0,0],[w-1,h-1])
            points = np.array([start,middle,end],np.int32)
            cv2.polylines(line,[points],False,255,int(rng.integers(1,4)))
            opacity = float(rng.uniform(.45,1.))
            value = float(rng.uniform(145,255) if rng.random()<.8 else rng.uniform(5,70))
            selected = line>0
            x[selected] = (1-opacity)*x[selected]+opacity*value
            mask = cv2.bitwise_or(mask,line)
        if rng.random() < .4:
            spot = np.zeros_like(mask)
            cv2.circle(spot,(int(rng.integers(w)),int(rng.integers(h))),
                       int(rng.integers(2,max(3,min(h,w)//24))),255,-1)
            x[spot>0] = rng.uniform(170,250)
            mask = cv2.bitwise_or(mask,spot)
    return np.uint8(np.clip(x,0,255)),mask


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
            cv2.line(mask,(int(x1),int(y1)),(int(x2),int(y2)),255,1)
    # A Hough line is evidence, not a filled repair region. Intersect its narrow
    # neighbourhood with actual contrast candidates so gaps retain real texture.
    support = cv2.dilate(mask, np.ones((5,5),np.uint8))
    mask = cv2.bitwise_and(support, candidate)
    # Short, isolated hairline scratches may not accumulate enough Hough votes.
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (_, _), (cw, ch), _ = cv2.minAreaRect(contour)
        length, width = max(cw,ch), min(cw,ch)
        if length >= max(12,min(gray.shape)//24) and length/max(width,1) >= 7:
            region = np.zeros_like(gray)
            cv2.drawContours(region,[contour],-1,255,-1)
            mask = cv2.bitwise_or(mask,cv2.bitwise_and(region,candidate))
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
    mask = cv2.bitwise_or(mask,cv2.bitwise_and(spots,white))
    chroma = image.max(axis=2).astype(float)-image.min(axis=2)
    if chroma.mean() < 8 and np.mean(mask > 0) < .005:
        mask = cv2.bitwise_or(mask,detect_faded_scratches(image))
    return mask


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


def restore_automatic(net, photo, device="cpu", return_mask=False):
    """GAN tone restoration plus narrow, original-image scratch interpolation.

    Inpainting the original avoids propagating GAN scratch halos into the fill.
    Never blend the damaged original back inside a selected scratch.
    """
    mask = detect_scratches(photo, 32)
    if np.mean(mask > 0) > .08:
        mask = detect_scratches(photo, 50)
    if np.mean(mask > 0) > .12:
        mask = np.zeros(photo.shape[:2], np.uint8)
    generated = neural_restore(net, photo, mask, device)
    # Match the fill to GAN-restored tones, while retaining original local detail.
    repaired = repair_scratches(photo, mask)
    # A photo with a healthy tonal range needs much less global GAN correction.
    # This protects intact texture in photos whose only defect is a scratch.
    gray = cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY)
    values = gray[mask == 0]
    low,high = np.percentile(values if values.size else gray,[1,99])
    strength = float(np.clip((185-(high-low))/45,.1,.9))
    intact = (mask == 0).astype(np.float32)[:,:,None]
    residual = (generated.astype(np.float32)-photo.astype(np.float32))*intact
    tone = cv2.GaussianBlur(residual, (0,0), 5) / np.maximum(
        cv2.GaussianBlur(intact[:,:,0], (0,0), 5)[:,:,None], .01)
    filled = np.clip(repaired.astype(np.float32)+strength*tone,0,255).astype(np.uint8)
    restored = cv2.addWeighted(generated,strength,repaired,1-strength,0)
    restored[mask > 0] = filled[mask > 0]
    if (photo.max(axis=2).astype(float)-photo.min(axis=2)).mean() < 3:
        restored = cv2.cvtColor(cv2.cvtColor(restored,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    return (restored, mask) if return_mask else restored


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
                   lambda_l1=100., damage_mask=None):
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
        # Scratches occupy few pixels: ordinary global L1 rewards leaving them.
        # Give damaged pixels nine times the weight, normalized by total weight.
        weights = 1 + 8 * (inp[:,3:4] if damage_mask is None else damage_mask)
        reconstruction = ((fake-target).abs()*weights).sum() / (3*weights.sum())
        # Preserve edges instead of reducing all uncertain detail to smooth fills.
        edge = .5 * (F.l1_loss(fake[:,:,:,1:]-fake[:,:,:,:-1], target[:,:,:,1:]-target[:,:,:,:-1]) +
                     F.l1_loss(fake[:,:,1:,:]-fake[:,:,:-1,:], target[:,:,1:,:]-target[:,:,:-1,:]))
        g_loss = adversarial + lambda_l1 * (reconstruction + .15*edge)
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
    def __init__(self, paths, size=256, samples=None, validation=False, realistic=False, automatic_mask_prob=.25):
        self.paths, self.size = paths, size
        self.samples, self.validation = samples or len(paths), validation
        self.realistic = realistic
        self.automatic_mask_prob = automatic_mask_prob

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
        # B&W restoration is paired B&W -> B&W, never forced colorization.
        if rng.random() < .3:
            clean = cv2.cvtColor(cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        corrupt = realistic_damage if self.realistic else damage
        degraded, mask = corrupt(clean, rng.randrange(2**31))
        true_mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()/255
        # Expose G to automatic-mask misses as well as exact synthetic masks.
        if not self.validation and rng.random() < self.automatic_mask_prob:
            mask = detect_scratches(degraded,32)
        pair = (torch.cat([tensor(degraded), torch.from_numpy(mask.copy()).unsqueeze(0).float()/255]), tensor(clean))
        return (*pair,true_mask) if self.realistic else pair


def metric_pair(pred, target):
    mse = np.mean((pred.astype(np.float64)/255-target.astype(np.float64)/255)**2)
    psnr = float(-10*np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(target, pred, channel_axis=2, data_range=255))
    return psnr, ssim


@torch.inference_mode()
def validate(net, loader, device, automatic=False):
    net.eval()
    rows = []
    for batch in loader:
        inp,target = batch[:2]
        if automatic:
            inp = inp.clone()
            for index,source in enumerate(inp):
                photo = np.uint8(source[:3].permute(1,2,0).numpy().clip(0,1)*255)
                inp[index,3] = torch.from_numpy(detect_scratches(photo,32)).float()/255
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
    realistic = getattr(args,"realistic_damage",False)
    mask_probability = getattr(args,"automatic_mask_prob",.25)
    automatic_validation = getattr(args,"automatic_validation",False)
    train_set = PhotoPairs(train_files, args.size, args.samples, realistic=realistic,
                           automatic_mask_prob=mask_probability)
    val_set = PhotoPairs(val_files, args.size, validation=True, realistic=realistic)
    loader_options = {"num_workers":args.workers,"persistent_workers":args.workers>0,
                      "pin_memory":device.type=="cuda"}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, batch_size=1, **loader_options)
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
        print("Resume restores both GAN networks and optimizers; requested LR is applied after loading.", flush=True)
        net.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        for opt in (optimizer, optimizer_d):
            for group in opt.param_groups:
                group["lr"] = args.lr
        start, best = checkpoint["epoch"], checkpoint["best_psnr"]
        if out.resolve() != Path(args.resume).resolve().parent:
            # A new fine-tuning branch may change validation/augmentation policy.
            best = -float("inf")
    (out/"config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out/"split.json").write_text(json.dumps({"train": [str(p) for p in train_files],
        "validation": [str(p) for p in val_files]}, indent=2), encoding="utf-8")
    if not (out/"starting_metrics.json").exists():
        starting = validate(net,val_loader,device,automatic_validation)
        (out/"starting_metrics.json").write_text(json.dumps(starting,indent=2),encoding="utf-8")
        print("Starting checkpoint on this exact validation curriculum: "+json.dumps(starting),flush=True)
    # Preserve the exact training implementation alongside checkpoints.
    (out/"training_source.py").write_text(Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    print(f"Device: {device}; {len(train_files)} train / {len(val_files)} validation images", flush=True)
    for epoch in range(start, args.epochs):
        net.train()
        discriminator.train()
        total = dict.fromkeys(["g_loss", "d_loss", "g_adversarial", "reconstruction"], 0.)
        seen = 0
        for step, batch in enumerate(train_loader):
            inp,target = batch[:2]
            truth = batch[2].to(device) if len(batch)>2 else None
            inp, target = inp.to(device), target.to(device)
            losses = gan_train_step(net, discriminator, optimizer, optimizer_d, inp, target,
                                    args.lambda_l1,truth)
            for name, value in losses.items():
                total[name] += value * len(inp)
            seen += len(inp)
            if step % 25 == 0:
                print(f"GAN epoch {epoch+1}/{args.epochs}, batch {step+1}/{len(train_loader)}, "
                      f"G {losses['g_loss']:.4f}, D {losses['d_loss']:.4f}", flush=True)
        metrics = validate(net, val_loader, device,automatic_validation)
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


def main():
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
    parser.add_argument("--realistic-damage",action="store_true",help="Use diverse faded/grey/bent creases and intact examples")
    parser.add_argument("--automatic-mask-prob",type=float,default=.25)
    parser.add_argument("--automatic-validation",action="store_true",help="Select G using predicted masks, matching deployment")
    args = parser.parse_args()
    if not 0 <= args.automatic_mask_prob <= 1 or args.lambda_l1 < 0 or args.size < 32 or args.size % 8 or min(args.epochs,args.samples,args.batch_size,args.patience) < 1:
        parser.error("size must be >=32 and divisible by 8; training counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
