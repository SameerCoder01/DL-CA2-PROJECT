"""Evaluate a checkpoint on held-out synthetic damage; export actual metrics and examples."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import ROOT, image_paths, read_rgb, damage, neural_restore, load_restorer, metric_pair, classical_restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT/"runs/pix2pix/best.pt"))
    parser.add_argument("--data", default=str(ROOT/"data/DIV2K_valid_HR"))
    parser.add_argument("--output", default=str(ROOT/"outputs/evaluation"))
    parser.add_argument("--limit",type=int,default=100)
    args = parser.parse_args()
    paths = image_paths(args.data)[:args.limit]
    if not paths:
        raise ValueError("No validation images found")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, checkpoint = load_restorer(args.checkpoint,device)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for i,path in enumerate(paths):
        image = read_rgb(path)
        h,w = image.shape[:2]
        # Fixed center crop, different corruption seed from checkpoint selection.
        side = min(h,w,512)
        image = image[(h-side)//2:(h-side)//2+side,(w-side)//2:(w-side)//2+side]
        degraded, mask = damage(image, 91000+i)
        restored = neural_restore(net,degraded,mask,device)
        baseline = classical_restore(degraded,mask)
        row = {"file":path.name}
        for name,pred in [("input",degraded),("opencv",baseline),("gan",restored)]:
            psnr,ssim = metric_pair(pred,image)
            row.update({name+"_psnr":psnr,name+"_ssim":ssim})
        rows.append(row)
        if i < 6:
            Image.fromarray(np.concatenate([degraded,baseline,restored,image],1)).save(out/f"comparison_{i:02}.png")
        print(json.dumps(row),flush=True)
    means = {key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "file"}
    report = {"checkpoint_epoch":checkpoint["epoch"],"count":len(rows),"means":means,"per_image":rows,
              "scope":"Held-out source photos with new synthetic corruption. Same validation split; not an independent real-world test set.",
              "comparison_order":"input | OpenCV | Pix2Pix GAN | clean reference"}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(means,indent=2))


if __name__ == "__main__":
    main()
