"""Download official academic DIV2K data and optional ECCV16 colorization assets."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ASSETS = {
    "colorization_deploy_v2.prototxt": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "pts_in_hull.npy": "https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy",
    "colorization_release_v2.caffemodel": "https://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
}


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f"Already downloaded: {destination}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix+".part")
    print(f"Downloading {url}", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DL-CA2-academic-project/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if temporary.stat().st_size == 0:
            raise ValueError("The download was empty")
        temporary.replace(destination)
    except Exception as error:
        raise RuntimeError(f"Download failed: {url}. Check internet access. Original error: {error}") from error


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination/item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--colorization", action="store_true")
    args = parser.parse_args()
    if not args.dataset and not args.colorization:
        parser.error("Choose --dataset and/or --colorization")
    manifest = {}
    if args.dataset:
        print("DIV2K: academic research only; original image copyright remains with its owners. "
              "Source and terms: https://data.vision.ee.ethz.ch/cvl/DIV2K/", flush=True)
        for split, expected in [("train",800),("valid",100)]:
            name = f"DIV2K_{split}_HR.zip"
            path = ROOT/"data/archives"/name
            download(BASE+name, path)
            safe_extract(path, ROOT/"data")
            images = list((ROOT/"data"/f"DIV2K_{split}_HR").glob("*.png"))
            if len(images) != expected:
                raise ValueError(f"Expected {expected} {split} images, found {len(images)}")
            manifest[name] = {"source": BASE+name, "sha256": sha256(path), "images": len(images)}
    if args.colorization:
        for name, url in ASSETS.items():
            path = ROOT/"weights/colorization"/name
            download(url,path)
            manifest[name] = {"source": url,"sha256":sha256(path)}
    path = ROOT/"data/download_manifest.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(existing | manifest, indent=2),encoding="utf-8")
    print("Download checksums recorded for provenance, not independently authenticated reference checksums.")


if __name__ == "__main__":
    main()
