"""Create smaller clean training copies for a quick classroom demonstration."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def resize_one(pair):
    source, target = pair
    if target.exists():
        return
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((512,512), Image.Resampling.LANCZOS)
        image.save(target, format="PNG", compress_level=1)


def main():
    for source_name, split, expected in [("DIV2K_train_HR","train",800),
                                         ("DIV2K_valid_HR","valid",100)]:
        sources = sorted((ROOT/"data"/source_name).glob("*.png"))
        if len(sources) != expected:
            raise ValueError(f"Expected {expected} source images for {split}")
        destination = ROOT/"data/fast"/split
        destination.mkdir(parents=True,exist_ok=True)
        pairs = [(source,destination/source.name) for source in sources]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index,_ in enumerate(pool.map(resize_one,pairs),1):
                if index % 100 == 0:
                    print(f"Prepared {index}/{expected} {split} photos",flush=True)
    print("Fast copies ready. Original DIV2K files were preserved.",flush=True)


if __name__ == "__main__":
    main()
