"""Install the accepted pretrained repair backends; no new dataset/training."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import download_restoration_assets
DETECTOR_SHA256 = 'b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95'
CONFIG = {'enabled':True,'threshold':.4,'geometric_color_mask':True,'tone_range_threshold':140,
          'selection':'Accepted learned detector + LaMa repair; optional local Pix2Pix tone correction for low-contrast images.',
          'trained_locally':'Pix2Pix epoch 40; no CelebA-trained checkpoint saved or deployed.'}


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(4*1024*1024):
            result.update(chunk)
    return result.hexdigest()


def download(name):
    previous = sys.argv
    try:
        sys.argv = ['download_restoration_assets.py',name]
        download_restoration_assets.main()
    finally:
        sys.argv = previous


def main():
    detector = ROOT/'weights/oldphoto/detection.pt'
    if not detector.is_file() or sha256(detector)!=DETECTOR_SHA256:
        download('oldphoto')
        detector.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(ROOT/'data/external/global_checkpoints.zip') as archive:
            name = 'checkpoints/detection/FT_Epoch_latest.pt'
            pending = detector.with_suffix('.pending')
            with archive.open(name) as source,pending.open('wb') as target:
                shutil.copyfileobj(source,target)
            if sha256(pending)!=DETECTOR_SHA256:
                raise ValueError('Scratch detector SHA256 mismatch')
            pending.replace(detector)
    download('lama')
    selection = ROOT/'weights/learned_pipeline.json'
    selection.parent.mkdir(parents=True,exist_ok=True)
    selection.write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    print('Accepted learned-repair backend installed. No CelebA dataset is required.')


if __name__=='__main__':
    main()
