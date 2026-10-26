"""Run GAN training here using the libraries already installed on this laptop.

This imports existing packages; it does not install or download anything.
"""
import sys
from pathlib import Path

project = Path(__file__).resolve().parent
packages = Path.home()/"Downloads/DL-CA2-PROJECT/.venv/Lib/site-packages"
if packages.is_dir():
    sys.path.insert(0, str(packages))
sys.path.insert(0, str(project))
if __name__ == "__main__":
    import model
    model.main()
