"""Open the local Streamlit app using this laptop's existing project packages."""
import runpy
import sys
from pathlib import Path

project = Path(__file__).resolve().parent
packages = Path.home()/"Downloads/DL-CA2-PROJECT/.venv/Lib/site-packages"
if packages.is_dir():
    sys.path.insert(0, str(packages))
sys.path.insert(0, str(project))
sys.argv = ["streamlit", "run", str(project/"App.py"), "--server.headless=true",
            "--server.address=127.0.0.1", "--server.port=8501", "--browser.gatherUsageStats=false"]
runpy.run_module("streamlit", run_name="__main__")
