"""Run full GAN training, then evaluation, recording progress for this local job."""
import json
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

project = Path(__file__).resolve().parent
packages = Path.home()/"Downloads/DL-CA2-PROJECT/.venv/Lib/site-packages"
if packages.is_dir():
    sys.path.insert(0, str(packages))
sys.path.insert(0, str(project))
output = project/"runs/pix2pix"
output.mkdir(parents=True, exist_ok=True)


def status(state, stage, error=None):
    record = {"state":state, "stage":stage,
              "updated_utc":datetime.now(timezone.utc).isoformat()}
    if error:
        record["error"] = error
    temporary = output/"job_status.tmp"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(output/"job_status.json")


try:
    status("running", "training")
    sys.argv[0] = str(project/"model.py")
    runpy.run_path(str(project/"model.py"), run_name="__main__")
    status("running", "evaluation")
    config = json.loads((output/"config.json").read_text(encoding="utf-8"))
    sys.argv = [str(project/"evaluate.py"), "--data", config["val_dir"], "--limit", "20"]
    runpy.run_path(str(project/"evaluate.py"), run_name="__main__")
    status("completed", "training_and_evaluation")
except BaseException as error:
    status("failed", "training_or_evaluation", str(error))
    traceback.print_exc()
    raise
