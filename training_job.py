"""Run full GAN training, then evaluation, recording progress for this local job."""
import json
import argparse
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
preview = argparse.ArgumentParser(add_help=False)
preview.add_argument("--output", default=str(project/"runs/pix2pix"))
job_options, _ = preview.parse_known_args()
output = Path(job_options.output).resolve()
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
    evaluation = project/"outputs"/("evaluation" if output.name == "pix2pix" else output.name+"_evaluation")
    sys.argv = [str(project/"evaluate.py"), "--checkpoint", str(output/"best.pt"),
                "--data", config["val_dir"], "--output", str(evaluation), "--limit", "20"]
    runpy.run_path(str(project/"evaluate.py"), run_name="__main__")
    status("completed", "training_and_evaluation")
except BaseException as error:
    status("failed", "training_or_evaluation", str(error))
    traceback.print_exc()
    raise
