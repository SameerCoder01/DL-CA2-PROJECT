"""Summarize GAN training phases without mixing validation criteria."""
import csv
import json
from pathlib import Path


def main():
    phases = [('runs/pix2pix-long-realistic','Known-mask validation'),
              ('runs/pix2pix-long-auto','Automatic-mask validation')]
    out = Path('outputs/long_training_report')
    out.mkdir(parents=True,exist_ok=True)
    summary = {}
    for directory,label in phases:
        root = Path(directory)
        with (root/'history.csv').open() as handle:
            rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(handle)]
        start = json.loads((root/'starting_metrics.json').read_text())
        best = max(rows,key=lambda r:r['psnr'])
        summary[directory] = {'validation_mode':label,'start':start,'best':best,
                              'last_epoch':int(rows[-1]['epoch']),'completed_epochs':len(rows)}
    summary['caution'] = 'Phases use different mask validation criteria; compare each only with its own start. Intact curriculum cases inflate the input PSNR mean; use damaged-only evaluation for restoration claims.'
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
