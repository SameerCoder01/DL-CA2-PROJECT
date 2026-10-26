"""Reproducible automatic-mask evaluation, including local damaged example.

Unlike evaluate.py's oracle masks, every restoration here detects its own mask.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import model


def old_pipeline(net, photo, device, module):
    mask = module.detect_scratches(photo, 32)
    if np.mean(mask > 0) > .08:
        mask = module.detect_scratches(photo, 50)
    if np.mean(mask > 0) > .12:
        mask = np.zeros(photo.shape[:2], np.uint8)
    generated = module.neural_restore(net, photo, mask, device)
    result = module.repair_scratches(cv2.addWeighted(generated,.65,photo,.35,0),mask)
    if (photo.max(2).astype(float)-photo.min(2)).mean() < 3:
        result = cv2.cvtColor(cv2.cvtColor(result,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    return result, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='runs/pix2pix/best.pt')
    parser.add_argument('--output',default='outputs/quality_baseline')
    parser.add_argument('--old',action='store_true')
    parser.add_argument('--limit',type=int,default=20)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--realistic',action='store_true')
    parser.add_argument('--backend',choices=['auto','legacy','learned'],default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, checkpoint = model.load_restorer(args.checkpoint,device)
    module = model
    if args.old:
        spec = importlib.util.spec_from_file_location('old_model','outputs/model_before_quality.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    cases = []
    for i,path in enumerate(model.image_paths('data/fast/valid')[args.start:args.start+args.limit],args.start):
        image = model.read_rgb(path)
        h,w = image.shape[:2]
        image = image[(h-256)//2:(h-256)//2+256,(w-256)//2:(w-256)//2+256]
        if i % 3 == 0:
            image = cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        corrupt = model.realistic_damage if args.realistic else model.damage
        photo, truth = corrupt(image, 191000+i)
        cases.append((path.stem,photo,image,truth))
    demo = Path('outputs/demo_test/input_photograph.png')
    if demo.exists():
        cases.append(('local_demo',model.read_rgb(demo),None,None))
    for i,(name,photo,clean,truth) in enumerate(cases):
        result, mask = old_pipeline(net,photo,device,module) if args.old else model.restore_automatic(net,photo,device,return_mask=True,backend=args.backend)
        row = {'name':name,'mask_fraction':float(np.mean(mask>0))}
        if clean is not None:
            row['intact_input'] = bool(np.array_equal(photo,clean))
            for label,pred in [('input',photo),('restored',result)]:
                psnr,ssim = model.metric_pair(pred,clean)
                row.update({label+'_psnr':psnr,label+'_ssim':ssim})
            row['scratch_mae'] = float(np.abs(result.astype(float)-clean)[truth>0].mean()) if truth.any() else None
            row['mask_recall'] = float(np.mean(mask[truth>0]>0)) if truth.any() else None
            row['mask_precision'] = float(np.mean(truth[mask>0]>0)) if mask.any() else 0.
        rows.append(row)
        Image.fromarray(result).save(out/(name+'_restored.png'))
        if i < 6 or clean is None:
            panels = [photo,result] + ([clean] if clean is not None else [])
            labels = ['Damaged input','Automatic restoration','Clean reference']
            canvas = Image.new('RGB',(photo.shape[1]*len(panels),photo.shape[0]+28),'#161b22')
            draw = ImageDraw.Draw(canvas)
            for j,panel in enumerate(panels):
                canvas.paste(Image.fromarray(panel),(j*photo.shape[1],28))
                draw.text((j*photo.shape[1]+8,8),labels[j],fill='white')
            canvas.save(out/(name+'_comparison.png'))
            Image.fromarray(mask).save(out/(name+'_mask.png'))
    measured = [r for r in rows if 'restored_psnr' in r]
    keys = [key for key in measured[0] if key not in {'name','intact_input'}]
    def averages(subset):
        return {key:float(np.mean(values)) if values else None for key in keys
                for values in [[r[key] for r in subset if r[key] is not None]]}
    means = averages(measured)
    report = {'checkpoint':args.checkpoint,'epoch':checkpoint['epoch'],'means':means,'rows':rows,
              'by_input_type':{label:{'count':len(subset),'means':averages(subset)}
                               for label in ['damaged','intact']
                               for subset in [[r for r in measured if r['intact_input'] == (label=='intact')]]},
              'realistic':args.realistic,
              'backend':args.backend,
              'scope':f'{len(measured)} validation sources from index {args.start}, fixed synthetic damage, automatic masks; grayscale subset. Not real historical-photo ground truth. Scratch MAE/recall exclude undamaged examples.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(means,indent=2),flush=True)


if __name__ == '__main__':
    main()
