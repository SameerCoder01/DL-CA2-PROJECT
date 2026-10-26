"""Separate faded, scratched, damaged and B&W checks using known references."""
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from model import image_paths, read_rgb, load_restorer, restore_automatic, metric_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='runs/pix2pix/best.pt')
    parser.add_argument('--output',default='outputs/category_review')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, checkpoint = load_restorer(args.checkpoint,device)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for i,path in enumerate(image_paths('data/fast/valid')[::8]):
        clean = read_rgb(path)
        h,w = clean.shape[:2]
        clean = clean[(h-256)//2:(h-256)//2+256,(w-256)//2:(w-256)//2+256]
        for category in ['faded','scratched','damaged','black_and_white']:
            target = clean.copy()
            if category == 'black_and_white':
                target = cv2.cvtColor(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
            if category in ['faded','black_and_white']:
                photo = np.clip(target.astype(float)*.65+45,0,255).astype(np.uint8)
            else:
                photo = target.copy()
                if category == 'scratched':
                    for start,end,color in [((25,10),(170,245),(255,255,255)),((10,190),(230,70),(15,15,15))]:
                        cv2.line(photo,start,end,color,2)
                else:
                    cv2.circle(photo,(110,130),7,(255,255,255),-1)
                    cv2.line(photo,(170,20),(180,235),(235,235,235),3)
            restored = restore_automatic(net,photo,device)
            row = {'file':path.name,'category':category}
            for label,array in [('input',photo),('restored',restored)]:
                psnr,ssim = metric_pair(array,target)
                row.update({label+'_psnr':psnr,label+'_ssim':ssim})
            if category == 'black_and_white':
                row['achromatic'] = bool(np.array_equal(restored[:,:,0],restored[:,:,1]) and np.array_equal(restored[:,:,1],restored[:,:,2]))
            rows.append(row)
            if i == 0:
                canvas = Image.new('RGB',(768,284),'#161b22')
                draw = ImageDraw.Draw(canvas)
                for j,(label,array) in enumerate([('Input: '+category,photo),('Restored',restored),('Clean reference',target)]):
                    canvas.paste(Image.fromarray(array),(j*256,28))
                    draw.text((j*256+8,8),label,fill='white')
                canvas.save(out/(category+'.png'))
    means = {}
    for category in ['faded','scratched','damaged','black_and_white']:
        subset = [r for r in rows if r['category']==category]
        means[category] = {key:float(np.mean([r[key] for r in subset])) for key in ['input_psnr','restored_psnr','input_ssim','restored_ssim']}
    report = {'epoch':checkpoint['epoch'],'sources_per_category':len(rows)//4,'means':means,'rows':rows,
              'scope':'Synthetic category checks on validation sources, not real historical photographs; category settings distinct from training corruption.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(means,indent=2))


if __name__ == '__main__':
    main()
