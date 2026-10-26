"""Local pretrained damage segmentation and context-aware LaMa inpainting.

These are imported research models, not weights trained by this project.
The project's separately trained Pix2Pix GAN remains in model.py.
"""
from functools import lru_cache
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from restoration_backends.oldphoto_network import UNet

ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT/'weights/oldphoto/detection.pt'
INPAINTER = ROOT/'weights/lama/big-lama.pt'
SELECTION = ROOT/'weights/learned_pipeline.json'


def available():
    return DETECTOR.is_file() and INPAINTER.is_file() and SELECTION.is_file() and json.loads(SELECTION.read_text()).get('enabled',False)


@lru_cache(maxsize=2)
def load_detector(device='cpu'):
    net = UNet(in_channels=1,out_channels=1,depth=4,conv_num=2,wf=6,
               padding=True,batch_norm=True,up_mode='upsample',with_tanh=False,
               sync_bn=False,antialiasing=True)
    checkpoint = torch.load(DETECTOR,map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint['model_state'])
    return net.to(device).eval()


@lru_cache(maxsize=2)
def load_inpainter(device='cpu'):
    return torch.jit.load(str(INPAINTER),map_location=device).eval()


@torch.inference_mode()
def damage_probability(photo,device='cpu',short_side=256):
    h,w = photo.shape[:2]
    scale = min(short_side/min(h,w),1024/max(h,w))
    height,width = max(32,round(h*scale/16)*16),max(32,round(w*scale/16)*16)
    gray = cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    condition = torch.from_numpy(gray.copy()).float()[None,None].to(device)/127.5-1
    probability = torch.sigmoid(load_detector(device)(condition))
    return F.interpolate(probability,size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()


def predicted_mask(photo,device='cpu',threshold=.4,short_side=256):
    probability = damage_probability(photo,device,short_side)
    mask = np.uint8(probability>=threshold)*255
    count,labels,stats,_ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]>=3:
            cleaned[labels==index] = 255
    return cv2.dilate(cleaned,np.ones((3,3),np.uint8)),probability


@torch.inference_mode()
def context_inpaint(photo,mask,device='cpu'):
    if not mask.any():
        return photo.copy()
    h,w = photo.shape[:2]
    scale = min(1.,768/max(h,w))
    height,width = max(8,round(h*scale)),max(8,round(w*scale))
    rgb = cv2.resize(photo,(width,height),interpolation=cv2.INTER_AREA) if scale<1 else photo
    selected = cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST) if scale<1 else mask
    image = rgb.astype(np.float32).transpose(2,0,1)/255
    condition = (selected>0).astype(np.float32)[None]
    pad = ((0,0),(0,(-height)%8),(0,(-width)%8))
    image = torch.from_numpy(np.pad(image,pad,mode='symmetric'))[None].to(device)
    condition = torch.from_numpy(np.pad(condition,pad,mode='symmetric'))[None].to(device)
    generated = load_inpainter(device)(image,condition)[0,:,:height,:width]
    generated = np.uint8(generated.cpu().permute(1,2,0).numpy().clip(0,1)*255)
    if scale<1:
        generated = cv2.resize(generated,(w,h),interpolation=cv2.INTER_CUBIC)
    result = photo.copy()
    result[mask>0] = generated[mask>0]
    return result


def restore_with_learned(net,photo,device='cpu',return_mask=False):
    """Pretrained scratch segmentation/LaMa, then the local GAN's tone residual.

    Only selected pixels are replaced by LaMa; the Pix2Pix stage receives the
    repaired photograph, so damaged white marks are not blended back into fills.
    """
    from model import detect_scratches, neural_restore
    config = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    mask,probability = predicted_mask(photo,device,threshold=config.get('threshold',.4))
    if np.mean(mask>0)>.4:
        mask,_ = predicted_mask(photo,device,threshold=.5)
    monochrome = (photo.max(axis=2).astype(float)-photo.min(axis=2)).mean()<3
    if not monochrome and config.get('geometric_color_mask',True):
        geometric = detect_scratches(photo,32)
        if np.mean(geometric>0)<.12:
            mask = cv2.bitwise_or(mask,geometric)
    repaired = context_inpaint(photo,mask,device)
    gray = cv2.cvtColor(repaired,cv2.COLOR_RGB2GRAY)
    low,high = np.percentile(gray,[1,99])
    # Keep the accepted natural portrait result. The local Pix2Pix model is used
    # for global tone correction only when the repaired tonal range is low.
    result = repaired
    if net is not None and high-low<config.get('tone_range_threshold',140):
        strength = float(np.clip((185-(high-low))/45,.1,.9))
        generated = neural_restore(net,repaired,np.zeros_like(mask),device)
        result = cv2.addWeighted(generated,strength,repaired,1-strength,0)
    if monochrome:
        result = cv2.cvtColor(cv2.cvtColor(result,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    return (result,mask) if return_mask else result
