
from itertools import product
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import torch
import torchvision as tv


def revert_imagenet_normalization(sample):
    """
    sample (Tensor): of size (nsamples,nchannels,height,width)
    """
    # Imagenet mean and std    
    mean = [.485,.456,.406]
    std = [.229,.224,.225]
    mean_tensor = torch.Tensor(mean).view(1,3,1,1).to(sample.device)
    std_tensor = torch.Tensor(std).view(1,3,1,1).to(sample.device)
    non_normalized_sample = sample*std_tensor + mean_tensor
    return non_normalized_sample

def draw_box(image, boxes, boxlabels, thickness=5):
    _, H, W = image.size()
    N, _ = boxes.size()
    colors = JOAKIM_TORCH_PALETTE4[boxlabels]
    for n in range(N):
        left   = int(boxes[n, 0].clamp(thickness, W - thickness))
        bottom = int(boxes[n, 3].clamp(thickness, H - thickness))
        right  = int(boxes[n, 2].clamp(thickness, W - thickness))
        top    = int(boxes[n, 1].clamp(thickness, H - thickness))
        image[:, top - thickness : bottom, left - thickness : left]     = colors[n]
        image[:, bottom : bottom + thickness, left - thickness : right] = colors[n]
        image[:, top : bottom + thickness, right : right + thickness]   = colors[n]
        image[:, top - thickness : top, left : right + thickness]       = colors[n]
    return image    

def draw_box_pil(image, boxes, labels=None, colors=None, color_ids=None, thickness=1,
                 text_font=None, text_size=10, overwrite=False):
    """ draw bounding box
    Args:
        image           (PIL.Image): image
        boxes              (Tensor): Of size (N,4), with each element [xmin,ymin,xmax,ymax]
        labels       (List<String>): Length N list, containing bounding box labels (eg. "person, 0.87")
        colors   (List<tuple<int>>): Length N list, containing a triplet of colors (in {0,...,255})
        color_ids      (LongTensor): Of size (N,), each an index used to select color from a palette
        thickness             (Int): thickness of bounding box
        text_font          (String): from available fonts or path to valid .ttf file
        text_size             (Int): text size
        overwrite         (Boolean): make copy of image or not (default=False)
    Returns:
        image
    """
    N, _ = boxes.size()
    if colors is None and color_ids is None:
        colors = N * [(0, 255, 0)]
    elif colors is None:
        colors = JOAKIM_TORCH_PALETTE4[color_ids].mul(255).long().view(-1, 3).tolist()
        colors = [tuple(col) for col in colors]
    if not overwrite:
        image = image.copy()
#    if text_font:
#        text_font = ImageFont.truetype(text_font, text_size)
#    else:
#        text_font = ImageFont.truetype('DejaVuSans-Oblique.ttf', text_size)
    text_font = ImageFont.load_default()

    draw = ImageDraw.Draw(image, 'RGB')
    for n in range(N):
        draw.rectangle(boxes[n].tolist(), fill=None, outline=colors[n], width=thickness)
        if labels is not None:
            label_box = boxes[n].clone()
            label_box[3] = label_box[1] + text_size * 2.0
            draw.rectangle(label_box.tolist(), fill=colors[n], outline=None)
            draw.text((boxes[n][0] + 5, boxes[n][1] + text_size / 2.0), labels[n], (255, 255, 255), font=text_font)
    return image

def save_vps_visualization_video(fpath, images, seg, boxes, boxlabels, boxtexts):
    B, L, _, H, W = images.size()
    images = revert_imagenet_normalization(images)
    
    if seg is not None:
        assert seg.size() == (B,L,H,W), f"seg of wrong size, {seg.size()}, expected ({H},{W})"
        seg = seg.view(B, L, 1, H, W).expand_as(images)
        colors = JOAKIM_TORCH_PALETTE4.to(images.device)
        for k in range(1, 32):
            color = colors[k].view(1,1,3,1,1).expand_as(images)
            images = images.where(seg != k, 0.5 * images + 0.5 * color)
    if boxes is not None and boxtexts is None:
        for b in range(B):
            for l in range(L):
                if boxlabels is None:
                    boxlabels = torch.arange(boxes[b][l].size(0))
                images[b,l] = draw_box(images[b,l], boxes[b][l], boxlabels[b][l])

    images = (images * 256).clamp(0, 255).cpu().byte()
    images = images.view(B*L, 3, H, W).permute(0, 2, 3, 1)

    if boxes is not None and boxtexts is not None:
        for b in range(B):
            for l in range(L):
                image_aspil = Image.fromarray(images[b*L + l].numpy(), mode='RGB')
                image_aspil = draw_box_pil(
                    image_aspil,
                    boxes[b][l],
                    boxtexts[b][l],
                    None,
                    boxlabels[b][l],
                    overwrite=True)
                images[b*L + l] = torch.as_tensor(np.asarray(image_aspil))

    tv.io.write_video(fpath, images, 6)

def save_vps_visualization_video2(fpath, images, seg, boxes, boxlabels, boxtexts):
    """Alters save_vps_visualization_video to instead plot track text on the bottom of the
    image, instead of plotting them on top of the box. Furthermore, multiple lines of track
    text can be plotted.
    """
    B, L, _, H, W = images.size()
    images = revert_imagenet_normalization(images)
    
    if seg is not None:
        assert seg.size() == (B,L,H,W), f"seg of wrong size, {seg.size()}, expected ({H},{W})"
        seg = seg.view(B, L, 1, H, W).expand_as(images)
        colors = JOAKIM_TORCH_PALETTE4.to(images.device)
        for k in range(1, 32):
            color = colors[k].view(1,1,3,1,1).expand_as(images)
            images = images.where(seg != k, 0.5 * images + 0.5 * color)

    images = (images * 256).clamp(0, 255).cpu().byte()
    images = torch.cat([images, torch.full((B, L, 3, 100, W), 200, dtype=torch.uint8)], dim=3)
    images = images.view(B*L, 3, H + 100, W).permute(0, 2, 3, 1)

    def draw_box(image, boxes, box_color_ids, boxtexts):
        N, _ = boxes.size()
        colors = JOAKIM_TORCH_PALETTE4[box_color_ids].mul(255).long().view(-1, 3).tolist()
        colors = [tuple(col) for col in colors]
        text_font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(image, 'RGB')
        for idx, m in enumerate(box_color_ids):
            left = 128 * (m - 1)
            right = 128 * m
            top = H
            draw.rectangle(boxes[idx].tolist(), outline=colors[idx], width=3)
            draw.rectangle((left, top, right, top + 10), fill=colors[idx])
            boxtext = boxtexts[idx]
            if isinstance(boxtext, str):
                boxtext = [boxtext]
            for jdx, text in enumerate(boxtext):
                draw.text((left + 5, top + 15 * (jdx + 1)), text, fill=(0,0,0), font=text_font)
        return image

    for b in range(B):
        for l in range(L):
            image_aspil = Image.fromarray(images[b*L + l].numpy(), mode='RGB')
            image_aspil = draw_box(
                image_aspil,
                boxes[b][l],
                boxlabels[b][l],
                boxtexts[b][l])
            images[b*L + l] = torch.as_tensor(np.asarray(image_aspil))

    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    tv.io.write_video(fpath, images, 6)

def tensor_to_visualizable(tensor):
    if tensor.dim() > 3:
        tensor = tensor.squeeze()
        assert tensor.dim() in (2,3)
    if tensor.dim() == 3:
        assert tensor.size(0) == 3
        tensor = tensor.transpose(0,1).transpose(1,2)
    npimg = tensor.numpy()
    return npimg

def plot_tensors(*tensors):
    fig = plt.figure()
    for i in range(len(tensors)):
        fig.add_subplot(1, len(tensors), i+1)
        image = tensor_to_visualizable(tensors[i])
        plt.imshow(image, cmap='seismic')
#    plt.waitforbuttonpress(0)
    while not plt.waitforbuttonpress(): pass
    plt.close(fig)
    plt.show()

def plot_nparray(*nparrays, cmap='seismic'):
    fig = plt.figure()
    for i in range(len(nparrays)):
        fig.add_subplot(1, len(nparrays), i+1)
        plt.imshow(nparrays[i], cmap=cmap)
    plt.show()

def plot_nparray_ifft2(*nparrays, cmap='seismic'):
    try:
        nparrays = [np.real(np.fft.ifft2(nparray)) for nparray in nparrays]
        plot_nparray(*nparrays, cmap=cmap)
    except:
        print("Error, nparrays[0]:", type(nparrays[0]))
        if isinstance(nparrays[0], np.ndarray):
            print(nparrays[0].shape)
        raise

def play_tracking_sequence(seq, result):
    return 0

JOAKIM_TORCH_PALETTE = torch.tensor(list(product(*(3*[[0., 0.25, 0.5, 0.75, 1.]])))).view(125,3,1,1)
JOAKIM_TORCH_PALETTE4 = torch.tensor(list(product(*(3*[[0., 0.33, 0.67, 1.]])))).view(64,3,1,1)
