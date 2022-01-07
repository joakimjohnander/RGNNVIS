import subprocess
import os, sys, time
import numpy as np
from PIL import Image
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
DEBUG = False
if DEBUG:
    from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torchvision as tv

from config import config
import rgnnvis.utils as utils

def prod(lst):
    prod = 1
    for elem in lst:
        prod *= elem
    return prod

def get_tensor_statistics_str(tensor, name="", formatting="standard"):
    """ Returns string of formatted tensor statistics, contains min, max, mean, and std"""
    if isinstance(tensor, (torch.FloatTensor, torch.cuda.FloatTensor)):
        if tensor.numel() == 0:
            string = "size: {}".format(tensor.size())
        elif formatting == "standard":
            string = "elem in [{:9.3f}, {:9.3f}]    mean: {:9.3f}    std: {:9.3f}    size: {}".format(tensor.min().item(), tensor.max().item(), tensor.mean().item(), tensor.std().item(), tuple(tensor.size()))
        elif formatting == "short":
            string = "[{:6.3f}, {:6.3f}]  mu: {:6.3f}  std: {:6.3f}  {!s:17} {: 6.1f}MB".format(tensor.min().item(), tensor.max().item(), tensor.mean().item(), tensor.std().item(), tuple(tensor.size()), 4e-6 * prod(tensor.size()))
    elif isinstance(tensor, (torch.LongTensor, torch.ByteTensor, torch.cuda.LongTensor, torch.cuda.ByteTensor)):
        if tensor.numel() == 0:
            string = "size: {}".format(tensor.size())
        else:
            tensor = tensor.to('cpu')
            string = "elem in [{}, {}]    size: {}    HIST BELOW:\n{}".format(tensor.min().item(), tensor.max().item(), tuple(tensor.size()), torch.stack([torch.arange(0, tensor.max()+1), tensor.view(-1).bincount()], dim=0))
    elif isinstance(tensor, (torch.BoolTensor, torch.cuda.BoolTensor)):
        if tensor.numel() == 0:
            string = "size: {}".format(tensor.size())
        else:
            tensor = tensor.to('cpu').long()
            string = f"BoolTensor with {tensor.sum()} True values out of {tensor.numel()}, size: {tensor.size()}"
    elif isinstance(tensor, (float, int, bool)):
        string = f"{type(tensor)}: {tensor}"
    else:
        tensor_type = tensor.type() if isinstance(tensor, torch.Tensor) else type(tensor)
        raise NotImplementedError(f"A type of tensor not yet supported was input. Expected torch.FloatTensor or torch.LongTensor, got: {tensor_type}")
    string = string + "    " + name
    return string

def print_tensor_statistics(tensor, name="", formatting="standard"):
    print(get_tensor_statistics_str(tensor, name, formatting))

def get_weight_statistics_str(layer, name="", formatting="standard"):
    return get_tensor_statistics_str(layer.weight, name, formatting)

def get_memory_str():
    return "{:.2f} MB".format(torch.cuda.memory_allocated() / 1e6)
def print_memory():
    print(get_memory_str())

def visualize_modules(module, space=""):
    if hasattr(module, 'weight'):
        print(space, type(module), get_tensor_statistics_str(module.weight.data))
    else:
        print(space, type(module))
    for child_module in module.children():
        visualize_modules(child_module, space=space+"-")

def get_model_size_str(model):
    nelem = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            nelem += module.weight.numel()
        if hasattr(module, 'bias'):
            nelem += module.weight.numel()
    size_str = "{:.2f} MB".format(nelem * 4 * 1e-6)
    return size_str

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def print_nvidia_smi():
    print(subprocess.check_output('nvidia-smi').decode('UTF-8'))

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

def draw_bbox(imgs, bboxes, thickness=5):
    N = imgs.size(0)
    H, W = imgs.size()[-2:]
    for i in range(N):
        left = int(bboxes[i,0,0].clamp(thickness, W-thickness))
        bottom = int((bboxes[i,0,1] + bboxes[i,0,3]).clamp(thickness, H - thickness))
        right = int((bboxes[i,0,0] + bboxes[i,0,2]).clamp(thickness, W - thickness))
        top = int((bboxes[i,0,1]).clamp(thickness, H - thickness))
        imgs[i,:,top-thickness:bottom,left-thickness:left]     = torch.Tensor([1.,0.,0.]).view(3,1,1)
        imgs[i,:,bottom:bottom+thickness,left-thickness:right] = torch.Tensor([1.,0.,0.]).view(3,1,1)
        imgs[i,:,top:bottom+thickness,right:right+thickness]   = torch.Tensor([1.,0.,0.]).view(3,1,1)
        imgs[i,:,top-thickness:top,left:right+thickness]       = torch.Tensor([1.,0.,0.]).view(3,1,1)
    return imgs

def debug_tsne(tensor, segmap):
    D,N = tensor.size()
    C,_ = segmap.size()
    seg = segmap.detach().cpu().argmax(dim=0, keepdim=False).numpy()
    colors = NUMPY_PALETTE[seg]
    array = tensor.detach().cpu().transpose(0,1).numpy() # s.t. tensors is num-samples x dimensionality
    embedded_array = TSNE(n_components=2).fit_transform(array) # of size Nsamples x 2
    print(colors.shape)
    plt.scatter(embedded_array[:,0], embedded_array[:,1], color=colors)
    plt.title("T-SNE embedding")
    plt.legend(["Class {}".format(c) for c in range(C)])
    plt.savefig(os.path.join(config['tempinfo_path'], 'tsne_plot.eps'))

def debug_plothist_multiclass(tensors, class_mask):
    """tensor of size C,N, class_mask of size K,N
    """
    C,N = tensors.size()
    K,_ = class_mask.size()
    tensors = tensors.detach().to('cpu')
    class_mask = class_mask.detach().to('cpu')
    mins = tensors.min(dim=1)[0]
    maxs = tensors.max(dim=1)[0]
    hists = torch.stack([tensor.histc() for tensor in tensors])
    for c in range(C):
        for k in range(K):
            hist = torch.masked_select(tensors[c], class_mask[k] > .5).histc()
            plt.plot(torch.linspace(mins[c].item(), maxs[c].item(), 100).numpy(), hist.numpy())
        plt.legend(['class = {}'.format(k) for k in range(K)])
        plt.xlabel('val')
        plt.ylabel('freq')
        plt.title('Tensor {}'.format(c))
        plt.savefig(os.path.join(config['tempinfo_path'], 'histogram{:03d}.eps'.format(c)))
        plt.close()

def debug_plothist(tensors, valrange=None):
    B,N = tensors.size()
    tensors = tensors.detach().to('cpu')
    if valrange is None:
        mins = tensors.min(dim=1)[0]
        maxs = tensors.max(dim=1)[0]
    hists = torch.stack([tensor.histc() for tensor in tensors])
    for b in range(B):
        if valrange is None:
            plt.plot(torch.linspace(mins[b].item(), maxs[b].item(), 100).numpy(), hists[b].numpy())
#        else:
#            plt.plot(torch.linspace(valrange[0], valrange[1], 100).numpy(), hists[b].numpy())
        plt.xlabel('val')
        plt.ylabel('freq')
        plt.title('Tensor {}'.format(b))
        plt.savefig(os.path.join(config['tempinfo_path'], 'histogram{:03d}.eps'.format(b)))
        plt.close()

def plothist(tensor, valrange, fname, normalize=False):
    tensor = tensor.detach().to('cpu')
    values = torch.linspace(valrange[0], valrange[1], 100)
    hist = tensor.histc(min=valrange[0], max=valrange[1])
    if normalize:
        Nelem = hist.sum()
        hist = hist * 100 / Nelem
        mean = (hist * values).sum()
    else:
        mean = (hist * values / Nelem).sum()
    print("Saving {}, mean is {:.3f}".format(fname, mean))
    plt.plot(values.numpy(), hist.numpy())
    plt.xlabel('values')
    plt.ylabel('freq')
    if normalize:
        plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(config['tempinfo_path'], fname))
    plt.close()

def debug_imgsegbbox(image, seg, bbox, name):
    """
    args:
        image (Tensor): of size (nframes, 3, height, width)
        seg (Tensor): of size (nframes, 1, sy*height, sx*width) where sy and sx are some scale parameters
        bbox (Tensor): of size (nframes, 1, 4)
    """
    assert image.dim() == 4
    assert image.size(-3) == 3
    assert seg.dim() == 4
    assert seg.size(-3) == 1
    if bbox is not None:
        assert bbox.dim() == 3
        assert bbox.size(-2) == 1
    L,_,H,W = image.size()
    img_visualization = revert_imagenet_normalization(image)
    upsampled_seg = F.interpolate(seg, (H,W))
    seg_visualization = torch.cat([upsampled_seg, torch.zeros_like(upsampled_seg), torch.zeros_like(upsampled_seg)],
                                  dim=-3)
#    visualization = (1 - upsampled_seg)*visualization + upsampled_seg*(.5*visualization + .5*seg_visualization)
    visualization = img_visualization + .5*upsampled_seg*(seg_visualization - img_visualization)
    if bbox is not None:
        visualization = draw_bbox(visualization, bbox)
    path = config['tempinfo_path']
    for l in range(L):
        pic = tv.transforms.functional.to_pil_image((visualization[l,:,:,:]*256).clamp(0,255).cpu().byte())
        fname = "{}{:05d}.png".format(name, l)
        pic.save(os.path.join(path, fname))

def save_image(non_normalized_image, fname, is_imagenet_normalized=False):
    if is_imagenet_normalized:
        non_normalized_image = revert_imagenet_normalization(non_normalized_image.unsqueeze(0)).squeeze(0)
    pic = tv.transforms.functional.to_pil_image((non_normalized_image*256).clamp(0,255).cpu().byte())
    pic.save(fname)

def save_seg(seg, K, fname):
    pic = tv.transforms.functional.to_pil_image(seg.cpu().byte())
    pic.putpalette(MULTISEG_PALETTE)
    pic.save(os.path.join(config['tempinfo_path'], fname))

def save_seg_soft(segmap, fname):
    """Saves a segmap softly. The dominant class chooses the colour, and its level of assignment the intensity
    args:
        segmap (Tensor): of size (K,H,W) where K is number of classes, H and W are height and width
        fname (str): entire path with filename where segmap will be saved
    """
    assert segmap.size(0) <= 16, "Only supporting up to 16 classes yet"
    seg_its, seg_ids = segmap.max(dim=0)
    seg_its = ((16 - 1e-5) * seg_its).byte()
    seg_ids = (seg_ids * 16).byte()
    seg = (seg_its + seg_ids).view(1, segmap.size(-2), segmap.size(-1))
    pic = tv.transforms.functional.to_pil_image(seg.cpu())
    pic.putpalette(SOFT_MULTISEG_PALETTE)
    pic.save(fname)

def debug_multiseg(img, seg, name, l="", soft=True):
    """Saves image and segm. separately into two different files, where segm. will follow a color palette
    args:
        img (Tensor): of size (nframes, 3, height, width)
        seg (Tensor): of size (nframes, K, height, width)
    """
    assert img.dim() == 4
    assert img.size(-3) == 3
    assert seg.dim() == 4
    assert img.size(0) == seg.size(0)
    B,_,H,W = img.size()
    _,K,_,_ = seg.size()
    img_vis = revert_imagenet_normalization(img)
    upsampled_seg = F.interpolate(seg, (H,W))
    seg_ids = upsampled_seg.argmax(dim=1, keepdim=True)
    for b in range(B):
        img_fname = os.path.join(config['tempinfo_path'], "{}_{:05d}_img_{}.png".format(name, b, l))
        seg_fname = os.path.join(config['tempinfo_path'], "{}_{:05d}_seg_{}.png".format(name, b, l))
        save_image(img_vis[b,:,:,:], img_fname)
        if soft:
            save_seg_soft(upsampled_seg[b,:,:,:], seg_fname)
        else:
            save_seg(seg_ids[b,:,:,:], K, seg_fname)

def debug_segresult(image, seg, name, l=""):
    """
    args:
        image (Tensor): of size (nframes, 3, height, width)
        seg (Tensor): of size (nframes, 1, sy*height, sx*width)
    """
    assert image.dim() == 4
    assert image.size(-3) == 3
    assert seg.dim() == 4
    assert seg.size(-3) == 1
    B,_,H,W = image.size()
    img_visualization = revert_imagenet_normalization(image)
    upsampled_seg = F.interpolate(seg, (H,W))
    seg_visualization = torch.cat([upsampled_seg, torch.zeros_like(upsampled_seg), torch.zeros_like(upsampled_seg)],
                                  dim=-3)
#    visualization = (1 - upsampled_seg)*visualization + upsampled_seg*(.5*visualization + .5*seg_visualization)
    visualization = img_visualization + .5*upsampled_seg*(seg_visualization - img_visualization)
    path = config['tempinfo_path']
    for b in range(B):
        pic = tv.transforms.functional.to_pil_image((visualization[b,:,:,:]*256).clamp(0,255).cpu().byte())
        fname = "{}_{:05d}_{}.png".format(name, b, l)
        pic.save(os.path.join(path, fname))

def debug_tensor_imageoverlay(tensor, image):
    """
    args:
        tensor (Tensor): of size (nchannels, height, width)
        image (Tensor): Imagenet normalized image of size (3, height, width)
    """
    assert tensor.dim() == 3
    assert image.dim() == 3
    height, width = image.size()[-2:]
    nchannels = tensor.size(0)
    vis_tensor = revert_imagenet_normalization(image.view(1,3,height,width)) # 1 x 3 x H W
    upsampled_tensor = F.interpolate(tensor.unsqueeze(1), (height, width)).squeeze() # nchannels x H x W
    vis_tensor = vis_tensor + torch.cat([.5 * upsampled_tensor.view(nchannels,1,height,width),
                                         torch.zeros(nchannels,2,height,width).to(tensor.device)], dim=1)
    debug_tensor(vis_tensor)

def debug_tensor(tensor, dynamic_range=False):
    assert tensor.dim() == 4, "Tensor must have 4 dimensions (nsamples, nchannels, height, width) where nchannels is 1 or 3, got size {}".format(tensor.size())
    if dynamic_range:
        tmin = tensor.min()
        tmax = tensor.max()
        pics = [tv.transforms.functional.to_pil_image(((subtensor-tmin)*255 / (tmax-tmin)).cpu().byte())
                for subtensor in tensor]
    else:
        pics = [tv.transforms.functional.to_pil_image((subtensor*256).clamp(0,255).cpu().byte())
                for subtensor in tensor]
    path = config['tempinfo_path']
    existing_debug_tensors = sorted([fname for fname in os.listdir(path) if(len(fname) > 12
                                                                            and fname[:12] == 'debug_tensor'
                                                                            and fname[12:-4].isdecimal()
                                                                            and fname[-4:] == '.png')])
    fnum = int(existing_debug_tensors[-1][12:-4]) + 1 if len(existing_debug_tensors) > 0 else 0

    for idx, pic in enumerate(pics):
        fname = "debug_tensor{:05d}.png".format(fnum + idx)
        pic.save(path + fname)
    
def single_object_dumper(targets):
    sm = utils.ReadSaveYTVOSChallengeLabels()
    t = str(int(time.time()*1e6))

    for idx, target in targets.items():
        fine, _ = target
        fine = torch.argmax(fine, dim=-3)
        filename = '_obj{:02d}.png'.format(idx)
        path = os.path.join(config['tempinfo_path'], 'single_object_dumper', t + filename)
        sm.save(fine.cpu().squeeze().numpy(), path)

def tensor_dumper(tensor, scale_factor):
    """ tensor_dumper(), dumping samples and channels of a 4d tensor, SxNxHxW, as 16 bit png images
    Args:
        tensor (FloatTensor): tensor to dump
        scale_factor (float): interpolation factor
    Returns:
        None
    """
    sm = utils.PngMono()
    t = int(time.time()*1e6)

    # Scale for visability
    tensor = F.interpolate(tensor, scale_factor=scale_factor, mode='bilinear')

    samples,channels,_,_ = tensor.size()
    for s in range(samples):
        for c in range(channels):
            tmp = tensor[s,c]

            # Dynamic
            tmin = tmp.min()
            tmax = tmp.max()
            if tmin.int() == 0 and tmax.int() == 0:
                tmp = tmp.long()
            else:
                tmp = ((2**16-1)*(tmp-tmin)/(tmax-tmin)).long()

            # Save
            filename = '{time:06d}_{sample:02d}_{channel:02d}.png'.format(time=t, sample=s, channel=c)
            path = os.path.join(config['tempinfo_path'], 'tensor_dumper', filename)
            sm.save(tmp.cpu().squeeze().numpy(), 16, path)


def spark_string(ints, fit_min=False):
    """Returns a spark string from given iterable of ints.

    Keyword Arguments:
    fit_min: Matches the range of the sparkline to the input integers
             rather than the default of zero. Useful for large numbers with
             relatively small differences between the positions
    """
    ticks = u' ▁▂▃▄▅▆▇█'
    min_range = min(ints) if fit_min else 0
    step_range = max(ints) - min_range
    step = (step_range / float(len(ticks) - 1)) or 1
    return u''.join(ticks[int(round((i - min_range) / step))] for i in ints)

def text_bargraph(values):
    blocks = np.array(('u', ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', 'o'))
    nsteps = len(blocks)-2 -1
    hstep = 1 / (2*nsteps)
    values = np.array(values)
    nans = np.isnan(values)
    values[nans] = 0  # '░'
    indices = ((values + hstep) * nsteps + 1).astype(np.int)
    indices[values < 0] = 0
    indices[values > 1] = len(blocks)-1
    graph = blocks[indices]
    graph[nans] = '░'
    graph = '\u250A'+str.join('', graph)+'\u250A'
    return graph

def draw_box(image, boxes, boxlabels, thickness=5):
    H, W = image.size()[-2:]
    boxids = torch.nonzero(boxlabels >= 0) # Only a boxlabel >= 0 corresponds to an actual detection
    boxids = boxids.squeeze(1) # Nonzero gives (num_nonzero, tensor_order), we remove tensor_order which is 1
    if boxids.dim() == 0: # No boxes found
        return image
    colors = JOAKIM_TORCH_PALETTE4[boxlabels[boxids]]
    boxes_filtered = boxes[boxids]
    for idx in range(boxids.size(0)):
        left   = int(boxes_filtered[idx,0].clamp(thickness, W - thickness))
        bottom = int(boxes_filtered[idx,3].clamp(thickness, H - thickness))
        right  = int(boxes_filtered[idx,2].clamp(thickness, W - thickness))
        top    = int(boxes_filtered[idx,1].clamp(thickness, H - thickness))
        image[:, top - thickness : bottom, left - thickness : left]     = colors[idx]
        image[:, bottom : bottom + thickness, left - thickness : right] = colors[idx]
        image[:, top : bottom + thickness, right : right + thickness]   = colors[idx]
        image[:, top - thickness : top, left : right + thickness]       = colors[idx]
    return image    

# We might want to plot: image, image+seg, seg, segmap, feats. Boxes can always be added
def debug_plot_image(fname, image, seg, boxes, boxlabels):
    H,W = image.size()[-2:]
    image = image.detach().cpu()
    image = revert_imagenet_normalization(image)[0]
    
    if seg is not None:
        seg = seg.detach().cpu()
#        seg = F.interpolate(seg.float().view(1, 1, seg.size()[0], seg.size()[1]), size=(H,W)).long().view(H,W)
        assert seg.size() == (H,W), f"seg of wrong size, {seg.size()}, expected ({H},{W})"
        K = seg.max()
        for k in range(1, K + 1):
            image[:,seg == k] = 0.5 * image[:,seg == k] + 0.5 * JOAKIM_TORCH_PALETTE4[k].view(3,1)
    if boxes is not None:
        if boxlabels is None:
#            boxlabels = torch.ones((boxes.size(0),), dtype=torch.int64)
            boxlabels = torch.arange(boxes.size(0))
        image = draw_box(image, boxes, boxlabels)
    pic = tv.transforms.functional.to_pil_image((image * 256).clamp(0,255).cpu().byte())
    pic.save(os.path.join(config['tempinfo_path'], fname))

def debug_plot_seg(fname, seg, size, boxes):
    if boxes is not None:
        raise NotImplementedError("Implement box plotting")
    save_seg(seg, None, fname)
    
def debug_plot_segmap(fname, segmap, size, boxes):
    segmap = F.interpolate(segmap.view(1,*segmap.size()), size=size).squeeze(0)
    if boxes is not None:
        raise NotImplementedError("Implement box plotting")
    save_seg_soft(segmap, os.path.join(config['tempinfo_path'], fname))    

def debug_plot_feats(fname, feats, boxes):
    raise NotImplementedError()

def debug_plot(fname, image=None, seg=None, segmap=None, boxes=None, boxlabels=None, feats=None, size=(480,864)):
    """
    args:
        fname (str): Name of file, will be saved in tempinfo dir
        image (Tensor): of size (3, height, width), RGB image that is imagenet normalized
        seg (LongTensor): of size (height, width), where sy and sx are scale params, values in {0,1,...,C-1}
        segmap (Tensor): of size (C, sy*height, sx*width) where sy and sx are scale params, values in [0,1]
        boxes (FloatTensor): of size (N, 4) where N is number of objects, in format (x1,y1,x2,y2)
        boxlabels (LongTensor): of size (N,) where N is number of objects
        feats (Tensor): of size (D, sy*height, sx*width), where sy and sx are scale params, will be normalized
        size (Tuple of int): Size to save segmap and feats (they will be interpolated)
    """
    if image is not None:
        assert segmap is None and feats is None
        debug_plot_image(fname, image, seg, boxes, boxlabels)
    elif seg is not None:
        assert segmap is None and feats is None
        debug_plot_seg(fname, seg, size, boxes)
    elif segmap is not None:
        assert feats is None
        debug_plot_segmap(fname, segmap, size, boxes)
    elif feats is not None:
        debug_plot_feats(fname, feats, boxes)
    else:
        raise ValueError("debug_plot called with everything None ...")



NUMPY_PALETTE = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
])

MULTISEG_PALETTE = 16*[
      0,   0,   0,
    128,   0,   0,
      0, 128,   0,
    128, 128,   0,
      0,   0, 128,
    128,   0, 128,
      0, 128, 128,
    128, 128, 128,
     64,   0,   0,
    191,   0,   0,
     64, 128,   0,
    191, 128,   0,
     64,   0, 128,
    191,   0, 128,
     64, 128, 128,
    191, 128, 128
]

MULTISEG_COLORS = 1 / 256 * torch.tensor(MULTISEG_PALETTE, dtype=torch.float32).view(256,3)

SOFT_MULTISEG_BASECOLOR_PALETTE = [
     16,   0,   0,
      0,  16,   0,
      0,   0,  16,
     16,  16,   0,
     16,   0,  16,
      0,  16,  16,
     16,  16,  16,
     16,   8,   0,
      8,  16,   0,
     16,   0,   8,
      8,   0,  16,
      0,  16,   8,
      0,   8,  16,
     16,  16,   8,
     16,   8,  16,
      8,  16,  16
]
SOFT_MULTISEG_PALETTE = [intensity * SOFT_MULTISEG_BASECOLOR_PALETTE[base_color_idx+coldim]
                         for base_color_idx in range(0,3*16,3) for intensity in range(0,16) for coldim in [0,1,2]]

JOAKIM_TORCH_PALETTE = torch.tensor(list(product(*(3*[[0., 0.25, 0.5, 0.75, 1.]])))).view(125,3,1,1)
JOAKIM_TORCH_PALETTE4 = torch.tensor(list(product(*(3*[[0., 0.33, 0.67, 1.]])))).view(64,3,1,1)


# We might want to plot: image, image+seg, seg, segmap, feats. Boxes can always be added
def debug_plot_image_withoutconfigdependence(image, seg, boxes, boxlabels):
    H, W = image.size()[-2:]
    image = image.detach().cpu()
    image = revert_imagenet_normalization(image)[0]

    if seg is not None:
        seg = seg.detach().cpu()
        #        seg = F.interpolate(seg.float().view(1, 1, seg.size()[0], seg.size()[1]), size=(H,W)).long().view(H,W)
        assert seg.size() == (H, W), f"seg of wrong size, {seg.size()}, expected ({H},{W})"
        #K = seg.max()
        #for k in range(1, K + 1):
        #    image[:, seg == k] = 0 #0.5 * image[:, seg == k] + 0.5 * JOAKIM_TORCH_PALETTE[k].view(3, 1)
        image = image*seg
    if boxes is not None:
        if boxlabels is None:
            boxlabels = torch.ones((boxes.size(0),), dtype=torch.int64)
        image = draw_box(image, boxes, boxlabels)
    pic = tv.transforms.functional.to_pil_image((image * 256).clamp(0, 255).cpu().byte())
    return pic

def vps_debug_plot(detections, sequence, path):
    """ wrapping of debug_plot, implements structure of vps_evaluator_v1
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        path                       (Str): final path for dumping
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    out = []
    for frame_idx, dets in enumerate(detections):
        iimg = sequence['images'][frame_idx]
        filename = os.path.join(path, '{:05d}.jpg'.format(frame_idx + sequence['split_idx'] * sequence['split_size']))

        # We must handle None since this is what is returned by the model
        # after postprocessing, see yolact_fullmask.py for an example.
        if dets['class'] is None:
            # Save original image without overlay
            iimg = iimg.detach().cpu()
            iimg = revert_imagenet_normalization(iimg)[0]
            oimg = tv.transforms.functional.to_pil_image((iimg * 256).clamp(0, 255).cpu().byte())
            out.append(0)
        else:
            seg = (torch.sum(dets['mask'], dim=-3) > 0).byte()
            box = dets['box']
            oimg = debug_plot_image_withoutconfigdependence(iimg, seg, box, dets['class'])
            out.append(box.shape[0])

        oimg.save(filename)

    return out


def plot_grad_flow(named_parameters, iteration, epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.clf()
    plt.figure(figsize=(35, 40))
    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.max())
            min_grads.append(p.grad.min())
    plt.bar(np.arange(len(min_grads)), min_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=3, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-10, top=10.0)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['min-gradient', 'max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(os.path.join(config['tempinfo_path'], 'grad_flow_iteration_{:06d}_epoch_{:03d}.png'.format(iteration, epoch)), dpi=200)
    plt.close()
