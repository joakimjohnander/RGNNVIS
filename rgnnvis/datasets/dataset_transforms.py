
import random
from math import ceil, floor

import torch
import torch.nn.functional as F
import torchvision as tv

import rgnnvis.utils.tensor_utils as tensor_utils

class ScaleCropAugment:
    def __init__(self,
                 get_scale_factors=(lambda: (.667, .675)),
                 imsize=(480, 864),
                 resample_boxes=False):
        self.get_scale_factors = get_scale_factors
        self.imsize = imsize
        self.resample_boxes = resample_boxes
    def __call__(self, images, active, boxes, instance_seg, labels, semantic_seg):
        # Scale
        L, _, H0, W0 = images.size()
        _, N, _ = boxes.size()
        scale_factors = self.get_scale_factors().tolist()
        images = F.interpolate(images, scale_factor=scale_factors, mode='bilinear')#, recompute_scale_factor=True)
        if not self.resample_boxes:
            box_scaling = torch.tensor([scale_factors[1], scale_factors[0]]).repeat(2).view(1, 1, 4)
            boxes = box_scaling * boxes
        instance_seg = F.interpolate(instance_seg.float().view(L, 1, H0, W0), scale_factor=scale_factors)
        semantic_seg = F.interpolate(semantic_seg.float().view(L, 1, H0, W0), scale_factor=scale_factors)

        # Pad
        _, _, H0, W0 = images.size()
        padding = (max(0, ceil((self.imsize[1] - W0)/2)),
                   max(0, floor((self.imsize[1] - W0)/2)),
                   max(0, ceil((self.imsize[0] - H0)/2)),
                   max(0, floor((self.imsize[0] - H0)/2)))
        images = F.pad(images, padding)
        box_translation = torch.tensor([padding[0], padding[2], padding[0], padding[2]], dtype=torch.float32)
        boxes = box_translation + boxes
        instance_seg = F.pad(instance_seg, padding)
        semantic_seg = F.pad(semantic_seg, padding)

        # Crop
        _, _, H0, W0 = images.size()
        x0 = random.choice(range(W0 - self.imsize[1] + 1))
        y0 = random.choice(range(H0 - self.imsize[0] + 1))
        x1 = x0 + self.imsize[1]
        y1 = y0 + self.imsize[0]
        images = images[:, :, y0:y1, x0:x1]
        instance_seg = instance_seg[:, :, y0:y1, x0:x1].long().view(L, *self.imsize)
        semantic_seg = semantic_seg[:, :, y0:y1, x0:x1].long().view(L, *self.imsize)
        if not self.resample_boxes:
            box_translation = torch.tensor([-x0, -y0, -x0, -y0], dtype=torch.float32)
            boxes = box_translation + boxes
        else:
            instance_segmap = (instance_seg.view(L, 1, *self.imsize) == torch.arange(N).view(1, N, 1, 1))
            boxes = tensor_utils.boolsegmap_to_boxes(instance_segmap)

        # Check active, boxes, and labels for disappeared instances
        _, _, H0, W0 = images.size()
        if not self.resample_boxes:
            disappeared_mask = ((W0 < (boxes[:,:,0] + boxes[:,:,2]) / 2)
                                + (H0 < (boxes[:,:,1] + boxes[:,:,3]) / 2)
                                + ((boxes[:,:,0] + boxes[:,:,2]) / 2 < 0)
                                + ((boxes[:,:,1] + boxes[:,:,3]) / 2 < 0))
            boxes[:,:,0] = boxes[:,:,0].clamp(0, W0)
            boxes[:,:,1] = boxes[:,:,1].clamp(0, H0)
            boxes[:,:,2] = boxes[:,:,2].clamp(0, W0)
            boxes[:,:,3] = boxes[:,:,3].clamp(0, H0)
        else:
            disappeared_mask = ~(instance_segmap.view(L, N, H0*W0).max(dim=2)[0])
        boxes[disappeared_mask] = -1
        for n in range(N):
            instance_seg[(instance_seg == n) * disappeared_mask[:,n].view(L, 1, 1)] = 0
        tmp_active = active
        active = ((active == 1) * ~disappeared_mask).long()
        has_been_active = active.cumsum(dim=0) > 0
        active[(active == 0) * has_been_active] = 2  # has been seen
        active[:, 0] = 3                             # background
        labels[(active == 0).max(dim=0)[0]] = 0
        return images, active, boxes, instance_seg, labels, semantic_seg
        
class HorizontalFlipAugment:
    def __call__(self, images, active, boxes, instance_seg, labels, semantic_seg):
        if random.random() > .5:
            return images, active, boxes, instance_seg, labels, semantic_seg
        images = images.flip(dims=[3])
        _, _, H, W = images.size()
        new_boxes = boxes.clone()
        new_boxes[:, :, 0] = W - boxes[:, :, 2]
        new_boxes[:, :, 2] = W - boxes[:, :, 0]
        instance_seg = instance_seg.flip(dims=[2])
        semantic_seg = semantic_seg.flip(dims=[2])
        return images, active, new_boxes, instance_seg, labels, semantic_seg

class ColorJitterAugment(tv.transforms.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, images, active, boxes, instance_seg, labels, semantic_seg):
        transform = tv.transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        images = transform(images)
        return images, active, boxes, instance_seg, labels, semantic_seg

class ComposeAugmentations:
    def __init__(self, *augmentations):
        self.augmentations = augmentations
    def __call__(self, *args):
        for augment in self.augmentations:
            args = augment(*args)
        return args
