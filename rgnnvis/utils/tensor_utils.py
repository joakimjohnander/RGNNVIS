import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import math
import numpy

import rgnnvis.utils
from pycocotools.mask import encode
#from itertools import groupby
#import scipy.ndimage as ndimage

def get_required_padding(height, width, div):
    height_pad = (div - height % div) % div
    width_pad = (div - width % div) % div
    padding = [(width_pad+1)//2, width_pad//2, (height_pad+1)//2, height_pad//2]
    return padding

def unpad(tensor, padding):
    _, _, _, height, width = tensor.size()
    tensor = tensor[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]
    return tensor

def pad_tensor(x, padding, mode):
    """ pad_tensor(), add padding to a tensor (image oriented but
                      could be working for volumes as well)
    Args:
        x (Tensor): tensor to pad
           padding: [left,right,top,bottom] amount of padding
              mode: "constant", "reflection" or "replication"
    Returns:
        tensor with padding added, height, width
    """

    if tuple(padding) != (0,0,0,0):
        y = F.pad(x, padding, mode=mode)
        _, _, height, width = y.size()
        return y, height, width
    else:
        _, _, height, width = x.size()
        return x, height, width

def unpad_tensor(tensor, padding):
    """ unpad_tensor(), remove padding from tensor
    Args:
        x (Tensor): tensor to unpad
           padding: [left,right,top,bottom] amount of padding
    Returns:
        tensor with padding removed
    """

    if tuple(padding) == (0, 0, 0, 0):
        return tensor
    else:
        _, _, height, width = tensor.size()
        return tensor[:, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]

def get_diff1d(a, b):
    """ get_diff1d(), simlar to numpy setdiff1d
    Args:
        a (ByteTensor): set of values
        b (ByteTensor): set of values
    Returns:
          (ByteTensor): regarding a and b as sets, a/b will be returned
    """

    indices = torch.ones_like(a)
    for elem in b:
        indices = indices & (a != elem)
    return a[indices]

def get_tensor_values(tensor, max_num_values=1e6):
    """Returns a list of all values in a tensor. If there are more than maxnum
        values in the tensor, an exception is raised. The function is intended
        for tensors with a small set of values, such as instance segmentation
        labels.
    Args:
        tensor (Tensor):
        max_num (int): Maximum number of values before breaking.
    Returns:
        list: Contains all values contained in a tensor.
    """
    val = tensor.min()
    maxval = tensor.max()
    values = [val.item()]
    while(val < tensor.max()):
        val = tensor[tensor > val].min()
        values.append(val.item())
    return values
#    val = tensor.max()
#    while (val >= minval) and (len(values) < max_num_values):
#        values.append(val)
#        tensor = tensor - (val - minval + 1)*((tensor == val).float())
#        val = tensor.max()
#    return values

def rectxywh_to_tensor(pos,size,tensor_size):
    """
    Args:
        pos (list): List of tuples (float,float), x- and y positions.
        size (list): List of tuples (float,float), widths and heights.
        tensor_size (int,int,int): Final tensor size (listlen,im_height,im_width)
    Returns:
        tensor: Of size tensor_size
    """
    # Handle case where a single pos, size and rect is used
    if not isinstance(pos[0], (list, tuple)):
        pos = [pos]
    if not isinstance(size[0], (list, tuple)):
        size = [size]
    if len(tensor_size) < 3:
        tensor_size = (1, tensor_size(0), tensor_size(1))
    try:
        tensor = torch.Tensor(tensor_size).zero_() 
        for i in range(tensor_size[0]):
            x0 = min(max(math.floor(pos[i][0]), 0), tensor_size[2]-1)
            x1 = min(max(math.ceil(pos[i][0] + size[i][0]), 1), tensor_size[2])
            y0 = min(max(math.floor(pos[i][1]), 0), tensor_size[1]-1)
            y1 = min(max(math.ceil(pos[i][1] + size[i][1]), 1), tensor_size[1])
            tensor[i, y0:y1, x0:x1] = 1.0
    except ValueError:
        print("x: {} -> {}, y: {} -> {}".format(x0,x1,y0,y1))
        print("pos: {}, size: {}, tensor_size: {}".format(pos,size,tensor_size))
        raise

    return tensor

def rect_to_tensor(rect, tensor_size, mode):
    assert mode in ('polygon','xywh','xyxy')
    if mode == 'polygon':
        x_positions = [rect[idx] for idx in range(0,len(rect), 2)]
        y_positions = [rect[idx] for idx in range(1,len(rect), 2)]
        pos = (min(x_positions), min(y_positions))
        rect_size = (max(x_positions) - pos[0], max(y_positions) - pos[1])
        return rectxywh_to_tensor(pos, rect_size, tensor_size)
    elif mode == 'xywh':
        return rectxywh_to_tensor(rect[0:2], rect[2:4], tensor_size)
    elif mode == 'xyxy':
        pos = (rect[0],rect[1])
        rect_size = (rect[2] - rect[0], rect[3] - rect[1])
        return rectxywh_to_tensor(pos, rect_size, tensor_size)
    else:
        raise ValueError("invalid mode, got {}".format(mode))

def get_centers_of_mass(tensor):
    """
    Args:
        tensor (Tensor): Size (*,height,width)
    Returns:
        Tuple (Tensor): Tuple of two tensors of sizes (*)
    """
    width = tensor.size(-1)
    height = tensor.size(-2)
    x_coord_im = torch.linspace(-1,1,width).repeat(height,1)
    y_coord_im = torch.linspace(-1,1,height).unsqueeze(0).transpose(0,1).repeat(1,width)
    x_mean = torch.mul(tensor,x_coord_im).sum(-1).sum(-1)/tensor.sum(-1).sum(-1)
    y_mean = torch.mul(tensor,y_coord_im).sum(-1).sum(-1)/tensor.sum(-1).sum(-1)
    return (x_mean, y_mean)
    
def scale_tensor(tensor):
    """
    Args:
        tensor (Tensor): Size (N,1,height,width)
    Returns:
        Tensor: Size (N,1,height,width)
    """
#    num_samples = tensor.size(0)
#    height = tensor.size(-2)
#    width = tensor.size(-1)

#    scales = torch.Tensor(num_samples,1,1,2).uniform_(0.9, 1.1)
#    grid = torch.stack([torch.linspace(-1,1,width).repeat(height,1), torch.linspace(-1,1,height).unsqueeze(0).transpose(0,1).repeat(1,width)], dim=-1)
#    scaled_grid = grid.mul(scales)
    
def translate_tensor(tensor):
    """ Randomly translate each image in a batch.
    Args:
        tensor (Tensor): Size (N,1,height,width)
    Returns:
        Tensor: Size (N,1,height,width)
    """

def augment_tensor_old0(tensor):
    """ Apply random affine transform to each image in a batch
    (scaling and translation).
    Args:
        tensor (Tensor): Size (N,1,height,width)
    Returns:
        Tensor: Size (N,1,height,width)
    """
    num_samples = tensor.size(0)
    height = tensor.size(-1)
    width = tensor.size(-2)
    transl = torch.Tensor(num_samples,2,1).uniform_(-0.1,0.1)
    scale = torch.eye(2) * torch.Tensor(num_samples,2,1).uniform_(0.9,1.1)
    
    affines = torch.cat([scale, transl], dim=-1)
    grids = torch.nn.functional.affine_grid(affines, tensor.size())
    augmented_tensor = torch.nn.functional.grid_sample(tensor, grids, mode='nearest')
    ndimage.binary_dilation
    return augmented_tensor.data

#def get_transl_tensor(t1,t2):
#    """Generate translation tensors.
#    Args:
#        t1 (Tensor): Of size(N,1)
#        t2 (Tensor): Of size(N,1)
#    Returns:
#        Tensor: Size (N,1,3,3)
#    """
#    t1 = t1.unsqueeze(-1).unsqueeze(-1)
#    t2 = t2.unsqueeze(-1).unsqueeze(-1)
#    size = t1.size()
#    result = torch.eye(3).repeat(size)
#    result[:,:,0,2] = t1
#    result[:,:,1,2] = t2
#    return base

#def get_scaling_tensor(s1,s2):
#    """Generate translation tensors.
#    Args:
#        t1 (Tensor): Of size(N,1)
#        t2 (Tensor): Of size(N,1)
#    Returns:
#        Tensor: Size (N,1,3,3)
#    """
#    s1 = s1.unsqueeze(-1).unsqueeze(-1)
#    s2 = s2.unsqueeze(-1).unsqueeze(-1)
#    size = s1.size()
#    result = torch.eye(3).repeat(size)
#    result[:,:,0,0] = s1
#    result[:,:,1,1] = s2
#    return base
#
#def get_rotation_tensor(phi):
#    """Generate translation tensors.
#    Args:
#        phi (Tensor): Of size(N,1)
#    Returns:
#        Tensor: Size (N,1,3,3)
#    """
#    phi = t1.unsqueeze(-1).unsqueeze(-1)
#    size = t1.size()
#    result = torch.eye(3).repeat(size)
#    result[:,:,0,0] = s1
#    result[:,:,1,1] = s2
#    return base

def get_structuring_elem(elem_type,size=5):
    if elem_type == 'disc':
        y,x = numpy.ogrid[-size:size+1,-size:size+1]
        if elem_type == 'disc':
            mask = (x**2+y**2 <= size**2)
        return torch.Tensor(mask.astype(float))
    else:
        print("### error in get_structuring_elem, received elem_type={}".format(elem_type))

def tensor_coarsen_gpu(tensor, structuring_elem_type='disc', radius=5):
    """Coarsens each image in tensor. This is done with a convolution using a
    kernel of ones and zeros. This is intended to work on images with
    elements 0.0, 1.0.
    Args:
        tensor (Tensor): Size (N,1,height,width)
        struct_elem (string): Only 'disc' implemented
        radius (int): Size of struct. element, convkernel of size 1+2*radius
    Returns:
        Tensor: Augmented images as a tensor of size (N,1,height,width)
    """
    num_samples = tensor.size(0)
    coarsener = torch.nn.Conv2d(1,1,11,padding=5,bias=False)
    coarsener.weight.data = get_structuring_elem('disc',size=5).unsqueeze(0).unsqueeze(0)
    coarsener.cuda()
#    coarsened_tensor = (coarsener(tensor)/(1.0)).float()
#    coarsened_tensor = (coarsener(tensor) >= 3.0).float()
    coarsened_tensor = coarsener(tensor)*0.5
#    coarsened_tensor = (coarsener((tensor >= 0.5).float()) >= 0.1).float()*255 #*0.5 + (coarsener((tensor >= 0.25).float()) >= 0.5).float()*0.25
    return coarsened_tensor

def tensor_coarsen(tensor, structuring_elem_type='disc', radius=5, blur=False):
    """Coarsens each image in tensor. This is done with a convolution using a
    kernel of ones and zeros. This is intended to work on images with
    elements 0.0, 1.0.
    Args:
        tensor (Tensor): Size (N,1,height,width)
        struct_elem (string): Only 'disc' implemented
        radius (int): Size of struct. element, convkernel of size 1+2*radius
    Returns:
        Tensor: Augmented images as a tensor of size (N,1,height,width)
    """
    num_samples = tensor.size(0)
    coarsener = torch.nn.Conv2d(1,1,2*radius+1,padding=radius,bias=False)
    coarsener.weight.data = get_structuring_elem('disc').unsqueeze(0).unsqueeze(0)
    coarsened_tensor = coarsener(tensor)
    if blur:
        coarsened_tensor = (1 - torch.exp(-0.1*coarsened_tensor)).data
    else:
        coarsened_tensor = (coarsened_tensor >= 1.0).data.float()
    return coarsened_tensor

#def tensor_blur(tensor, radius=5):
#    coarsener = torch.nn.Conv2d(1,1,radius*2+1, padding=radius, bias=false)
#    coarsener.weight.data.fill_(1.0)
#    normalizer = 

def tensor_affine_augment(tensor, translation=0.1, scaling=0.2, rotation=0.1,
                          center=True):
    """ Apply random affine transform to each image in a batch. This function
    is ugly beyond what is socially acceptable, but maybe it works...
    Args:
        tensor (Tensor): Size (N,1,height,width)
        translation (float): Translation of each image is sampled from a uniform
            distribution [-translation,translation]. Image coords are in [-1,1].
        scale (float): Scaling of each image is randomly sampled from a uniform
            distribution [1-scaling,1+scaling].
        rotation (float): Rotates each image by a value sampled from a uniform
            distribution [-pi rotation,pi rotation]. Note that rotations are
            given in a range of [-1,1] (and not in radians).
        center (float): Centers each image around its center of mass before
            applying the affine transformation.
    Returns:
        Tensor: Augmented images as a tensor of size (N,1,height,width)
    """
    num_samples = tensor.size(0)
    if center:
        offset = get_centers_of_mass(tensor)
        offset_x = offset[0].unsqueeze(-1)
        offset_y = offset[1].unsqueeze(-1)
    else:
        offset_x = torch.Tensor(num_samples,1,1).zero_()
        offset_y = torch.Tensor(num_samples,1,1).zero_()
    transl_x = torch.Tensor(num_samples,1,1).uniform_(-translation, translation)
    transl_y = torch.Tensor(num_samples,1,1).uniform_(-translation, translation)
    scale_x = torch.Tensor(num_samples,1,1).uniform_(1-scaling, 1+scaling)
    scale_y = torch.Tensor(num_samples,1,1).uniform_(1-scaling, 1+scaling)
    rot_angle = torch.Tensor(num_samples,1,1).uniform_(-rotation,rotation)*math.pi
    
    affines = torch.Tensor([[1,0,0],[0,1,0]]).repeat(num_samples,1,1)
    affines[:,0,0] = torch.mul(scale_x, torch.cos(rot_angle))
    affines[:,0,1] = torch.mul(scale_x, -torch.sin(rot_angle))
    affines[:,0,2] = (-torch.mul(offset_x, torch.mul(scale_x, torch.cos(rot_angle)))
                     + torch.mul(offset_y, torch.mul(scale_x, torch.sin(rot_angle)))
                     + transl_x
                     + offset_x)
    affines[:,1,0] = torch.mul(scale_y, torch.sin(rot_angle))
    affines[:,1,1] = torch.mul(scale_y, torch.cos(rot_angle))
    affines[:,1,2] = (-torch.mul(offset_x, torch.mul(scale_y, torch.sin(rot_angle)))
                     - torch.mul(offset_y, torch.mul(scale_y, torch.cos(rot_angle)))
                     + transl_y
                     + offset_y)

    grids = torch.nn.functional.affine_grid(affines, tensor.size())
    augmented_tensor = torch.nn.functional.grid_sample(tensor, grids, mode='bicubic').data
    augmented_tensor = tensor_coarsen(augmented_tensor, radius=5, blur=True)
#    coarsened_tensor = tensor_coarsen(augmented_tensor, structuring_elem_type='disc',radius=5)
#    coarsener = torch.nn.Conv2d(1,1,11,padding=5,bias=False)
#    coarsener.weight.data = get_structuring_elem('disc').unsqueeze(0).unsqueeze(0)
#    coarsened_tensor = (coarsener(Variable(augmented_tensor,volatile=True)) >= 1.0).data.float()
#    print("type: {},   size: {}".format(type(coarsened_tensor),coarsened_tensor.size()))
    return augmented_tensor

def video_augment(data, labels, video_crop_size=(256,448), video_rotation=(-0.2,0.2), frame_crop_noise=(-4,4), frame_rotation_noise=(-0.04,0.04), mirroring=True):
    """ Apply random affine transform to each image in a batch. This function
    is ugly beyond what is socially acceptable, but maybe it works...
    Args:
        data (Tensor): Size (num_frames,num_channels,height,width)
        labels (Tensor): Size (num_frames,1,height,width)
    Returns:
        Tensor: Augmented images as a tensor of size (N,1,height,width)
    """
    sample_len = data.size(0)
    num_channels = data.size(1)
    height = data.size(2)
    width = data.size(3)

    if mirroring and (torch.Tensor(1).uniform_(0,1)[0] < 0.5):
        data = torch.from_numpy(np.flip(data.numpy(),3).copy())
        labels = torch.from_numpy(np.flip(labels.numpy(),3).copy())

    rot_angles = torch.Tensor(1,1,1).uniform_(*video_rotation)*math.pi + torch.Tensor(sample_len,1,1).uniform_(*frame_rotation_noise)*math.pi
    
    affines = torch.Tensor([[1,0,0],[0,1,0]]).repeat(sample_len,1,1)
    aspect_change = width/height
    affines[:,0,0] = torch.cos(rot_angles)
    affines[:,0,1] = -torch.sin(rot_angles)/aspect_change
    affines[:,0,2] = 0
    affines[:,1,0] = aspect_change*torch.sin(rot_angles)
    affines[:,1,1] = torch.cos(rot_angles)
    affines[:,1,2] = 0

    grids_for_labels = torch.nn.functional.affine_grid(affines, labels.size())
    augmented_labels = torch.nn.functional.grid_sample(labels, grids_for_labels, mode='nearest').data
    grids_for_data = torch.nn.functional.affine_grid(affines, data.size())
    augmented_data = torch.nn.functional.grid_sample(data, grids_for_data, mode='nearest').data
    return (augmented_data, augmented_labels)

    
def tps_tensor(tensor):
    """ Applies a random thin plate spline to each image in a batch, using 5
    control points.
    Args:
        tensor (Tensor): Size (N,1,height,width)
    Returns:
        Tensor: Size (N,1,height,width)
    """
#    num_samples = tensor.size(0)
#    angles = torch.Tensor(5,1).uniform_(0, 2*math.pi/5)
#    angles = 

def get_error(predictions, gt):
    """ Calculates the classification error per image
    Args:
        predictions (Tensor): of size (video_len,batch,nclasses,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (video_len,batch,H,W)
    Returns:
        Tensor: of size (video_len,batch) with error for object class
    """
    videolen, batchsize, nclasses, h, w = predictions.size()
    prediction_max, prediction_argmax = predictions.max(-3)
    incorrect = gt.view(videolen, batchsize, h, w).ne(prediction_argmax).float().sum(-1).sum(-1)
    error = torch.div(incorrect, h*w)
    nframes = videolen*batchsize
    return error, nframes

def enumerated_to_onehot(tensor, nclasses):
    if tensor.dim() == 2:
        height, width = tensor.size()
        tensor = tensor.view(1, height, width)
    elif tensor.dim() == 3:
        nsamples, height, width = tensor.size()
        tensor = tensor.view(nsamples, 1, height, width)
    onehot = torch.cat([tensor == i for i in range(nclasses)], dim=-3).float()
    return onehot
    

def get_intersection_and_union(predictions, gt):
    """ Calculates the class intersections and unions of two tensors
    Args:
        predictions (Tensor): of size (nsamples,nclasses,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (nsamples,H,W)
    Returns:
        Tensor: of size (nsamples) with error for object class
    """
    nsamples, nclasses, height, width = predictions.size()
    prediction_max, prediction_argmax = predictions.max(-3)
    prediction_argmax = prediction_argmax.long()
    classes = torch.LongTensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1)
    pred_bin = (prediction_argmax.view(nsamples, 1, height, width) == classes)
    gt_bin = (gt.view(nsamples, 1, height, width) == classes)
    intersection = (pred_bin * gt_bin).float().sum(dim=0).sum(dim=-2).sum(dim=-1)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=0).sum(dim=-2).sum(dim=-1)
    intersection = intersection
    union = union
    return intersection, union

def get_intersection_over_union_mt(predictions, gt):
    """ Calculates the class intersections over unions of two tensors representing multiple targets
    Args:
        predictions (Tensor): of size (nsamples,nclasses,H,W), multinomial
        gt (Tensor): Ground truth segmentation, categorical of size (nsamples,1,H,W)
    Returns:
        Tensor: of size (nsamples) with error for object class
    """
    nsamples,nclasses,height,width = predictions.size()
    prediction_argmax = torch.argmax(predictions, dim=-3, keepdim=True) # returns torch.uint64

    classes = gt.new_tensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1)
    pred_bin = (prediction_argmax == classes)
    gt_bin = (gt == classes)
    intersection = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=-2).sum(dim=-1)
    assert (intersection > union).sum() == 0
    return intersection / (union + 1e-8)

def get_intersection_over_union(predictions, gt):
    """ Calculates the class intersections over unions of two tensors
    Args:
        predictions (Tensor): of size (nsamples,nclasses,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (nsamples,H,W)
    Returns:
        Tensor: of size (nsamples,nclasses) with error for each class
    """
    nsamples,nclasses,height,width = predictions.size()
    assert gt.size(0) == nsamples, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    assert gt.size(1) == height, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    assert gt.size(2) == width, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    prediction_max, prediction_argmax = predictions.max(-3)
    prediction_argmax = prediction_argmax.long()
    classes = gt.new_tensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1) # [1,K,1,1]
    pred_bin = (prediction_argmax.view(nsamples, 1, height, width) == classes)    # [N,K,H,W]
    gt_bin = (gt.view(nsamples, 1, height, width) == classes)                     # [N,K,H,W]
    intersection = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1)            # [N,K]
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=-2).sum(dim=-1)             # [N,K]
    assert (intersection > union).sum() == 0
    return (intersection + 1e-8) / (union + 1e-8)                                          # [N,K]

def get_intersection_over_union_bb(p, q):
    assert p.size() == q.size(), (p.size(), q.size())
    if p.dim() > 2:
        size = p.size()
        iou = get_intersection_over_union_bb(p.view(-1, 4), q.view(-1, 4))
        return iou.view(*size[:-1])
    xmin = torch.max(p[:,0], q[:,0])
    xmax = torch.min(p[:,0] + p[:,2], q[:,0] + q[:,2])
    ymin = torch.max(p[:,1], q[:,1])
    ymax = torch.min(p[:,1] + p[:,3], q[:,1] + q[:,3])
    intersection = (F.relu(xmax - xmin) * F.relu(ymax - ymin))
    union = p[:,2] * p[:,3] + q[:,2] * q[:,3] - intersection
    return (intersection + 1e-8) / (union + 1e-8)

def get_metrics2d(predictions, gt):
    """ Calculates the class intersections over unions of two tensors
    Args:
        predictions (Tensor): of size (nsamples,nclasses,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (nsamples,H,W)
    Returns:
        list of Tensor: each of size (nsamples), corresponding to iou, precision, and recall
    """
    nsamples,nclasses,height,width = predictions.size()
    assert gt.size(0) == nsamples, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    assert gt.size(1) == height, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    assert gt.size(2) == width, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    prediction_max, prediction_argmax = predictions.max(-3)
    prediction_argmax = prediction_argmax.long()
    classes = torch.LongTensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1)
    pred_bin = (prediction_argmax.view(nsamples, 1, height, width) == classes)
    gt_bin = (gt.view(nsamples, 1, height, width) == classes)

    true_positive = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1)
    false_positive = (pred_bin * (1 - gt_bin)).float().sum(dim=-2).sum(dim=-1)
    false_negative = ((1 - pred_bin) * gt_bin).float().sum(dim=-2).sum(dim=-1)
    true_negative = ((1 - pred_bin) * (1 - gt_bin)).float().sum(dim=-2).sum(dim=-1)
    assert (true_positive + false_positive + false_negative + true_negative == height*width).all(), "tp+fp+fn+tn is {}".format(true_positive + false_positive + false_negative + true_negative)

    return {'iou': true_positive / (true_positive + false_positive + false_negative + 1e-8),
            'pre': true_positive / (true_positive + false_positive + 1e-8),
            'rec': true_positive / (true_positive + false_negative + 1e-8)}

def get_iou(predictions, gt):
    """ Calculates the intersection over union.
    Args:
        predictions (Tensor): of size (video_len,batch,nchannels,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (video_len,batch,nchannels,H,W)
    Returns:
        Tensor: of size (video_len,batch) with iou for object class
    """
    intersection = torch.min(predictions, gt)
    intersection_area = torch.sum(torch.sum(intersection,dim=-1),dim=-1)
    union = torch.max(predictions, gt)
    union_area = torch.sum(torch.sum(union,dim=-1),dim=-1)
    iou = intersection_area/union_area

    return iou

def batch_many_to_many_box_iou(one, two, iou_on_empty=1):
    """
    Args:
        one (tensor): Of size (*,N,4)
        two (tensor): Of size (*,M,4)
        iou_on_empty (int): If two boxes are empty, do we set IoU to 0 or 1? Might want 1 if measuring performance
            as we might have correctly detected a fully occluded object. Might want 0 if doing object detector
            matching as non-defined bounding boxes are set to (-1,-1,-1,-1).
    Returns:
        tensor: Of size (*,N,M) with IoUs between all pairs of N and M boxes, with preceding dimensions
            treated as batch dimensions
    """
    batchsize_one = one.size()[:-2]
    batchsize_two = two.size()[:-2]
    assert batchsize_one == batchsize_two
    N = one.size()[-2]
    M = two.size()[-2]
    if N == 0 or M == 0:
        return torch.zeros((*batchsize_one, N, M), device=one.device)
    one = one.view(-1, N, 1, 4)
    two = two.view(-1, 1, M, 4)
    left   = torch.max(one[:,:,:,0], two[:,:,:,0])
    bottom = torch.min(one[:,:,:,3], two[:,:,:,3])
    right  = torch.min(one[:,:,:,2], two[:,:,:,2])
    top    = torch.max(one[:,:,:,1], two[:,:,:,1])
    area_intersection = F.relu(right - left) * F.relu(bottom - top)
    area_one = (one[:,:,:,2] - one[:,:,:,0]) * (one[:,:,:,3] - one[:,:,:,1])
    area_two = (two[:,:,:,2] - two[:,:,:,0]) * (two[:,:,:,3] - two[:,:,:,1])
    area_union = area_one + area_two - area_intersection
    if iou_on_empty == 1:
        IoU = (area_intersection + 1e-7) / (area_union + 1e-7)
    else:
        IoU = area_intersection / (area_union + 1e-7)
    IoU = IoU.view(*batchsize_one, N, M)
    return IoU


def batch_many_to_many_seg_iou(pred, anno, unassigned_iou=0.0, split_size=100000):
    """ Computes intersection over union, many to many. There are two ways to handle
        cuda out of memory. 1) Change split_size here or 2) the split_size regulating how
        many frames of the sequence that are processed in the evaluator (usually def. in
        the run file).
    Args:
        pred              (ByteTensor): one-hot tensor of size (B,L,Nmax,H,W)
        anno              (ByteTensor): one-hot tensor of size (B,L,Mmax,H,W)
        unassigned_iou         (Float): value for those iou that are nan
        split_size               (Int): split spatial dim. into chunks
    Returns:
        iou    (FloatTensor): intersection over union of size (B,L,Nmax,Mmax)
    """
    b, l, nmax, h, w = pred.shape
    _, _, mmax, _, _ = anno.shape

    # print('pred', pred.shape, 'anno', anno.shape)

    pred = pred.view(b, l, nmax, 1, h*w)
    anno = anno.view(b, l, 1, mmax, h*w)
    preds = torch.split(pred, split_size, dim=-1)
    annos = torch.split(anno, split_size, dim=-1)

    ntp = []
    nunion = []
    for p, a in zip(preds, annos):
        tp = p*a
        union = tp + p*(~a) + (~p)*a
        ntp.append(tp.float().sum(dim=-1, keepdim=True))
        nunion.append(union.float().sum(dim=-1, keepdim=True))

    ntp = torch.cat(ntp, dim=-1).sum(dim=-1)
    nunion = torch.cat(nunion, dim=-1).sum(dim=-1)
    iou = ntp/nunion

    # set nan to unassigned_iou value
    iou_nan = torch.isnan(iou)
    iou[iou_nan] = unassigned_iou

    return iou


def save_iou_as_images(predictions, gt, path):
    """ Takes two single channel tensors as input and stacks them in the
    channel dimension, yielding visualizations of predictions, gt, and
    their overlap.
    Args:
        predictions (Tensor): of size (video_len,batch,1,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (video_len,batch,1,H,W)
    """
    size = predictions.size()
    assert size[-3] == 1
    assert size == gt.size()
    tertiary_tensor = torch.FloatTensor(size).zero_()
    tensor = torch.cat([predictions, gt, tertiary_tensor], dim=-3)
    utils.save_tensor_as_images(tensor, path + "_iou")

    
def image_tensor_type_change(tensor, intype=('RGB',1.0,'standard'), outtype=('BGR',255.0,'imagenet')):
    """
    Args:
        tensor (Tensor): Size (nchannels, height, width)
    """
    assert intype[0] in ['RGB','BGR']
    assert outtype[0] in ['RGB','BGR']
    assert intype[2] in ['standard','imagenet']
    assert outtype[2] in ['standard','imagenet']

    if((intype[0] == 'RGB' and outtype[0] == 'BGR')
       or (intype[0] == 'BGR' and outtype[0] == 'RGB')):
        c1,c2,c3 = tensor.chunk(3,dim=0)
        tensor = torch.cat((c3,c2,c1), dim=0).contiguous()

    imnet_mean = (lambda x: x if outtype[0] == 'RGB' else x[::-1])([.485020, .457957, .407604])
    if(intype[2] == 'standard' and outtype[2] == 'imagenet'):
        mean = imnet_mean
    elif(intype[2] == 'imagenet' and outtype[2] == 'standard'):
        mean = [-val for val in imnet_mean]
    else:
        mean = [0,0,0]
    mean = [val * intype[1] for val in mean]
    std = (intype[1]/outtype[1],intype[1]/outtype[1],intype[1]/outtype[1])

    return tensor_normalize(tensor, mean, std)


def label_tensor_type_change(tensor, intype=('L',1.0), outtype=('L',255.0)):
    """
    Args:
        tensor (Tensor): Size (nchannels, height, width)
    """
    return tensor*outtype[1]/intype[1]


def tensor_normalize(tensor, mean, std):
    """ Taken from recent torchvision.transforms
    Args:
        tensor (Tensor): order 3 input tensor (nchannels, height, width)
        mean (tuple): nchannels values to be substracted from each channel
        std (tuple): nchannels values each channel is divided by
    Returns:
        Tensor: Normalized
    """
    return torch.cat(
        tuple(
            map(lambda im,mean,std: (im - mean)/std,
                tensor.chunk(tensor.size(0),dim=0),
                mean,
                std)),
        dim=0)

def get_laplace_tensor(size, mu, b):
    return torch.Tensor(*size).exponential_(1/b) - torch.Tensor(*size).exponential_(1/b) + mu

def cluster_emgmmdiagdiff(tensor, K, priors, init, niter):
    """
    args:
        tensor (Tensor): Of size (batchsize, nchannels, nsamples)
        K (int): Number of clusters
        priors (Tensor): Of size (batchsize, 1 (or none), nsamples, 1 (or none))
    returns:
        Tensor (batchsize, K, nsamples): Responsibility, or membership to each cluster (it is a winner-takes-it-all)
    """
    B, C, N = tensor.size() # batchsize x nchannels x nsamples
    tensor = tensor.transpose(1, 2) # B x N x C
    if isinstance(priors, float):
        priorval = priors
        priors = torch.zeros(B, 1, N, 1).fill_(priorval).to("cuda")
    elif isinstance(priors, torch.Tensor):
        priors = priors.view(B, 1, N, 1)
    if init == 'forgy':
        means = tensor[:,sorted(random.sample(range(N), K)),:].view(B, K, 1, C).detach()
        covs = torch.ones(B, K, 1, C).to("cuda")
    tensor = tensor.view(B, 1, N, C)
    for i in range(niter):
        responsibility = F.softmax(-1/2 * ((tensor-means)*(tensor-means)/(1e-8 + covs)).sum(-1, keepdim=True), dim=1)
        responsibility = responsibility * priors # B x K x N x 1
        means = (tensor * responsibility).sum(-2,keepdim=True) / (1e-8 + responsibility.sum(-2, keepdim=True))
        covs = 1. + (responsibility * ((tensor - means) * (tensor - means))).sum(-2, keepdim=True) / (1e-8 + responsibility.sum(-2, keepdim=True))
    
    return responsibility.float().view(B,K,N), means, covs

def cluster_emgmmdiag(tensor, K, priors, init, niter):
    """
    args:
        tensor (Tensor): Of size (batchsize, nchannels, nsamples)
        K (int): Number of clusters
        priors (Tensor): Of size (batchsize, 1 (or none), nsamples, 1 (or none))
    returns:
        Tensor (batchsize, K, nsamples): Responsibility, or membership to each cluster (it is a winner-takes-it-all)
        Tensor (batchsize, K, 1, C)
        Tensor (batchsize, K, 1, C)
    """
    B, C, N = tensor.size() # batchsize x nchannels x nsamples
    tensor = tensor.detach() # Do not calculate derivatives w.r.t. clustering. Does NOT copy, and may not be changed
    tensor = tensor.transpose(1, 2) # B x N x C
    if isinstance(priors, float):
        priorval = priors
        priors = torch.zeros(B, 1, N, 1).fill_(priorval).to("cuda")
    elif isinstance(priors, torch.Tensor):
        priors = priors.view(B, 1, N, 1)
    if init == 'forgy':
        means = tensor[:,sorted(random.sample(range(N), K)),:].view(B, K, 1, C)
        covs = torch.ones(B, K, 1, C).to("cuda")
    tensor = tensor.view(B, 1, N, C)
    for i in range(niter):
        responsibility = F.softmax(-1/2 * ((tensor-means)*(tensor-means)/(1e-8 + covs)).sum(-1, keepdim=True), dim=1)
        responsibility = responsibility * priors # B x K x N x 1
        means = (tensor * responsibility).sum(-2,keepdim=True) / (1e-8 + responsibility.sum(-2, keepdim=True))
        covs = 1. + (responsibility * ((tensor - means) * (tensor - means))).sum(-2, keepdim=True) / (1e-8 + responsibility.sum(-2, keepdim=True))
        for b in range(B):
            for k in range(K):
                if responsibility[b,k,:,:].sum(-2) == 0.0:
                    means[b,k,:,:] = tensor[b,:,random.randint(0,N-1),:]
                    covs = torch.ones(B, K, 1, C).to("cuda")
    return responsibility.float().view(B,K,N), means, covs

def cluster_kmeans(tensor, K, priors, init, niter):
    """
    args:
        tensor (Tensor): Of size (batchsize, nchannels, nsamples)
        K (int): Number of clusters
        priors (Tensor): Of size (batchsize, 1 (or none), nsamples, 1 (or none))
    returns:
        Tensor (batchsize, K, nsamples): Responsibility, or membership to each cluster (it is a winner-takes-it-all)
    """
    B, C, N = tensor.size() # batchsize x nchannels x nsamples
    tensor = tensor.detach() # Do not calculate derivatives w.r.t. clustering. Does NOT copy, and may not be changed
    tensor = tensor.transpose(1, 2) # B x N x C
#    print(B,C,N, priors.size())
    if isinstance(priors, float):
        priorval = priors
        priors = torch.zeros(B, 1, N, 1).fill_(priorval).to("cuda")
    elif isinstance(priors, torch.Tensor):
        priors = priors.view(B, 1, N, 1)
    if init == 'forgy':
        means = tensor[:,sorted(random.sample(range(N), K)),:].view(B, K, 1, C)
    tensor = tensor.view(B, 1, N, C)
    for i in range(niter):
        dists = ((tensor - means) * (tensor - means)).sum(dim=-1).view(B, K, N, 1)
        responsibility = (dists.argmin(dim=1).view(B,1,N,1) == torch.arange(K).view(1,K,1,1).to("cuda")).float()
        responsibility = responsibility * priors # B x K x N x 1
        means = (tensor * responsibility).sum(-2,keepdim=True) / (1e-8 + responsibility.sum(-2, keepdim=True))
        for b in range(B):
            for k in range(K):
                if responsibility[b,k,:,:].sum(-2) == 0.0:
                    means[b,k,:,:] = tensor[b,:,random.randint(0,N-1),:]
#    assert (responsibility.sum(-2) != 0.0).all(), "One cluster is empty"
    return responsibility.float().view(B,K,N)
        

def cluster(tensor, K, priors=1.0, strategy='kmeans', init='forgy', niter=10):
    """
    Args:
        tensor (Tensor): Of size (batchsize, nchannels, nsamples)
        K (int): Number of clusters
        strategy (str): in (kmeans, emgmm)
        init (str): initialization for means, in (forgy, randompartition)
    """
    if strategy == 'emgmmdiag':
        return cluster_emgmmdiag(tensor, K, priors, init, niter)
    elif strategy == 'emgmmdiagdiff':
        return cluster_emgmmdiagdiff(tensor, K, priors, init, niter)
    elif strategy == 'kmeans':
        return cluster_kmeans(tensor, K, priors, init, niter)
    raise NotImplementedError()
    return None

def convert_segmentations_to_bboxes(segmentation: torch.Tensor) -> torch.Tensor:
    """
    args:
        segmentation: size (*,1,H,W)
    """
    print("WARNING: I do not think this function works!")
    def to_bbox(seg):
        H, W = seg.size()
        dtype = seg.dtype
        device = seg.device
        target_ids = seg.unique().tolist()
        xvals = torch.arange(W, device=device, dtype=dtype).view(1,W)
        yvals = torch.arange(H, device=device, dtype=dtype).view(1,H)
        if 0 in target_ids: target_ids.remove(0)
        bboxes = - torch.ones(max_num_targets, 4)
        for n in target_ids:
            target_xvals = (xvals * (seg == n).max(dim=-2)[0].long()) # size B,W
            target_yvals = (yvals * (seg == n).max(dim=-1)[0].long()) # size B,H
            xmin = xvals.min(dim=1)[0]
            xmax = xvals.max(dim=1)[0]
            ymin = yvals.min(dim=1)[0]
            ymax = yvals.max(dim=1)[0]
            bboxes[n-1,:] = torch.tensor([xmin, ymin, xmax-xmin, ymax-ymin])
        return bboxes

    assert segmentation.dim() >= 3
    seg_tensorsize = segmentation.size()
    max_num_targets = segmentation.unique().max()
    bboxes = torch.stack([to_bbox(seg) for seg in segmentation.view(-1, *seg_tensorsize[-2:])], dim=0)
    bbox_tensorsize = seg_tensorsize[:-3] + (max_num_targets, 4)
    return bboxes.view(bbox_tensorsize)

def get_limited_window(reference_tensor, window_bbox):
    assert tensor.dim() == 4, tensor.dim()                       # nsamples, nchannels, height, width
    assert window_bbox.dim() == 3, window_bbox.dim()             # nsamples, height, width
    assert resampled_size.dim() in (2,3), resampled_size.dim()   # nsamples, height, width
    assert window_bbox.size(2) == 4, window_bbox.size(2)         # must be bbox
    window_bbox = torch.max(0, window_bbox)
    window_bbox[:,0] = torch.min(reference_tensor.size(-1) - 1, window_bbox[:,0])
    window_bbox[:,1] = torch.min(reference_tensor.size(-2) - 1, window_bbox[:,1])
    window_bbox[:,2] = torch.min(reference_tensor.size(-1) - window_bbox[:,0], window_bbox[:,2])
    window_bbox[:,3] = torch.min(reference_tensor.size(-2) - window_bbox[:,1], window_bbox[:,3])
    return window_bbox

def crop_tensor(tensor, window_bbox):
    x0 = window_bbox[:,0]
    x1 = window_bbox[:,0] + window_bbox[:,2]
    y0 = window_bbox[:,1]
    y1 = window_bbox[:,1] + window_bbox[:,3]
    B = tensor.size(0)
    tensor_crop_lst = [tensor[b,:,y0:y1,x0:x1] for b in range(B)]
    tensor_crop = torch.stack(tensor_crop_lst)
    return tensor_crop

def resample(tensor, window_bbox, resampled_size):
    assert tensor.dim() == 4, tensor.dim()                       # nsamples, nchannels, height, width
    assert window_bbox.dim() == 2, window_bbox.dim()             # nsamples, 4
    assert resampled_size.dim() == 2, resampled_size.dim()       # nsamples, 2
    assert window_bbox.size(1) == 4, window_bbox.size(1)         # must be bbox
    assert (window_bbox[:,0] >= 0.).all()
    assert (window_bbox[:,1] >= 0.).all()
    assert (window_bbox[:,0] + window_bbox[:,2] <= tensor.size(-1)).all()
    assert (window_bbox[:,1] + window_bbox[:,3] <= tensor.size(-2)).all()
    assert (window_bbox[:,2] >= 0).all()
    assert (window_bbox[:,3] >= 0).all()
    B = tensor.size(0)
    tensor_crop = crop_tensor(tensor, window_bbox)
    
def recursive_apply(x, fun):
    if isinstance(x, list):
        return [recursive_apply(elem, fun) for elem in x]
    elif isinstance(x, tuple):
        return tuple([recursive_apply(elem, fun) for elem in x])
    elif isinstance(x, dict):
        return {key: recursive_apply(val, fun) for key, val in x.items()}
    else:
        return fun(x)


def index_map_to_one_hot(tensor, classes, device):
    """ Transforms categorical/index map to one-hot encoding
    Args:
        tensor    (ByteTensor): indexes in dim=-3
        classes    (ByteTensor: one dimensional of classes, eg [0,1,2]
        device  (torch.device): cpu or gpu
    Returns:
        out (ByteTensor): one-hot encoded
    """

    tmp = torch.ones(tensor.dim()).byte()
    tmp[-3] = max(classes)+1
    cl = torch.tensor(range(max(classes)+1)).byte().to(device)
    to_shape = torch.Size(tmp)
    tmp = cl.view(to_shape)
    out = (tensor == tmp)
    return out


def compute_statistical_measures(prediction, ground_truth):
    """ Computes statistical measures true positives (tp), false positives (fp), etc.
    Args:
        prediction   (ByteTensor): one-hot encoding of the prediction
        ground_truth (ByteTensor): one-hot encoding of the ground truth
    Returns:
        measures      (dict[str]): see description
    """

    measures = {}
    measures['tp'] = prediction*ground_truth
    measures['fp'] = prediction*(~ground_truth)
    measures['tn'] = (~prediction) * (~ground_truth)
    measures['fn'] = (~prediction) * ground_truth

    # intersection = tp
    # union = tp+fp+fn

    return measures


def compute_intersection_over_union(intersection, union):
    """ Computes intersection over union
    Args:
        intersection (ByteTensor): true positives for objects, input size [C,H,W]
        union        (ByteTensor): tp+fp+fn for objects, input size [C,H,W]
    Returns:
        iou      (FloatTensor): intersection over union of size [C,N]
    """

    intersection = intersection.float().sum(dim=-2).sum(dim=-1)
    union = union.float().sum(dim=-2).sum(dim=-1)
    iou = intersection/union
    return iou

def match_boxes_injective(boxes_one, boxes_two, iou_threshold):
    """Injective here means that each box is matched with one or zero boxes in the other set. The function
    is greedy and finds the largest overlapping pair, matching these.
    Args:
        boxes_ones (FloatTensor): of size (N,4), boxes on (x1,y1,x2,y2) form
        boxes_two (FloatTensor): of size (M,4), boxes on (x1,y1,x2,y2) form
    Returns:
        (N,) LongTensor: For each box in boxes_one, an idx corresponding to the best matching box in boxes_two
    """
    N, _ = boxes_one.size()
    M, _ = boxes_two.size()
    if N == 0 or M == 0:
        return [], []
    ids_one = []
    ids_two = []
    iou = batch_many_to_many_box_iou(boxes_one, boxes_two, iou_on_empty=0) # (N,M) tensor
    for _ in range(M): # Greedy one-to-one matching
        N_best_iou, N_best_m = iou.max(dim=1) # (N,) tensors
        best_iou, best_n = N_best_iou.max(dim=0)
        best_m = N_best_m[best_n]

        if best_iou > iou_threshold:
            ids_one.append(best_n.item())
            ids_two.append(best_m.item())
            
            iou[best_n,:] = -1. # Box n taken, don't use again
            iou[:,best_m] = -1. # Box m taken, don't use again
        else:
            break
    return ids_one, ids_two

def resize_spatial_tensor(x, new_size, mode='bilinear'):
    assert x.dim() >= 2, f"Input must have size (*, H, W), got {x.size()}"
    old_size = x.size()
    x = F.interpolate(x.view(-1, 1, old_size[-2], old_size[-1]), size=new_size, mode=mode)
    x = x.view(*old_size[:-2], *new_size)
    return x

def resize_boxes(boxes, old_size, new_size):
    old_shape = boxes.size()
    assert old_shape[-1] == 4, f"Input must have size (*, 4), got {boxes.size()}"
    boxes = boxes.view(-1, 4)
    assert (boxes[:, [0, 1]] <= boxes[:, [2, 3]]).all(), f"Boxes expected in (x0,y0,x1,y1) form"
    scale_tensor = torch.tensor([new_size[1] / old_size[1],
                                 new_size[0] / old_size[0],
                                 new_size[1] / old_size[1],
                                 new_size[0] / old_size[0]],
                                dtype=torch.float32,
                                device=boxes.device)
    boxes = scale_tensor * boxes
    boxes = boxes.view(*old_shape)
    return boxes

def boolsegmap_to_boxes(segmap):
    """
    Args:
        segmap (Tensor): Of size (*, N, H, W)
    Returns:
        Tensor of size (*, N, 4) with boxes fit to the segmentation maps
    """
    old_shape = segmap.size()
    N, H, W = old_shape[-3:]
    device = segmap.device
    assert segmap.dim() >= 3, f"Input must have size (*, N, H, W), got {segmap.size()}"
    segmap = segmap.view(-1, N, H, W)
    B, _, _, _ = segmap.size()
    xgrid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    ygrid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xmin = xgrid.where(segmap, torch.tensor(W, device=device).float()).view(B, N, H*W).min(dim=2)[0]
    ymin = ygrid.where(segmap, torch.tensor(H, device=device).float()).view(B, N, H*W).min(dim=2)[0]
    xmax = xgrid.where(segmap, torch.tensor(0, device=device).float()).view(B, N, H*W).max(dim=2)[0] + 1
    ymax = ygrid.where(segmap, torch.tensor(0, device=device).float()).view(B, N, H*W).max(dim=2)[0] + 1
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)
#    obj_xcoords = (segmap * xgrid).view(-1, N, H*W)
#    obj_ycoords = (segmap * ygrid).view(-1, N, H*W)
#    boxes = torch.stack([obj_xcoords.min(dim=2)[0],
#                         obj_ycoords.min(dim=2)[0],
#                         obj_xcoords.max(dim=2)[0],
#                         obj_ycoords.max(dim=2)[0]],
#                        dim=2)
    boxes = boxes.view(*old_shape[:-3], N, 4)
#    print("segmap siez: ", segmap.size())
#    print("xgrid minimum at x in ( 0..10)", xgrid[0,0,0,:10].min())
#    print("xgrid minimum at x in (10..20)", xgrid[0,0,0,10:20].min())
#    print("xgrid minimum at x in (20..30)", xgrid[0,0,0,20:30].min())
#    print("segmap max at x in (0..10)", segmap[0,1,:,:10].max())
#    print("segmap max at x in (0..20)", segmap[0,1,:,:20].max())
#    print("segmap max at x in (0..30)", segmap[0,1,:,:30].max())
#    xgrid = torch.linspace(0, W, W, device=device).view(1, 1, 1, W)
#    tmp = xgrid.where(segmap, torch.tensor(W, device=device).float())
#    print("tmp size:", tmp.size())
#    print("after putting W where ~segmap we are in", tmp[0,1].min())
#    print(tmp[0,1].view(-1).long().bincount())
#    raise ValueError("Set num workers back to 11")
    return boxes
    
def mask_to_coco_rle(mask, transpose=True):
    """Encodes a boolean mask into a coco-style RLE. The RLE uses column major order and contains
    no values, instead encoding every other count as 0-value and every other as 1-value, starting
    with 0.
    Args:
        mask (BoolTensor): Of size (H,W)
    Returns:
        list: Contains counts of zero, one, zero, one, and so on.
    """
    assert mask.dim() == 2

    H, W = mask.size()
#    mask = mask.cpu().transpose(0,1).numpy()
#    old_rle = cocomask.encode(mask.cpu().transpose(0,1).numpy())
    mask_asnp = np.array(mask.cpu().numpy()[:, :, np.newaxis], order='F')
    rle = encode(mask_asnp)[0]
#    print(rle['counts'])
#    print(rle['counts'].decode())
#    raise ValueError()
#    rle['counts'] = str(rle['counts'])
#    print(type(rle['counts']))
    rle['counts'] = rle['counts'].decode()
    
#    print(rle)
    return rle
    
    # Apparently they, for the results but not the annotations, some odd additional encoding for the RLE ...
#    H, W = mask.size()
#    mask_flat = mask.transpose(0,1).reshape(-1)
#    HW = mask_flat.size(0)
#    device = mask.device
#    diffs = torch.logical_xor(mask_flat[0 : HW - 1], mask_flat[1 : HW])
#    switch_ids = diffs.nonzero().view(-1)
#    switch_ids = torch.cat([torch.tensor([-1], device=device), switch_ids, torch.tensor([HW - 1], device=device)])
#    rle2 = (switch_ids[1:] - switch_ids[:-1]).tolist()
#    rle2 = {"size": [H, W], "counts": rle2}
#    rle2 = cocomask.frPyObjects(rle2, H, W)
#    print(rle)
#    print(type(rle2["counts"].decode()))
    
#    return rle2

    # Adapted from MSCOCO ... really COCO, are you this slow, or did they pair the super-GPUs with pentium 3???
#    diffs = torch.logical_xor(mask[0 : HW - 1], mask[1 : HW])
#    counts_list = [1]
#    pos = 0
#    for diff in diffs:
#        if diff:
#            pos += 1
#            counts_list.append(1)
#        else:
#            counts_list[pos] += 1
#    if mask[0] == 1:
#        counts_list = [0] + counts_list
#    print(counts_list)
#    return counts_list
    
    # Adapted from
    # https://github.com/limingwu8/UNet-pytorch/blob/master/utils.py
    # who snagged it from
    # https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    # WAIT, WHAT IS THIS NONSTANDARD RLE???
#    pos_ids = mask.transpose(0,1).reshape(-1).nonzero().view(-1).cpu()
#    run_lengths = []
#    prev = -2
#    for idx in pos_ids:
#        if (idx - prev > 1): run_lengths.extend((idx + 1, 0))
#        run_lengths[-1] += 1
#        prev = idx
#    return run_lengths
    
    # TOOOO SLOW STILL!
#    rle = [sum(1 for _ in y) for (x, y) in groupby(mask)]
#    if mask[0]:
#        rle = [0] + rle
#    return rle
    
    # Below is waaaaaaay too slow!
#    count = 0
#    category = False
#    counts = []
#    for i in range(mask.size(0)):
#        if mask[i] != category:
#            counts.append(count)
#            category = not category
#        count = count + 1
#    counts.append(count)
#    return counts


def intersection1d(t1, t2):
    indices = torch.zeros_like(t1, dtype=torch.bool)
    for elem in t2:
        indices = indices | (t1 == elem)
    intersection = t1[indices]
    return intersection


def non_intersection1d(t1, t2):
    # returns t2 - t1
    indices = torch.ones_like(t1, dtype=torch.bool)
    for elem in t2:
        indices = indices & (t1 != elem)
    non_intersection = t1[indices]
    return non_intersection


def batch_many_to_many_class_overlap(pred, anno):
    """
    Args:
        pred (Tensor)    : Of size (*, M, C)
        anno (LongTensor): Of size (*, N)
    """
    batch_size = pred.size()[:-2]
    M, C = pred.size()[-2:]
    N = anno.size(-1)
    pred = pred.view(-1, M, 1, C).expand(-1, -1, N, -1) # (B, M, N, C)
    anno = anno.view(-1, 1, N, 1).expand(-1, M, -1, -1) # (B, M, N, 1)
    B = pred.size(0)
    
    pred_class_probs = F.softmax(pred, dim=3)
    pred_class_prob_per_anno = pred_class_probs.gather(3, anno).view(B, M, N)
    return pred_class_prob_per_anno
    
