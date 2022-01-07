import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import product
from math import sqrt
from typing import List
from rgnnvis.utils.debugging import print_tensor_statistics


def project_boxes(bboxes, image_size):
    """
    Batch based!
    Postprocesses the output of Yolact into a format that makes sense
    Args:
        detections           (torch.Tensor): bboxes for all detections for one frame
        image_size                  (Tuple): original image size
    Returns:
        bboxes_out           (torch.Tensor): projected and clamped coordinates
    """
    # [x1, y1, x2, y2]
    m = torch.zeros_like(bboxes[0:1])
    m[0, 0] = image_size[1]; m[0, 1] = image_size[0]; m[0, 2] = image_size[1]; m[0, 3] = image_size[0]
    bboxes_proj = bboxes * m
    if not ((bboxes_proj[:, 0] <= bboxes_proj[:, 2]) * (bboxes_proj[:, 1] <= bboxes_proj[:, 3])).all():
        violating_ids = (~((bboxes_proj[:, 0] <= bboxes_proj[:, 2]) * (bboxes_proj[:, 1] <= bboxes_proj[:, 3]))).nonzero()
        violating_boxes = bboxes_proj[violating_ids[:,0]]
        print("Got improper boxes, ids =")
        print(violating_ids)
        print("and boxes =")
        print(violating_boxes)
        raise ValueError()
#    bboxes_proj = (bboxes*m + 0.5)

#    x1 = torch.min(torch.stack([bboxes_proj[: ,0], bboxes_proj[:, 2]], dim=1), dim=1)[0]
#    y1 = torch.min(torch.stack([bboxes_proj[:, 1], bboxes_proj[:, 3]], dim=1), dim=1)[0]
#    x2 = torch.max(torch.stack([bboxes_proj[:, 0], bboxes_proj[:, 2]], dim=1), dim=1)[0]
#    y2 = torch.max(torch.stack([bboxes_proj[:, 1], bboxes_proj[:, 3]], dim=1), dim=1)[0]
    x1 = bboxes_proj[:, 0]
    y1 = bboxes_proj[:, 1]
    x2 = bboxes_proj[:, 2]
    y2 = bboxes_proj[:, 3]

    c1 = torch.clamp(torch.stack([x1, y1], dim=1), min=0)
    x2 = torch.clamp(x2, max=image_size[1]-1)
    y2 = torch.clamp(y2, max=image_size[0]-1)
    c2 = torch.stack([x2, y2], dim=1)

    bboxes_out = torch.cat([c1, c2], dim=1)
    return bboxes_out


def decode_boxes(loc, priors, use_yolo_regressors=False):
    """
    Batch based!
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.

    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).

    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]

    Returns: A tensor of decoded relative coordinates in point form
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])
        ), 1)

        # To point form
        boxes = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,      # xmin, ymin
                           boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax
    else:
        variances = [0.1, 0.2]

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

    return boxes


def make_priors_sa(height, width, aspect_ratios, scales,
                max_size=550, use_pixel_scales=True, use_square_anchors=True,
                device=torch.device('cpu')):
    """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
    prior_data = []

    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(height), range(width)):
        # +0.5 because priors are in center-size notation
        x = (i + 0.5) / width
        y = (j + 0.5) / height

        for ars in aspect_ratios:
            for scale in scales:
                for ar in ars:
                    ar = sqrt(ar)

                    if use_pixel_scales:
                        w = scale * ar / max_size
                        h = scale / ar / max_size
                    else:
                        w = scale * ar / width
                        h = scale / ar / height

                    # Yolact: This is for backward compatibility with a bug where
                    # they made everything square by accident
                    if use_square_anchors:
                        h = w
                    prior_data += [x, y, w, h]

    priors = torch.tensor(prior_data, device=device, requires_grad=False).view(-1, 4)
    return priors


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """

    def make_layer(layer_cfg):
        nonlocal in_channels

        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False,
                                              **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels


class InterpolateModule(nn.Module):
    """
    This is a module version of F.interpolate (rip nn.Upsampling).
    Any arguments you give it just get passed along for the ride.
    """
    def __init__(self, *args, **kwdargs):
        super().__init__()
        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0,
                 num_classes=81):
        super().__init__()
        self.use_mask_scoring = False
        self.use_instance_coeff = False
        self.mask_proto_split_prototypes_by_head = False
        self.mask_proto_coeff_gate = False
        self.eval_mask_branch = True
        self.use_yolo_regressors = False
        self.use_instance_coeff = False
        self.use_mask_scoring = False
        self.use_prediction_module = False
        self.extra_head_net = [(256, 3, {'padding': 1})]
        self.num_classes = num_classes
        self.mask_dim = 32  # Defined by Yolact
        self.num_priors = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent = [parent]  # Don't include this in the state dict
        self.index = index
        self.num_heads = 5  # Defined by Yolact
        self.extra_head_net = [(256, 3, {'padding':1})]
        self.num_instance_coeffs = 64
        
        if parent is None:

            if self.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, self.extra_head_net)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, 3, 1, 1)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, 3, 1, 1)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, 3, 1, 1)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code? ##################
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            extra_layers = (0, 0, 0)
            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in extra_layers]

            ############################################################################################################

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = {}
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if self.extra_head_net is not None:
            x = src.upfeature(x)
        
        if self.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        
        if self.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if self.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        if self.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_instance_coeffs)

        # See box_utils.decode for an explanation of this
        if self.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if self.eval_mask_branch:
            # Mask linear combination
            mask = torch.tanh(mask)
            if self.mask_proto_coeff_gate:
                gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                mask = mask * torch.sigmoid(gate)

        # Mask linear combination
        if self.mask_proto_split_prototypes_by_head and True:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode='constant', value=0)

        prior_key = (conv_h, conv_w)
        if prior_key not in self.priors.keys():
            prior = make_priors_sa(conv_h, conv_w,
                                   self.aspect_ratios, self.scales,
                                   max_size=500,
                                   use_square_anchors=False,
                                   device=x.device)
            self.priors.update({prior_key: prior})

        preds = {'yolact_loc': bbox, 'yolact_conf': conf, 'yolact_maskcoeffs': mask, 'yolact_priorboxes': self.priors[prior_key]}

        if self.use_mask_scoring:
            preds.update({'yolact_score': score})

        if self.use_instance_coeff:
            preds.update({'yolact_inst': inst})
        
        return preds


class FPN(nn.Module):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels=[512, 1024, 2048], num_features=256):
        super().__init__()

        self.interpolation_mode = 'bilinear'
        self.num_downsample = 2
        self.use_conv_downsample = True
        self.relu_downsample_layers = False
        self.relu_pred_layers = True
        self.num_features = num_features

        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, self.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        fpn_pad = True
        padding = 1 if fpn_pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if self.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(self.num_downsample)
            ])

    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            
            x = x + lat_layer(convouts[j])
            out[j] = x
        
        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)

        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out


class Yolact(nn.Module):
    """

    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝

    A better implementation without config.py.

    Create your backbone example:
        from torchvision.models._utils import IntermediateLayerGetter
        return_layers = {'layer1': 'bb_feats_s2', 'layer2': 'bb_feats_s4',
                         'layer3': 'bb_feats_s8', 'layer4': 'bb_feats_s16'}
        backbone = IntermediateLayerGetter(resnet101(pretrained=True), return_layers)

    """
    def __init__(self, num_classes=81, pretty_boxes=True,
                 output_score=False, freeze_bn=True, pred_aspect_ratios=None, pred_scales=None):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_bn = freeze_bn
        self.output_score = output_score
        self.pretty_boxes = pretty_boxes
        self.inference_mode = False

        # if self.freeze_bn:
        #     self.freeze_bn()

        # FPN returns a list of length 5, 3 backbone layers + 2 extra downsample layers,
        # each of 256 feature channels.
        self.fpn = FPN(in_channels=[512, 1024, 2048])  # layer 2,3 and 4 of the resnet101 backbone

        # ProtoNet
        # The include_last_relu=false here is because we might want to change it to another function
        mask_proto_net = [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
        self.proto_net, self.mask_dim = make_net(256, mask_proto_net, include_last_relu=False)

        # PredictionLayers
        # TODO: This module need refactoring!
        self.prediction_layers = nn.ModuleList()
        if pred_aspect_ratios is None:
            pred_aspect_ratios = [[[1.0/1.8, 0.5/1.8, 2.0/1.8]]] * 5
        if pred_scales is None:
            pred_scales = [[24], [48], [96], [192], [384]]
        for idx in range(5):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if True and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(256, 256,
                                    aspect_ratios=pred_aspect_ratios[idx],
                                    scales=pred_scales[idx],
                                    parent=parent,
                                    index=idx,
                                    num_classes=num_classes)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        # Use class existence loss
        # This comes from the smallest layer selected
        # Also note that num_classes includes background
        # self.class_existence_fc = nn.Linear(src_channels[-1], num_classes - 1)

        # Use semantic segmentation loss
        self.semantic_seg_conv = nn.Conv2d(256, num_classes - 1, kernel_size=1)

    def forward(self, features, image_shape):
        """ model forward
        Args:
            features       (Dict:torch.Tensor): features
            image_shape                (Tuple): (h,w)
        Returns:
            pred           (Dict:torch.Tensor): see code.
        """
        in_height, in_width = image_shape

        # FPN
        fpn_out = self.fpn(list(features.values()))

        # ProtoNet
        proto_out = self.proto_net(fpn_out[0])
        proto_out = F.relu(proto_out, inplace=True)

        # Move the features last so the multiplication is easy
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        # PredictionHead
        pred_outs = {'yolact_loc': [], 'yolact_conf': [], 'yolact_maskcoeffs': [], 'yolact_priorboxes': []}

        for idx, prediction_layer in enumerate(self.prediction_layers):

            # A hack for the way dataparallel works
            if prediction_layer is not self.prediction_layers[0]:
                prediction_layer.parent = [self.prediction_layers[0]]

            p = prediction_layer(fpn_out[idx])

            for k, v in p.items():
                pred_outs[k].append(v)

        # Concatenate lists in pred_outs
        for key, tl in pred_outs.items():
            pred_outs[key] = torch.cat(tl, dim=-2)

        # Add ProtoNet result
        pred_outs.update({'yolact_protosegmaps': proto_out})

        # Semantic segmentation objective during training
        #jj NO, BAD CODER! We want all losses when model is in eval-mode as well! Eval-mode in PyTorch is
        #jj used to deal with certain layers, like dropout and batchnorm. When we calculate loss on validation
        #jj set we want to freeze those (we DONT want to update batchnorm statistics on validation set), but
        #jj we still want the losses in order to plot them.
#        if self.training:
            # Class existence loss
            # pred_outs['classes'] = self.class_existence_fc(fpn_out[-1].mean(dim=(2, 3)))

        # Semantic segmentation loss
        if not self.inference_mode:
            pred_outs['yolact_semantic_segmap'] = self.semantic_seg_conv(fpn_out[0])

        # Pretty boxes
        if self.pretty_boxes:
            # Decode loc to relative point form of box
            loc = pred_outs['yolact_loc']
            rpf_boxes = torch.stack([decode_boxes(batch_loc, pred_outs['yolact_priorboxes']) for batch_loc in loc], dim=0)
            # Project to absolut point form of original image size
            apf_boxes = torch.stack([project_boxes(batch_box, (in_height, in_width)) for batch_box in rpf_boxes], dim=0)
            pred_outs.update({'yolact_boxes': apf_boxes})

            # Put priorboxes in (x1,y1,x2,y2) pixel coordinates
            priorboxes = pred_outs['yolact_priorboxes']
            priorbox_multiplier = torch.tensor([image_shape[1], image_shape[0], image_shape[1], image_shape[0]],
                                               dtype=torch.float32, device=loc.device)
            priorboxes = priorbox_multiplier * priorboxes
            priorboxes[:, 0:2] = priorboxes[:, 0:2] - 0.5 * priorboxes[:, 2:4]
            priorboxes[:, 2:4] = priorboxes[:, 0:2] + priorboxes[:, 2:4]
            pred_outs['yolact_priorboxes'] = priorboxes

        # conf softmaxed
        if self.output_score:
            pred_outs.update({'yolact_score': F.softmax(pred_outs['yolact_conf'], -1)})

        return pred_outs

    def inference(self, mode=True):
        self.train(False)
        self.inference_mode = mode

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_bn:
            self.freeze_batchnorm(mode)

    def freeze_batchnorm(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def load_state_dict_without_specific_classes(self, state_dict):
        """Wrap load_state_dict and skip some parameters that may change size for different dataset."""
        modified_state_dict = state_dict.copy() # Shallow copy of dict
        undesired_keys = ['semantic_seg_conv.weight', 'semantic_seg_conv.bias',
                          'prediction_layers.0.conf_layer.weight',
                          'prediction_layers.0.conf_layer.bias']
        for key in undesired_keys:
            modified_state_dict.pop(key)
        missing_keys, unexpected_keys = self.load_state_dict(modified_state_dict, strict=False)
        assert set(missing_keys) == set(undesired_keys), missing_keys
        assert len(unexpected_keys) == 0, unexpected_keys
