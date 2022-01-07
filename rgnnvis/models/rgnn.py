
from math import sqrt, ceil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

import rgnnvis.models.nms as nms
from rgnnvis.utils.debugging import print_tensor_statistics, get_memory_str
from rgnnvis.utils.tensor_utils import resize_spatial_tensor, resize_boxes, batch_many_to_many_box_iou
from rgnnvis.utils.recursive_functions import recursive_tensor_sizes


YTVIS_CATEGORY_NAMES = {0: "background", 1: "person", 2: "giant_panda", 3: "lizard", 4: "parrot", 5: "skateboard", 6: "sedan", 7: "ape", 8: "dog", 9: "snake", 10: "monkey", 11: "hand", 12: "rabbit", 13: "duck", 14: "cat", 15: "cow", 16: "fish", 17: "train", 18: "horse", 19: "turtle", 20: "bear", 21: "motorbike", 22: "giraffe", 23: "leopard", 24: "fox", 25: "deer", 26: "owl", 27: "surfboard", 28: "airplane", 29: "truck", 30: "zebra", 31: "tiger", 32: "elephant", 33: "snowboard", 34: "boat", 35: "shark", 36: "mouse", 37: "frog", 38: "eagle", 39: "earless_seal", 40: "tennis_racket"}
LOG2PI = 1.8379
GLOBALS = {}
GLOBALS['num iter'] = 0


def print_dict(name, elem):
    print(f"\n{name}")
    for key, val in elem.items():
        if isinstance(val, torch.Tensor):
            print_tensor_statistics(val, key)
        else:
            print(key, val)

def get_visualization_box_text2(lbscores, active):
    B, L, M, C = lbscores.size()
    lb = F.softmax(lbscores, dim=3)
    label_probs, semantic_labels = torch.topk(lb, k=3, dim=3)
    active_str = {1: "", 2: "INACTIVE"}
    boxtext = [[[[active_str[active[b,l,m].item()]]
                 + ["{} {:.2f}".format(YTVIS_CATEGORY_NAMES[semantic_labels[b,l,m,c].item()], label_probs[b,l,m,c])
                  for c in range(3)]
                 for m in range(M) if active[b,l,m] in (1, 2)]
                for l in range(L)]
               for b in range(B)]
    return boxtext
            
def get_visualization_box_text(lbscores, active):
    B, L, M, C = lbscores.size()
    semantic_labels = lbscores[:,:,:,1:].argmax(dim=3) + 1
    label_prob = (lbscores[:,:,:,1:].max(dim=3)[0] - lbscores.logsumexp(dim=3)).exp()
    boxtext = [[["{} {:.2f}".format(YTVIS_CATEGORY_NAMES[semantic_labels[b,l,m].item()], label_prob[b,l,m])
                 for m in range(M) if (active[b,l,m] == 1) or (active[b,l,m] == 2)]
                for l in range(L)]
               for b in range(B)]
    return boxtext

def asparam(tensor):
    return nn.Parameter(tensor, requires_grad=tensor.requires_grad)

def pad_and_stack(lst, pad_val, size, device, dtype):
    """Pads dimension 1 s.t. it has the same size for all elements in lst. Then stacks all elements in lst."""
    args = {'fill_value': pad_val, 'device': lst[0].device, 'dtype': dtype}
    pad_lst = [torch.full([size[1] - elem.size(0)] + list(size[2:]), **args) for elem in lst]
    return torch.stack([torch.cat([elem, padelem], dim=0) for elem, padelem in zip(lst, pad_lst)])

def box_to_seg(boxes, segsize, imsize, device):
    """
    Args:
        boxes (FloatTensor): of size (B, Nmax, 4)
        segsize (tuple): Height and width of the seg
        imsize (tuple): Height and width of original image
    """
    B, Nmax, _ = boxes.size()
    xvals = torch.linspace(0., imsize[1], segsize[1], device=device).view(1, 1, 1, -1).expand(B, Nmax, -1, -1)
    yvals = torch.linspace(0., imsize[0], segsize[0], device=device).view(1, 1, -1, 1).expand(B, Nmax, -1, -1)
    masks = ((boxes[:, :, 0].view(B, Nmax, 1, 1) <= xvals) *
             (boxes[:, :, 1].view(B, Nmax, 1, 1) <= yvals) *
             (xvals <= boxes[:, :, 2].view(B, Nmax, 1, 1)) *
             (yvals <= boxes[:, :, 3].view(B, Nmax, 1, 1))).float()
    return masks

def normalize_boxes(boxes):
    scale = torch.tensor([2/864, 2/480, 2/864, 2/480], device=boxes.device)
    return boxes * scale - 1.0


class Wrap(nn.Module):
    """Wraps standard module that as input takes a single argument, and that provides a single
    output. The wrapped module instead takes two input arguments, input and state, and provides
    two outputs, including an empty state."""
    def __init__(self, module_to_be_wrapped):
        super().__init__()
        self.with_state = True
        self.body = module_to_be_wrapped
    def forward(self, standard_input, state):
        return self.body(standard_input), None

class SequentialWithState(nn.Module):
    def __init__(self, layers, return_layers=None):
        super().__init__()
        self.with_state = True
        self.return_layers = return_layers
        if len(layers) > 0 and isinstance(layers, (tuple, list)):
            for idx, layer in enumerate(layers):
                self.add_module(str(idx), layer)
        elif len(layers) > 0 and isinstance(layers, dict):
            for name, layer in layers.items():
                self.add_module(name, layer)
    def forward(self, x, state):
        if state is None:
            state = {}
        new_state = {}
        if self.return_layers is not None:
            out = {}
        for name, module in self.named_children():
            if hasattr(module, 'with_state'):
                x, new_state[name] = module(x, state.get(name))
            else:
                x = module(x)
            if self.return_layers is not None and name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        if self.return_layers is not None:
            return out, new_state
        else:
            return x, new_state


class AppNet(nn.Module):
    def __init__(self, feat_net_dict):
        super().__init__()
        assert isinstance(feat_net_dict, nn.ModuleDict), f"expected ModuleDict, got {type(feat_net_dict)}"
        self.feat_nets = feat_net_dict
    def forward(self, feats):
        """
        Returns:
            list<Tensor>: list of feature maps at different strides
        """
        if isinstance(feats, (list, tuple)): # Hack to deal with some backbones and nn.ModuleDicts req. str keys
            feats = {f"{idx}": feat for idx, feat in enumerate(feats)}
        out = [net(feats[key]) for key, net in self.feat_nets.items()]
        return out


class MatcherScoreLayer(nn.ModuleDict):
    def __init__(self, layer_dict):
        super().__init__(layer_dict)
    def _get_trkdet_scores(self, feats, masks, B, M, N):
        trkdet_scores = self['trkdet'](feats['trkdet_edges'])
        trkdet_scores = trkdet_scores.view(B, M, N)
        trkdet_scores = trkdet_scores.where(masks['trkdet'], torch.full_like(trkdet_scores, -1e5))
        return trkdet_scores
    def _get_novel_track_scores(self, feats, masks, B, M, N):
        novel_track_scores = self['det'](feats['det_vertices'])
        novel_track_scores = novel_track_scores.view(B, N)
        novel_track_scores = novel_track_scores.where(masks['det'], torch.full_like(novel_track_scores, -1e5))
        return novel_track_scores
    def forward(self, feats, masks):
        B, M, N, _ = feats['trkdet_edges'].size()
        trkdet_scores = self._get_trkdet_scores(feats, masks, B, M, N)
        novel_track_scores = self._get_novel_track_scores(feats, masks, B, M, N)
        return trkdet_scores, novel_track_scores
class MatcherScoreLayerIP(MatcherScoreLayer):
    def _get_trkdet_scores(self, feats, masks, B, M, N):
        B, M, N, D = feats['trkdet_edges'].size()
        trkdet_scores = torch.einsum(
            'bmd,bnd->bmn',
            (1 / sqrt(D)) * self['trkdet_trk'](feats['trk_vertices']),
            self['trkdet_det'](feats['det_vertices']))
        trkdet_scores = trkdet_scores.where(masks['trkdet'], torch.full_like(trkdet_scores, -1e5))
        return trkdet_scores
class MatcherScoreLayerEdgeless(MatcherScoreLayer):
    def _get_trkdet_scores(self, feats, masks, B, M, N):
        trkdet_scores = self['trkdet'](torch.cat([
            feats['trk_vertices'].view(B, M, 1, -1).expand(-1, -1, N, -1),
            feats['det_vertices'].view(B, 1, N, -1).expand(-1, M, -1, -1)
        ], dim=3))
        return trkdet_scores
class MatcherScoreLayerEdges(MatcherScoreLayer):
    def _get_novel_track_scores(self, feats, masks, B, M, N):
        novel_track_scores = self['bkgdet'](feats['bkgdet_edges'])
        novel_track_scores = novel_track_scores.view(B, N)
        novel_track_scores = novel_track_scores.where(masks['det'], torch.full_like(novel_track_scores, -1e5))
        return novel_track_scores
class MatcherScoreLayerNograph(MatcherScoreLayer):
    def _get_novel_track_scores(self, feats, masks, B, M, N):
        trkdet_scores = feats['trkdet_edges'].view(B, M, N) # scores to be sigmoided
        trkdet_scores = trkdet_scores.detach()
        # Deal with tracks without score. If there are no tracks, or all with below -20 score, then
        # the detection will start a new track with score 20.
        trkdet_scores = trkdet_scores.where(masks['trkdet'], torch.full_like(trkdet_scores, -10))
        max_scores, _ = trkdet_scores.max(dim=1)    
        novel_track_scores = - max_scores             # if p = sigmoid(score), then 1 - p = sigmoid(-score)
        novel_track_scores = novel_track_scores.where(masks['det'], torch.full_like(novel_track_scores, -1e5))
        return novel_track_scores

    
class TrkNograph(nn.ModuleDict):
    def __init__(self, *args, gather_threshold=-100, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_threshold = gather_threshold
    def forward(self, feats, masks, B, M, N):
        """Let each track gather information directly from a single detection. It is assumed that
        trkdet_edges is the final matching probabilities.
        """
        trkdet_scores = feats['trkdet_edges'].view(B, M, N)
        trkdet_scores = trkdet_scores.where(masks['trkdet'], torch.full_like(trkdet_scores, -1e5))
        max_scores, max_ids = trkdet_scores.max(dim=2)
#        if GLOBALS['num iter'] == 201:
#            raise ValueError()
#        if GLOBALS['num iter'] in (100, 200,):
#            if GLOBALS['frame idx'] == 0:
#                print("\n---------\n")
#            print(trkdet_scores[0,[0,1,2,3,4]][:,[0,1,2,3,4,5,6]])
        det_feats = feats['det_vertices'].where(
            masks['det'].view(B, N, 1).expand(-1, -1, feats['det_vertices'].size(2)),
            torch.zeros_like(feats['det_vertices']))
        trk_new = det_feats.gather(1, max_ids.view(B, M, 1).expand(-1, -1, det_feats.size(2)))
        mask = masks['trk'].view(B, M, 1).expand_as(trk_new)
        mask = mask * (max_scores > self.gather_threshold).view(B, M, 1)
        trk_new = trk_new.where(mask, torch.zeros_like(trk_new))
        out = torch.cat([self['trk_in'](feats['trk_vertices']), self['det_in'](trk_new)], dim=2)
        return self['out'](out)
class BkgFanatic(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        bkg_vertex = self['bkg_in'](feats['bkg_vertex'])
        if self['bkgdet_in'] is not None:
            bkgdet_in = self['bkgdet_in'](feats['bkgdet_edges'])
            if self['bkgdet_sigmoid'] is not None:
                bkgdet_in = bkgdet_in * torch.sigmoid(self['bkgdet_sigmoid'](feats['bkgdet_edges']))
            mask = masks['det'].view(B, N, 1).expand_as(bkgdet_in)
            bkgdet_in = bkgdet_in.where(mask, torch.zeros_like(bkgdet_in))
            bkgdet_in = bkgdet_in.sum(dim=1)
            out = torch.cat([bkg_vertex, bkgdet_in], dim=1)
            if self['residual'] is not None:
                out = bkg_vertex + self['residual'](out)
        else:
            out = bkg_vertex
        return self['out'](out)
class TrkFanatic(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        trk_vertices = self['trk_in'](feats['trk_vertices'])
        if self['trkdet_in'] is not None:
            trkdet_in = self['trkdet_in'](feats['trkdet_edges'])
            if self['trkdet_sigmoid'] is not None:
                trkdet_in = trkdet_in * torch.sigmoid(self['trkdet_sigmoid'](feats['trkdet_edges']))
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet_in)
            trkdet_in = trkdet_in.where(mask, torch.zeros_like(trkdet_in))
            trkdet_in = trkdet_in.sum(dim=2)
            out = torch.cat([trk_vertices, trkdet_in], dim=2)
            if self['residual'] is not None:
                out = trk_vertices + self['residual'](out)
        else:
            out = trk_vertices
        return self['out'](out)
class DetFanatic(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        det_vertices = self['det_in'](feats['det_vertices'])
        if self['trkdet_in'] is not None:
            trkdet_in = self['trkdet_in'](feats['trkdet_edges'])
            if self['trkdet_sigmoid'] is not None:
                trkdet_in = trkdet_in * torch.sigmoid(self['trkdet_sigmoid'](feats['trkdet_edges']))
            mask_trkdet = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet_in)
            trkdet_in = trkdet_in.where(mask_trkdet, torch.zeros_like(trkdet_in))
            trkdet_in = trkdet_in.sum(dim=1)
            bkgdet_in = self['bkgdet_in'](feats['bkgdet_edges'])
            if self['bkgdet_sigmoid'] is not None:
                bkgdet_in = bkgdet_in * torch.sigmoid(self['bkgdet_sigmoid'](feats['bkgdet_edges']))
            out = torch.cat([det_vertices, bkgdet_in + trkdet_in], dim=2)
            if self['residual'] is not None:
                out = det_vertices + self['residual'](out)
        else:
            out = det_vertices
        return self['out'](out)

class TrkEuphoric(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        trk = self['trk_in'](feats['trk_vertices']) # (B,M,D)
        trk_aggr = self['message'](trk).sum(1, True).expand(-1, M, -1)
        return self['out'](trk + self['residual'](torch.cat([trk, trk_aggr], dim=2)))

class BkgDetDope(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        bkgdet = feats['bkgdet_edges'] # (B,N,D)
        if self['nodes_in'] is not None or self['mid'] is not None:
            nodes_in = self['nodes_in'](
                feats['bkg_vertex'].view(B, 1, -1).expand(-1, N, -1),
                feats['det_vertices'])
            if bkgdet is None:
                bkgdet = self['mid'](nodes_in) # (B,N,D)
            else:
                bkgdet = bkgdet + self['mid'](torch.cat([bkgdet, nodes_in], dim=2))
        term_lst = [bkgdet]
        if self['trk_gather'] is not None:
            trkdet = feats['trkdet_edges'] # (B,M,N,D)
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet)
            residual = trkdet.where(mask, torch.zeros_like(trkdet)).sum(1) # (B,N,D)
            residual = self['trk_gather'](torch.cat([bkgdet, residual], dim=2)) # (B,N,D)
            term_lst.append(residual)
        if self['det_gather'] is not None:
            mask = masks['det'].view(B, N, 1).expand_as(bkgdet)
            residual = bkgdet.where(mask, torch.zeros_like(bkgdet)).sum(1, True).expand_as(bkgdet) # (B,N,D)
            residual = self['det_gather'](torch.cat([bkgdet, residual], dim=2))
            term_lst.append(residual)
        return self['out'](sum(term_lst))
class TrkDetDope(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        trkdet = feats['trkdet_edges'] # (B,M,N,D)
        if self['nodes_in'] is not None or self['mid'] is not None:
            nodes_in = self['nodes_in'](
                feats['trk_vertices'].view(B, M, 1, -1).expand(-1, -1, N, -1),
                feats['det_vertices'].view(B, 1, N, -1).expand(-1, M, -1, -1))
            if trkdet is None:
                trkdet = self['mid'](nodes_in)
            else:
                trkdet = trkdet + self['mid'](torch.cat([trkdet, nodes_in], dim=3))
        term_lst = [trkdet]
        if self['trk_gather'] is not None:
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet)
            residual = trkdet.where(mask, torch.zeros_like(trkdet)).sum(1) # (B,N,D)
            residual = residual + feats['bkgdet_edges'].where(masks['det'].view(B, N, 1).expand_as(feats['bkgdet_edges']), torch.zeros_like(feats['bkgdet_edges'])) # (B,N,D)
            residual = residual.view(B, 1, N, -1).expand_as(trkdet)
            residual = self['trk_gather'](torch.cat([trkdet, residual], dim=3))
            term_lst.append(residual)
        if self['det_gather'] is not None:
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet)
            residual = trkdet.where(mask, torch.zeros_like(trkdet)).sum(2, True).expand_as(trkdet) # (B,M,N,D)
            residual = self['det_gather'](torch.cat([trkdet, residual], dim=3))
            term_lst.append(residual)
        return self['out'](sum(term_lst))
class BkgDope(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        bkg = feats['bkg_vertex']
        node_gates = self['node_gate'](feats['bkgdet_edges'])
        mask = masks['det'].view(B, N, -1).expand_as(node_gates)
        node_gates = node_gates.where(mask, torch.full_like(node_gates, -1e7)) # (B,N,D)
        node_gates = F.softmax(node_gates, dim=1)
        node_gates = node_gates.where(mask, torch.zeros_like(node_gates))
        if node_gates.size(2) == 1: # If we gate with a single channel
            nodes = torch.einsum('bn,bnd->bd', node_gates.view(B, N), feats['det_vertices'])
        else:
            nodes = torch.einsum('bnd,bnd->bd', node_gates, feats['det_vertices'])
        edges = self['bkgdet_in'](feats['bkgdet_edges']) # (B,N,D)
        edges = edges.sum(dim=1) # (B,D)
        return self['out'](bkg + self['residual'](torch.cat([bkg, nodes, edges], dim=1)))
class TrkDope(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        trk = feats['trk_vertices']
        node_gates = self['node_gate'](feats['trkdet_edges'])
        mask = masks['trkdet'].view(B, M, N, -1).expand_as(node_gates) # (B,M,N,D)
        node_gates = node_gates.where(mask, torch.full_like(node_gates, -1e7))
        node_gates = F.softmax(node_gates, dim=2)
        node_gates = node_gates.where(mask, torch.zeros_like(node_gates))
        if node_gates.size(3) == 1: # If we gate with a single channel
            nodes = torch.einsum('bmn,bnd->bmd', node_gates.view(B, M, N), feats['det_vertices'])
        else:
            nodes = torch.einsum('bmnd,bnd->bmd', node_gates, feats['det_vertices'])
        edges = self['trkdet_in'](feats['trkdet_edges']) # (B,M,N,D)
        edges = edges.sum(dim=2) # (B,M,D)
        return self['out'](trk + self['residual'](torch.cat([trk, nodes, edges], dim=2)))
class DetDope(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        det = feats['det_vertices']
        edges_bkg = self['bkgdet_in'](feats['bkgdet_edges'])
        edges_trk = self['trkdet_in'](feats['trkdet_edges'])
        mask = masks['trkdet'].view(B, M, N, -1).expand_as(edges_trk)
        edges_trk = edges_trk.where(mask, torch.zeros_like(edges_trk)) # (B,M,N,D)
        edges_trk = edges_trk.sum(1) # (B,N,D)
        edges = edges_bkg + edges_trk
        return self['out'](det + self['residual'](torch.cat([det, edges], 2)))

class BkgDetCrazy(nn.ModuleDict):
    def __init__(self, layer_dict, diff_mode):
        super().__init__(layer_dict)
        self.diff_mode = diff_mode
    def forward(self, feats, masks, B, M, N):
        bkgdet = feats['bkgdet_edges']
        if self['diff_in'] is not None:
            bkg = feats['bkg_vertex'].view(B, 1, -1)
            det = feats['det_vertices']
            if self.diff_mode == '-':
                diff = bkg - det
            elif self.diff_mode == '*':
                diff = bkg * det
            elif self.diff_mode == 'square diff':
                diff = (bkg - det)**2
            elif self.diff_mode == 'square diff accumulate 16':
                diff = (bkg - det).view(B, N, 16, -1)
                diff = torch.einsum('bnkd,bnkd->bnk', diff, diff)
            if bkgdet is None:
                bkgdet = self['diff_in'](diff)
            else:
                bkgdet = bkgdet + self['diff_in'](torch.cat([bkgdet, diff], dim=2))
        term_lst = [bkgdet]
        if self['trk_gather'] is not None:
            trkdet = feats['trkdet_edges']
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet)
            residual = trkdet.where(mask, torch.zeros_like(trkdet)).sum(1)
            residual = self['trk_gather'](torch.cat([bkgdet, residual], dim=2))
            term_lst.append(residual)
        if self['det_gather'] is not None:
            mask = masks['det'].view(B, N, 1).expand_as(bkgdet)
            residual = bkgdet.where(mask, torch.zeros_like(bkgdet)).sum(1, True).expand_as(bkgdet)
            residual = self['det_gather'](torch.cat([bkgdet, residual], dim=2))
            term_lst.append(residual)
        return self['out'](sum(term_lst))
class TrkDetCrazy(nn.ModuleDict):
    def __init__(self, layer_dict, diff_mode):
        super().__init__(layer_dict)
        self.diff_mode = diff_mode
    def forward(self, feats, masks, B, M, N):
        trkdet = feats['trkdet_edges']
        if self['diff_in'] is not None:
            trk = feats['trk_vertices'].view(B, M, 1, -1)
            det = feats['det_vertices'].view(B, 1, N, -1)
            if self.diff_mode == '-':
                diff = trk - det
            elif self.diff_mode == '*':
                diff = trk * det
            elif self.diff_mode == 'square diff':
                diff = 0.1 * (trk - det)**2
            elif self.diff_mode == 'square diff accumulate 16':
                diff = (trk - det).view(B, M, N, 16, -1)
                diff = 0.1 * torch.einsum('bmnkd,bmnkd->bmnk', diff, diff)
            if trkdet is None:
                trkdet = self['diff_in'](diff)
            else:
                trkdet = trkdet + self['diff_in'](torch.cat([trkdet, diff], dim=3))
        term_lst = [trkdet]
        if self['trk_gather'] is not None:
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet)
            residual = trkdet.where(mask, torch.zeros_like(trkdet)).sum(1)
            residual = residual + feats['bkgdet_edges'].where(masks['det'].view(B, N, 1).expand_as(feats['bkgdet_edges']), torch.zeros_like(feats['bkgdet_edges']))
            residual = residual.view(B, 1, N, -1).expand_as(trkdet)
            residual = self['trk_gather'](torch.cat([trkdet, residual], dim=3))
            term_lst.append(residual)
        if self['det_gather'] is not None:
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet)
            residual = trkdet.where(mask, torch.zeros_like(trkdet)).sum(2, True).expand_as(trkdet)
            residual = self['det_gather'](torch.cat([trkdet, residual], dim=3))
            term_lst.append(residual)
        return self['out'](sum(term_lst))
class BkgCrazy(nn.ModuleDict):
    def __init__(self, layer_dict, fix_aggregation_bug=False):
        super().__init__(layer_dict)
        self.fix_aggregation_bug = fix_aggregation_bug
    def forward(self, feats, masks, B, M, N):
        message_gate = self['message_gate'](feats['bkgdet_edges'])
        mask = masks['det'].view(B, N, -1).expand_as(message_gate)
        message_gate = message_gate.where(mask, torch.full_like(message_gate, -1e7))
        if self.fix_aggregation_bug:
            message_gate = F.softmax(message_gate, dim=1)
        else:
            message_gate = F.softmax(message_gate, dim=2)
        message_gate = message_gate.where(mask, torch.zeros_like(message_gate))
        messages = torch.einsum('bnd,bnd->bd', message_gate, feats['det_vertices'])
        out_gate = F.sigmoid(self['out_gate'](feats['bkgdet_edges'].sum(dim=1)))
        bkg = self['bkg_in'](feats['bkg_vertex'])
        return (1 - out_gate) * bkg + out_gate * messages
#        return self['out'](torch.cat([feats['bkg_vertex'], messages], dim=2))
class TrkCrazy(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        message_gate = self['message_gate'](feats['trkdet_edges'])
        mask = masks['trkdet'].view(B, M, N, -1).expand_as(message_gate)
        message_gate = message_gate.where(mask, torch.full_like(message_gate, -1e7))
        message_gate = F.softmax(message_gate, dim=2)
        message_gate = message_gate.where(mask, torch.zeros_like(message_gate))
        messages = torch.einsum('bmnd,bnd->bmd', message_gate, feats['det_vertices'])
        out_gate = F.sigmoid(self['out_gate'](feats['trkdet_edges'].sum(dim=2)))
        trk = self['trk_in'](feats['trk_vertices'])
        return (1 - out_gate) * feats['trk_vertices'] + out_gate * messages
#        return self['out'](torch.cat([feats['trk_vertices'], messages], dim=2))
#class DetCrazy(nn.ModuleDict):
#    def forward(self, feats, masks, B, M, N):
#        message_weights = self['message_weights'](feats['trkdet_edges'])
#        mask = masks['trkdet'].view(B, M, N, -1).expand_as(feats['trkdet_edges'])
#        message_weights = message_weights.where(mask, torch.full_like(message_weights, -1e7))
#        message_weights = F.softmax(message_weights, dim=2)
#        message_weights = message_weights.where(mask, torch.zeros_like(message_weights))
#        messages = torch.einsum('bmnd,bnd->bmd', message_weights, feats['det_vertices'])
#        return self['out'](torch.cat([feats['trk_vertices'], messages], dim=2))
        

class BkgAttendTrk(nn.ModuleDict):
    def __init__(self, layer_dict, num_heads=4):
        super().__init__(layer_dict)
        self.num_heads = num_heads
    def forward(self, feats, masks, B, M, N):
        queries = self['query'](feats['bkg_vertex']).view(B, self.num_heads, -1)
        keys = self['key'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        vals = self['val'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        alpha = torch.einsum('bkd,bmkd->bmk', queries, keys)
        alpha = alpha.where(masks['trk'].view(B, M, 1), torch.full_like(alpha, -1e7))
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.where(masks['trk'].view(B, M, 1), torch.zeros_like(alpha))
        result = torch.einsum('bmk,bmkd->bkd', alpha, vals)
        result = result.view(B, -1)
        x = torch.cat([feats['bkg_vertex'], result], dim=1)
        return self['out'](feats['bkg_vertex'] + self['result'](x))
class BkgAttendDet(nn.ModuleDict):
    def __init__(self, layer_dict, num_heads=4):
        super().__init__(layer_dict)
        self.num_heads = num_heads
    def forward(self, feats, masks, B, M, N):
        queries = self['query'](feats['bkg_vertex']).view(B, self.num_heads, -1)
        keys = self['key'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        vals = self['val'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        alpha = torch.einsum('bkd,bnkd->bnk', queries, keys)
        alpha = alpha.where(masks['det'].view(B, N, 1), torch.full_like(alpha, -1e7))
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.where(masks['det'].view(B, N, 1), torch.zeros_like(alpha))
        result = torch.einsum('bnk,bnkd->bkd', alpha, vals)
        result = result.view(B, -1)
        x = torch.cat([feats['bkg_vertex'], result], dim=1)
        return self['out'](feats['bkg_vertex'] + self['result'](x))
class TrkAttendTrk(nn.ModuleDict):
    def __init__(self, layer_dict, num_heads=4):
        super().__init__(layer_dict)
        self.num_heads = num_heads
    def forward(self, feats, masks, B, M, N):
        queries = self['query'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        keys = self['key'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        vals = self['val'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        alpha = torch.einsum('bmkd,bnkd->bmnk', queries, keys)
        mask = (masks['trk'].view(B, M, 1, 1) * masks['trk'].view(B, 1, M, 1)).expand(-1, -1, -1, self.num_heads)
        alpha = alpha.where(mask, torch.full_like(alpha, -1e7))
        alpha = F.softmax(alpha, dim=2)
        alpha = alpha.where(mask, torch.zeros_like(alpha))
        result = torch.einsum('bmnk,bnkd->bmkd', alpha, vals)
        result = result.reshape(B, M, -1)
        x = torch.cat([feats['trk_vertices'], result, feats['bkg_vertex'].view(B,1,-1).expand(-1,M,-1)], dim=2)
        return self['out'](feats['trk_vertices'] + self['result'](x))
class TrkAttendDet(nn.ModuleDict):
    def __init__(self, layer_dict, num_heads=4):
        super().__init__(layer_dict)
        self.num_heads = num_heads
    def forward(self, feats, masks, B, M, N):
        queries = self['query'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        keys = self['key'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        vals = self['val'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        alpha = torch.einsum('bmkd,bnkd->bmnk', queries, keys)
        alpha = alpha.where(masks['trkdet'].view(B, M, N, 1).expand(-1, -1, -1, self.num_heads), torch.full_like(alpha, -1e7))
        alpha = F.softmax(alpha, dim=2)
        alpha = alpha.where(masks['trkdet'].view(B, M, N, 1).expand(-1, -1, -1, self.num_heads), torch.zeros_like(alpha))
        result = torch.einsum('bmnk,bnkd->bmkd', alpha, vals)
        result = result.reshape(B, M, -1)
        x = torch.cat([feats['trk_vertices'], result], dim=2)
        return self['out'](feats['trk_vertices'] + self['result'](x))
class DetAttendTrk(nn.ModuleDict):
    def __init__(self, layer_dict, num_heads=4):
        super().__init__(layer_dict)
        self.num_heads = num_heads
    def forward(self, feats, masks, B, M, N):
        queries = self['query'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        keys = self['key'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        vals = self['val'](feats['trk_vertices']).view(B, M, self.num_heads, -1)
        alpha = torch.einsum('bnkd,bmkd->bmnk', queries, keys)
        alpha = alpha.where(masks['trkdet'].view(B, M, N, 1).expand(-1, -1, -1, self.num_heads), torch.full_like(alpha, -1e7))
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.where(masks['trkdet'].view(B, M, N, 1).expand(-1, -1, -1, self.num_heads), torch.zeros_like(alpha))
        result = torch.einsum('bmnk,bmkd->bnkd', alpha, vals)
        result = result.reshape(B, N, -1)
        x = torch.cat([feats['det_vertices'], result, feats['bkg_vertex'].view(B,1,-1).expand(-1,N,-1)], dim=2)
        return self['out'](feats['det_vertices'] + self['result'](x))
class DetAttendDet(nn.ModuleDict):
    def __init__(self, layer_dict, num_heads=4):
        super().__init__(layer_dict)
        self.num_heads = num_heads
    def forward(self, feats, masks, B, M, N):
        queries = self['query'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        keys = self['key'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        vals = self['val'](feats['det_vertices']).view(B, N, self.num_heads, -1)
        alpha = torch.einsum('bmkd,bnkd->bmnk', queries, keys)
        mask = (masks['det'].view(B, N, 1, 1) * masks['det'].view(B, 1, N, 1)).expand(-1, -1, -1, self.num_heads)
        alpha = alpha.where(mask, torch.full_like(alpha, -1e7))
        alpha = F.softmax(alpha, dim=2)
        alpha = alpha.where(mask, torch.zeros_like(alpha))
        result = torch.einsum('bmnk,bnkd->bmkd', alpha, vals)
        result = result.reshape(B, N, -1)
        x = torch.cat([feats['det_vertices'], result], dim=2)
        return self['out'](feats['det_vertices'] + self['result'](x))

class Edgeless(nn.Module):
    def forward(self, *args, **kwargs):
        return None
class BkgDetBase(nn.ModuleDict):    
    def forward(self, feats, masks, B, M, N):
        x = []
        if self['bkg_in'] is not None:
            x.append(self['bkg_in'](feats['bkg_vertex']).view(B, 1, -1).expand(-1, N, -1))
        if self['det_in'] is not None:
            x.append(self['det_in'](feats['det_vertices']))
        if self['bkgdet_in'] is not None:
            x.append(self['bkgdet_in'](feats['bkgdet_edges']))
        return self['out'](torch.cat(x, dim=2))
class TrkDetBase(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        x = []
        if self['trk_in'] is not None:
            x.append(self['trk_in'](feats['trk_vertices']).view(B, M, 1, -1).expand(-1, -1, N, -1))
        if self['det_in'] is not None:
            x.append(self['det_in'](feats['det_vertices']).view(B, 1, N, -1).expand(-1, M, -1, -1))
        if self['trkdet_in'] is not None:
            x.append(self['trkdet_in'](feats['trkdet_edges']))
        return self['out'](torch.cat(x, dim=3))
class BkgIdentity(nn.Module):
    def forward(self, feats, masks, B, M, N):
        return feats['bkg_vertex']
class TrkIdentity(nn.Module):
    def forward(self, feats, masks, B, M, N):
        return feats['trk_vertices']
class DetIdentity(nn.Module):
    def forward(self, feats, masks, B, M, N):
        return feats['det_vertices']
class BkgBase(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        bkg_vertex = self['bkg_in'](feats['bkg_vertex'])
        if self['bkgdet_in'] is not None:
            bkgdet_in = self['bkgdet_in'](feats['bkgdet_edges'])
            if self['bkgdet_sigmoid'] is not None:
                bkgdet_in = bkgdet_in * torch.sigmoid(self['bkgdet_sigmoid'](feats['bkgdet_edges']))
            mask = masks['det'].view(B, N, 1).expand_as(bkgdet_in)
            bkgdet_in = bkgdet_in.where(mask, torch.zeros_like(bkgdet_in))
            bkgdet_in = bkgdet_in.sum(dim=1)
            bkg_vertex = bkg_vertex + bkgdet_in
        return self['out'](bkg_vertex)
class TrkBase(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        trk_vertices = self['trk_in'](feats['trk_vertices'])
        if self['trkdet_in'] is not None:
            trkdet_in = self['trkdet_in'](feats['trkdet_edges'])
            if self['trkdet_sigmoid'] is not None:
                trkdet_in = trkdet_in * torch.sigmoid(self['trkdet_sigmoid'](feats['trkdet_edges']))
            mask = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet_in)
            trkdet_in = trkdet_in.where(mask, torch.zeros_like(trkdet_in))
            trkdet_in = trkdet_in.sum(dim=2)
            trk_vertices = trk_vertices + trkdet_in
        return self['out'](trk_vertices)
class DetBase(nn.ModuleDict):
    def forward(self, feats, masks, B, M, N):
        det_vertices = self['det_in'](feats['det_vertices'])
        if self['trkdet_in'] is not None:
            trkdet_in = self['trkdet_in'](feats['trkdet_edges'])
            if self['trkdet_sigmoid'] is not None:
                trkdet_in = trkdet_in * torch.sigmoid(self['trkdet_sigmoid'](feats['trkdet_edges']))
            mask_trkdet = masks['trkdet'].view(B, M, N, 1).expand_as(trkdet_in)
            trkdet_in = trkdet_in.where(mask_trkdet, torch.zeros_like(trkdet_in))
            trkdet_in = trkdet_in.sum(dim=1)
            bkgdet_in = self['bkgdet_in'](feats['bkgdet_edges'])
            if self['bkgdet_sigmoid'] is not None:
                bkgdet_in = bkgdet_in * torch.sigmoid(self['bkgdet_sigmoid'](feats['bkgdet_edges']))
            det_vertices = det_vertices + bkgdet_in + trkdet_in
        return self['out'](det_vertices)
class GraphNet(nn.ModuleDict):
    def __init__(self, layer_dict, num_iterations=1):
        super().__init__(layer_dict)
        self.num_iterations = num_iterations
    def get_updated_states(self, *args, **kwargs):
        return None
    def forward(self, feats, states, masks):
        B, M, N = masks['trkdet'].size()
        feats = feats.copy()
        for idx in range(self.num_iterations):
            feats['bkgdet_edges'] = self['bkgdet'](feats, masks, B, M, N)
            feats['trkdet_edges'] = self['trkdet'](feats, masks, B, M, N)
            feats['bkg_vertex'] = self['bkg'](feats, masks, B, M, N)
            feats['trk_vertices'] = self['trk'](feats, masks, B, M, N)
            feats['det_vertices'] = self['det'](feats, masks, B, M, N)
        return feats, None    
                
class MatcherSequential(nn.ModuleList):
    def __init__(self, layer_lst):
        assert all([isinstance(layer, (GraphNet,)) for layer in layer_lst])
        super().__init__(layer_lst)
    def update_states(self, states, assignment_ids, initialize_mask):
        """
        Args:
            states (list)
            assignment_ids (LongTensor): Of size (B, M)
            initialize_mask (BoolTensor): Of size (B, M)
        """
        return [layer.get_updated_states(states[idx], assignment_ids, initialize_mask)
                for idx, layer in enumerate(self)]
    def forward(self, feats, states, masks):
        """
        Args:
            feats (dict<Tensor>):
            states (list)
            masks (dict<Tensor>):
        """
        if states == None:
            states = [None for elem in self]
        else:
            states = states.copy()
        for idx, layer in enumerate(self):
            feats, states[idx] = layer(feats, states[idx], masks)
        return feats, states
    

class TrackDescriptor(nn.Module):
    def __init__(self, memory_layers, Dmem, out_layers, Dapp, init_var, init_var_background,
                 masktrackrcnn_update=False, masktrackrcnn_scoring=False, expavglbscores=False):
        super().__init__()
        self.memory_layers = memory_layers
        self.Dmem = Dmem
        self.out_layers = out_layers
        self.Dapp = Dapp
        self.init_var = asparam(init_var)
        self.init_var_background = asparam(init_var_background)
        self.masktrackrcnn_update = masktrackrcnn_update
        self.masktrackrcnn_scoring = masktrackrcnn_scoring
        self.expavglbscores = expavglbscores
    def get_init_state(self, sizes, device, background):
        B, M, C, H, W = sizes
        active = torch.zeros((B, M), dtype=torch.long, device=device)
        mask = torch.ones((B, M), dtype=torch.bool, device=device)
        lbscores = torch.zeros((B, M, C), device=device)
        mean = torch.zeros((B, M, self.Dapp), device=device)  # Placeholder
        var = F.softplus(self.init_var.repeat(B, M, 1))
        var_background = F.softplus(self.init_var_background.repeat(B, 1))
        boxes = torch.full((B, M, 4), -1., device=device)      # Placeholder
        tracks = {'lbscores': lbscores, 'time': 0, 'appearance_mean': mean, 'appearance_var': var,
                  'boxes': boxes, 'active': active,
                  'embedding': torch.zeros((B, M, self.Dmem), device=device),
                  'memory_cell': torch.zeros((B, M, self.Dmem), device=device),}
        background = {'appearance_mean': background['appearance_mean'], 'appearance_var': var_background,
                      'embedding': torch.zeros((B, self.Dmem), device=device),
                      'memory_cell': torch.zeros((B, self.Dmem), device=device)}

        if self.masktrackrcnn_scoring:
            tracks['masktrackrcnn_class_counts'] = torch.zeros((B, M, C), device=device, dtype=torch.long)
            tracks['masktrackrcnn_conf'] = torch.zeros((B, M), device=device)
        
        return tracks, background
    def update_memory(self, x, prev_out, prev_cell, mode):
        if mode == 'background':
            x = x.unsqueeze(1)
            prev_out = prev_out.unsqueeze(1)
            prev_cell = prev_cell.unsqueeze(1)
        fioc = self.memory_layers[mode]['x'](x) + self.memory_layers[mode]['prev_out'](prev_out)
        forget_gate = torch.sigmoid(fioc[:, :, 0:self.Dmem])
        input_gate = torch.sigmoid(fioc[:, :, self.Dmem:2*self.Dmem])
        output_gate = torch.sigmoid(fioc[:, :, 2*self.Dmem:3*self.Dmem])
        cell = forget_gate * prev_cell + input_gate * F.tanh(fioc[:, :, 3*self.Dmem:4*self.Dmem])
        out = output_gate * self.memory_layers[mode]['output_naf'](cell)
        if mode == 'background':
            out = out.squeeze(1)
            cell = cell.squeeze(1)
        return out, cell
    def _update_appearance(self, embedding, mask, mean_new, track_mean, track_var):
        mean_eta = torch.sigmoid(self.out_layers['mean_eta'](embedding)) # (B,M,1)
        var_eta = torch.sigmoid(self.out_layers['var_eta'](embedding))   # (B,M,1)
        var_new = F.softplus(self.out_layers['var_new'](embedding))      # (B,M,1)
        updated_mean = (1 - mean_eta) * track_mean + mean_eta * mean_new
        if self.masktrackrcnn_update:
            updated_mean = mean_new
        if mask is not None:
            mean = updated_mean.where(mask, track_mean)
        else:
            mean = updated_mean
        updated_var = ((1 - var_eta) * track_var
                       + var_eta * var_new
                       + (mean_eta * var_eta / (mean_eta + var_eta)) * (track_mean - mean_new) ** 2)
        if mask is not None:
            var = updated_var.where(mask, track_var)
        else:
            var = updated_var            
        return mean, var
        
    def forward(self, tracks, detections, background, new_background, trackdet_embedding,
                assignment_ids, initialize_mask):
        """
        Args:
            tracks (dict)    : {'appearance': {'mean': Size (B,M,D),
                                               'var' : Size (B,M,D)},
                                'scores'    : Size (B,M)}
            detections (dict): {'appearance': Size (B,N,D),
                                'lbscores'  : Size (B,N,C),
                                'scores'    : Size (B,N)}
            background (dict): {'appearance_mean': Size (B,),
                                'appearance_var' : Size (B,)}
            new_background (dict): {'appearance_mean': Size (B,)}
            trackdet_embedding (Dict)     : 
            assignment_ids (LongTensor)   : Size (B,M)
            initialize_mask (BoolTensor)  : Size (B,M)
        """
        B, M, Dapp = tracks['appearance_mean'].size()
#        _, _, _, Dtd = trackdet_embedding.size()
        _, _, Dtd = trackdet_embedding['trk_vertices'].size()
        _, N, C = detections['lbscores'].size()
        device = tracks['appearance_mean'].device

        embedding = torch.where(
            ((tracks['active'] == 1) + (tracks['active'] == 2)).view(B, M, 1).expand(-1, -1, Dtd),
            trackdet_embedding['trk_vertices'],
            torch.zeros((B, M, Dtd), device=device)) # (B,M,1,Dtd)
        embedding = torch.where(
            initialize_mask.view(B, M, 1).expand(-1, -1, Dtd),
            trackdet_embedding['det_vertices'].gather(
                1,
                assignment_ids.view(B, M, 1).expand(-1, -1, Dtd)),
            embedding)
        embedding = embedding.view(B, M, Dtd)
        if self.masktrackrcnn_update:
            memory_cell = embedding
        else:
            embedding, memory_cell = self.update_memory(
                embedding, tracks['embedding'], tracks['memory_cell'], 'tracks')
        
        mask = (tracks['active'] == 1) + (tracks['active'] == 2) + initialize_mask
        embedding = embedding.where(mask.view(B, M, 1).expand(-1, -1, self.Dmem), torch.zeros_like(embedding))
        memory_cell = memory_cell.where(mask.view(B, M, 1).expand(-1, -1, self.Dmem), torch.zeros_like(memory_cell))

        lbscores = self.out_layers['lbscores'](embedding)

        mean, var = self._update_appearance(
            embedding,
            (tracks['active'] == 1).view(B, M, 1).expand(-1, -1, Dapp),
            detections['appearance'].gather(1, assignment_ids.view(B, M, 1).expand(-1, -1, Dapp)),
            tracks['appearance_mean'],
            tracks['appearance_var'])
        
        bkg_embedding, bkg_memory_cell = self.update_memory(
            trackdet_embedding['bkg_vertex'], background['embedding'], background['memory_cell'], 'background')

        bkg_mean, bkg_var = self._update_appearance(
            bkg_embedding,
            None,
            new_background['appearance_mean'],
            background['appearance_mean'],
            background['appearance_var'])
        background = {
            'appearance_mean': bkg_mean,
            'appearance_var': bkg_var,
            'memory_cell': bkg_memory_cell,
            'embedding': bkg_embedding,
        }

        # ASSERTS
        assert torch.isfinite(embedding).all(), ("embedding", GLOBALS['frame idx'], embedding)
        assert torch.isfinite(lbscores).all(), ("lbscore", GLOBALS['frame idx'], lbscores)
#        assert torch.isfinite(mean_eta).all(), ("mean eta", GLOBALS['frame idx'], mean_eta)
#        assert torch.isfinite(var_eta).all(), ("var eta", GLOBALS['frame idx'], var_eta)
#        assert torch.isfinite(var_new).all(), ("var new", GLOBALS['frame idx'], var_new, self.out_layers['var_new']((state, mask))[0], mask)
        assert torch.isfinite(background['appearance_mean']).all()
        assert torch.isfinite(background['appearance_var']).all()

        new_tracks = {'lbscores': lbscores, 'time': tracks['time'] + 1, 'embedding': embedding,
                      'appearance_mean': mean, 'appearance_var': var,
                      'boxes': tracks['boxes'], 'active': tracks['active'],
                      'memory_cell': memory_cell}

        mask = (tracks['active'] == 1).view(B, M, 1)
        if self.masktrackrcnn_scoring:
            det_classes = 1 + detections['lbscores'][:,:,1:].argmax(dim=2)
            det_classes = F.one_hot(det_classes, num_classes=C)
            new_trk_classes = det_classes.gather(1, assignment_ids.view(B, M, 1).expand(-1, -1, C))
            new_tracks['masktrackrcnn_class_counts'] = torch.where(
                mask.expand(-1, -1, C),
                tracks['masktrackrcnn_class_counts'] + new_trk_classes,
                tracks['masktrackrcnn_class_counts'])
            det_conf = F.softmax(detections['lbscores'], dim=2)[:,:,1:].max(dim=2)[0]
            new_trk_conf = det_conf.gather(1, assignment_ids)
            new_tracks['masktrackrcnn_conf'] = torch.where(
                mask.view(B, M),
                tracks['masktrackrcnn_conf'] + new_trk_conf,
                tracks['masktrackrcnn_conf'])

            new_tracks['lbscores'] = torch.zeros_like(new_tracks['lbscores'])
            for b in range(B):
                for m in range(M):
                    max_class_idx = new_tracks['masktrackrcnn_class_counts'][b,m].argmax()
                    max_class_conf = (new_tracks['masktrackrcnn_conf'][b,m]
                                      / (new_tracks['masktrackrcnn_class_counts'][b,m].sum() + 1e-7))
                    max_class_conf = ((C * max_class_conf + 1e-7).log()
                                      - (1 - max_class_conf + 1e-7).log())
                    new_tracks['lbscores'][b, m, max_class_idx] = max_class_conf
        if self.expavglbscores:
            new_scores = detections['lbscores'].gather(1, assignment_ids.view(B, M, 1).expand(-1, -1, C))
            lbscores = torch.where(
                mask.expand(-1, -1, C),
                (0.9 * tracks['lbscores'] + 0.1 * new_scores),
                tracks['lbscores'])
            lbscores = torch.where(
                initialize_mask.view(B, M, 1).expand(-1, -1, C),
                new_scores,
                lbscores)
            new_tracks['lbscores'] = lbscores
        
        return new_tracks, background

class TrackDescriptorLSTMHack(TrackDescriptor):
    def __init__(self, *args, utilize_embedding=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_mean_x = nn.Linear(self.Dapp, 4*self.Dapp)
        self.layer_mean_h = nn.Linear(self.Dapp, 4*self.Dapp)
        self.layer_mean_out = nn.Linear(self.Dapp, self.Dapp)
        self.layer_var_x = nn.Linear(self.Dapp, 4*self.Dapp)
        self.layer_var_h = nn.Linear(self.Dapp, 4*self.Dapp)
        self.layer_var_out = nn.Linear(self.Dapp, self.Dapp)
    def get_init_state(self, sizes, device, background):
        B, M, C, H, W = sizes
        self.trk_mean_cell = torch.zeros((B, M, self.Dapp), device=device)
        self.trk_mean_out = torch.zeros((B, M, self.Dapp), device=device)
        self.trk_var_cell = torch.zeros((B, M, self.Dapp), device=device)
        self.trk_var_out = torch.zeros((B, M, self.Dapp), device=device)
        self.bkg_mean_cell = torch.zeros((B, self.Dapp), device=device)
        self.bkg_mean_out = torch.zeros((B, self.Dapp), device=device)
        self.bkg_var_cell = torch.zeros((B, self.Dapp), device=device)
        self.bkg_var_out = torch.zeros((B, self.Dapp), device=device)
        return super().get_init_state(sizes, device, background)
    def _lstm_bkg_update(self, x, cell, out, layer_x, layer_h):
        fioc = layer_x(x) + layer_h(out)
        forget_gate = torch.sigmoid(fioc[:, 0:self.Dapp])
        input_gate = torch.sigmoid(fioc[:, self.Dapp:2*self.Dapp])
        output_gate = torch.sigmoid(fioc[:, 2*self.Dapp:3*self.Dapp])
        cell = forget_gate * cell + input_gate * F.tanh(fioc[:, 3*self.Dapp:4*self.Dapp])
        out = output_gate * F.tanh(cell)
        return out, cell
    def _lstm_trk_update(self, x, cell, out, layer_x, layer_h):
        fioc = layer_x(x) + layer_h(out)
        forget_gate = torch.sigmoid(fioc[:, :, 0:self.Dapp])
        input_gate = torch.sigmoid(fioc[:, :, self.Dapp:2*self.Dapp])
        output_gate = torch.sigmoid(fioc[:, :, 2*self.Dapp:3*self.Dapp])
        cell = forget_gate * cell + input_gate * F.tanh(fioc[:, :, 3*self.Dapp:4*self.Dapp])
        out = output_gate * F.tanh(cell)
        return out, cell
    def _update_appearance(self, embedding, mask, mean_new, track_mean, track_var):
        if mask is None:
            mean_out, mean_cell = self._lstm_bkg_update(
                mean_new, self.bkg_mean_cell, self.bkg_mean_out, self.layer_mean_x, self.layer_mean_h)
            var_out, var_cell = self._lstm_bkg_update(
                mean_new, self.bkg_var_cell, self.bkg_var_out, self.layer_var_x, self.layer_var_h)
            self.bkg_mean_cell = mean_cell
            self.bkg_mean_out = mean_out
            self.bkg_var_cell = var_cell
            self.bkg_var_out = var_out
            return self.layer_mean_out(self.bkg_mean_out), F.softplus(self.layer_var_out(self.bkg_var_out))
        else:
            mean_out, mean_cell = self._lstm_trk_update(
                mean_new, self.trk_mean_cell, self.trk_mean_out, self.layer_mean_x, self.layer_mean_h)
            var_out, var_cell = self._lstm_trk_update(
                mean_new, self.trk_var_cell, self.trk_var_out, self.layer_var_x, self.layer_var_h)
            self.trk_mean_cell = mean_cell.where(mask, self.trk_mean_cell)
            self.trk_mean_out = mean_out.where(mask, self.trk_mean_out)
            self.trk_var_cell = var_cell.where(mask, self.trk_var_cell)
            self.trk_var_out = var_out.where(mask, self.trk_var_out)
            return self.layer_mean_out(self.trk_mean_out), F.softplus(self.layer_var_out(self.trk_var_out))        

class TrackDescriptorNoLSTM(TrackDescriptor):
    def update_memory(self, x, prev_out, prev_cell, mode):
        return x, prev_cell
class TrackDescriptorNoLSTMCell(TrackDescriptor):
    def update_memory(self, x, prev_out, prev_cell, mode):
        if mode == 'background':
            x = x.unsqueeze(1)
            prev_out = prev_out.unsqueeze(1)
            prev_cell = prev_cell.unsqueeze(1)
        oc = self.memory_layers[mode]['x'](x)
        output_gate = torch.sigmoid(oc[:, :, 0:self.Dmem])
        out = output_gate * F.tanh(oc[:, :, self.Dmem:2*self.Dmem])
        cell = prev_cell
        if mode == 'background':
            out = out.squeeze(1)
            cell = cell.squeeze(1)
        return out, cell
    
        
    
class DetectionDescriptorExtractor(nn.Module):
    def __init__(self, appearance_layers, feature_selection='mask pool'):
        super().__init__()
        self.appearance_layers = appearance_layers
        self.feature_selection = feature_selection
    def mask_pool(self, featmap, segscores):
        B, D, H, W = featmap.size()
        _, _, Hs, Ws = segscores.size()
        segscores = torch.cat([torch.full((B, 1, Hs, Ws), 10.0, device=segscores.device), segscores], dim=1)
        segmap = F.softmax(F.interpolate(segscores, size=(H,W), mode='bilinear'), dim=1)[:, 1:]
        segmap = segmap / (segmap.sum(dim=(2,3), keepdim=True) + 1e-4)
        return torch.einsum('bdhw,bnhw->bnd', featmap, segmap)
    def center(self, featmap, boxes, orig_image_size):
        H, W = orig_image_size
        B, D, Hs, Ws = featmap.size()
        _, N, _ = boxes.size()
        scale = torch.tensor([Ws / W, Hs / H], device=featmap.device)
        scale = scale.view(1, 1, 2)
        centers = 0.5 * scale * (boxes[:, :, 0:2] + boxes[:, :, 2:4])
        centers = centers.long()
        ids = centers[:,:,0] + Ws * centers[:,:,1] # (B, N)
        return featmap.view(B, D, Hs*Ws).gather(2, ids.view(B, 1, N).expand(-1, D, -1)).permute(0, 2, 1)
    def sigmoid_mask_pool(self, featmap, segscores):
        B, D, H, W = featmap.size()
        _, _, Hs, Ws = segscores.size()
        segmap = torch.sigmoid(F.interpolate(segscores - 10.0, size=(H,W), mode='bilinear'))
        segmap = segmap / (segmap.sum(dim=(2,3), keepdim=True) + 1e-4)
        return torch.einsum('bdhw,bnhw->bnd', featmap, segmap)
    def forward(self, appearance_feature_map_lst, detections, orig_image_size):
        """
        Args:
            appearance_feature_map_lst (list<Tensor>): Contains appearance feature maps (B, D, H, W)
            detections (dict): Contains filtered detections, their boxes, segscores, active-table, lbscores
        Returns
            detections (dict): Contains filtered detections, their boxes, segscores, active-table, lbscores,
                appearance
        """
        B, N = detections['active'].size()
        detections = detections.copy()
        if self.feature_selection == 'mask pool':
            app_feat_lst = [self.mask_pool(featmap, detections['segscores'])
                            for featmap in appearance_feature_map_lst] # (B,N,D)
        elif self.feature_selection == 'center':
            app_feat_lst = [self.center(featmap, detections['boxes'], orig_image_size)
                            for featmap in appearance_feature_map_lst] # (B,N,D)
        elif self.feature_selection == 'sigmoid mask pool':
            app_feat_lst = [self.sigmoid_mask_pool(featmap, detections['segscores'])
                            for featmap in appearance_feature_map_lst] # (B,N,D)            
        app_feats = self.appearance_layers(torch.cat(app_feat_lst, dim=2))
        app_feats = app_feats.where(detections['active'].view(B, N, 1) == 1, torch.zeros_like(app_feats))
        background_descriptor = torch.cat([
            F.adaptive_avg_pool2d(feature_map, (1,1)) for feature_map in appearance_feature_map_lst
        ], dim=1).view(B, -1)
        detections['appearance'] = app_feats
        if not torch.isfinite(app_feats).all():
            for idx, feature_map in enumerate(appearance_feature_map_lst):
                print_tensor_statistics(feature_map, f"feature map {idx}")
            print_tensor_statistics(detections['segscores'], "segscores")
            print_tensor_statistics(detections['segscores'][detections['active'] == 1])
            print_tensor_statistics(app_feats, "detection appearance features")
            for name, param in self.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"Layer {name} is nonfinite")
        return detections, {'appearance_mean': background_descriptor}


class TrackDetectionMatcher(nn.Module):
    def __init__(self, layers, score_layers,
                 det_stuff=('lbscores', 'boxes'),
                 bkg_stuff=(),
                 trk_stuff=(),
                 trkdet_stuff=('appearance gaussian loglikelihood', 'box iou'),
                 bkgdet_stuff=('appearance loglikelihood'),
                 masktrackrcnn_mode=False):
        super().__init__()
        self.layers = layers
        self.score_layers = score_layers
        self.det_stuff = det_stuff
        self.bkg_stuff = bkg_stuff
        self.trk_stuff = trk_stuff
        self.trkdet_stuff = trkdet_stuff
        self.bkgdet_stuff = bkgdet_stuff
        self.masktrackrcnn_mode = masktrackrcnn_mode
    def _get_trkdet_edges(self, tracks, detections):
        B, M, Dapp = tracks['appearance_mean'].size()
        _, N, _ = detections['appearance'].size()
        if len(self.trkdet_stuff) == 0:
            return None
        edges = []
        if 'appearance gaussian loglikelihood' in self.trkdet_stuff:
            mu = tracks['appearance_mean']
            var = tracks['appearance_var']
            x = detections['appearance']
            ivar = 1 / var
            ll = - 1/2 * (Dapp * LOG2PI
                          + var.log().sum(dim=2, keepdim=True)
                          + torch.einsum('bnd,bnd,bmd->bmn', x, x, ivar)
                          - 2 * torch.einsum('bnd,bmd->bmn', x, mu * ivar)
                          + torch.einsum('bmd,bmd->bm', mu, mu * ivar).view(B, M, 1)).view(B, M, N, 1)
            edges.append(-4.0 + 0.01 * ll) # Ugly hard-coded rescaling
        if 'appearance masktrackrcnn' in self.trkdet_stuff:
            mu = tracks['appearance_mean']
            x = detections['appearance']
            ll = torch.einsum('bnd,bmd->bmn', x, mu).view(B, M, N, 1)
            edges.append(ll)
        if 'box iou' in self.trkdet_stuff:
            iou = batch_many_to_many_box_iou(tracks['boxes'], detections['boxes']).view(B, M, N, 1)
            edges.append(-1.0 + 2.0 * iou.view(B, M, N, 1))
        return torch.cat(edges, dim=3) if len(edges) > 1 else edges[0]
    def _get_bkgdet_edges(self, background, detections):
        B, Dapp = background['appearance_mean'].size()
        _, N, _ = detections['appearance'].size()
        if len(self.bkgdet_stuff) == 0:
            return None
        edges = []
        if 'appearance gaussian loglikelihood' in self.bkgdet_stuff:
            mu = background['appearance_mean']
            var = background['appearance_var']
            x = detections['appearance']
            ivar = 1 / var
            ll = - 1/2 * (Dapp * LOG2PI
                          + var.log().sum(dim=1, keepdim=True)
                          + torch.einsum('bnd,bnd,bd->bn', x, x, ivar)
                          - 2 * torch.einsum('bnd,bd->bn', x, mu * ivar)
                          + torch.einsum('bd,bd->b', mu, mu * ivar).view(B, 1)).view(B, N, 1)
            edges.append(-4.0 + 0.01 * ll) # Ugly hard-coded rescaling
        return torch.cat(edges, dim=2) if len(edges) > 1 else edges[0]
    def _get_background_vertex(self, background):
        if len(self.bkg_stuff) == 0:
            return None
        vertex_parts = []
        if 'appearance' in self.bkg_stuff:
            vertex_parts.append(background['appearance_mean'])
        if 'embedding' in self.bkg_stuff:
            vertex_parts.append(background['embedding'])
        return torch.cat(vertex_parts, dim=1)
    def _get_track_vertices(self, tracks):
        if len(self.trk_stuff) == 0:
            return None
        vertex_parts = []
        if 'lbscores' in self.trk_stuff:
            vertex_parts.append(tracks['lbscores'])
        if 'boxes' in self.trk_stuff:
            vertex_parts.append(normalize_boxes(tracks['boxes']))
        if 'appearance' in self.trk_stuff:
            vertex_parts.append(tracks['appearance_mean'])
        if 'embedding' in self.trk_stuff:
            vertex_parts.append(tracks['embedding'])
#        B, M, _ = tracks['spatial']['boxes'].size()
#        device = tracks['spatial']['boxes'].device
#        return torch.ones((B, M, 1, 1), device=device, torch.float32)
        return torch.cat(vertex_parts, dim=2)
    def _get_detection_vertices(self, detections):
        B, N, _ = detections['lbscores'].size()
        device = detections['lbscores'].device
        if len(self.det_stuff) == 0:
            return None
        vertex_parts = []
        if 'lbscores' in self.det_stuff:
            vertex_parts.append(detections['lbscores'])
        if 'boxes' in self.det_stuff:
            vertex_parts.append(normalize_boxes(detections['boxes']))
        if 'appearance' in self.det_stuff:
            vertex_parts.append(detections['appearance'])

        return torch.cat(vertex_parts, dim=2)
    def update_state(self, state, assignment_ids, initialize_mask):
        """Moves states from detection vertices to track vertices IF that detection initializes
        that track this frame. Then sets the state for detection vertices to zero. Also asserts
        that bkgdet- and trkdet edges are stateless.
        Args:
            state (dict): Dict of list each with 3 elements, each a pair of tensors of size
                (B*N, D), (B*M*N, D), or (B*N, D), depending on which of the 3 elements it is.
            assignment_ids (LongTensor): Of size (B,M)
            initialize_mask (LongTensor): Of size (B,M), marking memory entries that are
                initialized this frame. assignment_ids of these rows mark the detections that
                initialize these tracks.
        """
        return self.layers.update_states(state, assignment_ids, initialize_mask)
    def forward(self, tracks, detections, background, state):
        """
        Args:
            tracks (Dict): lbscores (B,M,C), time (int), appearance_mean (B,M,D), appearance_var (B,M,D),
                boxes (B,M,D), active (B,M)
            detections (Dict): lbscores (B,N,C), appearance (B,N,D), boxes (B,M,D), active (B,N)
            background (Dict): appearance_mean (B,D), appearance_var (B,D)
            state (Dict): 
        Returns:
            matching scores (Tensor): Of size (B,M,N)
            new_state
        """
        B, M = tracks['active'].size()
        _, N = detections['active'].size()
        device = detections['active'].device

        masks = {
            'trk': ((tracks['active'] == 1) + (tracks['active'] == 2)).view(B, M),
            'det': (detections['active'] == 1).view(B, N),
        }
        masks['trkdet'] = masks['trk'].view(B, M, 1) * masks['det'].view(B, 1, N)

        feats = {
            'bkg_vertex': self._get_background_vertex(background), # (B, D)
            'trk_vertices': self._get_track_vertices(tracks), # (B, M, D)
            'det_vertices': self._get_detection_vertices(detections),
            'bkgdet_edges': self._get_bkgdet_edges(background, detections),
            'trkdet_edges': self._get_trkdet_edges(tracks, detections),
        }
#        if GLOBALS['num iter'] == 201:
#            raise ValueError()
#        if GLOBALS['num iter'] in (1, 10, 200):
#            if GLOBALS['frame idx'] == 0:
#                print()
#                print("----------")
#            print()
#            print(masks['det'].long()[0, [0,1,2,3,4,5,6,7]])
#            print(feats['bkgdet_edges'][0, [0,1,2,3,4,5,6,7], 0])
#            print(masks['trkdet'].long()[0, [0,1,2,3,4]][:, [0,1,2,3,4,5,6,7]])
#            print(feats['trkdet_edges'][0, [0,1,2,3,4]][:, [0,1,2,3,4,5,6,7], 0])
#            print(feats['trkdet_edges'][0, [0,1,2,3,4]][:, [0,1,2,3,4,5,6,7], 1])
#            print_tensor_statistics(background['appearance_var'], "bkg var")
#            print_tensor_statistics(tracks['appearance_var'][masks['trk']], "trk var")
#            print_tensor_statistics(feats['trk_vertices'][:,:,0:41][masks['trk']], 'trk lbscores')
#            print_tensor_statistics(feats['trk_vertices'][:,:,41:45][masks['trk']], 'trk boxes')
#            print_tensor_statistics(feats['det_vertices'][:,:,0:41][masks['det']], 'det lbscores')
#            print_tensor_statistics(feats['det_vertices'][:,:,41:45][masks['det']], 'det boxes')
#            print_tensor_statistics(feats['trkdet_edges'][:,:,:,0][masks['trkdet']], 'trkdet appearance')
#            print_tensor_statistics(feats['trkdet_edges'][:,:,:,1][masks['trkdet']], 'trkdet spatial')

        if self.masktrackrcnn_mode:
#            assert 'appearance gaussian loglikelihood' in self.trkdet_stuff
            assert 'appearance masktrackrcnn' in self.trkdet_stuff
            assert 'box iou' in self.trkdet_stuff
            app_sim = feats['trkdet_edges'][:,:,:,0]
            app_sim = app_sim.where(masks['trkdet'], torch.full_like(app_sim, -1e5))
            app_sim = torch.cat([app_sim, torch.full((B, 1, N), 0.0, device=device)], 1)
            app_sim = F.log_softmax(app_sim, dim=1)
            novel_sim = app_sim[:, -1, :]
            app_sim = app_sim[:, :-1, :]
            conf = F.softmax(detections['lbscores'], dim=2)[:, :, 1:].max(dim=2)[0].view(B, 1, N)
            iou = feats['trkdet_edges'][:,:,:,1] + 1
            cls_sim = 10*(detections['lbscores'][:, :, 1:].argmax(dim=2).view(B, 1, N) == tracks['lbscores'][:, :, 1:].argmax(dim=2).view(B, M, 1)).float()
            if self.training:
                sim = app_sim
            else:
                sim = app_sim + conf + iou + cls_sim
            novel_sim = novel_sim.where(masks['det'], torch.full_like(novel_sim, -1e5))
            sim = sim.where(masks['trkdet'], torch.full_like(sim, -1e5))

        matcher_encoding, state = self.layers(feats, state, masks)
        trackdet_scores, novel_track_scores = self.score_layers(matcher_encoding, masks)

        if self.masktrackrcnn_mode:
            trackdet_scores = sim
            novel_track_scores = novel_sim
#            if GLOBALS['frame idx'] == 0:
#                print("\n--------")
#            if GLOBALS['num iter'] == 31:
#                raise NotImplementedError("DEBUG")
#            if GLOBALS['num iter'] in (1, 10, 20, 30):
#                print()
#                print(app_sim[0,[0,1,2,3,4]][:,[0,1,2,3,4,5,6]])
#                print(iou[0,[0,1,2,3,4]][:,[0,1,2,3,4,5,6]])
#                print(conf[0,0,[0,1,2,3,4,5,6]])
#                print(cls_sim[0,[0,1,2,3,4]][:,[0,1,2,3,4,5,6]])
#                print(sim[0,[0,1,2,3,4]][:,[0,1,2,3,4,5,6]])
#                print(novel_sim[0,[0,1,2,3,4,5,6]])

#        if GLOBALS['num iter'] in (1, 10, 200):
#            print(trackdet_scores[0,[0,1,2,3,4]][:,[0,1,2,3,4,5,6,7]])
#            print(novel_track_scores[0,[0,1,2,3,4,5,6,7]])
#            print()
#            print_tensor_statistics(matcher_encoding['trk_vertices'][masks['trk']], 'trk')
#            print_tensor_statistics(matcher_encoding['det_vertices'][masks['det']], 'det')
#            print_tensor_statistics(matcher_encoding['trkdet_edges'][masks['trkdet']], 'trkdet')

        return trackdet_scores, novel_track_scores, state, matcher_encoding


class SuperGlueSinkhorn(nn.Module):
    """
        Sinkhorn algorithm from the SuperGlue paper
    """
    def __init__(self, max_iter=10, lam=100.0, alpha=1.0, train_alpha=True, verbose=False, device="cuda:0"):
        super(SuperGlueSinkhorn, self).__init__()
        self.max_iter = max_iter
        self.lam = lam
        self.alpha=alpha
        self.verbose = verbose
        if train_alpha:
            self.alpha = nn.Parameter(alpha * torch.ones(1))
        else:
            self.alpha = alpha * torch.ones((1,)).to(device)

    def forward(self, scores):
        if self.verbose:
            print("alpha:")
            print(self.alpha.item())
        Z = self.log_optimal_transport(scores*self.lam, self.alpha)
        #return Z[:, :-1, :-1].exp()
        return Z[:, :, :-1].exp()

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu):
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.max_iter):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(self, scores, alpha):
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu)
        Z = Z - norm  # multiply probabilities by M+N
        return Z


class SinkhornHardAssigner(nn.Module):
    """Assigner used when we train to predict probability that a track-detection pair constitute
    a TP track.
    """

    def __init__(self,
                 max_iter=10, lam=100.0, alpha=1.0, train_alpha=True,
                 track_acceptance_threshold=0.05,
                 novel_acceptance_threshold=0.05):
        super().__init__()
        self.track_acceptance_threshold = track_acceptance_threshold
        self.novel_acceptance_threshold = novel_acceptance_threshold
        self.sink = SuperGlueSinkhorn(max_iter, lam, alpha, train_alpha)

    def forward(self, trackdet_scores, novel_track_scores, track_active):
        """
        Args:
            trackdet_scores (Tensor): Size (B, M, N) with matching scores between tracks and detections
            novel_track_scores (Tensor): Size (B, N) with scores for detections being deemed as novel tracks
            track_active (LongTensor) : Size (B, M) telling us which memory locations are in use for what
                0 is unnused/free, 1 is seen object, 2 is not used but we may use it for fully
                occluded object, 3 is background, 4 is novel object.
        Returns:
            LongTensor: Size (B, M) indicating for each track, what detection idx fits it best (or 0 if no fit)
            BoolTensor: Size (B, M) true for already existing tracks which obtained a matching detection
            BoolTensor: Size (B, M) true for tracks that were initialized this frame
            LongTensor: Size (B, M) update of track_active
        """
        B, M, N = trackdet_scores.size()
        device = trackdet_scores.device
        new_track_active = track_active.clone()

        # trackdet_scores = trackdet_scores.clone().detach()
        # novel_track_scores = novel_track_scores.clone().detach()
        trackdetnov_scores = torch.cat([trackdet_scores, novel_track_scores.view(B, 1, N)], dim=1)  # (B,M+1,N)
        # trackdetnov_scores = trackdetnov_scores - trackdetnov_scores.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        # trackdetnov_scores = torch.exp(trackdetnov_scores)
        PA = self.sink(trackdetnov_scores)
        det_to_track_id = PA.argmax(dim=1)  # (B, N), indices [0, M+2)

        track_assignment_ids = torch.zeros((B, M+2), device=device, dtype=torch.long)
        for i in range(N):
            track_assignment_ids[range(B), det_to_track_id[:, i]] = i
        track_assignment_ids = track_assignment_ids[:, :-2]

        disappearance_mask = (track_active == 1) * (track_assignment_ids == 0)
        new_track_active[disappearance_mask] = 2
        reappearance_mask = (track_active == 2) * (track_assignment_ids != 0)
        new_track_active[reappearance_mask] = 1
        track_assignment_ids[new_track_active != 1] = -1

        confident_novel_det_mask = (det_to_track_id == M)
        available_track_slots = (track_active == 0)
        num_dets = confident_novel_det_mask.sum(dim=1)
        num_dets = torch.min(available_track_slots.sum(dim=1), num_dets)
        initialize_mask = torch.zeros((B, M), dtype=torch.bool, device=device)
        for b in range(B):
            det_ids = torch.nonzero(confident_novel_det_mask[b]).view(-1)[0: num_dets[b]]
            track_ids = torch.nonzero(available_track_slots[b]).view(-1)[0: num_dets[b]]
            track_assignment_ids[b, track_ids] = det_ids
            initialize_mask[b, track_ids] = True
        new_track_active[initialize_mask] = 1
        track_assignment_ids[new_track_active != 1] = 1  # Just put them to something that allows gather -> mask

        return track_assignment_ids, initialize_mask, new_track_active


    # def forward(self, trackdet_scores, novel_track_scores, track_active):
    #     """
    #     Args:
    #         trackdet_scores (Tensor): Size (B, M, N) with matching scores between tracks and detections
    #         novel_track_scores (Tensor): Size (B, N) with scores for detections being deemed as novel tracks
    #         track_active (LongTensor) : Size (B, M) telling us which memory locations are in use for what
    #             0 is unnused/free, 1 is seen object, 2 is not used but we may use it for fully
    #             occluded object, 3 is background, 4 is novel object.
    #     Returns:
    #         LongTensor: Size (B, M) indicating for each track, what detection idx fits it best (or 0 if no fit)
    #         BoolTensor: Size (B, M) true for already existing tracks which obtained a matching detection
    #         BoolTensor: Size (B, M) true for tracks that were initialized this frame
    #         LongTensor: Size (B, M) update of track_active
    #     """
    #     B, M, N = trackdet_scores.size()
    #     device = trackdet_scores.device
    #     new_track_active = track_active.clone()
    #
    #     # trackdet_scores = trackdet_scores.clone().detach()
    #     # novel_track_scores = novel_track_scores.clone().detach()
    #     novel_track_scores = torch.stack([torch.diag(d) for d in novel_track_scores], dim=0)
    #     trackdetnov_scores = torch.cat([trackdet_scores, novel_track_scores], dim=1)  # (B,M+N,N)
    #     trackdetnov_scores = torch.exp(trackdetnov_scores)
    #     PA = self.sink(trackdetnov_scores)
    #     max_scores, max_ids = PA.max(dim=2)  # (B, M+2)
    #     track_max_scores = max_scores[:, :M]
    #     track_assignment_ids = max_ids[:, :M]
    #     novel_max_scores = max_scores[:, M:]
    #     # novel_assignment_ids = max_ids[:, M:]
    #
    #     disappearance_mask = (track_active == 1) * (track_max_scores < self.track_acceptance_threshold)
    #     new_track_active[disappearance_mask] = 2
    #     reappearance_mask = (track_active == 2) * (track_max_scores >= self.track_acceptance_threshold)
    #     new_track_active[reappearance_mask] = 1
    #     track_assignment_ids[new_track_active != 1] = -1
    #
    #     confident_novel_det_mask = novel_max_scores > self.novel_acceptance_threshold
    #     available_track_slots = (track_active == 0)
    #     num_dets = confident_novel_det_mask.sum(dim=1)
    #     num_dets = torch.min(available_track_slots.sum(dim=1), num_dets)
    #     initialize_mask = torch.zeros((B, M), dtype=torch.bool, device=device)
    #     for b in range(B):
    #         det_ids = confident_novel_det_mask[b].nonzero().view(-1)[0: num_dets[b]]
    #         track_ids = available_track_slots[b].nonzero().view(-1)[0: num_dets[b]]
    #         track_assignment_ids[b, track_ids] = det_ids
    #         initialize_mask[b, track_ids] = True
    #     new_track_active[initialize_mask] = 1
    #     track_assignment_ids[new_track_active != 1] = 1  # Just put them to something that allows gather -> mask
    #
    #     return track_assignment_ids, initialize_mask, new_track_active

    
class HardAssigner(nn.Module):
    """Assigner used when we train to predict probability that a track-detection pair constitute
    a TP track.
    """
    def __init__(self,
                 track_acceptance_threshold=-2.0,
                 novel_acceptance_threshold=0.0,
                 masktrackrcnn_mode=False):
        super().__init__()
        self.track_acceptance_threshold = track_acceptance_threshold
        self.novel_acceptance_threshold = novel_acceptance_threshold
        self.masktrackrcnn_mode         = masktrackrcnn_mode
    def forward(self, trackdet_scores, novel_track_scores, track_active):
        """
        Args:
            trackdet_scores (Tensor): Size (B, M, N) with matching scores between tracks and detections
            novel_track_scores (Tensor): Size (B, N) with scores for detections being deemed as novel tracks
            track_active (LongTensor) : Size (B, M) telling us which memory locations are in use for what
                0 is unnused/free, 1 is seen object, 2 is not used but we may use it for fully
                occluded object, 3 is background, 4 is novel object.
        Returns:
            LongTensor: Size (B, M) indicating for each track, what detection idx fits it best (or 0 if no fit)
            BoolTensor: Size (B, M) true for already existing tracks which obtained a matching detection
            BoolTensor: Size (B, M) true for tracks that were initialized this frame
            LongTensor: Size (B, M) update of track_active
        """
        B, M, N = trackdet_scores.size()
        device = trackdet_scores.device
        new_track_active = track_active.clone()

        trackdet_scores = trackdet_scores.clone().detach()
        novel_track_scores = novel_track_scores.clone().detach()
        track_max_scores, assignment_ids = trackdet_scores.max(dim=2) # (B, M)
        if self.masktrackrcnn_mode: # We assign each detection to only one track
            det_max_scores = torch.max(trackdet_scores.max(dim=1)[0], novel_track_scores) # (B, N)
            sec_pos_mask = track_max_scores < det_max_scores.gather(1, assignment_ids) # (B, M)
            track_max_scores[sec_pos_mask] = -100
            track_max_scores[(track_max_scores > -10) * ~sec_pos_mask] = 100
            sec_pos_mask = novel_track_scores < det_max_scores
            novel_track_scores[sec_pos_mask] = -100
            novel_track_scores[(novel_track_scores > -10) * ~sec_pos_mask] = 100
        disappearance_mask = (track_active == 1) * (track_max_scores < self.track_acceptance_threshold)
        new_track_active[disappearance_mask] = 2
        reappearance_mask  = (track_active == 2) * (track_max_scores >= self.track_acceptance_threshold)
        new_track_active[reappearance_mask] = 1
        assignment_ids[new_track_active != 1] = -1

        confident_novel_det_mask = novel_track_scores > self.novel_acceptance_threshold
        available_track_slots = (track_active == 0)
        num_dets = confident_novel_det_mask.sum(dim=1)
        num_dets = torch.min(available_track_slots.sum(dim=1), num_dets)
        initialize_mask = torch.zeros((B, M), dtype=torch.bool, device=device)
        for b in range(B):
            det_ids = confident_novel_det_mask[b].nonzero().view(-1)[0 : num_dets[b]]
            track_ids = available_track_slots[b].nonzero().view(-1)[0 : num_dets[b]]
            assignment_ids[b, track_ids] = det_ids
            initialize_mask[b, track_ids] = True
        new_track_active[initialize_mask] = 1
        assignment_ids[new_track_active != 1] = 1 # Just put them to something that allows gather -> mask
            
        return assignment_ids, initialize_mask, new_track_active

    
class TrackModule(nn.Module):
    def __init__(self, seg_layers=None,
                 embedding_to_seg_layers=None, use_raw_plus_box_segscore=True,
                 use_appearance_correlation=False, use_background_correlation=False,
                 masktrackrcnn_mode=False):
        super().__init__()
        self.seg_layers = seg_layers
        self.embedding_to_seg_layers = embedding_to_seg_layers
        self.use_raw_plus_box_segscore = use_raw_plus_box_segscore
        self.use_appearance_correlation = use_appearance_correlation
        self.use_background_correlation = use_background_correlation
        self.masktrackrcnn_mode = masktrackrcnn_mode
    def get_segscores(self, segscores, embedding, appearance_correlation, background_correlation):
        if self.seg_layers is None:
            return segscores
        if self.use_raw_plus_box_segscore:
            B, M, H, W = segscores.size()
            x = [segscores.view(B*M, 1, H, W)]
        else:
            B, M, _, H, W = segscores.size()
            x = [segscores.view(B*M, 2, H, W)]
        if self.embedding_to_seg_layers is not None:
            from_embedding = self.embedding_to_seg_layers(embedding)
            from_embedding = from_embedding.view(B*M, -1, 1, 1).expand(-1, -1, H, W)
            x.append(from_embedding)
        if self.use_appearance_correlation:
            x = x + [elem.view(B*M, 1, H, W) for elem in appearance_correlation]
        if self.use_background_correlation:
            x = x + [elem.view(B*M, 1, H, W) for elem in background_correlation]
        x = torch.cat(x, dim=1)
        x = self.seg_layers(x).view(B, M, H, W)
        return x
    def forward(self, tracks, detections, assignment_ids, feature_maps, background):
        """
        Args:
            tracks
            detections (dict): Contains boxes, segscores, lbscores
            assignment_ids (LongTensor): Size (B,M)
        Returns:
            dict: {'boxes': Tensor, 'segscores': Tensor, 'lbscores': Tensor}
        """
        B, M = assignment_ids.size()
        _, _, H, W = detections['segscores'].size()
        _, _, C = detections['lbscores'].size()
        device = detections['lbscores'].device
        track_has_det = (tracks['active'] == 1)
        boxes = detections['boxes'].gather(1, assignment_ids.view(B, M, 1).expand(-1, -1, 4))
        boxes = boxes.where(track_has_det.view(B, M, 1).expand(-1, -1, 4), torch.zeros_like(boxes))
        if self.use_raw_plus_box_segscore:
            segscores = detections['segscores'].gather(1, assignment_ids.view(B, M, 1, 1).expand(-1, -1, H, W))
            segscores = segscores.where(track_has_det.view(B, M, 1, 1).expand(-1, -1, H, W),
                                        torch.full_like(segscores, -20.))
        else:
            segscores = torch.stack([detections['raw_segscores'], detections['box_segscores']], dim=2) # (B,N,2,H,W)
            segscores = torch.where(
                track_has_det.view(B, M, 1, 1, 1).expand(-1, -1, 2, H, W),
                segscores.gather(1, assignment_ids.view(B, M, 1, 1, 1).expand(-1, -1, 2, H, W)),
                torch.full((B, M, 2, H, W), -20., device=device)) # (B,M,2,H,W)
        if not torch.isfinite(segscores).all():
            print("\nActive and segscores")
            print(tracks['active'][:,0:6])
            print(detections['active'][:, 0:7])
            print(assignment_ids[:,0:6])
            print(track_has_det[:,0:6])
            print(segscores[:,0:5,0].mean(dim=(2,3)))
            print(segscores[:,0:5,1].mean(dim=(2,3)))
            print("inf at (b,m)")
            print((~torch.isfinite(segscores.mean(dim=(2,3,4)))).nonzero())
            inf_ids = (~torch.isfinite(segscores.mean(dim=(2, 3, 4)))).nonzero()
            for ids in inf_ids:
                b, m = ids.to_list()
                print(tracks['active'][b, m])
                print(assignment_ids[b, m])
                det_idx = assignment_ids[b, m]
                print_tensor_statistics(segscores[b, det_idx], 'segscores')
        assert torch.isfinite(segscores).all(), print_tensor_statistics(segscores, "segscores in")
        assert torch.isfinite(tracks['embedding']).all(), print_tensor_statistics(tracks['embedding'], "track embedding")
        if self.use_appearance_correlation:
            means = tracks['appearance_mean'].split([f.size(1) for f in feature_maps], dim=2)
            ivars = (1 / tracks['appearance_var']).split([f.size(1) for f in feature_maps], dim=2)
            appearance_correlation = [
                0.01 * (-1/2) * (mean.size(2) * LOG2PI
                                 - ivar.log().sum(dim=2).view(B, M, 1, 1)
                                 + torch.einsum('bdhw,bdhw,bmd->bmhw', feature_map, feature_map, ivar)
                                 - 2 * torch.einsum('bdhw,bmd->bmhw', feature_map, mean * ivar)
                                 + torch.einsum('bmd,bmd->bm', mean, mean * ivar).view(B, M, 1, 1))
                for mean, ivar, feature_map in zip(means, ivars, feature_maps)]
            appearance_correlation = [
                F.interpolate(elem, size=(H, W), mode='bilinear').view(B, M, 1, H, W)
                for elem in appearance_correlation]
        else:
            appearance_correlation = None
        if self.use_background_correlation:
            means = background['appearance_mean'].split([f.size(1) for f in feature_maps], dim=1)
            ivars = (1 / background['appearance_var']).split([f.size(1) for f in feature_maps], dim=1)
            background_correlation = [
                0.01 * (-1/2) * (mean.size(1) * LOG2PI
                                 - ivar.log().sum(dim=1).view(B, 1, 1)
                                 + torch.einsum('bdhw,bdhw,bd->bhw', feature_map, feature_map, ivar)
                                 - 2 * torch.einsum('bdhw,bd->bhw', feature_map, mean * ivar)
                                 + torch.einsum('bd,bd->b', mean, mean * ivar).view(B, 1, 1)).unsqueeze(1)
                for mean, ivar, feature_map in zip(means, ivars, feature_maps)]
            background_correlation = [
                F.interpolate(elem, size=(H, W), mode='bilinear').view(B, 1, 1, H, W).repeat(1, M, 1, 1, 1)
                for elem in background_correlation]
        else:
            background_correlation = None
            
        if self.masktrackrcnn_mode:
            assert self.use_raw_plus_box_segscore
            assert not self.use_appearance_correlation
            assert not self.use_background_correlation
        else:
            segscores = self.get_segscores(segscores, tracks['embedding'],
                                           appearance_correlation,
                                           background_correlation)
        
        segscores = segscores.where((tracks['active'] == 1).view(B, M, 1, 1).expand(-1, -1, H, W),
                                    torch.full_like(segscores, -20))
        assert torch.isfinite(segscores).all(), print_tensor_statistics(segscores, "segscores out")
        lbscores = tracks['lbscores']
        tracks['boxes'] = boxes
        tracks['segstate'] = segscores.view(B*M, -1, H, W)
        return {'boxes': boxes, 'segscores': segscores, 'lbscores': lbscores, 'active': tracks['active']}, tracks

        
class RGNN(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 appnet,
                 detection_descriptor_extractor,
                 track_detection_matcher,
                 track_descriptor,
                 hard_assigner,
                 track_module,
                 raw_det_threshold,
                 nms_iou_threshold,
                 max_num_detections,
                 max_num_tracks,
                 num_classes,
                 num_maskcoeffs,
                 box_to_seg_weight=2.0,
                 box_to_seg_bias=-1.0,
                 bg_segscore=0.0,
                 freeze_detector_condition=(lambda frame_idx: frame_idx != 0),
                 freeze_batchnorm=False,
                 track_lbscore_mode='mean',
                 always_visualize=False,
                 visualization_box_text=False,
                 detector_type='yolact',
                 backbone_droprate=0,
                 debug_mode=False):
        super().__init__()
        if not hasattr(backbone, 'with_state'):
            self.backbone = Wrap(backbone)
        else:
            self.backbone = backbone
        self.detector                       = detector
        self.appnet                         = appnet
        self.detection_descriptor_extractor = detection_descriptor_extractor
        self.track_detection_matcher        = track_detection_matcher
        self.track_descriptor               = track_descriptor
        self.hard_assigner                  = hard_assigner
        self.track_module                   = track_module
        self.raw_det_threshold = raw_det_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.box_to_seg_weight = box_to_seg_weight
        self.box_to_seg_bias   = box_to_seg_bias
        self.bg_segscore       = bg_segscore
        self.max_num_detections = max_num_detections
        self.max_num_tracks     = max_num_tracks
        self.num_classes       = num_classes
        self.num_maskcoeffs    = num_maskcoeffs
        self.freeze_detector_condition = freeze_detector_condition
        self.freeze_batchnorm          = freeze_batchnorm
        self.track_lbscore_mode = track_lbscore_mode
        self.always_visualize = always_visualize
        self.visualization_box_text = visualization_box_text
        self.detector_type = detector_type
        self.backbone_droprate = backbone_droprate
        self.debug_mode       = debug_mode

    def train(self, mode=True):
        returnval = super().train(mode)
        if self.freeze_batchnorm:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train(False)
        return returnval

    def pad_input(self, x, given_segs, given_bbs, divisibility):
        required_padding = get_required_padding(*x.size()[-2:], divisibility)
        if required_padding != [0,0,0,0]:
            assert not self.training, "Input required padding while we are in training mode"
            x, given_segs, given_bbs = apply_padding(x, given_segs, given_bbs, required_padding)
        return x, given_segs, given_bbs, required_padding

    def _get_confident_detection_ids(self, raw_detections):
        if self.detector_type in ('sipmask', 'yangmaskrcnn'):
            return raw_detections['confident_ids']
        elif self.detector_type == 'sipmask_add_extra_filter':
            confscores = [conf[:,1:].max(dim=1)[0] if conf.size(0) > 0 else conf[:, 0]
                          for conf in raw_detections['conf']] # Är detta korrekt?
            B = len(confscores)
            device = confscores[0].device
        elif self.detector_type == 'sipmask_add_extra_filter_with_sigmoid':
            confscores = [F.logsigmoid(conf[:,1:]).max(dim=1)[0] if conf.size(0) > 0 else conf[:, 0]]
            B = len(confscores)
            device = confscores[0].device
        elif self.detector_type == 'yolact':
            conf_in = raw_detections['conf']
            confscores = conf_in[:,:,1:].max(dim=2)[0] - torch.logsumexp(conf_in, dim=2) # (B,N) tensor
            B, N = confscores.size()
            device = confscores.device

        with torch.no_grad():
            confident_ids = [(confscores[b] > self.raw_det_threshold).nonzero().view(-1) for b in range(B)] # B tensors of size (N_b,)
            confident_ids = [confident_ids[b][nms.cc_fast_nms(
                confscores[b][confident_ids[b]], raw_detections['boxes'][b][confident_ids[b]],
                iou_threshold = self.nms_iou_threshold)]
                for b in range(B)] # put in function, note that these are sorted in order of conf
            confident_ids = [confident_ids[b][:self.max_num_detections] for b in range(B)]
        return confident_ids
    
    def filter_detections(self, raw_detections):
        """Filter the raw detections from YOLACT, producing what we call potential detections.
        Aim is to get lower than 32 potential detections.
        """
        conf_in = raw_detections['conf']
        boxes_in = raw_detections['boxes']

        confident_ids = self._get_confident_detection_ids(raw_detections)
        B = len(confident_ids)
        device = conf_in[0].device
            
        active = torch.full((B, self.max_num_detections), fill_value=0, device=device, dtype=torch.int64)
        for b in range(B):
            active[b, 0 : len(confident_ids[b])] = 1

        boxes = [boxes_in[b][confident_ids[b]].detach() for b in range(B)]
        boxes = pad_and_stack(boxes, pad_val=-1, size=(B, self.max_num_detections, 4),
                              device=device, dtype=torch.float32)

        if self.detector_type == 'yolact':
            maskcoeffs_in = raw_detections['maskcoeffs']
            protosegmaps_in = raw_detections['protosegmaps']
            _, H4, W4, _ = protosegmaps_in.size()
            H1, W1 = 4*H4, 4*W4
            segscores = [maskcoeffs_in[b, confident_ids[b]].detach() for b in range(B)]
            segscores = [torch.einsum('nd,hwd->nhw' , segscores[b], protosegmaps_in[b].detach())
                         if segscores[b].size(0) > 0
                         else torch.empty((0, H4, W4), device=device)
                         for b in range(B)]
            raw_segscores = pad_and_stack(segscores, pad_val=float('-inf'),
                                          size=(B, self.max_num_detections, H4, W4),
                                          device=device, dtype=torch.float32)
            box_segscores = box_to_seg(boxes, (H4, W4), (H1, W1), device)
            segscores = raw_segscores + self.box_to_seg_weight * box_segscores
        elif self.detector_type in ('sipmask', 'sipmask_add_extra_filter'):
            _, H4, W4 = raw_detections['segscores'][b].size()
            H1, W1 = 4*H4, 4*W4
            segscores = [raw_detections['segscores'][b][confident_ids[b]] for b in range(B)]
            raw_segscores = pad_and_stack(segscores, pad_val=float('-inf'),
                                          size=(B, self.max_num_detections, H4, W4),
                                          device=device, dtype=torch.float32)
            box_segscores = box_to_seg(boxes, (H4, W4), (H1, W1), device)
            segscores = raw_segscores + self.box_to_seg_weight # Sipmask has already cropped
        elif self.detector_type in ('yangmaskrcnn',):
            _, H4, W4 = raw_detections['segscores'][0].size()
            H1, W1 = 4*H4, 4*W4
            segscores = [raw_detections['segscores'][b][confident_ids[b]] for b in range(B)]
            #print([elem.size() for elem in segscores])
            #print(H1, W1, H4, W4)
            #raise ValueError()
            raw_segscores = pad_and_stack(segscores, pad_val=float('-inf'),
                                          size=(B, self.max_num_detections, H4, W4),
                                          device=device, dtype=torch.float32)
            box_segscores = box_to_seg(boxes, (H4, W4), (H1, W1), device)
            segscores = raw_segscores + self.box_to_seg_weight # MaskRCNN has already cropped
            #for b in range(B):
            #    for n in range(5):
            #        print_tensor_statistics(segscores[b][n], f"seg {b} {n}")
            #raise ValueError()

        lbscores = [conf_in[b][confident_ids[b]].detach() for b in range(B)]
        lbscores = pad_and_stack(lbscores, pad_val=-100, size=(B, self.max_num_detections, self.num_classes),
                                 device=device, dtype=torch.float32)

        return {'active': active, 'boxes': boxes, 'segscores': segscores, 'lbscores': lbscores,
                'raw_segscores': raw_segscores, 'box_segscores': box_segscores}


    def get_track_initializers(self, track_initializers, initialize_mask, assignment_ids, l):
        """
        Args:
            track_initializers (LongTensor): Output of this function from last frame
            initialize_mask (BoolTensor): Size (B, M) true for tracks that were initialized this frame
            assignment_ids (LongTensor): Size (B, M) indicating for which track, what detection idx fits it best
            l (int): frame idx
        Returns:
            LongTensor: Size (B, M, 2) containing identifiers for the detections that initialized
                each track in memory.
        """
        B, M = initialize_mask.size()
        device = initialize_mask.device
        if track_initializers is None:
            track_initializers = torch.zeros((B, M, 2), dtype=torch.int64, device=initialize_mask.device)
            track_initializers[:, :, 1] = -1
        else:
            track_initializers = track_initializers.clone()
        new_initializers = torch.stack([torch.full_like(assignment_ids, l), assignment_ids], dim=2)
        new_initializers = new_initializers[initialize_mask]
        track_initializers[initialize_mask] = new_initializers
        return track_initializers


    def _pack_for_assignment_loss(self, track_active, trackdet_scores, novel_track_scores, track_initializers):
        B, M, N = trackdet_scores.size()
        device = trackdet_scores.device
        track_active = torch.cat([
            torch.full((B, 1), 3, dtype=torch.long, device=device),
            track_active,
            torch.full((B, 1), 4, dtype=torch.long, device=device),
        ], dim=1)
        assignment_scores = torch.cat([
            torch.full((B, M+2, 1), float('-inf'), device=device),
            torch.cat([
                torch.full((B, 1, N), float('-inf'), device=device),
                trackdet_scores,
                novel_track_scores.view(B, 1, N)
            ], dim=1)
        ], dim=2)
        track_initializers[:, :, 1] = track_initializers[:, :, 1] + 1
#        track_initializers = track_initializers + 1
        track_initializers = torch.cat([
            torch.zeros((B, 1, 2), dtype=torch.long, device=device),
            track_initializers,
            torch.zeros((B, 1, 2), dtype=torch.long, device=device),
        ], dim=1) # Ugly fix
        return track_active, assignment_scores, track_initializers

    def _pack_detections(self, detections):
        B, N, C = detections['lbscores'].size()
        device = detections['lbscores'].device
        _, _, H, W = detections['segscores'].size()
        detections = {
            'active': torch.cat([
                torch.full((B, 1), 3, dtype=torch.long, device=device),
                detections['active'],
            ], dim=1),
            'lbscores': torch.cat([
                torch.zeros((B, 1, C), dtype=torch.float, device=device),
                detections['lbscores'],
            ], dim=1),
            'boxes': torch.cat([
                4 * torch.tensor([0, 0, W, H], dtype=torch.float, device=device).view(1, 1, 4).expand(B, 1, -1),
                detections['boxes'],
            ], dim=1),
            'segscores': torch.cat([
                torch.full((B, 1, H, W), self.box_to_seg_weight, dtype=torch.float, device=device),
                detections['segscores'],
            ], dim=1),
        }
        return detections

    def _pack_video_instance_output(self, video_instance_active, video_instance_boxes, video_instance_segscores,
                                    video_instance_lbscores, aux_vis_lbscores):
        B, L, M, H, W = video_instance_segscores.size()
        device = video_instance_segscores.device
        _, _, C = video_instance_lbscores.size()
        return (
            torch.cat([
                torch.full((B, L, 1), 3, dtype=torch.long, device=device),
                video_instance_active,
                torch.full((B, L, 1), 4, dtype=torch.long, device=device),
            ], dim=2),
            torch.cat([
                4 * torch.tensor([0,0,H,W], dtype=torch.float, device=device).view(1, 1, 1, 4).expand(B, L, -1, -1),
                video_instance_boxes,
                torch.full((B, L, 1, 4), -1.0, dtype=torch.float, device=device),
            ], dim=2),
            torch.cat([
                torch.full((B, L, 1, H, W), 10.0, dtype=torch.float, device=device),
                video_instance_segscores,
                torch.full((B, L, 1, H, W), -1e5, dtype=torch.float, device=device),
            ], dim=2),
            torch.cat([
                torch.zeros((B, 1, C), dtype=torch.float, device=device),
                video_instance_lbscores,
                torch.zeros((B, 1, C), dtype=torch.float, device=device),
            ], dim=1),
            torch.cat([
                torch.zeros((B, L, 1, C), dtype=torch.float, device=device),
                aux_vis_lbscores,
                torch.zeros((B, L, 1, C), dtype=torch.float, device=device),
            ], dim=2),
        )
    
    
    def forward(self, images, *args, state=None, epoch=None, visualize=False, **kwargs):
        """Run YOLACT. Output YOLACT auxillary stuffs needed for its loss function. Make hard decision on
        probably detections, and initialize tracks for those.
        Args:
            images (FloatTensor (B,L,D,H,W))
        Returns (dict) {
#            'vis_confscores': FloatTensor (B,L,Nmax,2) of confidences
            'vis_confscores': FloatTensor (B,L,Nmax,C) of confidences
            'vis_boxes'     : FloatTensor (B,L,Nmax,4) of detected and tracked object boxes
#            'vis_lbscores'  : FloatTensor (B,L,Nmax,C) of class/category scores for the objects
            'vis_isscores'  : FloatTensor (B,L,Nmax,H4,W4)
            'vis_final'     : LongTensor (B,L,H,W) final segmentation
            'vis_active'    : LongTensor (B,L,Nmax) of 0 (inactive), 1 (active), 2 (disappeared), 3 (background)
#            'is_confscores': FloatTensor (B,L,Nmax,2) of confidence for a potential detection
            'is_confscores': FloatTensor (B,L,Nmax,C) of confidence for a potential detection
            'is_boxes'     : FloatTensor (B,L,Nmax,4) of detected objects, and each should belong to a track
#            'is_lbscores'  : FloatTensor (B,L,Nmax,C) of class/category scores for detections
            'is_isscores'  : FloatTensor (B,L,Nmax,H4,W4)
            'is_final'     : LongTensor (B,L,H,W) final segmentation
            'is_active'    : LongTensor (B,L,Nmax) of 0 (inactive), 1 (active), 2 (disappeared)
            'yolact_conf'       : FloatTensor (B,L,Nmax,C) of confidences
            'yolact_boxes'      : FloatTensor (B,L,Nmax,4) with boxes, converted to our form from yolact form
            'yolact_maskcoeff'  : FloatTensor (B,L,Nmax,Dmask)
            'yolact_proto'      : FloatTensor (B,L,H4,W4,Dmask)
            'yolact_priorsboxes': FloatTensor (Nmax,4) with boxes, converted to our form from yolact form
            'yolact_ss'         : FloatTensor (B,L,C,H,W)
            'assign_scores'     : FloatTensor (B, L*L*Nmax*Nmax) score between each pair of detections
        }
        """
        H0, W0 = images.size()[-2:]
        if (H0, W0) != (480, 864):
            assert not self.training, f"We always expect 480x864 images during training, got {images.size()}"
            images = resize_spatial_tensor(images, (480, 864), 'bilinear')
        #if (H0, W0) != (384, 640):
        #    assert not self.training, f"We always expect 384x640 images during training, got {images.size()}"
        #    images = resize_spatial_tensor(images, (384, 640), 'bilinear')

        # We may use annotations during training
        if self.training: # or True: # @todo DEBUGGING:
            annos = {key: kwargs[key] for key in ('odannos', 'lbannos', 'isannos', 'active')}
        else:
            annos = None

        GLOBALS['num iter'] = GLOBALS['num iter'] + 1
        
        # Perhaps add padding and rescaling
        B, L, _, H1, W1 = images.size()
        H16, W16 = int(H1 / 16 + 0.5), int(W1 / 16 + 0.5)
        H4, W4 = int(H1 / 4 + 0.5), int(W1 / 4 + 0.5)
        C = self.num_classes
        device = images.device

        active_object_lst = []
        obj_boxes_lst = []
        isscore_lst = []
        label_lst = []

        # Final output
        track_lst = []
        detection_lst = []

        # For the Assignment loss
        track_active_lst = []
        track_detection_assignment_lst = []
        track_initializer_lst = []

        all_yolact_out = {key: [] for key in ('yolact_loc', 'yolact_boxes', 'yolact_conf', 'yolact_maskcoeffs',
                                              'yolact_protosegmaps', 'yolact_semantic_segmap', 'yolact_priorboxes')}

        if state is None:
            state = {}
            GLOBALS['frame idx'] = -1
        for l in range(L):
            GLOBALS['frame idx'] += 1

            # Run YOLACT
            if self.training and self.freeze_detector_condition(l):
                self.train(False)
                with torch.no_grad():
                    feats, state['backbone'] = self.backbone(images[:, l], state.get('backbone'))
                    if self.backbone_droprate > 0 and self.training:
                        feats = {key: F.dropout2d(val, self.backbone_droprate) for key, val in feats.items()}
                    yolact_out = self.detector(feats, image_shape=(H1, W1))
                self.train(True)
            else:
                feats, state['backbone'] = self.backbone(images[:, l], state.get('backbone'))
                if self.backbone_droprate > 0 and self.training:
                    feats = {key: F.dropout2d(val, self.backbone_droprate) for key, val in feats.items()}
                yolact_out = self.detector(feats, image_shape=(H1, W1))
                
            # Store yolact output for loss calculation
            if self.detector_type == 'yolact':
                for key, val in yolact_out.items():
                    all_yolact_out[key].append(val)
                    raw_detections = {
                        'conf'        : yolact_out['yolact_conf'],
                        'boxes'       : yolact_out['yolact_boxes'],
                        'protosegmaps': yolact_out['yolact_protosegmaps'],
                        'maskcoeffs'  : yolact_out['yolact_maskcoeffs'],
                    }
            elif self.detector_type in ('sipmask', 'sipmask_add_extra_filter', 'yangmaskrcnn'):
                raw_detections = yolact_out

            # Threshold and NMS
            detections = self.filter_detections(raw_detections)
            detection_lst.append(self._pack_detections(detections))

            appearance_feats = self.appnet(feats)
            detections, new_background = self.detection_descriptor_extractor(appearance_feats, detections, (H0, W0))

            if state.get('tracks') is None:
                sizes = (B, self.max_num_tracks, self.num_classes, H4, W4)
                state['tracks'], state['background'] = self.track_descriptor.get_init_state(sizes, device, new_background)

            out = self.track_detection_matcher(
                state['tracks'],
                detections,
                state['background'],
                state.get('track_detection_matcher'),
            )
            trackdet_scores                  = out[0]
            novel_track_scores               = out[1]
            state['track_detection_matcher'] = out[2]
            matcher_embedding                = out[3]

            track_active_lst.append(state['tracks']['active']) # Used for ass loss, we need it before we update it
            out = self.hard_assigner(trackdet_scores, novel_track_scores, state['tracks']['active'])
            assignment_ids           = out[0]
            initialize_mask          = out[1]
            state['tracks']['active'] = out[2]

            state['tracks'], state['background'] = self.track_descriptor(
                state['tracks'],
                detections,
                state['background'],
                new_background,
                matcher_embedding,
                assignment_ids,
                initialize_mask)
            track, state['tracks'] = self.track_module(
                state['tracks'],
                detections,
                assignment_ids,
                appearance_feats,
                state['background'])
   
            state['track_detection_matcher'] = self.track_detection_matcher.update_state(
                state['track_detection_matcher'],
                assignment_ids,
                initialize_mask)

            track_lst.append(track)
            track_detection_assignment_lst.append((trackdet_scores, novel_track_scores))
            state['track_initializers'] = self.get_track_initializers(state.get('track_initializers'),
                                                                     initialize_mask,
                                                                     assignment_ids,
                                                                     l) # @todo

            # We add the track initializers, but not for those initialized in this frame. The
            # assignment loss wants to compare new detections to detections in previous frames,
            # and hence must discard the newly initialized tracks.
            track_initializer_lst.append(state['track_initializers'].where(~initialize_mask.view(B, -1, 1).expand(-1, -1, 2), torch.tensor([0, -1], dtype=torch.long, device=device).view(1, 1, 2).expand(B, self.max_num_tracks, -1)))
            
        # Put together YOLACT output
        if self.detector_type == 'yolact':
            for key in ['yolact_boxes', 'yolact_conf', 'yolact_maskcoeffs',
                        'yolact_protosegmaps', 'yolact_semantic_segmap', 'yolact_loc']:
                all_yolact_out[key] = torch.stack(all_yolact_out[key], dim=1)
            all_yolact_out['yolact_priorboxes'] = all_yolact_out['yolact_priorboxes'][0]
        else:
            all_yolact_out = {key: None for key in ['yolact_boxes', 'yolact_conf', 'yolact_maskcoeffs',
                                                    'yolact_protosegmaps', 'yolact_semantic_segmap',
                                                    'yolact_loc', 'yolact_priorboxes']}

        # Put together VIS output
        video_instance_active = torch.stack([track['active'] for track in track_lst], dim=1)
        video_instance_boxes = torch.stack([track['boxes'] for track in track_lst], dim=1)
        video_instance_segscores = torch.stack([track['segscores'] for track in track_lst], dim=1)
        if self.track_lbscore_mode == 'mean':
            video_instance_lbscores = torch.stack([track['lbscores'] for track in track_lst], dim=1).mean(dim=1)
        else:
            video_instance_lbscores = track['lbscores'] # (B,M,C)
        aux_vis_lbscores = torch.stack([track['lbscores'] for track in track_lst], dim=1) # (B,L,M,C)
        out = self._pack_video_instance_output(video_instance_active,
                                               video_instance_boxes,
                                               video_instance_segscores,
                                               video_instance_lbscores,
                                               aux_vis_lbscores)
        video_instance_active = out[0]
        video_instance_boxes = out[1]
        video_instance_segscores = out[2]
        video_instance_lbscores = out[3]
        aux_vis_lbscores = out[4]

        # Put together Detection output
        detection_active = torch.stack([det['active'] for det in detection_lst], dim=1)
        detection_boxes = torch.stack([det['boxes'] for det in detection_lst], dim=1)
        detection_segscores = torch.stack([det['segscores'] for det in detection_lst], dim=1)
        detection_lbscores = torch.stack([det['lbscores'] for det in detection_lst], dim=1)

        # For the assignment loss
        for l in range(L):
            out = self._pack_for_assignment_loss(track_active_lst[l],
                                                 *track_detection_assignment_lst[l],
                                                 track_initializer_lst[l])
            track_active_lst[l] = out[0]
            track_detection_assignment_lst[l] = out[1]
            track_initializer_lst[l] = out[2]
        track_active = torch.stack(track_active_lst, dim=1)
        track_detection_assignments = torch.stack(track_detection_assignment_lst, dim=1)
        track_initializers = torch.stack(track_initializer_lst, dim=1)

        # Resize what should be resized
        if (H0, W0) != (H1, W1):
            assert not self.training
            video_instance_boxes = resize_boxes(video_instance_boxes, (H1, W1), (H0, W0))
            video_instance_segscores = resize_spatial_tensor(video_instance_segscores, (H0, W0), 'bilinear')
            detection_boxes = resize_boxes(detection_boxes, (H1, W1), (H0, W0))
            detection_segscores = resize_spatial_tensor(detection_segscores, (H0, W0), 'bilinear')
        else:
            video_instance_segscores = resize_spatial_tensor(video_instance_segscores, (H1, W1), 'bilinear')
            detection_segscores = resize_spatial_tensor(detection_segscores, (H1, W1), 'bilinear')
                                                          
        # Put together final segmentations
        with torch.no_grad():
            video_instance_segmentation = video_instance_segscores.argmax(dim=2)
            detection_segmentations = detection_segscores.argmax(dim=2)
            
        to_visualize = {}
        if visualize or self.always_visualize:
            to_visualize['detection'] = {
                'seg': detection_segmentations, # Raw detections
                'boxes': [[detection_boxes[b,l][detection_active[b,l] == 1]
                           for l in range(L)] for b in range(B)],
                'boxlabels': [[(detection_active[b,l] == 1).nonzero().view(-1)
                               for l in range(L)] for b in range(B)]
            }
            to_visualize['track'] = {
                'seg': video_instance_segmentation,
                'boxes': [[video_instance_boxes[b,l][(video_instance_active[b,l] == 1) + (video_instance_active[b,l] == 2)]
                           for l in range(L)] for b in range(B)],
                'boxlabels': [[((video_instance_active[b,l] == 1) + (video_instance_active[b,l] == 2)).nonzero().view(-1)
                               for l in range(L)] for b in range(B)],
            }
            if self.visualization_box_text == 'version 2':
                to_visualize['detection']['boxtexts'] = get_visualization_box_text2(
                    detection_lbscores, detection_active)
                to_visualize['track']['boxtexts'] = get_visualization_box_text2(
                    aux_vis_lbscores, video_instance_active)
            elif self.visualization_box_text:
                to_visualize['detection']['boxtexts'] = get_visualization_box_text(
                    detection_lbscores, detection_active)
                to_visualize['track']['boxtexts'] = get_visualization_box_text(
                    aux_vis_lbscores, video_instance_active)

        output = {
            'yolact_boxes'          : all_yolact_out['yolact_boxes'],
            'yolact_loc'            : all_yolact_out['yolact_loc'],
            'yolact_conf'           : all_yolact_out['yolact_conf'],
            'yolact_maskcoeffs'     : all_yolact_out['yolact_maskcoeffs'],
            'yolact_priorboxes'     : all_yolact_out['yolact_priorboxes'],
            'yolact_protosegmaps'   : all_yolact_out['yolact_protosegmaps'],
            'yolact_semantic_segmap': all_yolact_out['yolact_semantic_segmap'],
            'active_objects': video_instance_active, # This active should be used in final output
            'object_boxes'  : video_instance_boxes,
            'isscore'       : video_instance_segscores,
            'instance_segs' : video_instance_segmentation,
            'lbscores'      : video_instance_lbscores,
            'detection_active'           : detection_active,
            'detection_boxes'            : detection_boxes,
            'detection_lbscores'         : detection_lbscores,
            'detection_segs'             : detection_segmentations,
            'track_active'               : track_active, # Does not include newly found objects, used for loss
            'track_detection_assignments': track_detection_assignments,
            'track_initializers'         : track_initializers,
            'aux_vis_lbscores'           : aux_vis_lbscores,
            'to_visualize': to_visualize,
        }
        
        if self.debug_mode:
            raise ValueError("Quitting after complete forward as we are in debug mode")
        
        return output, state

