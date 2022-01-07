
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgnnvis.utils.tensor_utils import batch_many_to_many_box_iou, match_boxes_injective, batch_many_to_many_class_overlap
from rgnnvis.utils.debugging import print_tensor_statistics

from third_party.lovasz_losses import lovasz_softmax, lovasz_softmax_flat

GLOBALS = {'num iter': 0}

    
class ComposeObjective:
    """Used to compose multiple objectives. Added support for scenarios where a single subobjective contains
    multiple objectives (such as yolact_objective which comprises multiple subobjectives, B, M, C, and S)."""
    def __init__(self, objectives):
        self.objectives = objectives
#    def __call__(self, model_output, provides_seganno, provides_bboxanno, segannos, bboxannos, **kwargs): #@todo args
    def get_idfs(self):
        idfs = []
        for name, objective in self.objectives.items():
            if hasattr(objective, 'get_idfs'):
                idfs = idfs + [f'{name} {idf}' for idf in objective.get_idfs()]
            else:
                idfs.append(name)
        return idfs
    def __call__(self, model_output, annos, mode=None):
        GLOBALS['num iter'] += 1
        partial_losses = {}
        for name, objective in self.objectives.items():
            result = objective(model_output, annos, mode)
            if hasattr(objective, 'get_idfs'):
                for subname, subresult in result.items():
                    partial_losses[f'{name} {subname}'] = subresult
            else:
                partial_losses[name] = result
        total_loss = sum([loss['val'] for key, loss in partial_losses.items()])
        return total_loss, partial_losses

    def update(self):
        self.objectives['detector'].update()

class Objective:
    def __init__(self, weight, keys=None, device='cuda'):
        self.weight = weight
        self.keys = keys
        self.device = device


class VISObjective(Objective):
    """Calculates losses for track-detection assignment, novel object assignment, track scoring,
    and track segmentation.
    """
    def __init__(self, detection_iou_threshold, trkdetass, newass, trkcls, trkseg,
                 consider_class_for_overlap=False, **parent_args):
        super().__init__(**parent_args)
        self.detection_iou_threshold = detection_iou_threshold
        self.trkdetass = trkdetass
        self.newass = newass
        self.trkcls = trkcls
        self.trkseg = trkseg
        self.consider_class_for_overlap = consider_class_for_overlap
        self.assignment_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.trkseg_loss = SegLoss(mode=trkseg['mode'])
        self.trkcls_loss = nn.CrossEntropyLoss(reduction='none')
    def get_idfs(self):
        return ['trkdetass', 'newass', 'trkcls', 'trkseg']
    def _assignment_loss(self,
                         all_track_anno_ids,
                         primary_detection_anno_ids,
                         detection_anno_ids,
                         size,
                         all_pred_tracks_active,
                         all_pred_detections_active,
                         assignment):
        """Constructs a (B, L, M, N) tensor that is 1 for a track-detection pair that share the
        same nonzero annotation idx. If they have different ids, or any of them are zero, we want
        the ground-truth to be 0. That is, we want to find POSITIVES!
        """
        B, L, M, N = size
        if self.trkdetass['detection_match_mode'] == 'primary':
            det_matches = (all_track_anno_ids.view(B, L, M, 1) == primary_detection_anno_ids.view(B, L, 1, N))
        elif self.trkdetass['detection_match_mode'] == 'all':
            det_matches = (all_track_anno_ids.view(B, L, M, 1) == detection_anno_ids.view(B, L, 1, N))
            raise ValueError("There are some unsolved issues here...")
        det_matches[:, :, -1, :] = True
        det_matches_ids = torch.min(
            det_matches.long() * torch.arange(-M, 0, device=self.device).long().view(1, 1, M, 1),
            dim=2)[0] + M # (B,L,N)
        det_matches = torch.zeros_like(det_matches).scatter_(
            2,
            det_matches_ids.view(B, L, 1, N),
            torch.ones_like(det_matches))        
        targets = det_matches * ((all_track_anno_ids.view(B, L, M, 1) > 0)         # TP Track-Det-pair
                                 + (all_pred_tracks_active.view(B, L, M, 1) == 4)) # Novel track
        trk_mask = (all_pred_tracks_active == 1) + (all_pred_tracks_active == 2) + (all_pred_tracks_active == 4)
        det_mask = (all_pred_detections_active == 1)
        loss_mask = trk_mask.view(B, L, M, 1) * det_mask.view(B, L, 1, N)

        loss = self.assignment_loss(assignment, targets.float()) # Calculate loss
        loss = loss.where(loss_mask, torch.zeros_like(loss))
        if self.trkdetass['secondary_match_weight'] < 1.0:
            secondary_matches = (
                targets.any(dim=3, keepdim=True) # Only tracks which has a primary match
                * (~targets) # Not the primary match
                * (all_track_anno_ids.view(B, L, M, 1) == detection_anno_ids.view(B, L, 1, N))) # All matches
            secondary_matches[:, :, -1, :] = 0 # Don't reweight novel dets
            old_loss = loss
            loss = (self.trkdetass['secondary_match_weight'] * loss).where(secondary_matches, loss)
#            if 486 < GLOBALS['num iter'] < 500:
#                print(f"{GLOBALS['num iter']} - loss: {loss.sum():.3f}, old loss: {old_loss.sum():.3f}, N primary: {targets.sum()}, N secondary {secondary_matches.sum()}, N total: {loss_mask.sum()}")
#                if GLOBALS['num iter'] == 489:
#                    for l in range(3):
#                        print(primary_detection_anno_ids[0,l,[0,1,2,3,4,5,6]])
#                        print(targets[0,l,[0,1,2,3,4,5,6,-1]][:,[0,1,2,3,4,5,6]].float())
#                        print(secondary_matches[0,l,[0,1,2,3,4,5,6,-1]][:,[0,1,2,3,4,5,6]].long())
#        if GLOBALS['num iter'] == 499:
#            raise ValueError("Debugging")

#        print("\n\nAssignment Losses")
#        for l in range(L):
#            print()
#            print(loss[0,l,[0,1,2,3,4,5,-1]][:,[0,1,2,3,4,5,6,7]])
#            print(targets[0,l,[0,1,2,3,4,5,-1]][:,[0,1,2,3,4,5,6,7]])
#            print(assignment[0,l,[0,1,2,3,4,5,-1]][:,[0,1,2,3,4,5,6,7]])
        
        if self.trkdetass['normalization'] == 'BL':
            trkdetass_normalization = B * L
        elif self.trkdetass['normalization'] == 'BLM':
            trkdetass_normalization = trk_mask.float().sum()
        elif self.trkdetass['normalization'] == 'BLN':
            trkdetass_normalization = det_mask.float().sum()
        elif self.trkdetass['normalization'] == 'BLMN':
            trkdetass_normalization = loss_mask.float().sum()
        else:
            raise ValueError(f"Invalid trkdetass normalization: {self.trkdetass['normalization']}")
        trkdetass_loss = loss[:, :, :-1, :].sum() / (trkdetass_normalization + 1e-5)
        
        newass_loss = loss[:, :, -1]
        if self.newass['normalization'] == 'BL':
            newass_normalization = B * L
        elif self.newass['normalization'] == 'BLN':
            newass_normalization = det_mask.float().sum()
        else:
            raise ValueError(f"Invalid newass normalization: {self.newass['normalization']}")
        newass_loss = loss[:, :, -1, :].sum() / (newass_normalization + 1e-5)

        assert torch.isfinite(trkdetass_loss).all()
        assert torch.isfinite(newass_loss).all()
        return {'trkdetass': {'val': self.trkdetass['weight'] * trkdetass_loss, 'N': trkdetass_normalization},
                'newass': {'val': self.newass['weight'] * newass_loss, 'N': newass_normalization}}

    def _vis_loss(self,
                  size,
                  all_track_anno_ids,
                  all_anno_is,
                  all_anno_labels,
                  all_pred_is,
                  all_pred_tracks_active,
                  all_pred_lbscores):
        B, L, M, N, Nanno = size
        _, _, H, W = all_anno_is.size()

        # We find for each annotation the corresponding track (if one exists)
        # We find for each track the corresponding annotation (if one exists)
        anno_track_ids = torch.zeros((B, Nanno), dtype=torch.int64, device=self.device)
        track_anno_ids = all_track_anno_ids[:, -1, :].clone()
        for b in range(B):
            for m in range(M):
                anno_idx = track_anno_ids[b, m]
                if anno_idx != 0:
                    assert (anno_track_ids[b, anno_idx] == 0).all(), (b, m, anno_track_ids)
                    anno_track_ids[b, track_anno_ids[b, m]] = m
                    track_anno_ids[b, track_anno_ids[b, :] == track_anno_ids[b, m]] = 0

        # We map the annotation ids in the segmentations to track ids (or 0 if no corresponding track exists)
        all_anno_is = anno_track_ids.view(B, 1, Nanno, 1, 1).expand(-1, L, -1, H, W).gather(2, all_anno_is.view(B, L, 1, H, W)) # (B,L,H,W)

        # We gather for each track the corresponding annotation ground truths
        all_anno_labels = all_anno_labels.gather(1, all_track_anno_ids[:, -1, :])

        trkseg_loss = self.trkseg_loss(all_pred_is, all_anno_is)

        active_track_mask = (all_pred_tracks_active == 1) + (all_pred_tracks_active == 2)
        trkcls_loss = self.trkcls_loss(
            all_pred_lbscores.view(B*L*M, -1),
            all_anno_labels.view(B, 1, M).expand(-1, L, -1).reshape(B*L*M)
        ).view(B, L, M)
        trkcls_loss = trkcls_loss.where(active_track_mask, torch.zeros_like(trkcls_loss))
        
        if self.trkcls['normalization'] == 'mask sum':
            trkcls_normalization = active_track_mask[:, -1].float().sum()
            trkcls_loss = trkcls_loss.sum() / (1e-5 + trkcls_normalization)
        elif self.trkcls['normalization'] == 'num frames':
            trkcls_normalization = B * L
            trkcls_loss = trkcls_loss.sum() / (1e-5 + trkcls_normalization)
        elif isinstance(self.trkcls['normalization'], torch.Tensor): # Assume it is a tensor of L weights
            trkcls_normalization = B / (self.trkcls['normalization'][-L:].sum())
            trkcls_loss = (trkcls_loss.sum(dim=(0,2)) * self.trkcls['normalization'][-L:]).sum() / B

        if not (torch.isfinite(trkcls_loss) and torch.isfinite(trkseg_loss)):
            print("VISObjective became non-finite")
            print(all_pred_is[:, :, [0,1,2,3,4,5,6,33]].sum(dim=(3,4)))
            print(all_anno_is[:, :, 0, :, :].sum(dim=(2,3)))
            
        return {'trkcls': {'val': self.trkcls['weight'] * trkcls_loss, 'N': trkcls_normalization},
                'trkseg': {'val': self.trkseg['weight'] * trkseg_loss, 'N': B*L}}

    def _get_detection_anno_ids(self,
                                size,
                                all_pred_detections_active,
                                all_pred_boxes,
                                pred_lbscores,
                                all_anno_active,
                                all_anno_boxes,
                                anno_classes):
        """Assign all detections to an annotation track index. If no anno matches a detection, we set the
        detection anno idx to zero (background).
        Args:
            size                       (tuple)     :
            all_pred_detections_active (LongTensor): Size (B,L,N)
            all_pred_boxes             (Tensor)    : Size (B,L,N,4)
            pred_lbscores              (Tensor)    : Size (B,L,N,C)
            all_anno_active            (LongTensor): Size (B,N)
            all_anno_boxes             (Tensor)    : Size (B,L,N,4)
            anno_classes               (LongTensor): Size (B,N)
        """
        B, L, N, Nanno = size
        iou = batch_many_to_many_box_iou(all_pred_boxes, all_anno_boxes) # (B, L, N, Nanno)
        if self.consider_class_for_overlap:
            iou[:, :, :, 0] = 0.5
            class_overlap = batch_many_to_many_class_overlap(
                pred_lbscores, anno_classes.view(B, 1, N).expand(-1, L, -1))
            iou = iou + class_overlap * (iou > 0.5).float()
        det_mask = (all_pred_detections_active == 1) # (B, L, N)
        anno_mask = (all_anno_active == 1)           # (B, L, Nanno)
        detanno_mask = det_mask.view(B, L, N, 1) * anno_mask.view(B, L, 1, Nanno)
        iou = iou.where(detanno_mask, torch.zeros_like(iou)) # (B, L, N, Nanno)
        assert (iou[:, :, :, 0] == 0).all() # first spot should not contain an actual annotation (active is 3)
        assert (iou[:, :, 0, :] == 0).all() # first spot should not contain an actual detection (active is 3)
        iou[:, :, :, 0] = 0.5
        anno_ids = iou.argmax(dim=3) # (B, L, N), anything not over 0.5 will have match idx 0
        pred_ids = iou.argmax(dim=2) # (B, L, Nanno)
        primary_matches = det_mask * (pred_ids.gather(2, anno_ids)
                                      == torch.arange(N, dtype=torch.float, device=iou.device).view(1, 1, N))
        primary_anno_ids = anno_ids.where(primary_matches, torch.zeros_like(anno_ids))
        return primary_anno_ids, anno_ids

    def __call__(self, model_output, annos, *args, **kwargs):
        """
        Args:
        """
        B, L, N, _ = model_output[self.keys['pred_boxes']].size()
        _, _, M = model_output[self.keys['pred_track_active']].size()
        all_pred_boxes = model_output[self.keys['pred_boxes']]                 # (B,L,N,4)
        all_pred_tracks_active = model_output[self.keys['pred_track_active']]  # (B,L,N)
        all_track_initializers = model_output[self.keys['track_initializers']] # (B,L,N,2) tensor, where the last dimension contains the (l,n) tuple that identifies what detection gave rise to this track. Note that we might remove low probability tracks at the end of each time-step, and a track might therefore change its index over time.
        all_pred_detections_active = model_output[self.keys['pred_detection_active']] # (B,L,N)

        all_anno_boxes = annos[self.keys['anno_boxes']]   # (B,L,N,4)
        all_anno_active = annos[self.keys['anno_active']] # (B,L,N)
        _, _, Nanno = all_anno_active.size()

        primary_detection_anno_ids, detection_anno_ids = self._get_detection_anno_ids(
            (B, L, N, Nanno),
            all_pred_detections_active,
            all_pred_boxes,
            model_output['detection_lbscores'], # @todo not consistent, but I hate my idea with keys ...
            all_anno_active,
            all_anno_boxes,
            annos[self.keys['anno_labels']],
        )

        # Next, we construct the (B,L,M) tensor containing the anno idx of each track. Inactive
        # tracks and tracks at index 0 or -1 have initializers (0, 0). They will hence not match
        # any object annotation and get anno_idx = 0.
        all_track_anno_ids = torch.stack([
            torch.stack([
                primary_detection_anno_ids[b, all_track_initializers[b,l,:,0], all_track_initializers[b,l,:,1]]
                for l in range(L)])
            for b in range(B)])
        assert (all_track_anno_ids[:, :, 0] == 0).all()
        assert (all_track_anno_ids[:, :, -1] == 0).all()

        assignment_loss = self._assignment_loss(
            all_track_anno_ids         = all_track_anno_ids,
            primary_detection_anno_ids = primary_detection_anno_ids,
            detection_anno_ids         = detection_anno_ids,
            size                       = (B, L, M, N),
            all_pred_tracks_active     = all_pred_tracks_active,
            all_pred_detections_active = all_pred_detections_active,
            assignment                 = model_output[self.keys['pred_track_detection_assignments']])

        vis_loss = self._vis_loss(
            size                   = (B, L, M, N, Nanno),
            all_track_anno_ids     = all_track_anno_ids,
            all_anno_is            = annos[self.keys['anno_instance_segmentation']], # (B,L,H,W) with anno ids
            all_anno_labels        = annos[self.keys['anno_labels']], # (B,N)
            all_pred_is            = model_output[self.keys['pred_instance_segscores']], # (B,L,N,H,W)
            all_pred_tracks_active = all_pred_tracks_active,
            all_pred_lbscores      = model_output[self.keys['aux_vis_lbscores']]) # (B,L,M,C)

        loss = {}
        loss.update(assignment_loss)
        loss.update(vis_loss)
        GLOBALS['num iter'] = GLOBALS['num iter'] + 1
        return loss

    
class SegLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
    def forward(self, predicted_scores, targets):
        B, L, C, H, W = predicted_scores.size()
        if self.mode == 'CE':
            return F.cross_entropy(predicted_scores.view(B*L, C, H, W), targets.view(B*L, H, W), ignore_index=255)
        if self.mode == 'Lovasz per image':
            predictions = F.softmax(predicted_scores, dim=2)
            predictions = predictions.permute(0, 1, 3, 4, 2).contiguous().view(B, L, H*W, C)
            targets = targets.view(B, L, H*W)
            masks = targets != 255
            losses = [lovasz_softmax_flat(predictions[b, l, masks[b, l]], targets[b, l, masks[b, l]])
                      for b in range(B) for l in range(L)]
            return torch.stack(losses).mean()
        if self.mode == 'Lovasz per video':
            predictions = F.softmax(predicted_scores, dim=2)
            predictions = predictions.permute(0, 1, 3, 4, 2).contiguous().view(B, L*H*W, C)
            targets = targets.view(B, L*H*W)
            masks = targets != 255
            losses = [lovasz_softmax_flat(predictions[b, masks[b]], targets[b, masks[b]])
                      for b in range(B)]
            return torch.stack(losses).mean()
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented")

        
