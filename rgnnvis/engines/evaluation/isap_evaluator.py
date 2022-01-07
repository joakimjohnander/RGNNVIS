from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from rgnnvis.utils.tensor_utils import index_map_to_one_hot
from rgnnvis.utils.tensor_utils import batch_many_to_many_seg_iou

import os, sys


class ISAPEvaluatorV1(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_sequence_results(pred, anno, iou_thresholds, spatial_split_size=100000, use_old_TP_code=False):
        """
        Args:
            pred
                                 (Dict): {'detection_segs':    (FloatTensor): (L,H,W), values in [0 Nmax-1]
                                          'detection_lbscores': (FloatTensor): (L,N,C),
                                          'detection_active:     (LongTensor): (L,N) }

            anno                 (Dict): { 'isannos': (FloatTensor): (L,H,W), values in [0 Mmax-1]
                                          'lbannos':  (LongTensor): (Mmax),
                                           'active':  (ByteTensor): (L,Mmax) }

            iou_thresholds       (List): Typically [.5] or varied as [.50, .55, ..., .95]
        Returns:
            Dict, see code.
        """
        active_pred = pred['detection_active'].squeeze(0) == 1
        conf_pred = pred['detection_lbscores'].squeeze(0)
        seg_pred = pred['detection_segs'].squeeze(0)

        # Make conf zero and boxes -1 for non active predictions
        l, n, c = conf_pred.shape
        device = conf_pred.device

        # conf score and active
        conf_pred = F.softmax(conf_pred, dim=2)
        conf_pred = conf_pred.where(active_pred.view(l, n, 1), torch.tensor(0., device=device)) # set value when condition is false

        # Get max conf out of num_classes
        scores_pred = torch.max(conf_pred[:, :, 1:], dim=2)[0]  # Remove background from class conf dim
        labels_pred = torch.argmax(conf_pred, dim=2).where(active_pred, torch.tensor(0, device=device))
        sorted_idx = torch.argsort(scores_pred, dim=-1, descending=True)
        scores_pred = scores_pred.gather(1, sorted_idx)
        labels_pred = labels_pred.gather(1, sorted_idx)
        
        # Annotations (ground truths)
        _, h, w = anno['isannos'].shape
        _, m = anno['active'].shape
        seg_anno = anno['isannos']
        lb = anno['lbannos'].expand(l, m)
        labels_anno = ((anno['active'] == 1) * lb)
        num_objects = torch.bincount(labels_anno.flatten(), minlength=c)
        # For the broadcast to give false on category 0 which is not a category, A = {1,...,C}.
        labels_anno[labels_anno == 0] = 255
        label_mask = labels_pred.view(l, -1, 1) == labels_anno.view(l, 1, -1)  # [l,n,m]

        # pred to onehot
        pred_onehot_seg = index_map_to_one_hot(seg_pred.unsqueeze(-3),  # size now (L,1,H,W)
                                               torch.tensor(range(n), dtype=torch.uint8),
                                               device)

        # pred masks on active, background will be canceled here since n=0 is background instance and always False
        pred_onehot_seg = pred_onehot_seg.where(active_pred.view(l, n, 1, 1),
                                                torch.tensor(0, dtype=torch.bool, device=device))
                                                # where sets value when condition is false

        # sort pred masks
        pred_onehot_seg_sorted = torch.stack([ps[sorted_idx[i]] for i, ps in enumerate(pred_onehot_seg)], dim=0)

        # anno to onehot
        anno_onehot_seg = index_map_to_one_hot(seg_anno.unsqueeze(-3),  # size now (L,1,H,W)
                                               torch.tensor(range(m), dtype=torch.uint8),
                                               device)

        # pred_onehot_seg is of size (1,L,Nmax,H,W) and anno_onehot_seg is of size (1,L,Mmax,H,W)
        # output size (1,L,Nmax,Mmax)
        iou = batch_many_to_many_seg_iou(pred_onehot_seg_sorted.view(1, l, n, h, w), anno_onehot_seg.view(1, l, m, h, w),
                                         unassigned_iou=0.0, split_size=spatial_split_size)
        iou = iou.view(l, n, m)

        # Threshold masks and stack
        iou_masks = torch.stack([iou > threshold for threshold in iou_thresholds], dim=-1)

        matching = iou_masks & label_mask.view(l, n, m, 1)

        # Count TPs
        if use_old_TP_code:
            is_tp = torch.zeros((l, n, len(iou_thresholds)), dtype=torch.bool, device=seg_pred.device)
            taken = torch.zeros((l, m, len(iou_thresholds)), dtype=torch.bool, device=seg_pred.device)
            for i in range(n):
                if matching[:, i].any():
                    tmp = matching[:, i]
                    tidx = torch.max(iou[:, i].view(*iou[:, i].shape, 1)*tmp, dim=1, keepdim=True)[1]
                    is_tp[:, i, :] = (~taken & tmp).any(dim=1)  # This can be fishy! Weakly validated!
                    taken.scatter_(1, tidx, 1)
        else:
            # Set overlap to 0 if the classes don't match
            iou = iou * label_mask.float()

            T = len(iou_thresholds)
            iou = iou.view(l, n, m, 1).repeat(1, 1, 1, T) # (l, n, m, num_iou_thresh)
            iou_thresholds = torch.tensor(iou_thresholds, device=iou.device).view(1, -1).expand(l, -1)
            is_tp = torch.zeros((l, n, T), dtype=torch.bool, device=iou.device)
            for i in range(n):
                best_anno_iou, best_anno_idx = iou[:, i].max(dim=1) # (l, num_iou_thresholds)
                is_tp[:,i] = (best_anno_iou >= iou_thresholds)
                best_anno_mask = torch.scatter(
                    torch.zeros((l, m, T), device=iou.device, dtype=torch.bool),
                    dim = 1,
                    index = best_anno_idx.view(l, 1, T),
                    value = True
                ).view(l, 1, m, T).expand(-1, n, -1, -1)
                claimed_anno_mask = best_anno_mask * is_tp[:, i].view(l, 1, 1, T)
                iou[claimed_anno_mask] = 0.0

        return {'tp': is_tp, 'scores': scores_pred, 'labels': labels_pred, 'num_objects': num_objects}

    @staticmethod
    def calculate_AP(results, iou_thresholds, num_classes):
        perclass_num_objects = [res['num_objects'] for res in results]
        perclass_num_objects = torch.stack(perclass_num_objects, dim=0)
        perclass_num_objects = torch.sum(perclass_num_objects, dim=0)

        perclass_results = []
        for c in range(1, num_classes):
            tps = []
            scores = []
            for res in results:
                lmask = res['labels'] == c
                tp = res['tp'][lmask]
                sc = res['scores'][lmask]
                if tp.shape[0]:
                    tps.append(tp)
                    scores.append(sc)
                else:
                    tps.append(torch.empty(0).bool())
                    scores.append(torch.empty(0))
            tps = torch.cat(tps, dim=0)
            scores = torch.cat(scores)
            perclass_results.append((tps, scores))

        # Calculate AP for a single threshold and a single class
        apr = {int(t*100 + 0.5): torch.zeros(num_classes-1, dtype=torch.float32) for t in iou_thresholds}
        iou_thresholds = list(apr.keys())

        for c in range(1, num_classes):
            tp, scores = perclass_results[c-1]
            if tp.shape[0]:
                tpsp = list(torch.split(tp, 1, -1))
                for i, t in enumerate(iou_thresholds):
                    M = perclass_num_objects[c].float().item()
                    if M > 0:
                        tpsp[i] = tpsp[i].view(tpsp[i].shape[0])
                        apr[t][c-1] = ISAPEvaluatorV1.calculate_perclass_AP(M, tpsp[i].float(), scores)

        return apr, perclass_num_objects[1:]

    @staticmethod
    def calculate_perclass_AP(M, istp, scores):
        """Calculate AP for a single threshold and a single class
        Args:
            M: int, number of annotated objects of this class label
            result: (N,3) tensor, with fields (isTP, confidence, class label)
        Returns:
            Average Precision
        """
        sorted_ids = torch.argsort(scores, dim=0, descending=True)
        istp = istp[sorted_ids]
        precisions = torch.cumsum(istp, 0) / torch.arange(1., len(istp) + 1)
        v = 0.
        for idx in range(len(precisions) - 1, -1, -1):
            v = max(v, precisions[idx])
            precisions[idx] = v
        return (torch.sum(precisions*istp)/M).item()
