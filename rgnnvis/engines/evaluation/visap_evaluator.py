from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from rgnnvis.utils.tensor_utils import index_map_to_one_hot
from rgnnvis.utils.tensor_utils import batch_many_to_many_seg_iou

import os, sys


def vis_many_to_many_intersection_union(pred, anno, split_size=100000):
    """ Computes intersection over union, many to many. There are two ways to handle
        cuda out of memory. 1) Change split_size here or 2) the split_size regulating how
        many frames of the sequence that are processed in the evaluator (usually def. in
        the run file).
    Args:
        pred              (ByteTensor): one-hot tensor of size (B,L,Nmax,H,W)
        anno              (ByteTensor): one-hot tensor of size (B,L,Mmax,H,W)
        split_size               (Int): split spatial dim. into chunks
    Returns:
        intersection     (FloatTensor): (B,Nmax,Mmax)
        union            (FloatTensor): (B,Nmax,Mmax)
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

    return ntp, nunion

class VISAPEvaluatorV1(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_sequence_results(pred, anno, spatial_split_size=100000):
        """
        Args:
            pred
                                 (Dict): {'instance_segs':    (FloatTensor): (B,H,W), values in [0 Nmax-1]
                                          'lbscores':         (FloatTensor): (B,N,C),
                                          'active_objects:    (LongTensor): (B,N) }

            anno                 (Dict): { 'isannos': (FloatTensor): (L,H,W), values in [0 Mmax-1]
                                          'lbannos':  (LongTensor): (Mmax),
                                           'active':  (ByteTensor): (L,Mmax) }
        Returns:
            Dict, see code.
        """
        active_pred = (pred['active_objects'].squeeze(0) == 1) + (pred['active_objects'].squeeze(0) == 2) # (L, N)
        conf_pred = pred['lbscores'].squeeze(0)
        seg_pred = pred['instance_segs'].squeeze(0)

        l, h, w = seg_pred.shape
        n, c = conf_pred.shape
        device = conf_pred.device

        # num class tracks, is the same for each sequence (lazy)
        #num_objects = torch.bincount(anno['lbannos'], minlength=c)

        # conf score
        conf_pred = F.softmax(conf_pred, dim=-1)
        conf_pred = torch.where(
            (~torch.isnan(conf_pred)) * active_pred.view(l, n, 1).any(dim=0),
            conf_pred,
            torch.tensor([1.0] + 40 * [0.], device=device)
        ) # Set confidence to zero if we have nan anywhere. Can happen where active is 0.

        # Get max conf out of num_classes
        scores_pred = torch.max(conf_pred[:, 1:], dim=1)[0]  # Remove background from class conf dim

        # When we predict the maximum class score and idx we exclude background
#        labels_pred = torch.argmax(conf_pred, dim=1) * active_pred.any(dim=0)
        labels_pred = (torch.argmax(conf_pred[:, 1:], dim=1) + 1) * active_pred.any(dim=0)

        # TODO accumulate active_pred and mask in video_eval?

        #sorted_idx = torch.argsort(scores_pred, dim=0, descending=True)
        #scores_pred = scores_pred.gather(0, sorted_idx)
        #labels_pred = labels_pred.gather(0, sorted_idx)
        
        # Annotations (ground truths)
        _, h, w = anno['isannos'].shape
        _, m = anno['active'].shape
        active_anno = anno['active'] == 1
        labels_anno = anno['lbannos'] * active_anno.any(dim=0)

        # For the broadcast to give false on category 0 which is not a category, A = {1,...,C}.
        #labels_anno[labels_anno == 0] = 255
        #labels_mask = labels_pred.view(-1, 1) == labels_anno.view(1, -1)

        # pred to onehot
        pred_onehot_seg = index_map_to_one_hot(seg_pred.unsqueeze(-3),  # size now (L,1,H,W)
                                               torch.tensor(range(n), dtype=torch.uint8),
                                               device)

        # pred masks on active, background will be canceled here since n=0 is background instance and always False
        pred_onehot_seg = pred_onehot_seg.where(active_pred.view(l, n, 1, 1),
                                                torch.tensor(0, dtype=torch.bool, device=device))
                                                # where sets value when condition is false

        # sort pred masks
        #pred_onehot_seg = pred_onehot_seg[:, sorted_idx]

        # anno to onehot
        seg_anno = anno['isannos']
        anno_onehot_seg = index_map_to_one_hot(seg_anno.unsqueeze(-3),  # size now (L,1,H,W)
                                               torch.tensor(range(m), dtype=torch.uint8),
                                               device)

        # pred_onehot_seg is of size (1,L,Nmax,H,W) and anno_onehot_seg is of size (1,L,Mmax,H,W)
        # output size (1,L,Nmax,Mmax)
        intersection, union = vis_many_to_many_intersection_union(pred_onehot_seg.view(1, l, n, h, w),
                                                                  anno_onehot_seg.view(1, l, m, h, w),
                                                                  split_size=spatial_split_size)
        intersection = intersection.view(l, n, m)
        union = union.view(l, n, m)

        # DEBUG
#        print(f"\nTaking {l} frames")
#        print(f"active at ids    {active_pred[-1].nonzero().view(-1).tolist()}")
#        print(f"anno active ids  {active_anno.any(dim=0)[[0,1,2,3,4,5]].nonzero().view(-1).tolist()}")
#        print(f"IOU = \n{(intersection.sum(0) / union.sum(0))[active_pred[-1]][:, [0, 1, 2, 3, 4, 5]].cpu()}")
#        print(f"scores = {scores_pred[active_pred[-1]].cpu()}")
#        print(f"labels = {labels_pred[active_pred[-1]].cpu()}")

        return {'intersection': intersection, 'union': union, 'scores_pred': scores_pred, 'labels_pred': labels_pred,
                #'active_pred': active_pred,
                'lbannos': anno['lbannos']}

    @staticmethod
    def evaluate_video(video_results, iou_thresholds, num_classes, use_old_TP_code, ignore_class):

        # active pred
        #active_pred = [res['active_pred'] for res in video_results]
        #active_pred = torch.cat(active_pred, dim=0)
        #active_pred_mask = active_pred.any(dim=0)

        lastr = video_results[-1]

        # scores and labels
        scores_pred = lastr['scores_pred']
        labels_pred = lastr['labels_pred'] #* active_pred_mask

        # sort scores and labels
        sorted_idx = torch.argsort(scores_pred, dim=0, descending=True)
        scores_pred = scores_pred.gather(0, sorted_idx)
        labels_pred = labels_pred.gather(0, sorted_idx)

        # num class tracks, is the same for each sequence (lazy)
        labels_anno = lastr['lbannos']
        cnum_inst = torch.bincount(labels_anno, minlength=num_classes)

        # intersection (sum over L)
        intersection = [res['intersection'] for res in video_results]
        intersection = torch.cat(intersection, dim=0)
        intersection = intersection.sum(dim=0)

        # union (sum over L)
        union = [res['union'] for res in video_results]
        union = torch.cat(union, dim=0)
        union = union.sum(dim=0)

        # DEBUG
#        print()
#        for res in video_results:
#            print(torch.stack([res['intersection'][:,1,1], res['union'][:,1,1]]))

        # intersection over union
        iou = intersection/union
        iou_nan = torch.isnan(iou)
        iou[iou_nan] = 0.0 # @todo If annotation has 0 pixels, perhaps we should raise an error?

        # sort iou
        iou = iou[sorted_idx]

        # label
        lac = labels_anno.clone()
        lac[lac == 0] = 255
        labels_mask = labels_pred.view(-1, 1) == lac.view(1, -1)
        raw_iou = iou # DEBUG, (N,M)
        N, M = iou.shape
        if not ignore_class:
            iou = iou * labels_mask.float()
        else:
            iou = iou * (labels_anno != 0).view(1, -1).expand(N, -1).float()

        # Threshold masks
        iou_masks = torch.stack([iou > threshold for threshold in iou_thresholds], dim=-1)

        matching = iou_masks # (N, M, num_iou_thresholds), 1 where iou breaks threshold

        # Count TPs
        if use_old_TP_code:
            is_tp = torch.zeros((N, len(iou_thresholds)), dtype=torch.bool, device=iou.device)
            taken = torch.zeros((M, len(iou_thresholds)), dtype=torch.bool, device=iou.device)
            for n in range(N):
                if matching[n].any():
                    tmp = matching[n] # (M, num_iou_thresholds)
        
                    # For this prediction, find for each threshold the best matching annotation. This
                    # may be the last annotation if there is no match (as indicated by matching[i])
                    tidx = torch.max(iou[n].view(*iou[n].shape, 1)*tmp, dim=0, keepdim=True)[1] # (num_iou_thresh,)
        
                    # For this prediction and an IOU threshold, mark as TP if there is any matching
                    # annotation that is not yet taken.
                    is_tp[n, :] = (~taken & tmp).any(dim=0)  # This can be fishy! Weakly validated!
        
                    # Mark, for each IOU threshold, the annotation marked by tidx as taken
                    # BUG: We might have matching[i,0] be True, but matching[i,-1] be False. Then
                    # tidx[-1] will be M-1, which will be marked as taken. HOWEVER, this won't
                    # matter since the last annotation slot is probably never used. We never have
                    # that many objects.
                    # BUG: We might have a scenario where best match is taken but we have another
                    # sufficiently good match that is not yet taken. Now we set is_tp to False and
                    # again mark the best match as taken. Instead we should mark is_tp as True and
                    # mark the second best as taken. HOWEVER, this won't happen for our current
                    # model since it assigns only one class to each pixel.
                    taken.scatter_(0, tidx, 1)
        else:
            T = len(iou_thresholds)
            iou = iou.view(N, M, 1).repeat(1, 1, T) # (n, m, num_iou_thresh)
            iou_thresholds = torch.tensor(iou_thresholds, device=iou.device)
            is_tp = torch.zeros((N, T), dtype=torch.bool, device=iou.device)
#            print(iou[[0,1,2,3,4,5,6,7]][:,[0,1,2,3,4,5,6]][:,:,0])
            for n in range(N):
                best_anno_iou, best_anno_idx = iou[n].max(dim=0) # (num_iou_thresholds,)
                is_tp[n] = (best_anno_iou >= iou_thresholds) # Mark as positive if it matches anything

                # Set iou to 0.0 for each (annotation, iou_threshold) if we claim the anno for that iou
                best_anno_mask = torch.scatter(
                    torch.zeros((M, T), device=iou.device, dtype=torch.bool),
                    dim = 0,
                    index = best_anno_idx.view(1, T),
                    value = True).view(1, M, T).expand(N, -1, -1)           # (N,M,T)
                claimed_anno_mask = best_anno_mask * is_tp[n].view(1, 1, T) # (N,M,T)
                iou[claimed_anno_mask] = 0.0

            # DEBUG
            anno_ids = labels_anno.nonzero().view(-1)
            anno_labels = labels_anno[labels_anno != 0]
            
            if is_tp[:, 0].sum() != len(anno_ids):
                print(f"Found {is_tp[:, 0].sum()} out of {len(anno_ids)} objects")
                for m, anno_label in zip(anno_ids, anno_labels):
                    best_pred_iou, n = raw_iou[:, m].max(dim=0)
                    print(f"* Anno {m} with class {anno_label:2d} overlaps with a track by {best_pred_iou:.3f}, classified as {labels_pred[n]:2d} scored with {scores_pred[n]:.3f}.", "Marked as TP" if is_tp[n, 0] else "")
                if is_tp[:, 0].sum() > len(anno_ids):
                    print("* labels_anno")
                    print(labels_anno)
                    print("* is_tp:")
                    print(is_tp[[0,1,2,3,4,5,6,7], 0])
                    print("* overlap")
                    print(raw_iou[[0,1,2,3,4,5,6,7]][:,[0,1,2,3,4,5,6]])
                    raise ValueError("Found more positives than annotations, CODE IS BUGGED")
                    
            

        # DEBUG        
#        print("Raw IOU:      ")
#        print(raw_iou[[0,1,2,3,4,5,6,7]][:,[0,1,2,3,4]].transpose(0,1).cpu())
#        print("Is TP at IOU > 50%:", is_tp[[0,1,2,3,4,5,6,7]][:,0].cpu())
#        print("Scores:            ", scores_pred[[0,1,2,3,4,5,6,7]].cpu())
#        print("Labels:            ", labels_pred[[0,1,2,3,4,5,6,7]].cpu())
#        print("Track idx:         ", sorted_idx[[0,1,2,3,4,5,6,7]].cpu())

        return {'tp': is_tp, 'scores': scores_pred, 'labels': labels_pred, 'num_objects': cnum_inst}


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
                        apr[t][c-1] = VISAPEvaluatorV1.calculate_perclass_AP(M, tpsp[i].float(), scores)

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
