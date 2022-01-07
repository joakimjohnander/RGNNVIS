from scipy.optimize import linear_sum_assignment
import json

import torch
import torch.nn.functional as F

from rgnnvis.utils.debugging import print_tensor_statistics
from rgnnvis.utils.tensor_utils import batch_many_to_many_box_iou
#with torch.no_grad():

#from .optflow_consist import opt_flow_consist

import os, sys


def calculate_bbox_iou(box_one, box_two):
    """Assumes bbox of size (N,4) for box_one and box_two. This function broadcasts, and as such it is possible
    to test one to many relations
    """
    assert box_one.dim() == box_two.dim()
#    size = box_one.size()
#    box_one = box_one.view(-1, 4)
#    box_two = box_two.view(-1, 4)
    left   = torch.max(box_one[:,0], box_two[:,0])
    top    = torch.max(box_one[:,1], box_two[:,1])
#    right  = torch.min(x[:,0] + x[:,2], y[:,0] + y[:,2])
#    bottom = torch.min(x[:,1] + x[:,3], y[:,1] + y[:,3])
    right  = torch.min(box_one[:,2], box_two[:,2])
    bottom = torch.min(box_one[:,3], box_two[:,3])
    intersection_area = F.relu(right - left) * F.relu(bottom - top)
    area_one = (box_one[:,2] - box_one[:,0]) * (box_one[:,3] - box_one[:,1])
    area_two = (box_two[:,2] - box_two[:,0]) * (box_two[:,3] - box_two[:,1])
    union_area = area_one + area_two - intersection_area

    IoU = (intersection_area + 1e-7) / (union_area + 1e-7)

    return IoU


def calculate_box_iou(boxes_one, boxes_two):
    """Takes boxes of size (N,4) and (M,4), and for each of the NM pairs calculate the iou between the boxes.
    This function is a bit more general than the one above, and the naming is consistent with torchvision's.
    """
    assert False, "This should not ever be called. video_od_evaluator: Line 37"
    N = boxes_one.size(0)
    M = boxes_two.size(0)
    boxes_one = boxes_one.view(N,1,4)
    boxes_two = boxes_two.view(1,M,4)
    left   = torch.max(boxes_one[:,:,0], boxes_two[:,:,0])
    top    = torch.max(boxes_one[:,:,1], boxes_two[:,:,1])
    right  = torch.min(boxes_one[:,:,2], boxes_two[:,:,2])
    bottom = torch.min(boxes_one[:,:,3], boxes_two[:,:,3])
    iou = F.relu(right - left) * F.relu(bottom - top)
    return iou


def calculate_box_overlap(boxes_one, boxes_two, iou_threshold):
    """Compares two set of boxes, and returns the number of correspondences (and the number of boxes that lacks
    correspondences). A box from set one and a box from set two correspond if their iou is larger than the
    iou threshold.
    """
    N = boxes_one.size(0)
    M = boxes_two.size(0)
    iou = calculate_box_iou(boxes_one, boxes_two)
    cost = 1 * (iou >= iou_threshold).float() + N*M * (iou < iou_threshold)
    row_ids, col_ids = linear_sum_assignment(cost.numpy())
    num_matches = (iou[torch.tensor(row_ids), torch.tensor(col_ids)] >= iou_threshold).long().sum().float()
    return num_matches




class VPSODEvaluatorV2(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_sequence_results(pred, anno, iou_thresholds, use_old_TP_code, ignore_class):
        """
        Args:
            pred
                                 (Dict): {'detection_boxes':    (FloatTensor): (L,N,4)
                                          'detection_lbscores': (FloatTensor): (L,N,C)
                                          'detection_active:     (LongTensor): (L,N)

            anno                 (Dict): { 'odannos': (FloatTensor): (L,Mmax,4),
                                          'lbannos':  (LongTensor): (Mmax),
                                           'active':  (ByteTensor): (L,Mmax) }

            iou_thresholds       (List): Typically [.5] or varied as [.50, .55, ..., .95]
        Returns:
            results       (FloatTensor): of size (*, N x 3) with each elem (is true positive, confidence, label)
            num_objects    (LongTensor): Tensor of size num_classes, describing number of objects of that class (Histogram)
        """
        active_pred = pred['detection_active'].squeeze(0) == 1
        conf_pred = pred['detection_lbscores'].squeeze(0)
        boxes_pred = pred['detection_boxes'].squeeze(0)

        # DEBUG
        # active_pred = torch.zeros_like(anno['active']).byte()
        # boxes_pred = -1*torch.ones_like(anno['odannos'])
        # active_pred[:, 0:4] = anno['active'][:, 0:4] == 1
        # boxes_pred[:, 0:4] = anno['odannos'][:, 0:4]
        # l, m, p = anno['odannos'].shape
        # lb_tmp = anno['lbannos'].expand(l, m)
        # labels_pred = (active_pred * lb_tmp)
        # scores_pred = torch.rand(l, m).to(active_pred.device)
        # sorted_idx = torch.argsort(scores_pred, dim=-1, descending=True)
        # scores_pred = scores_pred.gather(1, sorted_idx)
        # labels_pred = labels_pred.gather(1, sorted_idx)
        #######################################################################

        # Make conf zero and boxes -1 for non active predictions
        l, n, c = conf_pred.shape
        #n = 16  # DEBUG
        device = conf_pred.device

        conf_pred = F.softmax(conf_pred, dim=2)
        conf_pred = conf_pred.where(active_pred.view(l, n, 1), torch.tensor(0., device=device)) # set value when condition is false
        boxes_pred = boxes_pred.where(active_pred.view(l, n, 1), torch.tensor(-1., device=device)) # set value when condition is false

        # Get max conf out of num_classes
        scores_pred = torch.max(conf_pred[:, :, 1:], dim=2)[0]  # Remove background from class conf dim
        labels_pred = torch.argmax(conf_pred, dim=2).where(active_pred, torch.tensor(0, device=device))
        sorted_idx = torch.argsort(scores_pred, dim=-1, descending=True)
        scores_pred = scores_pred.gather(1, sorted_idx)
        labels_pred = labels_pred.gather(1, sorted_idx)
        
        # Annotations (ground truths)
        _, m, p = anno['odannos'].shape
        boxes_anno = anno['odannos']
        lb = anno['lbannos'].expand(l, m)
        labels_anno = ((anno['active'] == 1) * lb)
        num_objects = torch.bincount(labels_anno.flatten(), minlength=c)

        # For the broadcast to give false on category 0 which is not a category, A = {1,...,C}.
        labels_anno[labels_anno == 0] = 255
        label_mask = labels_pred.view(l, -1, 1) == labels_anno.view(l, 1, -1)  # [l,n,m]

        boxes_pred_sorted = torch.stack([pb[sorted_idx[i]] for i, pb in enumerate(boxes_pred)], dim=0)

        iou = batch_many_to_many_box_iou(boxes_pred_sorted, boxes_anno, iou_on_empty=0) # (l, n, m)

        # Threshold masks and stack
        iou_masks = torch.stack([iou > threshold for threshold in iou_thresholds], dim=-1)

        matching = iou_masks & label_mask.view(l, n, m, 1)

        # Count TPs
        if use_old_TP_code:
            tp = torch.zeros((l, n, len(iou_thresholds)), dtype=torch.bool, device=boxes_pred.device)
            taken = torch.zeros((l, m, len(iou_thresholds)), dtype=torch.bool, device=boxes_pred.device)
            for i in range(n):
                if matching[:, i].any():
                    tmp = matching[:, i]
                    tidx = torch.max(iou[:, i].view(*iou[:, i].shape, 1)*tmp, dim=1, keepdim=True)[1]
                    tp[:, i, :] = (~taken & tmp).any(dim=1) # This can be fishy! Weakly validated!
                    taken.scatter_(1, tidx, 1)
        else:
            # Set overlap to 0 if the classes don't match
            if not ignore_class:
                iou = iou * label_mask.float()
            else:
                iou = iou * (anno['active'] == 1).view(l, 1, -1).expand(-1, n, -1).float()
            
            T = len(iou_thresholds)
            iou = iou.view(l, n, m, 1).repeat(1, 1, 1, T) # (l, n, m, num_iou_thresh)
            iou_thresholds = torch.tensor(iou_thresholds, device=iou.device).view(1, T).expand(l, -1)
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

        # DEBUG
#        print(f"  num TP = {is_tp[:, :, 0].sum()}")
#        print(f"  TP labels = {labels_pred[is_tp[:, :, 0]].unique()}")

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

            # DEBUG
#            YTVIS_CATEGORY_NAMES = {1: "person", 2: "giant_panda", 3: "lizard", 4: "parrot", 5: "skateboard", 6: "sedan", 7: "ape", 8: "dog", 9: "snake", 10: "monkey", 11: "hand", 12: "rabbit", 13: "duck", 14: "cat", 15: "cow", 16: "fish", 17: "train", 18: "horse", 19: "turtle", 20: "bear", 21: "motorbike", 22: "giraffe", 23: "leopard", 24: "fox", 25: "deer", 26: "owl", 27: "surfboard", 28: "airplane", 29: "truck", 30: "zebra", 31: "tiger", 32: "elephant", 33: "snowboard", 34: "boat", 35: "shark", 36: "mouse", 37: "frog", 38: "eagle", 39: "earless_seal", 40: "tennis_racket"}
#            print(f"\nClass {c} ({YTVIS_CATEGORY_NAMES[c]})")
#            print(f"{perclass_num_objects[c]} annotated objects")
#            if tps.shape[0]:
#                print(f"Got {tps.size()[0]} detections, {tps[:, 0].sum().item()} TP at IOU > 50%, and {tps[:, 5].sum().item()} at IOU > 75%")
#        raise ValueError()

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
                        apr[t][c-1] = VPSODEvaluatorV2.calculate_perclass_AP(M, tpsp[i].float(), scores)

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

    ##################################################################################################################

class VPSODEvaluatorV1(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_sequence_results_w(boxes_pred, conf_pred, anno, num_classes, evaluation_state):
        """
        Args:
            boxes_pred    (LongTensor): (L,N,4)
            conf_pred    (FloatTensor): (L,N,C)
            anno                (Dict): { 'odannos': (FloatTensor): (L,Mmax,4),
                                         'lbannos':  (LongTensor): (Mmax),
                                          'active':  (ByteTensor): (L,Mmax) }
        Returns:
            results      (FloatTensor): of size (*, N x 3) with each elem (is true positive, confidence, label)
            num_objects   (LongTensor): Tensor of size num_classes, describing number of objects of that class (Histogram)
        """
        l, n, c = conf_pred.shape

        # Get max conf out of num_classes
        scores_pred, labels_pred = torch.max(conf_pred, dim=-1)

        # Annotations (ground truths)
        l, m, p = anno['odannos'].shape
        boxes_anno = anno['odannos']
        lb = anno['lbannos'].expand(l, m)
        labels_anno = (anno['active'] * lb)

        pred = []
        for bp, lp, sp in zip(boxes_pred, labels_pred, scores_pred):
            pred.append({'boxes': bp, 'labels': lp, 'scores': sp})

        targets = []
        for ba, la in zip(boxes_anno, labels_anno):
            targets.append({'boxes': ba, 'labels': la.long()})

        return VPSODEvaluatorV1.evaluate_batch(pred, targets, evaluation_state, num_classes)


    @staticmethod
    def get_image_results(ODpred, ODanno, iou_threshold, num_classes):
        """
        Args:
            ODpred: {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
            ODanno: {'boxes': (N,4) tensor, 'labels': (N,) tensor}
            iou_threshold: Typically .5 or varied as {.50, .55, ..., .95}
        Returns:
            Tensor of size N x 3 with each elem (is true positive, confidence, label)
            Tensor of size num_classes, describing number of objects of that class
        """
        sorted_ids = torch.argsort(ODpred['scores'], dim=0, descending=True)
        boxes_pred = ODpred['boxes'][sorted_ids, :]
        labels_pred = ODpred['labels'][sorted_ids]
        scores_pred = ODpred['scores'][sorted_ids]
        N = sorted_ids.size(0)
        M = ODanno['boxes'].size(0)

        results = torch.zeros((N, 3))
        results[:, 1] = scores_pred
        results[:, 2] = labels_pred
        ODanno_taken = torch.zeros((M,), dtype=torch.uint8)
        for n in range(N):
            iou = batch_many_to_many_box_iou(boxes_pred[n:n + 1].unsqueeze(0), ODanno['boxes'].unsqueeze(0), iou_on_empty=0).squeeze()
            #iou = calculate_bbox_iou(boxes_pred[n:n + 1], ODanno['boxes'])
            assert iou.size(0) == M, (iou, M)
            matches = []
            for m in range(M):
                if iou[m] > iou_threshold and (labels_pred[n] == ODanno['labels'][m]) and not ODanno_taken[m]:
                    matches.append((m, iou[m]))
            if len(matches) > 0:
                best_match = max(matches, key=(lambda x: x[1]))
                best_match_idx = best_match[0]
                results[n, 0] = 1.
                ODanno_taken[best_match_idx] = 1
        num_objects = torch.bincount(ODanno['labels'], minlength=num_classes)
        return results, num_objects

    @staticmethod
    def get_batch_results(ODpred_lst, ODanno_lst, iou_threshold, num_classes):
        """
        Args:
            ODpred: list of {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
            ODanno: list of {'boxes': (N,4) tensor, 'labels': (N,) tensor}
            iou_threshold: Typically .5 or varied as {.50, .55, ..., .95}
        Returns:
            list of (Tensor of size N x 3 with each elem (is true positive, confidence, label), Mobj)
        """
        results = [VPSODEvaluatorV1.get_image_results(ODpred, ODanno, iou_threshold, num_classes)
                   for ODpred, ODanno in zip(ODpred_lst, ODanno_lst)]
        return results

    @staticmethod
    def evaluate_batch(model_predictions, targets, evaluation_state, num_classes):
        """
        Args:
            model_predictions : list of {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
            targets           : list of {'boxes': (N,4) tensor, 'labels': (N,) tensor}
            evaluation_state  : {iou_threshold: list of image_results}, each image_result is
                                (tensor of size (N,3) with each elem (is true positive, confidence, label), Mobj)
        """
        if evaluation_state is None:
            # dict of N,2 tensors (last dim is TP/FP and confidence)
            evaluation_state = {int(100*iou_threshold+0.5): []
                                for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        new_evaluation_state = {int(100*iou_threshold+0.5): VPSODEvaluatorV1.get_batch_results(model_predictions, targets, iou_threshold, num_classes)
                                for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        evaluation_state = {int(100*iou_threshold+0.5): evaluation_state[int(100*iou_threshold+0.5)] + new_evaluation_state[int(100*iou_threshold+0.5)]
                            for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        return evaluation_state


    @staticmethod
    def calculate_perclass_AP(M, result):
        """Calculate AP for a single threshold and a single class
        Args:
            M: int, number of annotated objects of this class label
            result: (N,3) tensor, with fields (isTP, confidence, class label)
        Returns:
            Average Precision
        """
        sorted_ids = torch.argsort(result[:, 1], dim=0, descending=True)
        result = result[sorted_ids]
        istp = result[:, 0]
        precisions = torch.cumsum(istp, 0) / torch.arange(1., len(istp) + 1)
        v = 0.
        for idx in range(len(precisions) - 1, -1, -1):
            v = max(v, precisions[idx])
            precisions[idx] = v
        return torch.sum(precisions * istp / M).item()

    @staticmethod
    def calculate_AP(result, num_classes):
        """Calculate every class AP for a given threshold
        """
        perclass_num_objects = [sum([Mimg[c] for res, Mimg in result]) for c in range(num_classes)]
        perclass_result = [torch.cat([res[res[:, 2] == c, :] for res, Mimg in result])
                           for c in range(num_classes)]
        return [VPSODEvaluatorV1.calculate_perclass_AP(M, res) for M, res in zip(perclass_num_objects, perclass_result)]
    

class VideoODEvaluator(object):
    def __init__(self, num_classes=10, device='cuda', consistency_measure='oracle_v0'):
        self._num_classes = num_classes
        self._device = device
        self._consistency_measure = consistency_measure

    def get_image_results(self, ODpred, ODanno, iou_threshold):
        """
        Args:
            ODpred: {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
            ODanno: {'boxes': (N,4) tensor, 'labels': (N,) tensor}
            iou_threshold: Typically .5 or varied as {.50, .55, ..., .95}
        Returns:
            Tensor of size N x 3 with each elem (is true positive, confidence, label)
            Tensor of size num_classes, describing number of objects of that class
        """
        sorted_ids = torch.argsort(ODpred['scores'], dim=0, descending=True)
        boxes_pred = ODpred['boxes'][sorted_ids,:]
        labels_pred = ODpred['labels'][sorted_ids]
        scores_pred = ODpred['scores'][sorted_ids]
        N = sorted_ids.size(0)
        M = ODanno['boxes'].size(0)

        results = torch.zeros((N,3))
        results[:,1] = scores_pred
        results[:,2] = labels_pred
        ODanno_taken = torch.zeros((M,), dtype=torch.uint8)
        for n in range(N):
            iou = calculate_bbox_iou(boxes_pred[n:n+1], ODanno['boxes'])
            assert iou.size(0) == M, (iou, M)
            matches = []
            for m in range(M):
                if iou[m] > iou_threshold and (labels_pred[n] == ODanno['labels'][m]) and not ODanno_taken[m]:
                    matches.append((m, iou[m]))
            if len(matches) > 0:
                best_match = max(matches, key=(lambda x: x[1]))
                best_match_idx = best_match[0]
                results[n,0] = 1.
                ODanno_taken[best_match_idx] = 1
        num_objects = torch.bincount(ODanno['labels'], minlength=self._num_classes)
        return results, num_objects
        
    def get_batch_results(self, ODpred_lst, ODanno_lst, iou_threshold):
        """
        Args:
            ODpred: list of {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
            ODanno: list of {'boxes': (N,4) tensor, 'labels': (N,) tensor}
            iou_threshold: Typically .5 or varied as {.50, .55, ..., .95}
        Returns:
            list of (Tensor of size N x 3 with each elem (is true positive, confidence, label), Mobj)
        """
        results = [self.get_image_results(ODpred, ODanno, iou_threshold)
                   for ODpred, ODanno in zip(ODpred_lst, ODanno_lst)]
        return results
        
    def evaluate_batch(self, model_predictions, targets, evaluation_state):
        """
        Args:
            model_predictions : list of {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
            targets           : list of {'boxes': (N,4) tensor, 'labels': (N,) tensor}
            evaluation_state  : {iou_threshold: list of image_results}, each image_result is
                                (tensor of size (N,3) with each elem (is true positive, confidence, label), Mobj)
        """
        if evaluation_state is None:
            # dict of N,2 tensors (last dim is TP/FP and confidence)
            evaluation_state = {iou_threshold: []
                                for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        new_evaluation_state = {iou_threshold: self.get_batch_results(model_predictions, targets, iou_threshold)
                                for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        evaluation_state = {iou_threshold: evaluation_state[iou_threshold] + new_evaluation_state[iou_threshold]
                            for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        return evaluation_state

    def calculate_perclass_AP(self, M, result):
        """Calculate AP for a single threshold and a single class
        Args:
            M: int, number of annotated objects of this class label
            result: (N,3) tensor, with fields (isTP, confidence, class label)
        Returns:
            Average Precision
        """
        sorted_ids = torch.argsort(result[:,1], dim=0, descending=True)
        result = result[sorted_ids]
        istp = result[:,0]
        precisions = torch.cumsum(istp, 0) / torch.arange(1., len(istp) + 1)
        v = 0.
        for idx in range(len(precisions) - 1, -1, -1):
            v = max(v, precisions[idx])
            precisions[idx] = v
        return torch.sum(precisions * istp / M).item()
    
    def calculate_AP(self, result):
        """Calculate every class AP for a given threshold
        """
        perclass_num_objects = [sum([Mimg[c] for res, Mimg in result]) for c in range(self._num_classes)]
        perclass_result = [torch.cat([res[res[:,2] == c, :] for res, Mimg in result])
                           for c in range(self._num_classes)]
        return [self.calculate_perclass_AP(M, res) for M, res in zip(perclass_num_objects, perclass_result)]
    
    def calculate_dataset_performance(self, evaluation_state):
        """Calculate per-class AP50 and mscoco mAP measures. First calculates every class AP for every iou-threshold
        """
        AP = {iou_threshold: self.calculate_AP(evaluation_state[iou_threshold])
              for iou_threshold in torch.arange(.50, 1., .05).tolist()}
        class_AP50 = AP[.50]
        coco_mAP = sum([sum(AP[iou_threshold]) for iou_threshold in torch.arange(.50, 1., .05).tolist()])
        coco_mAP = coco_mAP / (self._num_classes * 10)
        return class_AP50, coco_mAP

    
    def get_consistency_oracle_v0(self, ODpreds, ODannos):
        """Compares predicted boxes to ground truth, and returns consistency as TP/(TP+FP+FN) which is a measure
        on how well the model performs. Here, we ignore the class labels.
        """
        result_lists = [self.get_image_results(ODpred, ODanno, .3) for ODpred, ODanno in zip(ODpreds, ODannos)]

        def per_image_score(result, M):
            Mtot = M.sum()
#            THRESHOLD = .3
#            result = result[result[:,1] > THRESHOLD]
            TP = result[:,0].sum()
#            FP = result[:,1].sum()
            FP = (1 - result[:,0]).sum()
            FN = Mtot - TP
            return (TP + 1e-7) / (TP + FP + FN + 1e-7)
            
        return [per_image_score(result, M) for result, M in result_lists]

    def get_consistency_temporal_v0(self, ODpreds, iscontiguous):
        L = len(ODpreds)
        def per_image_score(result, M):
            Mtot = M.sum()
#            THRESHOLD = .3
#            result = result[result[:,1] > THRESHOLD]
            TP = result[:,0].sum()
#            FP = result[:,1].sum()
            FP = (1 - result[:,0]).sum()
            FN = Mtot - TP
            return (TP + 1e-7) / (TP + FP + FN + 1e-7)
        
        forward_consistency = [per_image_score(*self.get_image_results(ODpreds[l], ODpreds[l+1], .3))
                               for l in range(L - 1)] + [None]
        backward_consistency = [None] + [per_image_score(*self.get_image_results(ODpreds[l+1], ODpreds[l], .3))
                                         for l in range(L - 1)]
        consistency = list(range(L))
        for l in range(L):
            if (l == 0 and iscontiguous[l+1] == 0) or (l == L-1 and iscontiguous[l] == 0):
                consistency[l] = 1.
            elif l == 0:
                consistency[l] = forward_consistency[l]
            elif l == L-1:
                consistency[l] = backward_consistency[l]
            elif (iscontiguous[l] == 0 and iscontiguous[l+1] == 1):
                consistency[l] = forward_consistency[l]
            elif (iscontiguous[l] == 1 and iscontiguous[l+1] == 0):
                consistency[l] = backward_consistency[l]
            elif 0 < l and l < L-1 and iscontiguous[l] == 1 and iscontiguous[l+1] == 1:
                consistency[l] = .5 * (forward_consistency[l] + backward_consistency[l])
            else:
                consistency[l] = 1.
                print("Found a frame with no temporal neighbours ...")
                print("iscontiguous: ", iscontiguous)
                print(f"l = {l}")
#                print("iscontiguous: ", iscontiguous)
#                print(f"l = {l}")
#                raise ValueError("This should not happen")
        for elem in consistency:
            if not isinstance(elem, (float, torch.Tensor)) or not (0. <= elem and elem <= 1.):
                print(consistency)
                print(forward_consistency)
                print(backward_consistency)
                print(iscontiguous)
                raise ValueError()
        return consistency

    def get_consistency_temporal_v1(self, ODpreds, iscontiguous, images):
        L = len(ODpreds)
        boxes = [ODpred['boxes'] for ODpred in ODpreds]
        print(len(images))
        print([im.size() for im in images])
        print(len(boxes))
        print([box.size() for box in boxes])
        print(iscontiguous)
        forward_boxes = opt_flow_consist(images, boxes, iscontiguous, direction='forward')
        backward_boxes = opt_flow_consist(images, boxes, iscontiguous, direction='backward')
        ODpreds_forward = [{'boxes': forward_boxes[l],
                            'labels': ODpreds[l]['labels'],
                            'scores': ODpreds[l]['scores']}
                           for l in range(0,L-1)]
        ODpreds_backward = [{'boxes': backward_boxes[l-1],
                             'labels': ODpreds[l]['labels'],
                             'scores': ODpreds[l]['scores']}
                            for l in range(1,L)]
        assert len(forward_boxes) == L-1
        assert len(backward_boxes) == L-1
        
        def per_image_score(result, M):
            Mtot = M.sum()
            TP = result[:,0].sum()
            FP = (1 - result[:,0]).sum()
            FN = Mtot - TP
            return (TP + 1e-7) / (TP + FP + FN + 1e-7)

        forward_consistency = [per_image_score(*self.get_image_results(
            ODpreds_forward[l], ODpreds[l+1], .3))
                               for l in range(L - 1)] + [None]
        backward_consistency = [None] + [per_image_score(*self.get_image_results(
            ODpreds_backward[l], ODpreds[l], .3))
                                         for l in range(L - 1)]
        consistency = list(range(L))
        for l in range(L):
            if (l == 0 and iscontiguous[l+1] == 0) or (l == L-1 and iscontiguous[l] == 0):
                consistency[l] = 1
            elif l == 0:
                consistency[l] = forward_consistency[l]
            elif l == L-1:
                consistency[l] = backward_consistency[l]
            elif (iscontiguous[l] == 0 and iscontiguous[l+1] == 1):
                consistency[l] = forward_consistency[l]
            elif (iscontiguous[l] == 1 and iscontiguous[l+1] == 0):
                consistency[l] = backward_consistency[l]
            elif 0 < l and l < L-1 and iscontiguous[l] == 1 and iscontiguous[l+1] == 1:
                consistency[l] == .5 * (forward_consistency[l] + backward_consistency[l])
            else:
                consistency[l] = 1
                print("Found a frame with no temporal neighbours ...")
                print("iscontiguous: ", iscontiguous)
                print(f"l = {l}")
                
        return consistency
    
    def get_consistency(self, model_predictions, targets=None, iscontiguous=None, images=None):
        """This function is called by the trainer on all training data to determine how "consistent" predictions
        are. Results should be model predictions corresponding to one temporally contiguous sequence of images.
        Args:
            model_predictions: list of {'boxes': (N,4) tensor, 'labels': (N,) tensor, 'scores': (N,) tensor}
        """
        if self._consistency_measure == 'oracle_v0':
            return self.get_consistency_oracle_v0(model_predictions, targets)
        if self._consistency_measure == 'oracle_easiest_v0':
            consistency = self.get_consistency_oracle_v0(model_predictions, targets)
            return [1 - val for val in consistency]
        elif self._consistency_measure == 'temporal_v0':
            return self.get_consistency_temporal_v0(model_predictions, iscontiguous)
        elif self._consistency_measure == 'temporal_v1':
            return self.get_consistency_temporal_v1(model_predictions, iscontiguous, images)


def main():
    # Anno
    nc = 41
    od1 = -torch.ones(5, 4)
    od1[1] = torch.tensor([20, 20, 100, 100])
    od1[2] = torch.tensor([20, 200, 60, 240])
    od2 = od1.clone()
    od2[3] = torch.tensor([500, 500, 520, 520])
    od = torch.stack([od1, od2], dim=0)

    lb = torch.tensor([0, 1, 2, 3, 0]).long()

    ac1 = torch.zeros(5)
    ac1[1] = 1
    ac1[2] = 1
    ac2 = torch.zeros(5)
    ac2[1] = 1
    ac2[2] = 1
    ac2[3] = 1
    ac = torch.stack([ac1, ac2], dim=0)

    anno = {'odannos': od, 'lbannos': lb, 'active': ac}
    for k, v in anno.items():
        print(k)
        print(v)

    ###########################################################################################################

    # Pred1

    db = od.clone()
    db[0, 1] = torch.tensor([35, 35, 125, 125])

    dlb1 = torch.zeros(5, nc)
    dlb1[1, 1] = 4
    dlb1[2, 2] = 2
    dlb2 = torch.zeros(5, nc)
    dlb2[1, 1] = 2
    dlb2[2, 2] = 4
    dlb2[3, 3] = 5
    dlb = torch.stack([dlb1, dlb2], dim=0)

    dac1 = torch.zeros(5)
    dac1[1] = 1
    dac1[2] = 1
    dac2 = torch.zeros(5)
    dac2[1] = 1
    dac2[2] = 1
    dac2[3] = 1
    dac = torch.stack([dac1, dac2], dim=0)

    pred1 = {'detection_boxes': db, 'detection_lbscores': dlb, 'detection_active': dac}

    ###########################################################################################################
    th = [0.5, 0.6, 0.7]
    odres = VPSODEvaluatorV2.get_sequence_results(pred1, anno, th)
    ap = VPSODEvaluatorV2.calculate_AP([odres], th, nc)

    out = dict.fromkeys(ap.keys())
    for key, value in ap.items():
        out[key] = torch.mean(value).item()

    print(json.dumps(out, indent=4))

    # VPSODEvaluatorV1
    results = VPSODEvaluatorV1.get_sequence_results_w(pred1['detection_boxes'], pred1['detection_lbscores'], anno, nc, None)
    ap50 = VPSODEvaluatorV1.calculate_AP(results[50], nc)
    print('end')


if __name__ == "__main__":
    main()
