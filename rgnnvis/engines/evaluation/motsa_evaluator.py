import torch
import torch.nn.functional as F
from rgnnvis.utils.tensor_utils import index_map_to_one_hot
from rgnnvis.utils.tensor_utils import batch_many_to_many_seg_iou


class MOTSAEvaluatorV2:
    """
    MOTSAEvaluatorV2 is a class with static functions for computing MOTSA evaluation metrics. Storage and summary of
    functions' outputs must be handles elsewhere. See VPSEvaluatorV1 for an example of how this class is used.
    https://arxiv.org/pdf/1902.03604.pdf
    """

    @staticmethod
    def get_sequence_results(pred, anno, num_classes, iou_threshold=0.5, spatial_split_size=100000):
        """
        Args:
            pred
                                 (Dict): {  'instance_segs':  (FloatTensor):  (B,L,H,W)    values in [0 Nmax-1]
                                                 'lbscores':   (ByteTensor):  (B,L,C)      C=num_classes
                                           'active_objects':   (ByteTensor):  (B,L,Nmax)   Nmax=maximum num of hypot.
                                         }

            anno                 (Dict): { 'isannos':  (FloatTensor):  (B,L,H,W),
                                            'active':   (ByteTensor):  (B,L,Mmax)          Mmax=maximum num of possible
                                                                                                tracks
                                         }

            iou_thresholds       (Float): Should always be 0.5 according to paper.
            spatial_split_size     (Int): spatial split size
        Returns:
            dict                  (Dict): see Code.
        """
        # pred['instance_segs'] = anno['isannos']  # Debug

        device = pred['active_objects'].device
        b, l, nmax = pred['active_objects'].shape
        pred_onehot_seg = index_map_to_one_hot(pred['instance_segs'].unsqueeze(-3),  # size now (B,L,1,H,W)
                                               torch.tensor(range(nmax), dtype=torch.uint8),
                                               device)

        _, _, mmax = anno['active'].shape
        anno_onehot_seg = index_map_to_one_hot(anno['isannos'].unsqueeze(-3),  # size now (B,L,1,H,W)
                                               torch.tensor(range(mmax), dtype=torch.uint8),
                                               device)

        # pred_onehot_seg is of size (B,L,Nmax,H,W) and anno_onehot_seg is of size (B,L,Mmax,H,W)
        # output size (B,L,Nmax,Mmax)
        iou = batch_many_to_many_seg_iou(pred_onehot_seg, anno_onehot_seg, unassigned_iou=0.0,
                                         split_size=spatial_split_size)

        # establish correspondence
        # actives (B,L,Nmax,Mmax), torch.Bool, remove background
        pred_active = (pred['active_objects'].view(b, l, nmax, 1) == 1)
        anno_active = (anno['active'].view(b, l, 1, mmax) == 1)
        actives = pred_active & anno_active

        # labels
        conf_pred = pred['lbscores']
        conf_pred = F.softmax(conf_pred, dim=-1)
        conf_pred = conf_pred.where(~torch.isnan(conf_pred), torch.tensor(0., device=device)) # set value when condition is false
        labels_pred = torch.argmax(conf_pred, dim=-1) * pred_active.any(dim=1).view(nmax)

        # counting m cardinality
        lb = anno['lbannos'].expand(l, mmax)
        lb = ((anno['active'] == 1) * lb)
        num_objects = torch.bincount(lb.flatten(), minlength=num_classes)

        # label mask
        labels_anno = anno['lbannos']
        lac = labels_anno.clone()
        lac[lac==0] = 255
        labels_mask = labels_pred.view(-1, 1) == lac.view(1, -1)


        # iou remove background
        #iou = iou[:, :, 1:, 1:]

        # iou threshold
        iou_mask = (iou > iou_threshold) & actives & labels_mask.view(1, 1, nmax, mmax)
        iou[~iou_mask] = 0.0

        # pick out tp, fp, correpondences
        tp = iou_mask.any(dim=-1, keepdim=True) & pred_active
        fp = ~tp & pred_active
        fn = ~iou_mask.any(dim=-2, keepdim=True) & anno_active
        iou_max, iou_max_idx = torch.max(iou, dim=-1, keepdim=True)
        iou_max_idx = iou_max_idx * tp + -1 * ~tp

        return {'tp': tp.view(l, nmax), 'fp': fp.view(l, nmax), 'fn': fn.view(l, mmax),
                'iou': iou_max.view(l, nmax), 'correspondence': iou_max_idx.view(l, nmax),
                'labels_pred': labels_pred.view(nmax), 'num_objects': num_objects}

    @staticmethod
    def calculate_motsa(results, num_classes):
        """ Computes mostsa metrics for entire dataset input
        Args:
            results           (List): of dicts
            num_classes        (Int): number of classes
        Returns:
            dict              (Dict): see code.
        """

        # m
        m = [res['num_objects'] for res in results]
        m = torch.stack(m, dim=0).sum(dim=0)

        cr = {}
        for c in range(1, num_classes):
            tp_cardinality = torch.tensor([0], dtype=torch.int64)
            fp_cardinality = torch.tensor([0], dtype=torch.int64)
            ids_cardinality = torch.tensor([0], dtype=torch.int64)  # number of times there is a idx switch
            tps = torch.tensor([0.0], dtype=torch.float32)
            for re in results:
                cmask = (re['labels_pred'] == c)
                tp_cardinality += re['tp'][:, cmask].sum()
                fp_cardinality += re['fp'][:, cmask].sum()
                tps += re['iou'][:, cmask].sum()

                # number of idx switches
                crs = torch.split(re['correspondence'][:, cmask], 1, dim=1)
                for i in range(1, len(crs)):
                    if i == 0:
                        continue
                    else:
                        diff = crs[i] != crs[i-1]
                        ids_cardinality += diff.sum()
            cr.update({c: {'tpc': tp_cardinality, 'fpc': fp_cardinality,
                           'idsc': ids_cardinality, 'tps': tps}})

        motsa = []
        motsp = []
        smotsa = []
        for c, value in cr.items():
            if m[c]:
                motsa.append((value['tpc'] - value['fpc'] - value['idsc']).float() / m[c].float())
                smotsa.append((value['tps'] - value['fpc'].float() - value['idsc'].float()) / m[c].float())
            else:
                motsa.append(torch.tensor([0.0]))
                smotsa.append(torch.tensor([0.0]))

            if value['tpc']:
                motsp.append(value['tps'] / value['tpc'].float())
            else:
                motsp.append(torch.tensor([0.0]))


        m_mask = m[1:] > 0
        motsa = torch.cat(motsa, dim=0)
        mmotsa = motsa[m_mask].mean(dim=0)
        motsp = torch.cat(motsp, dim=0)
        mmotsp = motsp[m_mask].mean(dim=0)
        smotsa = torch.cat(smotsa, dim=0)
        msmotsa = smotsa[m_mask].mean(dim=0)

        return {'mMOTSA': mmotsa.item(), 'MOTSA': motsa.tolist(),
                'mMOTSP': mmotsp.item(), 'MOTSP': motsp.tolist(),
                'msMOTSA': msmotsa.item(), 'sMOTSA': smotsa.tolist()}


class MOTSAEvaluatorV1:
    """
    MOTSAEvaluatorV1 is a class with static functions for computing MOTSA evaluation metrics. Storage and summary of
    functions' outputs must be handles elsewhere. See VPSEvaluatorV1 for an example of how this class is used.
    https://arxiv.org/pdf/1902.03604.pdf
    """

    @staticmethod
    def get_sequence_results(pred, anno, iou_threshold=0.5, spatial_split_size=100000):
        """
        Args:
            pred
                                 (Dict): {  'instance_segs':  (FloatTensor):  (B,L,H,W)    values in [0 Nmax-1]
                                                 'lbscores':   (ByteTensor):  (B,L,C)      C=num_classes
                                           'active_objects':   (ByteTensor):  (B,L,Nmax)   Nmax=maximum num of hypot.
                                         }

            anno                 (Dict): { 'isannos':  (FloatTensor):  (B,L,H,W),
                                            'active':   (ByteTensor):  (B,L,Mmax)          Mmax=maximum num of possible
                                                                                                tracks
                                         }

            iou_thresholds       (Float): Should always be 0.5 according to paper.
            spatial_split_size     (Int): spatial split size
        Returns:
            dict                  (Dict): see Code.
        """
        #TODO: make it per class

        # pred['instance_segs'] = anno['isannos']  # Debug

        device = pred['active_objects'].device
        b, l, nmax = pred['active_objects'].shape
        pred_onehot_seg = index_map_to_one_hot(pred['instance_segs'].unsqueeze(-3),  # size now (B,L,1,H,W)
                                               torch.tensor(range(nmax), dtype=torch.uint8),
                                               device)

        _, _, mmax = anno['active'].shape
        anno_onehot_seg = index_map_to_one_hot(anno['isannos'].unsqueeze(-3),  # size now (B,L,1,H,W)
                                               torch.tensor(range(mmax), dtype=torch.uint8),
                                               device)

        # pred_onehot_seg is of size (B,L,Nmax,H,W) and anno_onehot_seg is of size (B,L,Mmax,H,W)
        # output size (B,L,Nmax,Mmax)
        iou = batch_many_to_many_seg_iou(pred_onehot_seg, anno_onehot_seg, unassigned_iou=0.0,
                                         split_size=spatial_split_size)

        # establish correspondence
        # actives (B,L,Nmax,Mmax), torch.Bool, remove background
        pred_active = (pred['active_objects'].view(b, l, nmax, 1) == 1)[:, :, 1:]
        anno_active = (anno['active'].view(b, l, 1, mmax) == 1)[:, :, :, 1:]
        actives = pred_active & anno_active

        # iou remove background
        iou = iou[:, :, 1:, 1:]

        # iou threshold
        iou_mask = (iou > iou_threshold) & actives
        iou[~iou_mask] = 0.0

        # pick out tp, fp, correpondences
        tp = iou_mask.any(dim=-1, keepdim=True) & pred_active
        fp = ~tp & pred_active
        fn = ~iou_mask.any(dim=-2, keepdim=True) & anno_active
        iou_max, iou_max_idx = torch.max(iou, dim=-1, keepdim=True)
        iou_max_idx = iou_max_idx * tp + -1 * ~tp

        return {'tp': tp, 'fp': fp, 'fn': fn, 'iou': iou_max, 'correspondence': iou_max_idx, 'm': anno_active.sum()}

    @staticmethod
    def calculate_motsa(results, num_classes):
        """ Computes mostsa metrics for entire dataset input
        Args:
            results           (List): of dicts
            num_classes        (Int): number of classes
        Returns:
            dict              (Dict): see code.
        """

        tp_cardinality = torch.tensor([0], dtype=torch.int64)
        fp_cardinality = torch.tensor([0], dtype=torch.int64)
        #fn_cardinality = torch.tensor([0], dtype=torch.LongTensor)
        m_cardinality = torch.tensor([0], dtype=torch.int64)
        ids_cardinality = torch.tensor([0], dtype=torch.int64)  # number of times there is a idx switch
        tps = torch.tensor([0.0], dtype=torch.float32)
        for re in results:
            tp_cardinality += re['tp'].sum()
            fp_cardinality += re['fp'].sum()
            #fn_cardinality += re['fn'].sum()
            m_cardinality += re['m'].sum()
            tps += re['iou'].sum()

            # number of idx switches
            crs = torch.split(re['correspondence'], 1, dim=1)
            for i in range(1, len(crs)):
                if i == 0:
                    continue
                else:
                    diff = crs[i] != crs[i-1]
                    ids_cardinality += diff.sum()

        motsa = ((tp_cardinality - fp_cardinality - ids_cardinality).float() / m_cardinality.float()).item()
        motsp = (tps / tp_cardinality.float()).item()
        smotsa = ((tps - fp_cardinality.float() - ids_cardinality.float()) / m_cardinality.float()).item()

        return {'MOTSA': motsa, 'MOTSP': motsp, 'sMOTSA': smotsa}



