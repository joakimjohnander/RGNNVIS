import time
import os
from threading import Thread
import math
import json
import torch
import torch.nn.functional as F

from rgnnvis.utils.tensor_utils import index_map_to_one_hot
from rgnnvis.utils.recursive_functions import recursive_to
from rgnnvis.engines.evaluation.video_od_evaluator import VPSODEvaluatorV2
from rgnnvis.engines.evaluation.motsa_evaluator import MOTSAEvaluatorV1, MOTSAEvaluatorV2
from rgnnvis.engines.evaluation.isap_evaluator import ISAPEvaluatorV1
from rgnnvis.engines.evaluation.visap_evaluator import VISAPEvaluatorV1
from rgnnvis.utils.visualization import save_vps_visualization_video, save_vps_visualization_video2
from rgnnvis.utils.tensor_utils import mask_to_coco_rle
from rgnnvis.utils.visualization import revert_imagenet_normalization

from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.io.video import write_video


def motsa_results(detections, sequence, num_classes, spatial_split_size):
    """ computes mosta results for each sequence
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        num_classes                (Int): number of classes
        spatial_split_size         (Int): split spatial dim. into chunks
    Returns:
        ret, see Args.
    """
    # Predictions
    pkeys = ['instance_segs', 'lbscores', 'active_objects']
    pred = {k: detections[k] for k in pkeys}

    # Annotations
    akeys = ['isannos', 'active', 'lbannos']
    anno = {k: sequence[k] for k in akeys}

    out = MOTSAEvaluatorV2.get_sequence_results(pred, anno, num_classes, spatial_split_size=spatial_split_size)
    out = recursive_to(out, 'cpu')

    return out


def motsa_results_summary(video_results, parameters):
    """ summarizes the motsa results into the final metrics
    Args:
        video_results     (List): of all video results
        parameters        (Dict): parameters for motsa_results function
    Returns:
        mot               (Dict): of motsa measures
    """
    motsar = []
    for vr in video_results:
        motsar += vr['motsa_results']

    motsa_measures = MOTSAEvaluatorV2.calculate_motsa(motsar, parameters['num_classes'])

    return motsa_measures

def visap_results(detections, sequence, iou_thresholds, num_classes, spatial_split_size, **kwargs):
    """ computes visap results for each sequence
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        iou_thresholds            (List): float values
        num_classes                (Int): number of classes
    Returns:
        ret, see Args.
    """
    # Predictions
    pkeys = ['active_objects', 'instance_segs', 'lbscores']
    pred = {k: detections[k] for k in pkeys}

    # Annotations
    akeys = ['isannos', 'lbannos', 'active']
    anno = {k: sequence[k].squeeze(0) for k in akeys}

    out = VISAPEvaluatorV1.get_sequence_results(pred, anno, spatial_split_size=spatial_split_size)
    out = recursive_to(out, 'cpu')

    return out


def visap_results_summary(video_results, parameters):
    """ summarizes the visap results into the final metrics
    Args:
        video_results     (List): of all video results
        parameters        (Dict): parameters for od_results function
    Returns:
        apm               (Dict): of AP measures
    """
    odr = []
    for vr in video_results:
        odr.append(VISAPEvaluatorV1.evaluate_video(vr['visap_results'], parameters['iou_thresholds'], parameters['num_classes'], use_old_TP_code=parameters.get('use_old_TP_code'), ignore_class=parameters.get('ignore_class')))

    apall, num_class_inst_dist = VISAPEvaluatorV1.calculate_AP(odr, parameters['iou_thresholds'], parameters['num_classes'])
    num_class_dist_mask = num_class_inst_dist > 0
    out = {'num_class_inst_dist': num_class_inst_dist.tolist()}
    out.update(dict.fromkeys(apall.keys()))

    mean_ap = 0.0
    num_thresholds = len(parameters['iou_thresholds'])
    for key, value in apall.items():
        ap = torch.mean(value[num_class_dist_mask]).item()
        mean_ap += ap
        out[key] = {'AP': ap, 'AP_per_class': value.tolist()}
    out.update({'mAP': mean_ap/num_thresholds})

    return out

def isap_results(detections, sequence, iou_thresholds, num_classes, spatial_split_size, **kwargs):
    """ computes isap results for each sequence
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        iou_thresholds            (List): float values
        num_classes                (Int): number of classes
    Returns:
        ret, see Args.
    """
    # Predictions
    pkeys = ['detection_active', 'detection_segs', 'detection_lbscores']
    pred = {k: detections[k] for k in pkeys}

    # Annotations
    akeys = ['isannos', 'lbannos', 'active']
    anno = {k: sequence[k].squeeze(0) for k in akeys}

    out = ISAPEvaluatorV1.get_sequence_results(pred, anno, iou_thresholds, spatial_split_size=spatial_split_size,
                                               use_old_TP_code=kwargs.get('use_old_TP_code'))
    out = recursive_to(out, 'cpu')

    return out


def isap_results_summary(video_results, parameters):
    """ summarizes the isap results into the final metrics
    Args:
        video_results     (List): of all video results
        parameters        (Dict): parameters for od_results function
    Returns:
        apm               (Dict): of AP measures
    """
    odr = []
    for vr in video_results:
        odr += vr['isap_results']

    apall, num_class_inst_dist = ISAPEvaluatorV1.calculate_AP(odr, parameters['iou_thresholds'], parameters['num_classes'])
    num_class_dist_mask = num_class_inst_dist > 0
    out = {'num_class_inst_dist': num_class_inst_dist.tolist()}
    out.update(dict.fromkeys(apall.keys()))

    mean_ap = 0.0
    num_thresholds = len(parameters['iou_thresholds'])
    for key, value in apall.items():
        ap = torch.mean(value[num_class_dist_mask]).item()
        mean_ap += ap
        out[key] = {'AP': ap, 'AP_per_class': value.tolist()}
    out.update({'mAP': mean_ap/num_thresholds})

    return out


def od_results(detections, sequence, iou_thresholds, num_classes, **kwargs):
    """ computes od results for each sequence
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        iou_thresholds            (List): float values
        num_classes                (Int): number of classes
    Returns:
        ret, see Args.
    """
    # Predictions
    pkeys = ['detection_active', 'detection_boxes', 'detection_lbscores']
    pred = {k: detections[k] for k in pkeys}

    # Annotations
    akeys = ['odannos', 'lbannos', 'active']
    anno = {k: sequence[k].squeeze(0) for k in akeys}

    out = VPSODEvaluatorV2.get_sequence_results(pred, anno, iou_thresholds,
                                                kwargs.get('use_old_TP_code'),
                                                kwargs.get('ignore_class'))
    out = recursive_to(out, 'cpu')

    return out


def od_results_summary(video_results, parameters):
    """ summarizes the od results into the final metrics
    Args:
        video_results     (List): of all video results
        parameters        (Dict): parameters for od_results function
    Returns:
        apm               (Dict): of AP measures
    """
    odr = []
    for vr in video_results:
        odr += vr['od_results']

    apall, num_class_inst_dist = VPSODEvaluatorV2.calculate_AP(odr, parameters['iou_thresholds'], parameters['num_classes'])
    num_class_dist_mask = num_class_inst_dist > 0
    out = {'num_class_inst_dist': num_class_inst_dist.tolist()}
    out.update(dict.fromkeys(apall.keys()))

    mean_ap = 0.0
    num_thresholds = len(parameters['iou_thresholds'])
    for key, value in apall.items():
        ap = torch.mean(value[num_class_dist_mask]).item()
        mean_ap += ap
        out[key] = {'AP': ap, 'AP_per_class': value.tolist()}
    out.update({'mAP': mean_ap/num_thresholds})

    return out


def association_json_save(video_results, filename):
    """ computes visap results for each sequence
    Args:
        video_results
        filename
    Returns:
        ret, see Args.
    """
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    seq = sum(video_results, [])
    with open(filename, 'w') as fp:
        json.dump(seq, fp)

def association_json_frame(detections, sequence, **kwargs):
    """ computes visap results for each sequence
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        kwargs
    Returns:
        ret, see Args.
    """

    n_digits = 5
    out = []

    active_tracks = detections['active_objects'][0, :, 1:]
    track_scores = detections['aux_vis_lbscores'][0, :, 1:].softmax(dim=2)

    for score, active in zip(track_scores, active_tracks):
        tracks = {}
        for track_idx, (tsc, tac) in enumerate(zip(score, active)):
            # tsc = (tsc * 10 ** n_digits).round() / (10 ** n_digits)
            tracks.update({track_idx: {'active': tac.item(), 'score': tsc.tolist()}})
        out.append(tracks)

    return out


def emibr12_detection_video_save(video_results, filename):
    """ save detections as a video
    Args:
        video_results     (List): of all video results
        filename
    Returns:
        None
    """

    # check filename path
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    image_sequence = torch.cat([torch.stack(split, dim=0) for split in video_results], dim=0)
    image_sequence = (image_sequence.permute(0, 2, 3, 1)*255).byte()
    write_video(filename, image_sequence, fps=6)


def emibr12_detection_video(detections, sequence, draw_anno, category_names, path):
    """ Compute a sequence of PIL images
    Args:
        detections                (Dict): of detections, (B,L,*)
        sequence                  (Dict): sequence data
        draw_anno
        category_names
        path
    Returns:
        sub_sequence         (PIL.Image): subset of sequence
    """
    out = []

    # annos
    images = revert_imagenet_normalization(sequence['images']).squeeze(0)
    anno_active = (sequence['active'] == 1)
    _, l, manno = anno_active.shape
    anno_active = anno_active.view(l, manno)
    anno_labels = sequence['lbannos'].view(1, manno).expand_as(anno_active) * (anno_active == 1)
    anno_boxes = sequence['odannos'].view(l, manno, 4)

    # predictions
    pred_active = detections['detection_active']
    _, _, mpred = pred_active.shape
    pred_active = (pred_active.view(l, mpred) == 1)
    pred_boxes = detections['detection_boxes'].view(l, mpred, 4)
    pred_lbscores = torch.softmax(detections['detection_lbscores'].view(l, mpred, -1), dim=2)
    pred_scores, pred_labels = pred_lbscores.max(dim=2)

    pred_masks = detections['detection_segs']
    #pred_masks = pred_masks.sigmoid().gt(0.5).float()
    #_, _, _, h1, w1 = pred_masks.shape
    pred_masks = (pred_masks.unsqueeze(2) == torch.arange(mpred, device=pred_masks.device).view(1, 1, mpred, 1, 1))
    # pred_masks = index_map_to_one_hot(pred_masks.unsqueeze(2), torch.tensor(range(mpred), dtype=torch.uint8), pred_masks.device)
    _, _, _, h1, w1 = pred_masks.shape
    pred_masks = pred_masks.view(l, mpred, h1, w1)

    for i, image in enumerate(images):
        image = to_pil_image((image * 255 + 0.5).cpu().byte())

        # draw anno boxes
        if draw_anno:
            active_idx = anno_active[i].nonzero().flatten().tolist()
            boxes = []
            labels = []
            for aidx in active_idx:
                boxes.append(anno_boxes[i, aidx].tolist())
                label = anno_labels[i, aidx].item()
                category_name = category_names[label]
                text = '{}, (anno)'.format(category_name)
                labels.append(text)

            image = draw_boxes(image, boxes, labels, color=(0, 0, 0), palette=None, thickness=2,
                               text_font=TEXT_FONT, text_size=10, overwrite=True)

        # draw pred.
        active_idx = pred_active[i].nonzero().flatten().tolist()
        boxes = []
        labels = []
        masks = []
        for aidx in active_idx:
            boxes.append(pred_boxes[i, aidx].tolist())
            label = pred_labels[i, aidx].item()
            score = pred_scores[i, aidx].item()
            category_name = 'background'
            if not label == 0:
                category_name = category_names[label]
            text = '{}, {:.02f}'.format(category_name, score)
            labels.append(text)
            #print(pred_masks[i, aidx].dtype, pred_masks[i, aidx].min(), pred_masks[i, aidx].max(), pred_masks[i, aidx].sum())
            mask = to_pil_image(pred_masks[i, aidx].cpu().byte()*255)
            masks.append(mask)

        image = draw_boxes_masks(image, boxes, masks, labels, color=None, palette=YTVOS_DAVIS_MIX_PALETTE, thickness=2,
                                 text_font=TEXT_FONT, text_size=13, overwrite=True)
        image = image.convert('RGB')

        image = to_tensor(image)
        out.append(image)

    return out


def save_detection_visualization(detections, sequence, path, version=1):
    track_visualization_part = detections['to_visualize']['detection'].copy() # Shallow copy of dict
    track_visualization_part['images'] = sequence['images']
    return track_visualization_part

def save_detection_visualization_video(track_visualization_parts, video_name, path, version=1):
    B, _, _, _, _ = track_visualization_parts[0]['images'].size()
    track_visualization = {
        'images': torch.cat([part['images'] for part in track_visualization_parts], dim=1),
        'seg': torch.cat([part['seg'] for part in track_visualization_parts], dim=1),
        'boxes': [[subpart for part in track_visualization_parts for subpart in part['boxes'][b]]
                  for b in range(B)],
        'boxlabels': [[subpart for part in track_visualization_parts for subpart in part['boxlabels'][b]]
                      for b in range(B)],
        'boxtexts': [[subpart for part in track_visualization_parts for subpart in part['boxtexts'][b]]
                     for b in range(B)] if track_visualization_parts[0].get('boxtexts') is not None else None,
    }
    fpath = f"{path}_det_{video_name}.mp4"
    if version == 1:
        save_vps_visualization_video(fpath, **track_visualization)
    elif version == 2:
        save_vps_visualization_video2(fpath, **track_visualization)


def save_track_visualization(detections, sequence, path, version=1):
    track_visualization_part = detections['to_visualize']['track'].copy() # Shallow copy of dict
    track_visualization_part['images'] = sequence['images']
    return track_visualization_part

def save_track_visualization_video(track_visualization_parts, video_name, path, version=1):
    B, _, _, _, _ = track_visualization_parts[0]['images'].size()
    track_visualization = {
        'images': torch.cat([part['images'] for part in track_visualization_parts], dim=1),
        'seg': torch.cat([part['seg'] for part in track_visualization_parts], dim=1),
        'boxes': [[subpart for part in track_visualization_parts for subpart in part['boxes'][b]]
                  for b in range(B)],
        'boxlabels': [[subpart for part in track_visualization_parts for subpart in part['boxlabels'][b]]
                      for b in range(B)],
        'boxtexts': [[subpart for part in track_visualization_parts for subpart in part['boxtexts'][b]]
                     for b in range(B)] if track_visualization_parts[0].get('boxtexts') is not None else None,
    }
    fpath = f"{path}_track_{video_name}.mp4"
    if version == 1:
        save_vps_visualization_video(fpath, **track_visualization)
    elif version == 2:
        save_vps_visualization_video2(fpath, **track_visualization)


def save_predictions(detections, sequence, path, save_method):
    """ save predictions according to save_method
    Args:
        detections                (List): of detections
        sequence                  (Dict): sequence data
        path                       (Str): base path for dumping
        save_method           (function): function pointer to the method used for save
    Returns:
        ret, see Args.
    """
    save_path = os.path.join(path, *sequence['identifier'])
    sout = save_method(detections, sequence, save_path)

    return sout


def save_predictions_ytvis(model_output, video_chunk, fpath):
    """
    Returns:
        list<dict>: M dicts each being a prediction for one object on one chunk
    """
    B, M, C = model_output['lbscores'].size()
    _, L, H, W = model_output['instance_segs'].size()
    category_ids = model_output['lbscores'][:, :, 1:].argmax(dim=2).view(M).cpu() + 1 # We map 0-39 to 1-40
    segmentations = model_output['instance_segs'].view(L, H, W)
#    scores = (model_output['lbscores'][:, :, 1:].logsumexp(dim=2) - model_output['lbscores'][:, :, 0]).view(M).cpu()
#    scores = (model_output['lbscores'][:, :, 1:].logsumexp(dim=2) - model_output['lbscores'].logsumexp(dim=2)).view(M).cpu()
    scores = (model_output['lbscores'][:, :, 1:].max(dim=2)[0] - model_output['lbscores'].logsumexp(dim=2)).view(M).cpu()
#    active = (model_output['active_objects'][:, -1] != 0).view(M).cpu()
    active = ((model_output['active_objects'][:, -1] == 1) + (model_output['active_objects'][:, -1] == 2)).view(M).cpu() # This seems beneficial (it should be, we should exclude background)
#    active = (model_output['active_objects'][:, -1] == 1).view(M).cpu() # This is how we have done it for minival
    result = []
    for m in range(1, M - 1):
        result.append({
            "video_id": video_chunk['video_id'].item(),
            "category_id": category_ids[m].item(),
            "segmentations": [mask_to_coco_rle(segmentations[l] == m) if active[m] else [H*W]
                              for l in range(L)],
            "score": scores[m].item(),
            'active': active[m],
        })
    return result

def save_predictions_ytvis_video(subresult_lst, fpath):
    """
    Args:
        subresult_lst (list<list<dict>>): num_chunks length list, each element (up to) an M length
            list, each element a prediction for one object on one chunk
    Returns:
        list<dict>: (up to) M length list, each element a prediction for object M
    """
    results = []
    M = len(subresult_lst[0])
    for m in range(M):
        if subresult_lst[-1][m]['active']:
            results.append({
                "video_id": subresult_lst[-1][m]["video_id"],
                "category_id": subresult_lst[-1][m]["category_id"],
                "segmentations": [seg for subresult in subresult_lst for seg in subresult[m]['segmentations']],
                "score": subresult_lst[-1][m]["score"],
            })
#            print("Added video {}, cat {}, with {} segmentations and score {}".format(results[-1]['video_id'], results[-1]['category_id'], len(results[-1]['segmentations']), results[-1]['scores']))
    return results

def save_predictions_ytvis_summary(video_result_lst, fpath):
    result = [elem for video_result in video_result_lst for elem in video_result]
#    print([(elem["video_id"], elem["category_id"]) for elem in result])
    print(result[0])
#    print(result[10])
#    print(result[20])
#    print(result[30])
    print(result[40])
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    with open(fpath, 'w') as fp:
        json.dump(result, fp)
    return None


def split_video(video, split_size):
    """ evaluate model on dataset according to parameters
    Args:
        video             (Dict): follows common architecture of dataloaders (see ytvis)
        split_size         (Int): split in chunks of split_size
    Returns:
        seqs         (List:Dict): returns a list with dict following the common architecture of dataloaders (see ytvis)
    """
    if video.get('lbannos') is None:
        video_length = video['images'].size(0)
        num_chunks = math.ceil(video_length / split_size)
        out = [{'images': video['images'][i : i + split_size],
                'identifier': video['identifier'],
                'video_id': video['video_id']} for i in range(num_chunks)]
        return out
    valid_keys = ['images', 'ssannos', 'isannos', 'odannos', 'active',
                  'provides_ss', 'provides_is', 'provides_od', 'provides_lb']
    splited = dict.fromkeys(valid_keys)
    sl = []
    for key, value in video.items():
        if key in valid_keys:
            splited[key] = torch.split(value, split_size, dim=1)  # Split over L
            sl.append(len(splited[key]))
    assert len(set(sl)) == 1

    out = []
    for i in range(sl[0]):
        emp = dict.fromkeys(valid_keys)
        emp.update({'split_size': split_size})
        emp.update({'split_idx': i})
        emp.update({'lbannos': video['lbannos']})
        emp.update({'identifier': video['identifier']})
        for key, value in splited.items():
            emp[key] = value[i]
        out.append(emp)

    return out


class VISEvaluator:
    def __init__(self, device):
        self.device = device

        # measure fps
        self._total_num_of_frames = None
        self._accumulated_time = None
        self._cs = torch.cuda.Stream(self.device)
        self._cevent_start = torch.cuda.Event(enable_timing=True)
        self._cevent_stop = torch.cuda.Event(enable_timing=True)


    def __call__(self, model, dataset, parameters, args):
        return self.evaluate(model, dataset, parameters, args)

    def evaluate(self, model, dataset, parameters, args):
        """ evaluate model on dataset according to parameters
        Args:
            model                     (torch.nn.Module): model
            dataset          (torch.utils.data.Dataset): dataset
            parameters                           (Dict): parameters
        Returns:
            see code
        """

        # If you forgot
        model.to(self.device)
        model.eval()

        # fps reset for new dataset
        self._total_num_of_frames = 0
        self._accumulated_time = 0.0

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        out = dict.fromkeys(parameters.keys(), None)
        video_results = []  # Results of all videos
        args.update({'video_idx': [0, len(dataset)]})
        with torch.no_grad():

            for video_idx, video in enumerate(dataloader): #video_idx in range(len(dataset)):
#                if 10 <= video_idx:
#                    break
#                if 'eaa99191cb' in video['identifier'][1][0]:
#                    print('found')
#                else:
#                    print(video_idx, *video['identifier'])
#                    continue
                args['video_idx'][0] = video_idx+1
                out_tmp = self.evaluate_video(model, video, parameters, args)
                video_results.append(out_tmp)

            # Calculate OD AP measures
            if 'od_results' in out.keys():
                out['od_results'] = od_results_summary(video_results, parameters['od_results'])

            # Calculate IS AP measures
            if 'isap_results' in out.keys():
                out['isap_results'] = isap_results_summary(video_results, parameters['isap_results'])

            # Calculate VIS AP measures
            if 'visap_results' in out.keys():
                out['visap_results'] = visap_results_summary(video_results, parameters['visap_results'])

            # Calculate MOTSA measures
            if 'motsa_results' in out.keys():
                out['motsa_results'] = motsa_results_summary(video_results, parameters['motsa_results'])

            # Store YTVIS json
            if 'save_predictions_ytvis' in out.keys():
                save_predictions_ytvis_summary([elem['save_predictions_ytvis'] for elem in video_results],
                                               **parameters['save_predictions_ytvis'])

        # extra output, fps
        fps = self._total_num_of_frames/self._accumulated_time
        out.update({'fps': fps})

        return out

    def evaluate_video(self, model, video, parameters, args):
        # Returns a list of length L. Each frame in the list contains a dict with the keys class, score, box and mask.
        # Each key stores a value of torch.Tensor with n detections in the first dimension.
        print('Evaluating  {}/{}, video {:04d}/{:04d}'.format(*video['identifier'], *args['video_idx']))
        results = {key: [] for key in parameters.keys()}
        video = recursive_to(video, self.device)
        video_splited = split_video(video, args['split_size'])
        model_state = None
        for seq in video_splited:
            out, model_state = self.evaluate_sequence(model, seq, parameters, args, model_state)
            for key, value in out.items():
                results[key] += value

        if 'association_json_frame' in out.keys():
            video_name = '{}_{}_assdata.json'.format(video['identifier'][0][0], video['identifier'][1][0])
            filename = os.path.join(parameters['association_json_frame']['path'], video_name)
            association_json_save(results['association_json_frame'], filename)
            results.pop('association_json_frame')

        if 'emibr12_detection_video' in out.keys():
            video_name = '{}_{}_detection.mp4'.format(video['identifier'][0][0], video['identifier'][1][0])
            filename = os.path.join(parameters['emibr12_detection_video']['path'], video_name)
            emibr12_detection_video_save(results['emibr12_detection_video'], filename)
            results.pop('emibr12_detection_video')

        if 'save_predictions_ytvis' in out.keys():
            results['save_predictions_ytvis'] = save_predictions_ytvis_video(
                results['save_predictions_ytvis'],
                **parameters['save_predictions_ytvis'])
        if 'save_track_visualization' in out.keys():
            # Why is idf list<list<str>> and not list<str> or str?
            video_idf_str = video['identifier'][0][0] + "_" + video['identifier'][1][0]
            save_track_visualization_video(results['save_track_visualization'],
                                           video_idf_str,
                                           **parameters['save_track_visualization'])
            results.pop('save_track_visualization')
        if 'save_detection_visualization' in out.keys():
            # Why is idf list<list<str>> and not list<str> or str?
            video_idf_str = video['identifier'][0][0] + "_" + video['identifier'][1][0]
            save_detection_visualization_video(results['save_detection_visualization'],
                                               video_idf_str,
                                               **parameters['save_detection_visualization'])
            results.pop('save_detection_visualization')

        return results

    def evaluate_sequence(self, model, sequence, parameters, args, model_state=None):
        self._cevent_start.record(stream=self._cs)
        detections, model_state = model(state=model_state, **sequence)
        self._cevent_stop.record(stream=self._cs)

        # Waits for everything to finish running
        torch.cuda.synchronize(device=self.device)

        self._accumulated_time += self._cevent_start.elapsed_time(self._cevent_stop)/1000.0
        self._total_num_of_frames += sequence['images'].shape[1]
        # print(self._cevent_start.elapsed_time(self._cevent_stop))

        # What to do with the detections is defined in parameters
        func_ret = {key: [] for key in parameters.keys()}  # Do not use dict.fromkeys!
        for func, fargs in parameters.items():
            if func in globals():
                # print('running {}'.format(func))
                ret = globals()[func](detections, sequence, **fargs)
                func_ret[func].append(ret)

        return func_ret, model_state

