
import os
import sys
import argparse
import datetime
import time
import json
import random
from collections import OrderedDict

import torch
torch.set_printoptions(edgeitems=4, linewidth=117)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
from PIL import Image

import rgnnvis.models.rgnn as rgnn
import rgnnvis.models.custom_modules as cm
import rgnnvis.models.yolact.yolact
from rgnnvis.models.yolact.yolact import Yolact
from rgnnvis.models.losses import ComposeObjective, VISObjective
from rgnnvis.datasets.ytvis import YTVIS
from rgnnvis.datasets.dataset_utils import IMAGENET_MEAN, IMAGENET_STD, LabelToLongTensor, MultitaskWeightedRandomSampler, ConcatDatasetWithIdx
from rgnnvis.datasets.dataset_transforms import ComposeAugmentations, ScaleCropAugment, HorizontalFlipAugment
from rgnnvis.engines.vis_evaluator import VISEvaluator
from rgnnvis.engines.vis_trainer import VISTrainer
from rgnnvis.utils.debugging import debug_plot, get_model_size_str
from config import config


# Detector
from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter


##################
# Training
##################
def train(model):
    #actual_validation_data = YTVIS(config['ytvis_path'], split='valid', stride=5, L=32, start_frame='first')
    validation_data = YTVIS(config['ytvis_path'], split='train', stride=5, L=10, start_frame='first')
    validation_data = validation_data.get_subsplit(video_idx_threshold=2019)[1]
    print(f"Using {len(validation_data)} validation sequences")
    val_loader = DataLoader(validation_data, shuffle=False, batch_size=8, num_workers=11)

    # We have already trained the detector, and train only the associator
    parameters = [{'params': [param for name, param in model.named_parameters()
                              if (param.requires_grad
                                  and name[:8]  != 'backbone'
                                  and name[:8]  != 'detector')]},
    ]
    optimizer = optim.Adam(parameters, lr=2e-4, weight_decay=1e-4)
    lr_sched = optim.lr_scheduler.LambdaLR(optimizer, (lambda epoch: 1.0))
    objective = ComposeObjective({
        'vps': VISObjective(
            detection_iou_threshold = 0.5,
            trkdetass = {'normalization': 'BL', 'weight': 4.0, 'secondary_match_weight': 1.0,
                         'detection_match_mode': 'primary'},
            newass = {'normalization': 'BL', 'weight': 1.0},
            trkcls = {'normalization': 0.285 * 0.8 ** torch.arange(10, 0, -1, dtype=torch.float, device='cuda'),
                      'weight': 1.0},
            trkseg = {'mode': 'Lovasz per image', 'weight': 1.0},
            weight = 1.5,
            keys = {
                'pred_boxes'                      : 'detection_boxes',
                'pred_track_active'               : 'track_active',
                'track_initializers'              : 'track_initializers',
                'pred_detection_active'           : 'detection_active',
                'pred_track_detection_assignments': 'track_detection_assignments',
                'pred_instance_segscores'         : 'isscore',
                'pred_instance_lbscores'          : 'lbscores',
                'aux_vis_lbscores'                : 'aux_vis_lbscores',
                'anno_boxes' : 'odannos',
                'anno_active': 'active',
                'anno_instance_segmentation': 'isannos',
                'anno_labels': 'lbannos',
            },
        )
    })
    training_data = YTVIS(config['ytvis_path'],
                          split         = 'train',
                          stride        = 5,
                          L             = 10,
                          start_frame   = 'random',
                          imsize        = None,
                          augmentations = ComposeAugmentations(
                              ScaleCropAugment(
                                  get_scale_factors = (lambda: 0.475 + 0.525 * torch.rand(1).expand(2)),
                                  imsize            = (480, 864)),
                              HorizontalFlipAugment()))
    training_data = training_data.get_subsplit(video_idx_threshold=2019)[0]
    print(f"Using {len(training_data)} training sequences")
    train_loader = DataLoader(training_data, shuffle=True, batch_size=4, num_workers=11, drop_last=True)
    trainer = VISTrainer(model          = model,
                         optimizer      = optimizer,
                         data_transform = None,
                         objective      = objective,
                         lr_sched       = lr_sched,
                         train_loader   = train_loader,
                         val_loaders    = [val_loader],
                         checkpoint_path              = config['checkpoint_path'],
                         save_name                    = os.path.splitext(os.path.basename(__file__))[0],
                         device                       = 'cuda',
                         checkpoint_epochs            = [50, 75, 100, 150, 200],
                         print_interval               = 25,
                         visualization_epochs         = [2, 10, 50],
                         visualization_loss_threshold = 100.,
                         gradient_clip_value          = None, #1.,
                         bptt_chunk_length            = 1000,
                         print_param_on_training_end  = True)
    trainer.load_checkpoint()
    trainer.train(150)


##################
# Testing
##################
def custom_print_dict(data, indent):
    def print_rec(data_structure, ind):
        idx = 0
        for key, value in data_structure.items():
            if isinstance(value, dict):
                print('{}"{}": {}'.format(' '*ind, str(key), '{'))
                print_rec(value, ind+indent)
                if idx < len(data_structure)-1:
                    print('{}{}'.format(' ' * ind, '},'))
                else:
                    print('{}{}'.format(' ' * ind, '}'))
            else:
                if idx < len(data_structure)-1:
                    sv = '{},'.format(str(value))
                else:
                    sv = str(value)
                print('{}"{}": {}'.format(' ' * ind, str(key), sv))
            idx += 1
    print('{')
    print_rec(data, indent)
    print('}')

def test_model(model, result_fpath=None, debug_sequences_dict=None,
               save_predictions=False, device=torch.device('cuda'),
               ignore_class=False):
    ytvis_dataset = YTVIS(root_path=config['ytvis_path'], split='train',
                          stride=5, L=1, imsize=(720, 1280), start_frame='first',
                          data_purpose='evaluation')
    ytvis_tvalid = ytvis_dataset.get_subsplit(video_idx_threshold=2019)[1]
    num_classes = 41
    ap_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95]

    datasets = {
        'ytvis (train)': {
            'data': ytvis_tvalid,

            # func. name         params
            'parameters': {
                'save_track_visualization': {
                    'path': os.path.join(config['tempinfo_path'],
                                         os.path.splitext(os.path.basename(__file__))[0],
                                         os.path.splitext(os.path.basename(__file__))[0]),
                    'version': 2,
                },
                'save_detection_visualization': {
                    'path': os.path.join(config['tempinfo_path'],
                                         os.path.splitext(os.path.basename(__file__))[0],
                                         os.path.splitext(os.path.basename(__file__))[0]),
                    'version': 2,
                },
                # 'save_predictions': {'path': os.path.join(config['tempinfo_path'], args.filestart_id),
                #                     'save_method': vps_debug_plot},
                'od_results'   : {'iou_thresholds': ap_thresholds,
                                  'num_classes': num_classes,
                                  'ignore_class': ignore_class},
                'isap_results' : {'iou_thresholds': ap_thresholds,
                                  'num_classes': num_classes,
                                  'spatial_split_size': 100000},
                'visap_results': {'iou_thresholds': ap_thresholds,
                                  'num_classes': num_classes,
                                  'spatial_split_size': 100000,
                                  'ignore_class': ignore_class},
                'motsa_results': {'num_classes': num_classes,
                                  'spatial_split_size': 100000},
                'association_json_frame': {
                    'path': os.path.join(config['tempinfo_path'],
                                         os.path.splitext(os.path.basename(__file__))[0],
                                         'val_trackass_json')}
            }
        }
    }

    # Initialize evaluator
    evaluator = VISEvaluator(device)

    # Run evaluation for all datasets
    since = time.time()
    arguments = {'verbose': True, 'split_size': 32}
    for dataset_name, setup in datasets.items():
        eval_res = evaluator(model, setup['data'], setup['parameters'], arguments)
        print('Dataset: {}'.format(dataset_name))
#        print(json.dumps(eval_res, indent=4))
        custom_print_dict(eval_res, indent=4)
        print(f'Brief result: {eval_res["od_results"][50]["AP"]:.3f}/{eval_res["od_results"][75]["AP"]:.3f}')

    time_elapsed = time.time() - since
    time_elapsed = str(datetime.timedelta(seconds=time_elapsed))
    print('Evaluation complete in {}'.format(time_elapsed))


def ytvis_test_model(model, result_fpath=None, device=torch.device('cuda')):
    ytvis_valid_data = YTVIS(config['ytvis_path'], split='valid', stride=5, L=1, imsize=(720, 1280),
                                   start_frame='first', data_purpose='evaluation')

    datasets = {
        'ytvis (valid)': {
            'data': ytvis_valid_data,
            'parameters': {
                'save_track_visualization': {
                    'path': os.path.join(config['tempinfo_path'],
                                         os.path.splitext(os.path.basename(__file__))[0],
                                         os.path.splitext(os.path.basename(__file__))[0]),
                    'version': 2,
                },
                'save_detection_visualization': {
                    'path': os.path.join(config['tempinfo_path'],
                                         os.path.splitext(os.path.basename(__file__))[0],
                                         os.path.splitext(os.path.basename(__file__))[0]),
                    'version': 2,
                },
                'save_predictions_ytvis': {
                    'fpath': os.path.join(result_fpath, 'results.json'),
                },
                'association_json_frame': {
                    'path': os.path.join(config['tempinfo_path'],
                                         os.path.splitext(os.path.basename(__file__))[0],
                                         'test_trackass_json')}
            }
        }
    }

    # Initialize evaluator
    evaluator = VISEvaluator(device)

    # Run evaluation for all datasets
    since = time.time()
    arguments = {'verbose': True, 'split_size': 6}
    for dataset_name, setup in datasets.items():
        eval_res = evaluator(model, setup['data'], setup['parameters'], arguments)
        print('Dataset: {}'.format(dataset_name))
        custom_print_dict(eval_res, indent=4)
    time_elapsed = time.time() - since
    time_elapsed = str(datetime.timedelta(seconds=time_elapsed))
    print('Evaluation complete in {}'.format(time_elapsed))


def run_image_folder_demo(model, device=torch.device('cuda')):
    transform = tv.transforms.Compose([
        tv.transforms.Resize((480, 864)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = tv.datasets.ImageFolder(args.image_folder_demo[0], transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=12, num_workers=11)
    for idx, data in enumerate(loader):
        images = data[0].to(device).unsqueeze(1) # (B, L=1, D, H=480, W=864)
        with torch.no_grad():
            model_output, model_state = model(images)
            pred = model_output['to_visualize']['detection']
            for b in range(images.size(0)):
                pred_fname = f"demo_idx{idx:03d}_b{b:03d}.png"
                debug_plot(image=images[b,0], seg=pred['seg'][b,0], boxes=pred['boxes'][b][0],
                           boxlabels=pred['boxlabels'][b][0], fname=pred_fname)


##################
# Main block
##################
def torch_init(to_device):
    """ Setup torch
    Args:
        to_device                 (Str): device name eg. cuda:0 or cpu
    Returns:
        cuda_avail            (Boolean): if cuda is available
        device           (torch.Device): pointer to device
    """
    cuda_avail = torch.cuda.is_available()
#    torch.cuda.manual_seed(0) # Not sure that we want this!
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cpu")
    if cuda_avail and 'cuda' in to_device:
        device = torch.device(to_device)
        torch.cuda.set_device(device)

    return cuda_avail, device


def list_cuda_devices():
    """ list cuda devices
    Args:
        None
    Returns:
        None
    """
    if torch.cuda.is_available():
        print("cuda available")
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            print("    cuda:" + str(i), torch.cuda.get_device_name(i))
    else:
        print("cuda not available")


def main(args):
    print("Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__))

    # List devices
    list_cuda_devices()

    # Init torch
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)

#    torch.cuda.manual_seed(0) # Not sure whether we should manual_seed, should prolly seed other stuffs as well then
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Yolact-550 ResNet-50 with FPN #################################################################################
#    yolact_backbone_sd = torch.load(os.path.join(config['nn_weights_path'],
#                                                 'yolact',
#                                                 'yolact_resnet50_54_800000_backbone.pth'))
#    yolact_base_sd = torch.load(os.path.join(config['nn_weights_path'],
#                                             'yolact',
#                                             'yolact_resnet50_54_800000_base.pth'))

    yolact_bb_return_layers = {'layer2': 'bb_feats_s4', 'layer3': 'bb_feats_s8', 'layer4': 'bb_feats_s16'}
    backbone = IntermediateLayerGetter(resnet50(pretrained=False), yolact_bb_return_layers)
#    backbone.load_state_dict(yolact_backbone_sd, strict=True)

    backbone = rgnn.SequentialWithState(
        {'conv1'  : backbone['conv1'],
         'bn1'    : backbone['bn1'],
         'relu'   : backbone['relu'],
         'maxpool': backbone['maxpool'],
         'layer1' : backbone['layer1'],
         'layer2' : backbone['layer2'],
         'layer3' : backbone['layer3'],
         'layer4' : backbone['layer4']},
        {'layer2': 'bb_feats_s4',
         'layer3': 'bb_feats_s8',
         'layer4': 'bb_feats_s16'})
    
    detector = Yolact(num_classes=41)
#    detector.load_state_dict_without_specific_classes(yolact_base_sd)

    saved_weights = torch.load(os.path.join(config['nn_weights_path'], 'vps063_v4_ep0120.pth.tar'))['net']
    backbone_weights = {key[9:]: val for key, val in saved_weights.items() if key[:9] == 'backbone.'}
    detector_weights = {key[9:]: val for key, val in saved_weights.items() if key[:9] == 'detector.'}
    backbone.load_state_dict(backbone_weights)
    detector.load_state_dict(detector_weights)

    model = rgnn.RGNN(
        backbone                       = backbone,
        detector                       = detector,
        appnet                         = rgnn.AppNet(nn.ModuleDict({
            'bb_feats_s8': nn.Sequential(
                nn.Conv2d(1024, 256, 1, 1, 0),),
            'bb_feats_s16': nn.Sequential(
                nn.Conv2d(2048, 256, 1, 1, 0),),
        })),
        detection_descriptor_extractor = rgnn.DetectionDescriptorExtractor(
            appearance_layers = cm.Residual(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 512),
            ),
        ),
        track_detection_matcher        = rgnn.TrackDetectionMatcher(
            layers = rgnn.MatcherSequential([
                rgnn.GraphNet(dict(
                    bkgdet = rgnn.BkgDetBase({
                        'bkg_in': nn.Identity(),
                        'det_in': nn.Identity(),
                        'bkgdet_in': nn.Identity(),
                        'out': nn.Sequential(
                            nn.Linear(41 + 4 + 1 + 128, 128),
                            nn.ReLU(inplace=True),
                            cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))),
                    }),
                    trkdet = rgnn.TrkDetBase({
                        'trk_in': nn.Identity(),
                        'det_in': nn.Identity(),
                        'trkdet_in': nn.Identity(),
                        'out': nn.Sequential(
                            nn.Linear(41 + 4 + 2 + 128, 128),
                            nn.ReLU(inplace=True),
                            cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))),
                    }),
                    bkg = rgnn.BkgFanatic({
                        'bkgdet_in': nn.Identity(),
                        'bkgdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'bkg_in': nn.Identity(),
                        'residual': None,
                        'out': nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
                    }),
                    trk = rgnn.TrkFanatic({
                        'trkdet_in': nn.Identity(),
                        'trkdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'trk_in': nn.Identity(),
                        'residual': None,
                        'out': nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
                    }),
                    det = rgnn.DetFanatic({
                        'bkgdet_in': nn.Identity(),
                        'bkgdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'trkdet_in': nn.Identity(),
                        'trkdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'det_in': nn.Identity(),
                        'residual': None,
                        'out': nn.Sequential(nn.Linear(41 + 4 + 128, 128), nn.ReLU(inplace=True))
                    }),
                )),
                rgnn.GraphNet(dict(
                    bkgdet = rgnn.BkgDetBase({
                        'bkg_in': None,
                        'det_in': None,
                        'bkgdet_in': nn.Identity(),
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128)),
                    }),
                    trkdet = rgnn.TrkDetBase({
                        'trk_in': None,
                        'det_in': None,
                        'trkdet_in': nn.Identity(),
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128)),
                    }),
                    bkg = rgnn.BkgFanatic({
                        'bkgdet_in': None,
                        'bkgdet_sigmoid': None,
                        'bkg_in': nn.Identity(),
                        'residual': None,
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))
                    }),
                    trk = rgnn.TrkFanatic({
                        'trkdet_in': None,
                        'trkdet_sigmoid': None,
                        'trk_in': nn.Identity(),
                        'residual': None,
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))
                    }),
                    det = rgnn.DetFanatic({
                        'bkgdet_in': None,
                        'bkgdet_sigmoid': None,
                        'trkdet_in': None,
                        'trkdet_sigmoid': None,
                        'det_in': nn.Identity(),
                        'residual': None,
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))
                    }),
                )),
                rgnn.GraphNet(dict(
                    bkgdet = rgnn.BkgDetBase({
                        'bkg_in': nn.Identity(),
                        'det_in': nn.Identity(),
                        'bkgdet_in': nn.Identity(),
                        'out': nn.Sequential(nn.Linear(384, 128), nn.ReLU(inplace=True)),
                    }),
                    trkdet = rgnn.TrkDetBase({
                        'trk_in': nn.Identity(),
                        'det_in': nn.Identity(),
                        'trkdet_in': nn.Identity(),
                        'out': nn.Sequential(nn.Linear(384, 128), nn.ReLU(inplace=True)),
                    }),
                    bkg = rgnn.BkgFanatic({
                        'bkgdet_in': nn.Identity(),
                        'bkgdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'bkg_in': nn.Identity(),
                        'residual': None,
                        'out': nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
                    }),
                    trk = rgnn.TrkFanatic({
                        'trkdet_in': nn.Identity(),
                        'trkdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'trk_in': nn.Identity(),
                        'residual': None,
                        'out': nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
                    }),
                    det = rgnn.DetFanatic({
                        'bkgdet_in': nn.Identity(),
                        'bkgdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'trkdet_in': nn.Identity(),
                        'trkdet_sigmoid': nn.Sequential(nn.Linear(128,32), nn.ReLU(inplace=True), nn.Linear(32,128)),
                        'det_in': nn.Identity(),
                        'residual': None,
                        'out': nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
                    }),
                )),
                rgnn.GraphNet(dict(
                    bkgdet = rgnn.BkgDetBase({
                        'bkg_in': None,
                        'det_in': None,
                        'bkgdet_in': nn.Identity(),
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128)),
                    }),
                    trkdet = rgnn.TrkDetBase({
                        'trk_in': None,
                        'det_in': None,
                        'trkdet_in': nn.Identity(),
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128)),
                    }),
                    bkg = rgnn.BkgFanatic({
                        'bkgdet_in': None,
                        'bkgdet_sigmoid': None,
                        'bkg_in': nn.Identity(),
                        'residual': None,
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))
                    }),
                    trk = rgnn.TrkFanatic({
                        'trkdet_in': None,
                        'trkdet_sigmoid': None,
                        'trk_in': nn.Identity(),
                        'residual': None,
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))
                    }),
                    det = rgnn.DetFanatic({
                        'bkgdet_in': None,
                        'bkgdet_sigmoid': None,
                        'trkdet_in': None,
                        'trkdet_sigmoid': None,
                        'det_in': nn.Identity(),
                        'residual': None,
                        'out': cm.Residual(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 128))
                    }),
                )),
            ]),
            score_layers = rgnn.MatcherScoreLayerEdges({
                'trkdet': nn.Linear(128, 1),
                'bkgdet': nn.Linear(128, 1),
            }),
            bkg_stuff = ('embedding',),
            det_stuff = ('lbscores', 'boxes',),
            trk_stuff = ('embedding',),
            trkdet_stuff=('appearance gaussian loglikelihood', 'box iou'),
            bkgdet_stuff=('appearance gaussian loglikelihood'),
        ),
        track_descriptor               = rgnn.TrackDescriptor(
            memory_layers = nn.ModuleDict({
                'tracks': nn.ModuleDict({
                    'x': nn.Linear(128, 4*128),
                    'prev_out': nn.Linear(128, 4*128),
                    'output_naf': nn.Tanh(),
                }),
                'background': nn.ModuleDict({
                    'x': nn.Linear(128, 4*128),
                    'prev_out': nn.Linear(128, 4*128),
                    'output_naf': nn.Tanh(),
                }),
            }),
            Dmem = 128,
            out_layers  = nn.ModuleDict({
                'lbscores': nn.Linear(128, 41),
                'mean_eta': nn.Linear(128, 1),
                'var_eta': nn.Linear(128, 1),
                'var_new': nn.Linear(128, 1),
            }),
            Dapp                = 512,
            init_var            = torch.full((512,), -7.),
            init_var_background = torch.full((512,), -7.),
        ),
        hard_assigner                  = rgnn.HardAssigner(
            novel_acceptance_threshold = -2.0,
        ),
        track_module                   = rgnn.TrackModule(
            embedding_to_seg_layers = nn.Sequential(
                nn.Linear(128, 16),
                nn.ReLU(inplace=True)),
            seg_layers               = nn.Sequential(
                nn.Conv2d(18, 16, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, 1, 1)),
            use_raw_plus_box_segscore = False,
        ),
        raw_det_threshold = -3.5,
        nms_iou_threshold = 0.7,
        box_to_seg_weight = 10.0,
        bg_segscore       = 0.0,
        max_num_detections = 16,
        max_num_tracks     = 24,
        num_classes       = 41,
        num_maskcoeffs    = 32,
        freeze_detector_condition = (lambda frame_idx: True),
        freeze_batchnorm          = True,
        track_lbscore_mode = 'from track',
        detector_type      = 'yolact',
        backbone_droprate = 0.1,
        debug_mode      = args.debug,
    )
    for module in model.track_descriptor.out_layers['var_new'].modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, val=-7.0)
    nn.init.xavier_uniform_(model.track_detection_matcher.score_layers['trkdet'].weight, gain=1.0)
    nn.init.constant_(model.track_detection_matcher.score_layers['trkdet'].bias, val=-2.1)
    nn.init.xavier_uniform_(model.track_detection_matcher.score_layers['bkgdet'].weight, gain=1.0)
    nn.init.constant_(model.track_detection_matcher.score_layers['bkgdet'].bias, val=-2.1)
    model = model.to(device)
    print("Network model {} loaded, (size: {})".format(model.__class__.__name__, get_model_size_str(model)))

    runfile_name = os.path.splitext(os.path.basename(__file__))[0]
    # checkpoint_ids = [50, 75, 100, 150]
    checkpoint_ids = [150]
    if args.last:
        checkpoint_ids = checkpoint_ids[-1:]
    if args.train:
        model.train()
        train(model)
    if args.test is not None:
        model.eval()
        model.always_visualize = True
        model.visualization_box_text = 'version 2'
        for idx in checkpoint_ids:
            checkpoint_name = '{}_ep{:04d}.pth.tar'.format(runfile_name, idx)
            print('Loading checkpoint {}'.format(checkpoint_name))
            file_path = os.path.join(config['checkpoint_path'], checkpoint_name)
            model.load_state_dict(torch.load(file_path)['net'])
            result_fpath = os.path.join(config['output_path'], '{}_ep{:04d}'.format(runfile_name, idx))
            test_model(model, result_fpath=result_fpath, device=device)
    if args.image_folder_demo is not None:
        model.eval()
        model.always_visualize = True
        model.visualization_box_text = 'version 2'
        checkpoint_name = '{}_ep{:04d}.pth.tar'.format(runfile_name, checkpoint_ids[-1])
        print('Loading checkpoint {}'.format(checkpoint_name))
        file_path = os.path.join(config['checkpoint_path'], checkpoint_name)
        model.load_state_dict(torch.load(file_path)['net'])
        run_image_folder_demo(model, device=device)
    if args.ytvis_test:
        model.eval()
        model.always_visualize = True
        model.visualization_box_text = 'version 2'
        checkpoint_name = '{}_ep{:04d}.pth.tar'.format(runfile_name, checkpoint_ids[-1])
        print('Loading checkpoint {}'.format(checkpoint_name))
        file_path = os.path.join(config['checkpoint_path'], checkpoint_name)
        model.load_state_dict(torch.load(file_path)['net'])
        result_fpath = os.path.join(config['output_path'], '{}_ep{:04d}'.format(runfile_name, checkpoint_ids[-1]))
        ytvis_test_model(model, result_fpath=result_fpath, device=device)
    if args.test_last_noclass:
        model.eval()
        model.always_visualize = True
        model.visualization_box_text = 'version 2'
        checkpoint_name = '{}_ep{:04d}.pth.tar'.format(runfile_name, checkpoint_ids[-1])
        print('Loading checkpoint {}'.format(checkpoint_name))
        file_path = os.path.join(config['checkpoint_path'], checkpoint_name)
        model.load_state_dict(torch.load(file_path)['net'])
        test_model(model, device=device, ignore_class=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--last", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store", nargs="*")
    parser.add_argument("--image_folder_demo", nargs=1)
    parser.add_argument("--ytvis_test", action="store_true", default=False)
    parser.add_argument("--test_last_noclass", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.",
                        type=str, default="cuda:0")
    args = parser.parse_args()

    args.fstart_id = '{dt}_{filename}'.format(dt=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), filename=os.path.splitext(os.path.basename(__file__))[0])
    
    main(args)
