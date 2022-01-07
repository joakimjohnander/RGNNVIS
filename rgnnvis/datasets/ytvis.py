import os
import json
import random
from math import ceil, floor

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import torchvision as tv

from rgnnvis.datasets.dataset_utils import IMAGENET_MEAN, IMAGENET_STD, LabelToLongTensor
import rgnnvis.utils.tensor_utils as tensor_utils


YTVIS_CATEGORY_NAMES = {0: "background", 1: "person", 2: "giant_panda", 3: "lizard", 4: "parrot", 5: "skateboard", 6: "sedan", 7: "ape", 8: "dog", 9: "snake", 10: "monkey", 11: "hand", 12: "rabbit", 13: "duck", 14: "cat", 15: "cow", 16: "fish", 17: "train", 18: "horse", 19: "turtle", 20: "bear", 21: "motorbike", 22: "giraffe", 23: "leopard", 24: "fox", 25: "deer", 26: "owl", 27: "surfboard", 28: "airplane", 29: "truck", 30: "zebra", 31: "tiger", 32: "elephant", 33: "snowboard", 34: "boat", 35: "shark", 36: "mouse", 37: "frog", 38: "eagle", 39: "earless_seal", 40: "tennis_racket"}



def get_nonnone_span(lst):
    first_idx = None
    end_idx = None
    for idx in range(0, len(lst)):
        if lst[idx] is not None and first_idx is None and end_idx is None:
            first_idx = idx
        elif lst[idx] is None and first_idx is not None and end_idx is None:
            last_idx  = idx - 1
        elif lst[idx] is not None and end_idx is not None:
            raise ValueError(f"List seems to contain multiple chunks: {lst}")
    return first_idx, end_idx

def apply_binary_rle_to_seg(binary_rle, value, seg):
    pointer = 0
    apply_value = False
    for count in binary_rle:
        if apply_value:
            seg[pointer : pointer + count] = value
        apply_value = not apply_value
        pointer = pointer + count
    return seg


class YTVIS(Dataset):
    def __init__(self, root_path, split, stride, L, imsize=(480, 864), start_frame='first', data_purpose='training',
                 max_num_objects=16, augmentations=None, image_augmentations=None):
        self.root_path = root_path
        self.split = split
        assert split in ('train', 'valid', 'test')
        self.stride = stride
        self.L = L
        self.imsize = imsize
        assert stride in (5,), "We run either with all frames, or all annotated frames"
        self.start_frame = start_frame
        self.image_path = os.path.join(root_path, f'{split}', 'JPEGImages')
        self.data_purpose = data_purpose
        assert data_purpose in ('training', 'evaluation')

        self.max_num_objects = max_num_objects
        self.augmentations = augmentations
        self.image_augmentations = image_augmentations if image_augmentations is not None else []

        print("\nYT-VIS dataset initializing")
        self._init_getters()
        self._init_data()

    def _init_getters(self):
        if self.data_purpose == 'training' and self.imsize is not None:
            self.image_transform = tv.transforms.Compose(
                [tv.transforms.Resize(self.imsize, interpolation=Image.BILINEAR)]
                + self.image_augmentations
                + [tv.transforms.ToTensor(),
                   tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        else:
            self.image_transform = tv.transforms.Compose(
                self.image_augmentations
                + [tv.transforms.ToTensor(),
                   tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def _init_data(self):
        with open(os.path.join(self.root_path, f'{self.split}.json'), 'r') as fp:
            raw_annos = json.load(fp)

        videos = raw_annos['videos']
        print(f"* Loaded {len(raw_annos['videos'])} videos from {self.split} split")
        if self.data_purpose == 'training':
            videos = self._filter_videos(videos)

        self.videos = {elem['id']: elem for elem in videos}
        self.video_ids = {dataset_idx: video_idx for dataset_idx, video_idx in enumerate(self.videos.keys())}
        video_length_tensor = torch.tensor([len(video['file_names']) for video in self.videos.values()])
        print("* Histogram over video length:")
        print(video_length_tensor.bincount())
        
        if raw_annos.get('annotations') is not None:
            self._init_annos(raw_annos)

    def _filter_videos(self, videos):
        videos = [elem for elem in videos if 5*elem['length'] >= 1 + (self.L-1)*self.stride]
        print(f"* {len(videos)} remaining after filtering on length")
        videos = [elem for elem in videos if elem['width'] == 1280 and elem['height'] == 720]
        print(f"* {len(videos)} remaining after keeping 720p ones")
        return videos

    def _init_annos(self, raw_annos):
        self.annos = {key: {} for key in self.videos.keys()}
        for anno in raw_annos['annotations']:
            # Do not add annos corresponding to videos removed during filtering
            if self.annos.get(anno['video_id']) is not None:
                # inst_idx may be larger than 255, but tensors are LONG and we use -1 as DONTCARE
                inst_idx = len(self.annos[anno['video_id']]) + 1
                self.annos[anno['video_id']][inst_idx] = anno

        num_obj_tensor = torch.tensor([len(self.annos[video_idx]) for video_idx in self.videos.keys()])
        print("* Bincount over num objects in each sequence:")
        print(num_obj_tensor.bincount())

        track_length_tensor = torch.tensor([len(anno['segmentations']) for video_idx in self.videos.keys() for anno in self.annos[video_idx].values()])
        print("* Bincount over object track lengths:")
        print(track_length_tensor.bincount())

        categories_tensor = torch.tensor([anno['category_id'] for video_idx in self.videos.keys() for anno in self.annos[video_idx].values()])
        self.class_counts = categories_tensor.bincount()
        print("* Bincount over object categories:")
        print(self.class_counts)

        # Debug
#        box_sizes = torch.tensor([bbox[2:4] for video_idx in self.videos.keys() for anno in self.annos[video_idx].values() for bbox in anno['bboxes'] if bbox is not None])
#        print("\nwidth:\n", box_sizes[:, 0].histc(bins=83, min=-24, max=1304).long())
#        print("\nheight:\n", box_sizes[:, 1].histc(bins=48, min=-24, max=744).long())
#        print("\naspect:\n", (box_sizes[:, 0] / box_sizes[:, 1]).histc(bins=101, min=-0.05, max=10.05).long())
#        raise ValueError()

        # Check that the video length is the same as the corresponding annotation lengths
        for video_idx, video_anno in self.annos.items():
            lengths = [len(self.videos[video_idx]['file_names'])] + [len(anno['segmentations']) for anno in video_anno.values()]
            assert len(set(lengths)) == 1, f"{video_idx} \n{video_anno} \n{self.videos[video_idx]} \n{lengths}"
            #for anno in video_anno.values():
#           #     print(f"{video_idx} {anno['category_id']}")
            #    for seg in anno['segmentations']:
            #        assert seg is None or (sum(seg['counts']) == 720*1280 and seg['size'] == [720, 1280]), seg
                    
#        print(raw_annos.keys())
#        print(raw_annos['info'])
#        print(raw_annos['licenses'])
#        print(raw_annos['categories'])
#        print(raw_annos['videos'][0])
#        print(raw_annos['annotations'][0])
#        print(raw_annos['videos'])
#        print(len(raw_annos['videos']), len(raw_annos['annotations']))
#        print([None if elem is None else "something" for elem in raw_annos['annotations'][0]['segmentations']])
#        print([None if elem is None else "something" for elem in raw_annos['annotations'][1]['segmentations']])
#        print([None if elem is None else "something" for elem in raw_annos['annotations'][2]['segmentations']])

#        trn_video_ids = [video_idx for dataset_idx, video_idx in self.video_ids.items()
#                         if video_idx < 2000]
#        val_video_ids = [video_idx for dataset_idx, video_idx in self.video_ids.items()
#                         if video_idx >= 2000]
#        trn_categories = torch.tensor([anno['category_id'] for video_idx in trn_video_ids for anno in self.annos[video_idx].values()])
#        val_categories = torch.tensor([anno['category_id'] for video_idx in val_video_ids for anno in self.annos[video_idx].values()])
#        print("Bincount over object categories for idx < 2000:")
#        print(trn_categories.bincount())
#        print("Bincount over object categories for idx >= 2000:")
#        print(val_categories.bincount())       
        

    def get_subsplit(self, video_idx_threshold=2000):
        left = [dataset_idx for dataset_idx, video_idx in self.video_ids.items()
                if video_idx < video_idx_threshold]
        right = [dataset_idx for dataset_idx, video_idx in self.video_ids.items()
                 if video_idx >= video_idx_threshold]

        print("\nSplitting an YT-VIS dataset")
        categories_tensor = torch.tensor([anno['category_id'] for video_idx in self.videos.keys() for anno in self.annos[video_idx].values() if video_idx < video_idx_threshold])
        print("* Bincount over object categories for left (first) set:")
        print(categories_tensor.bincount())
        categories_tensor = torch.tensor([anno['category_id'] for video_idx in self.videos.keys() for anno in self.annos[video_idx].values() if video_idx >= video_idx_threshold])
        print("* Bincount over object categories for right (second) set:")
        print(categories_tensor.bincount())
        num_obj_tensor = torch.tensor([len(self.annos[video_idx]) for video_idx in self.videos.keys() if video_idx < video_idx_threshold])
        print("* Bincount over num objects in each sequence for left (first) set:")
        print(num_obj_tensor.bincount())
        num_obj_tensor = torch.tensor([len(self.annos[video_idx]) for video_idx in self.videos.keys() if video_idx >= video_idx_threshold])
        print("* Bincount over num objects in each sequence for right (second) set:")
        print(num_obj_tensor.bincount())

        return Subset(self, left), Subset(self, right)

    def get_class_balancing_weights(self, video_idx_threshold=2000, base=3):
        total_count = self.class_counts.sum()
        class_weights = [base ** (total_count.float() / count.float()).log10() for count in self.class_counts]
        sample_weights = [max(
            [
                class_weights[anno['category_id']] for anno in self.annos[video_idx].values()
            ]) for video_idx in self.video_ids.values() if video_idx < video_idx_threshold]
        normalization = 1 / sum(sample_weights)
        sample_weights = [normalization * weight for weight in sample_weights]
#        assert torch.isfinite(torch.tensor(sample_weights)).all()
#        print(torch.tensor(sample_weights).histc(bins=100, min=-.01, max=10.01))
        return sample_weights        
    
    def get_image(self, path):
        pic = Image.open(path)
        image_as_tensor = self.image_transform(pic)
        return image_as_tensor

    def get_isannos(self, annos, frame_ids, L):
        first_objanno = list(annos.values())[0]
        H0 = first_objanno['height']
        W0 = first_objanno['width']
        isannos = torch.zeros((L, W0, H0), dtype=torch.int64).view(L, W0 * H0) # Decode anno in raw size
        for obj_idx, obj_anno in annos.items():
            for l, frame_idx in enumerate(frame_ids):
                segmentation = obj_anno['segmentations'][frame_idx]
                if segmentation is not None:
                    isannos[l] = apply_binary_rle_to_seg(segmentation['counts'], obj_idx, isannos[l])
        isannos = isannos.view(L, 1, W0, H0).transpose(2,3)
        if self.data_purpose == 'training' and self.imsize is not None:
            isannos = F.interpolate(isannos.float(), size=self.imsize, mode='nearest').view(-1, *self.imsize).long()
        else:
            isannos = isannos.view(L, H0, W0)                
        return isannos

    def get_ssannos(self, isannos, annos):
        ssannos = torch.full_like(isannos, fill_value=-1)
        for obj_idx, obj_anno in annos.items():
            ssannos[isannos == obj_idx] = obj_anno['category_id']
        return ssannos

    def get_odannos(self, annos, frame_ids, L):
        first_objanno = list(annos.values())[0]
        H0 = first_objanno['height']
        W0 = first_objanno['width']
        boxes = torch.full((L, self.max_num_objects, 4), fill_value=-1.)
        for obj_idx, obj_anno in annos.items():
            for l, frame_idx in enumerate(frame_ids):
                box_anno = obj_anno['bboxes'][frame_idx]
                if box_anno is not None:
                    boxes[l,obj_idx] = torch.tensor(box_anno, dtype=torch.float32)
                    boxes[l,obj_idx,2:4] = boxes[l,obj_idx,0:2] + boxes[l,obj_idx,2:4] # xywh to xyxy
        
        if self.data_purpose == 'training' and self.imsize is not None:
            self.odanno_resize_tensor = torch.tensor(self.imsize).float() / torch.tensor([H0, W0]).float()
            self.odanno_resize_tensor = self.odanno_resize_tensor[[1,0,1,0]].view(1,1,4)
            boxes = self.odanno_resize_tensor * boxes

        return boxes

    def get_lbannos(self, annos):
        lbannos = torch.full((self.max_num_objects,), fill_value=0, dtype=torch.int64)
        for obj_idx, anno in annos.items():
            lbannos[obj_idx] = anno['category_id']
        return lbannos

    def get_active(self, annos, frame_ids, L):
        """Active is a tensor marking the state of each object (or object position in the annotations).
        The state may be 0 (inactive, not an object, negative), 1 (active, is an object), and 2 (has been
        active previously but has disappeared), and 3 (corresponds to background).
        @todo  Perhaps also some kind of dont-care that is -1.
        """
        active = torch.zeros((L, self.max_num_objects), dtype=torch.uint8)
        active[:, 0] = 3
        for obj_idx, anno in annos.items():
            has_been_active = False
            for l, frame_idx in enumerate(frame_ids):
                has_box = anno['bboxes'][frame_idx] is not None
                has_seg = anno['segmentations'][frame_idx] is not None
                assert has_box == has_seg, anno
                if has_box:
                    active[l, obj_idx] = 1
                    has_been_active = True
                elif has_been_active:
                    active[l, obj_idx] = 2
        return active
#            print("\n", obj_idx)
#            print(" ".join(["YES" if elem is not None else "..." for elem in anno['segmentations']]))
#            print(" ".join(["YES" if elem is not None else "..." for elem in anno['bboxes']]))
#        print(frame_ids)
#        print(L)
#        raise ValueError()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Returns: dict containing
            images: (L,3,H,W)  FloatTensor
            ssannos: (L,H,W)   LongTensor
            isannos: (L,H,W)   LongTensor
            odannos: (L,Mmax,4) FloatTensor, describing boxes of all annotated objects as (x1,y1,x2,y2) coordinates
            lbannos: (L,Mmax)   LongTensor, describing classes/categories of all annotated objects
        """
        video_idx = self.video_ids[idx]
        video = self.videos[video_idx]
        frame_ids = list(range(len(video['file_names'])))
        if self.data_purpose == 'training':
            if self.start_frame == 'first':
                frame_ids = list(range(0, self.L))
            elif self.start_frame == 'random':
                frame_startidx = random.choice(frame_ids[0 : 1 - self.L] if self.L > 1 else frame_ids)
                frame_ids = list(range(frame_startidx, frame_startidx + self.L))

        images = torch.stack([self.get_image(os.path.join(self.image_path, video['file_names'][frame_idx])) for frame_idx in frame_ids])
        L = len(frame_ids)  # Will be self.L in training mode, and the sequence length in evaluation mode
        provides_anno = torch.ones((L,))
        seqname = os.path.dirname(video['file_names'][0])
        if self.split == 'train':
            annos = self.annos[video_idx]
            isannos = self.get_isannos(annos, frame_ids, L)
            ssannos = self.get_ssannos(isannos, annos)
            odannos = self.get_odannos(annos, frame_ids, L)
            lbannos = self.get_lbannos(annos)
            active  = self.get_active(annos, frame_ids, L)

            if self.augmentations is not None:
                augmented_data = self.augmentations(images, active, odannos, isannos, lbannos, ssannos)
                images, active, odannos, isannos, lbannos, ssannos = augmented_data

#        obj_ids = {obj_idx: get_nonnone_span(anno['segmentations']) for obj_idx, anno in annos.items()}
            return {'images': images, 'ssannos': ssannos, 'isannos': isannos, 'odannos': odannos,
                    'lbannos': lbannos, 'active': active,
                    'provides_ss': provides_anno, 'provides_is': provides_anno, # unsure if these are needed
                    'provides_od': provides_anno, 'provides_lb': provides_anno, # unsure if these are needed
                    'identifier': ['YTVIS', seqname], 'video_id': video_idx}
        else:
            return {'images': images, 'identifier': ['YTVIS', seqname], 'video_id': video_idx}


from torchvision.utils import make_grid
from numpy import prod


class YTVISMosaic(Dataset):

    def __init__(self, parent_dataset, video_idx_threshold=2019, mosaic_size=(2, 2)):
        self.parent_dataset = parent_dataset
        self.video_idx_threshold = video_idx_threshold
        self.mosaic_size = mosaic_size

        # generate new ids
        self.ioi = prod(mosaic_size)
        p = torch.ones(self.video_idx_threshold).bool()
        self.new_ids = []
        while p.any():
            r = torch.multinomial(p.float(), self.ioi, replacement=False).tolist()
            self.new_ids.append(r)
            p[r] = 0

    def __len__(self):
        return len(self.new_ids)

    def mosaic_concat(self, ioi_data):
        images = torch.stack([make_grid([d['images'][l] for d in ioi_data], nrow=self.mosaic_size[1], padding=0)
                              for l in range(self.parent_dataset.L)], dim=0)
        ssannos = make_grid([d['ssannos'] for d in ioi_data], nrow=self.mosaic_size[1], padding=0)
        isannos = make_grid([d['isannos'] for d in ioi_data], nrow=self.mosaic_size[1], padding=0)

        lbannos = []
        odannos = []
        active = []
        seqname = []
        imsize = self.parent_dataset.augmentations.augmentations[0].imsize
        for i, d in enumerate(ioi_data):
            uv = torch.tensor([i % self.mosaic_size[1], i // self.mosaic_size[1]])
            # assuming that boxes are in point-form and (1,1) correspond to the center of topleft pixel.
            B, M = d['active'].shape
            transl = (torch.tensor([imsize[1], imsize[0]])*uv).repeat(B, M, 2)*(d['active'] == 1).view(B, M, 1)
            odannos.append(d['odannos'] + transl)
            lbannos.append(d['lbannos'])
            active.append(d['active'])
            seqname.append(d['identifier'][1])
        odannos = torch.cat(odannos, dim=1)
        lbannos = torch.cat(lbannos, dim=0)
        active = torch.cat(active, dim=1)
        seqname = '_'.join(seqname)

        return images, active, odannos, isannos, lbannos, ssannos, seqname

    def __getitem__(self, idx):
        """
        Returns: dict containing
            images: (L,3,H,W)  FloatTensor
            ssannos: (L,H,W)   LongTensor
            isannos: (L,H,W)   LongTensor
            odannos: (L,Mmax*ioi,4) FloatTensor, describing boxes of all annotated objects as (x1,y1,x2,y2) coordinates
            lbannos: (L,Mmax*ioi)   LongTensor, describing classes/categories of all annotated objects
        """
        ioi_idx = self.new_ids[idx]
        ioi_data_list = [self.parent_dataset[ioi] for ioi in ioi_idx]  # containing dict data from parent
        images, active, odannos, isannos, lbannos, ssannos, seqname = self.mosaic_concat(ioi_data_list)

        provides_anno = torch.ones((self.parent_dataset.L,))
        if self.parent_dataset.split == 'train':

            if self.parent_dataset.augmentations is not None:
                augmented_data = self.parent_dataset.augmentations(images, active, odannos, isannos, lbannos, ssannos)
                images, active, odannos, isannos, lbannos, ssannos = augmented_data

            #        obj_ids = {obj_idx: get_nonnone_span(anno['segmentations']) for obj_idx, anno in annos.items()}
            return {'images': images, 'ssannos': ssannos, 'isannos': isannos, 'odannos': odannos,
                    'lbannos': lbannos, 'active': active,
                    'provides_ss': provides_anno, 'provides_is': provides_anno,  # unsure if these are needed
                    'provides_od': provides_anno, 'provides_lb': provides_anno,  # unsure if these are needed
                    'identifier': ['YTVIS', seqname], 'video_id': idx}
        else:
            return {'images': images, 'identifier': ['YTVIS', seqname], 'video_id': idx}


if __name__ == "__main__":
    ytvis = YTVIS('/my_data/ytvis', 'train', 5, L=13)
    sample = ytvis[0]
    
