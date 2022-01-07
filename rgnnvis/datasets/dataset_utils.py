import os
import sys
import random
random.seed()
from itertools import accumulate
import bisect
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision as tv

IMAGENET_MEAN = [.485,.456,.406]
IMAGENET_STD = [.229,.224,.225]

class VOTPathsHelper(object):
    def __init__(self, master_path, slave_path):
        self.video_list = self.read_paths(master_path, slave_path)

    def __len__(self):
        return len(self.video_list)

    def read_paths(self,master_path, slave_path):
        """ Read paths
        Args:
            master_path  : path to master sequences, determine the sequences/files
                           which are loaded from slave_path (if they exists)
            slave_path   : path to slave sequences
        Returns:
            paths        : list of lists
        """
        sequences = sorted(os.listdir(master_path))
        res = []
        for seq in sequences:
            samples = []
            master_dir = os.path.join(master_path, seq)
            slave_dir = os.path.join(slave_path, seq)
            master_files = sorted(os.listdir(master_dir))
            slave_files = sorted(os.listdir(slave_dir))
            slave_files_no_ext = [".".join(f.split(".")[:-1]) for f in sorted(os.listdir(slave_dir)) if
                                  os.path.isfile(os.path.join(slave_dir, f))]
            for i, mf in enumerate(master_files):
                mfne = ".".join(mf.split(".")[:-1])
                try:
                    sf = [slave_path, seq, slave_files[slave_files_no_ext.index(mfne)]]
                except ValueError:
                    sf = None
                samples.append([[master_path, seq, mf], sf])
            res.append(samples)
        return res # [[0,#seqs], [0,#samples], [image, annotation], [root_path, seq-parts, filename]]

    def get_paths(self, idx):
        """ Return paired sting paths for a sequence
        Args:
            idx   : sequence id
        Returns:
            paths : [[strmaster0, str_slave0],...,[strmasterN, str_slaveN]]
        """
        paths = []
        seq = self.video_list[idx]
        for sample in seq:
            mp = os.path.join(*sample[0])
            if sample[1] is not None:
                sp = os.path.join(*sample[1])
            else:
                sp = None
            paths.append([mp, sp])
        return paths

    def get_seq_idx(self, seq):
        for i in range(len(self)):
            if seq == self.get_seq_name(i):
                return i

    def get_filenames(self, idx):
        """Returns master filenames with extensions of sequence idx"""
        filenames = []
        for sample in self.video_list[idx]:
            filenames.append(sample[0][2])
        return filenames

    def get_seq_name(self, idx):
        """ Return name of a sequence
        Args:
            idx     : sequence id
        Returns:
            string  : name of sequence
        """
        return self.video_list[idx][0][0][1]

    def get_raw_paths(self):
        """ Return raw structure
        Args:
            None
        Returns:
            paths  : [[0,#seqs], [0,#samples], [image, annotation], [root_path, seq-parts, filename]]
        """
        return self.video_list

    def get_num_seq(self):
        return len(self.video_list)

class Video(object):
    """Object which keeps tracks of image and label paths. Created to handle poor dataset structures."""
    def __init__(self, image_glob, label_glob, label_idx):
        self.image_paths = image_glob
        self.label_paths = label_glob
        self.label_indices = label_idx

    def split(self, splitidx):
        left_images = self.image_paths[:splitidx]
        left_label_indices = [x for x in filter((lambda x: x < splitidx), self.label_indices)]
        left_labels = self.label_paths[:len(left_label_indices)]
        right_images = self.image_paths[splitidx:]
        right_label_indices = [x - splitidx for x in filter((lambda x: x >= splitidx), self.label_indices)]
        right_labels = self.label_paths[len(left_label_indices):]
        return (Video(left_images, left_labels, left_label_indices),
                Video(right_images, right_labels, right_label_indices))

    def get_labeled_images(self):
        return [self.image_paths[idx] for idx in self.label_indices]

    def get_labels(self):
        return self.label_paths

    def getnum_images(self):
        return len(self.image_paths)

    def getnum_labels(self):
        return len(self.label_paths)

class WeightedConcatDataset(torch.utils.data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
        sim_lengths (sequence): List of the simulated lengths of each dataset
    """
    def __init__(self, datasets, sim_lengths):
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.sim_lengths = sim_lengths
        self.sim_lengths = [min(simlen, len(dataset)) for dataset, simlen in zip(datasets, sim_lengths)]
#        self.cumulative_lengths = list(accumulate(lengths))
        self.cumulative_sim_lengths = list(accumulate([0] + sim_lengths))

    def __len__(self):
        return self.cumulative_sim_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sim_lengths, idx) - 1
        requested_idx = idx - self.cumulative_sim_lengths[dataset_idx]
        assert requested_idx < len(self.datasets[dataset_idx]), "dataset idx {}, req idx {}, dataset len {}".format(dataset_idx, requested_idx, len(self.datasets[dataset_idx]))
        sample_idx = random.choice([idx for idx in range(requested_idx, len(self.datasets[dataset_idx]), self.sim_lengths[dataset_idx])])
#        print("idx {}, dataset idx {}, req idx {}, sample idx {}".format(idx, dataset_idx, requested_idx, sample_idx))
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class ConcatDatasetWithIdx(torch.utils.data.ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample = self.datasets[dataset_idx][sample_idx]
        sample['dataset_idx'] = dataset_idx
        sample['sample_idx']  = sample_idx
        return sample

class LabelToLongTensor(object):
    """From Tiramisu github"""
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        elif pic.mode == '1':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        elif pic.mode == 'I':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            if pic.mode == 'LA': # Hack to remove alpha channel if it exists
                label = label.view(pic.size[1], pic.size[0], 2)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                label = label.view(1, label.size(0), label.size(1))
            else:
                label = label.view(pic.size[1], pic.size[0], -1)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
#            print(label.size(), label.min(), label.max(), label.sum(), ((label==255) + (label==0)).sum())
        return label

class LabelTranslateIndices(object):
    def __init__(self, translation):
        self.translation = translation
    def __call__(self, label):
        return label + self.translation

class LabelColoursToClasses(object):
    def __init__(self, class_colours):
#        self.class_colours = torch.LongTensor(class_colours).unsqueeze(-1).unsqueeze(-1)
        self.class_colours = class_colours
    def __call__(self, rgb_label):
        c,h,w = rgb_label.size()
        r = rgb_label[0,:,:]
        g = rgb_label[1,:,:]
        b = rgb_label[2,:,:]
        label = torch.zeros(1,h,w).long()
        for idx,colour in self.class_colours:
            label[(r==colour[0])*(g==colour[1])*(b==colour[2])] = idx
        return label
        
class LabelLongTensorToFloat(object):
    def __call__(self, label):
        return label.float()

class PadToDivisible(object):
    def __init__(self, divisibility):
        self.div = divisibility
        
    def __call__(self, tensor):
        size = tensor.size()
        assert tensor.dim() == 4
        height, width = size[-2:]
        height_pad = (self.div - height % self.div) % self.div
        width_pad = (self.div - width % self.div) % self.div
        padding = [(width_pad+1)//2, width_pad//2, (height_pad+1)//2, height_pad//2]
        tensor = F.pad(tensor, padding, mode='reflect')
        return tensor, padding

class RandomBlurViaResize:
    def __init__(self, scale_factor_range):
        self.scale_factor_range = scale_factor_range
        assert 0.0 < scale_factor_range[0]
        assert scale_factor_range[0] < scale_factor_range[1]
        assert scale_factor_range[1] <= 1.0
    def __call__(self, img):
        W, H = img.size
        scale_factor = (torch.rand(1) * (self.scale_factor_range[1] - self.scale_factor_range[0])
                        + self.scale_factor_range[0])
        img = tv.transforms.functional.resize(img, int((H * scale_factor), int(W * scale_factor)), interpolation=2)
        img = tv.transforms.functional.resize(img, (H, W), interpolation=2)
        return img

class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, images, labels):
        for t in self.transforms:
            images, labels = t(images, labels)
        return images, labels
            
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
            format_string += '\n)'
        return format_string
                
class JointRandomCrop(object):
    def __init__(self, size, padding=(0,0)):
        """Size and padding as (height,width)"""
        self.size = size
        self.padding = padding

    def __call__(self, *args):
        out = []
        for tensor in args:
            tensor_size = tensor.size()
            tensor = tensor.view(-1, tensor_size[-2], tensor_size[-1])
            padded_tensor = F.pad(tensor,(self.padding[1],self.padding[1],self.padding[0],self.padding[0]))
            randx = random.randrange(padded_tensor.size(-1) - self.size[1])
            randy = random.randrange(padded_tensor.size(-2) - self.size[0])
            cropped_tensor = padded_tensor[:, randy:randy+self.size[0], randx:randx+self.size[1]]
            result_tensor = cropped_tensor.view(tensor_size[:-2] + cropped_tensor.size()[-2:]).data
            out.append(result_tensor)
        return out
                
class JointCrop(object):
    def __init__(self, pos, size):
        """Size and padding as (height,width)"""
        self.pos = pos
        self.size = size

    def __call__(self, *args):
        out = []
        for tensor in args:
            tensor_size = tensor.size()
            tensor = tensor.view(-1, tensor_size[-2], tensor_size[-1])
            cropped_tensor = tensor[:, self.pos[0]:self.pos[0]+self.size[0], self.pos[1]:self.pos[1]+self.size[1]]
            result_tensor = cropped_tensor.view(tensor_size[:-2] + cropped_tensor.size()[-2:])
            out.append(result_tensor)
        return out


#    def __call__(self, images, labels):
#        """images and labels are expected to be 4th order tensors (time,channels,height,width)
#        """
#        padded_images = F.pad(images, (self.padding[1],self.padding[1],self.padding[0],self.padding[0])).data
#        padded_labels = F.pad(labels, (self.padding[1],self.padding[1],self.padding[0],self.padding[0])).data
#        randx = random.randrange(padded_images.size(-1) - self.size[1])
#        randy = random.randrange(padded_images.size(-2) - self.size[0])
#        cropped_images = padded_images[:, randy:randy+self.size[0], randx:randx+self.size[1]]
#        cropped_labels = padded_labels[:, randy:randy+self.size[0], randx:randx+self.size[1]]
#        return (cropped_images, cropped_labels)

class JointRandomHorizontalFlip(object):
    def __call__(self, *args):
        if random.choice([True, False]):
            out = []
            for tensor in args:
                idx = [i for i in range(tensor.size(-1)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                tensor_flip = tensor.index_select(-1, idx)
                out.append(tensor_flip)
            return out
        else:
            return args

class JointRandomTemporalFlip(object):
    def __call__(self, image_tensor, label_tensor):
        assert image_tensor.dim() == 4
        assert label_tensor.dim() == 3
        if random.choice([True, False]):
            idx = [i for i in range(image_tensor.size(0)-1, -1, -1)]
            idx = torch.LongTensor(idx)
            image_tensor_flip = image_tensor.index_select(0, idx)
            label_tensor_flip = label_tensor.index_select(0, idx)
            return image_tensor_flip, label_tensor_flip
        else:
            return image_tensor, label_tensor

class JointRandomZeroLabels(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image_tensor, label_tensor):
        """
        image_tensor (Tensor): Image tensor, not used
        label_tensor (Tensor): Label tensors of size (height,width), where labels are denoted as integers
        """
        min_classid = label_tensor.min()
        max_classid = label_tensor.max()
#        if min_classid != 0:
#            raise ValueError("Expected minimum class id in labels to be 0, received: {}".format(min_classid))
        for classid in range(min_classid, max_classid + 1):
            if random.random() < self.prob:
                label_tensor[label_tensor == classid] = 0
        return image_tensor, label_tensor
        
#    def __call__(self, images, labels):
#        if random.choice([True, False]):
#            idx = [i for i in range(images.size(-1)-1, -1, -1)]
#            idx = torch.LongTensor(idx)
#            images_flip = images.index_select(-1, idx)
#            labels_flip = labels.index_select(-1, idx)
#            return (images_flip, labels_flip)
#        else:
#            return (images, labels)                              
        
        
def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix

class JointRandomAffine(tv.transforms.RandomAffine):
    def __call__(self, image, label):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.size)
        return (
            torchvision.transforms.functional.affine(
                image, *ret, resample=Image.BILINEAR, fillcolor=self.fillcolor),
            torchvision.transforms.functional.affine(
                label, *ret, resample=self.Image.NEAREST, fillcolor=self.fillcolor))

def centercrop(tensor, cropsize):
    _, _, H, W = tensor.size()
    A, B = cropsize
#    print((H,W), (A,B), (H-A)//2, (H+A)//2
    return tensor[:,:,(H-A)//2:(H+A)//2,(W-B)//2:(W+B)//2]

class JointRandomScale(object):
    def __call__(self, images, labels):
        L, _, H, W = images.size()
        scales = ((1.0 + (torch.rand(1) < .5).float()*torch.rand(1)*.1)*torch.ones(L)).cumprod(0).tolist()
        images = torch.cat([centercrop(F.interpolate(images[l:l+1,:,:,:], scale_factor=scales[l], mode='bilinear', align_corners=False), (H, W)) for l in range(L)], dim=0)
        labels = torch.cat([centercrop(F.interpolate(labels[l,:,:].view(1,1,H,W).float(), scale_factor=scales[l], mode='nearest').long(), (H,W)) for l in range(L)], dim=0).view(L,H,W)
        return images, labels

def convert_segannos_to_bboxannos(segmentation: torch.Tensor, occlusion_threshold=0) -> torch.Tensor:
    """
    args:
        segmentation: Of size (L,1,H,W)
    returns:
        tensor of size (L,N,4)
    """
    assert segmentation.dim() == 4
    L,_,H,W = segmentation.size()
    device = segmentation.device
    target_ids = segmentation.unique().tolist()
    if 0 in target_ids: target_ids.remove(0)
    N = max(target_ids)

    bboxes = -torch.ones((L,N,4), device=device)
    
    xvals = torch.arange(0., W, device=device).view(1,1,1,W)
    yvals = torch.arange(0., H, device=device).view(1,1,H,1)
    for n in target_ids:
        seg = (segmentation == n).float()
        xmin = ((1-seg) * (W+1) + seg * xvals).min(dim=-1)[0].min(dim=-1)[0]
        xmax = ((1-seg) * (-1) + seg * xvals).max(dim=-1)[0].max(dim=-1)[0]
        ymin = ((1-seg) * (H+1) + seg * yvals).min(dim=-1)[0].min(dim=-1)[0]
        ymax = ((1-seg) * (-1) + seg * yvals).max(dim=-1)[0].max(dim=-1)[0]
        bboxes[:,n-1,:] = torch.cat([xmin,ymin,xmax-xmin+1,ymax-ymin+1], dim=-1)
        bboxes[(xmin + occlusion_threshold > xmax).squeeze(),n-1,:] = -1
    return bboxes


class MultitaskWeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, weight_vectors, num_samples, replacement=True):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not (isinstance(replacement, bool) or all(isinstance(elem, bool) for elem in replacement)):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        if isinstance(replacement, bool):
            replacement = [replacement for _ in weight_vectors]
        self.weight_vectors = [torch.tensor(weights) for weights in weight_vectors]
        self.task_offsets = list(accumulate([0] + [len(elem) for elem in weight_vectors]))
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        lists = [(torch.multinomial(weights, self.num_samples, replacement) + offset).tolist()
                 for weights, replacement, offset in zip(self.weight_vectors, self.replacement, self.task_offsets)]
#        print("lists", [(min(lst), max(lst)) for lst in lists])
#        print("Weight vectors", [weights.size() for weights in self.weight_vectors])
#        print("replacement", self.replacement)
#        print("offsets", self.task_offsets)
        round_robin_list = [elem for lst in zip(*lists) for elem in lst]
        return iter(round_robin_list)

    def __len__(self):
        return self.num_samples * len(self.weight_vectors)


def PrintDatasetStatistics(dataset):
    print(f"Analyzing instance of {dataset.__class__.__name__}")
    raise NotImplementedError("Would probably be great, partially because it can easily respect subsets")


if __name__ == "__main__":
    import unittest
    class TestConvertSegannosToBBoxannos(unittest.TestCase):
        def test_single_target(self):
            seganno = torch.zeros((8,1,240,432), dtype=torch.uint8)
            for l in range(8):
                seganno[l,:,10*l:11*l+10,100+l:140+l] = 1
            seganno[4] = 0
            bboxes = convert_segannos_to_bboxannos(seganno)
            gt = torch.Tensor([[100,0,40,10],[101,10,40,11],[102,20,40,12],[103,30,40,13],
                               [-1,-1,-1,-1],[105,50,40,15],[106,60,40,16],[107,70,40,17]])
            gt = gt.view(8,1,4)
            self.assertTrue((bboxes == gt).all(), (bboxes, gt))
        def test_multiple_targets(self):
            seganno = torch.zeros((8,1,240,432), dtype=torch.uint8)
            for l in range(8):
                seganno[l,:,10*l:11*l+10,100+l:140+l] = 1
                seganno[l,:,200:240,0:10] = 2
            seganno[4][seganno[4] == 1] = 0
            bboxes = convert_segannos_to_bboxannos(seganno)
            gt0 = torch.Tensor([[100,0,40,10],[101,10,40,11],[102,20,40,12],[103,30,40,13],
                                [-1,-1,-1,-1],[105,50,40,15],[106,60,40,16],[107,70,40,17]])
            gt1 = torch.Tensor([[0,200,10,40]]).repeat((8,1))
            gt = torch.stack([gt0, gt1], dim=1)
            self.assertTrue((bboxes == gt).all(), (bboxes, gt))

    class TestSamplers(unittest.TestCase):
        def test_multitask_weighted_random_sampler(self):
            sampler = MultitaskWeightedRandomSampler([5*[1/5] + 15*[1/15], 3*[1/3]], 20, True)
            ids = [idx for idx in sampler]
            print("In TestSamplers: Drawn ids are {}".format(ids))
            for i in range(3):
                self.assertTrue(ids[2*i] in range(20))
                self.assertTrue(ids[2*i + 1] in range(20,23))

    unittest.main()

