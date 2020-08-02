import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2


class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the nubmer of channels for output image is 2 (ab).
        """
        parser.set_defaults(input_nc=1, output_nc=2)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(self.make_dataset())
        self.transform_A = get_transform(self.opt, convert=False)
        self.transform_R = get_transform(self.opt, convert=False, must_resize=True)
        assert(opt.input_nc == 1 and opt.output_nc == 2)

    def __getitem__(self, index):
        path_A, path_R = self.AB_paths[index]
        im_A_l, im_A_ab = self.process_img(path_A, self.transform_A)
        im_R_l, im_R_ab = self.process_img(path_R, self.transform_R)
        hist_ab = im_R_ab
        label = torch.Tensor([0]).long()

        im_dict = {
            'A_l': im_A_l,
            'A_ab': im_A_ab,
            'R_l': im_R_l,
            'R_ab': im_R_ab,
            'hist_ab': hist_ab,
            'labels': label,
            'A_paths': path_A
        }
        return im_dict

    def make_dataset(self, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(self.dir), '%s is not a valid directory' % self.dir

        with open(os.path.join(self.dir, self.opt.paired_file)) as f:
            for line in f:
                line = line.strip().split('\t')
                line = [os.path.join(self.dir, i) for i in line]
                images.append(tuple(line))
        return images[:min(max_dataset_size, len(images))]

    def process_img(self, im_path, transform):
        im = Image.open(im_path).convert('RGB')
        im = transform(im)
        im = np.array(im)
        ims = [im]
        for i in [0.5, 0.25]:
            ims = [cv2.resize(im, None, fx=i, fy=i, interpolation=cv2.INTER_AREA)] + ims
        l_ts, ab_ts = [], []
        for im in ims:
            lab = color.rgb2lab(im).astype(np.float32)
            lab_t = transforms.ToTensor()(lab)
            l_ts.append(lab_t[[0], ...] / 50.0 - 1.0)
            ab_ts.append(lab_t[[1, 2], ...] / 110.0)
        return l_ts, ab_ts

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
