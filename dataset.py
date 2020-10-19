from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from transforms import Rescale,AddGaussianNoise
import fnmatch
from PIL import Image
from torchvision.transforms import ToTensor
from copy import deepcopy
# Ignore warnings
import warnings
import random
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode




class FluoDataset(Dataset):

    def __init__(self,sample_dir,gt_dir,transform=None,target_transform=None):

        self.sample_dir = sample_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = "man_seg"
        last_dir = os.path.basename(os.path.normpath(gt_dir))

        if last_dir == "TRA":
            self.prefix = "man_track"

        self.gt_count = len(fnmatch.filter(os.listdir(gt_dir), '*.tif'))  # dir is your directory path as string
        self.sample_count = len(fnmatch.filter(os.listdir(sample_dir), '*.tif'))

        if self.gt_count != self.sample_count:
            print("Amount of files dont match! gt: {} | smpl: {}".format(self.gt_count,self.sample_count))

    def __len__(self):
        return self.sample_count

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = torch.tolist()

        item = f'{item:03d}'

        sample = "t"+item+".tif"
        gt = self.prefix+item+".tif"

        sample = os.path.join(self.sample_dir,sample)
        gt = os.path.join(self.gt_dir,gt)

        if not os.path.isfile(gt):
            return None

        s_image = (Image.open(sample))
        print(s_image.mode)
        gt_image = np.array(Image.open(gt))
        gt_image = np.where(gt_image==0.0,gt_image,1)
        gt_image = Image.fromarray(np.uint8(gt_image*255))
        out = {"image":s_image,"ground_truth":gt_image}

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        if self.transform:
            s_image = self.transform(s_image)
            out['image'] = s_image

        random.seed(seed)
        torch.manual_seed(seed)
        if self.target_transform:
            gt_image = self.target_transform(gt_image)
            out['ground_truth'] = gt_image

        return out



    # ax = plt.subplot(1, 2, 1)
    # x = sample["image"]
    # x_vis = x.permute(1, 2, 0)
    # plt.imshow(x_vis,cmap="gray")



