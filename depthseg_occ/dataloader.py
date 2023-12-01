#from utils.calib_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

import bqplot.scales
import ipyvolume as ipv
import ipywidgets as widgets

from PIL import Image
import argparse
import matplotlib.pyplot as plt




def rsplit(set, perc=0.7):
    split_idx = int(perc * len(set))
    return random_split(set, [split_idx, len(set) - split_idx])


class PreprocessedDataset(Dataset):
    def __init__(self, file_path):
        self.vox_origin = torch.tensor([0, 128, 10])
        self.img_width = 1241
        self.img_height = 376

        self.class_names = [
            "empty", #0
            "vehicles", #1
            "building", #2
            "road", #3
            "sidewalk", #4
            "vegetation", #5
            "others", #6
            "unknown", #7
        ]
        self.num_classes = len(self.class_names)
        
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as hf:
            self.length = len(hf.keys()) // 3  # Assuming each sample has 3 keys (l/r images and label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hf:
            left_img = hf[f'left_image_{idx}'][:]
            right_img = hf[f'right_image_{idx}'][:]
            gt = hf[f'gt_{idx}'][:]
            return torch.from_numpy(left_img), torch.from_numpy(right_img), torch.from_numpy(gt)

    def get_batched_data(self, idx):
        left_img, right_img, gt = self.__getitem__(idx)
        left_img = left_img.unsqueeze(0)
        right_img = right_img.unsqueeze(0)
        gt = gt.unsqueeze(0)
        return left_img, right_img, gt


# calib = read_calib("/workspace/HKU-OccNet/calib.txt")
# calib_proj = get_projections(img_width, img_height, calib)


# vf_mask = calib_proj['fov_mask_1'].view(256, 256, 32)
# prj_pix = calib_proj['projected_pix_1'].view(256, 256, 32, 2)
# pix_z = calib_proj['pix_z_1'].view(256, 256, 32)

# cull_mask = torch.zeros((256, 256, 32)).bool()
# cull_mask[:int(0.5 * 256), :, :] = True






