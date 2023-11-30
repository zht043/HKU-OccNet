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



def voxel_to_coordinates(voxel_data, voxel_size=0.01, threshold = 0):
    xx, yy, zz = voxel_data.shape
    x_coords = []
    y_coords = []
    z_coords = []
    tags = []
    for x in range(xx):
        for y in range(yy):
            for z in range(zz):
                tag = voxel_data[x, y, z]
                if tag > threshold:
                    x_coords.append(x * voxel_size)
                    y_coords.append(y * voxel_size)
                    z_coords.append(z * voxel_size)
                    tags.append(tag)

    return np.array(x_coords), np.array(y_coords), np.array(z_coords), np.array(tags)

def visualize_labeled_array3d(voxel_data, num_classes=7, size = None, marker = None):
    voxel_data = voxel_data.astype(np.uint16)
    x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0])
    color_scale = bqplot.scales.ColorScale(min=0, max=num_classes, colors=["#f00", "#00f", "#000000", "#808080", "#0f0", "#800080"])
    fig = ipv.figure()
    unique_tags = np.unique(tags)

    for tag in unique_tags:
        mask = tags == tag
        x_filtered, y_filtered, z_filtered, tags_f = x[mask], y[mask], z[mask], tags[mask]
        
        ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="len({})={}".format(str(tag),x_filtered.shape[0]))
    ipv.xyzlabel('y','x','z')
    ipv.view(0, -50, distance=2.5)
    ipv.show()

def plot_tensor2d(img_tensor):
    tensor = img_tensor.permute(1, 2, 0)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    plt.imshow(tensor)

def plot_voxel3d(voxel_tensor, shape=(64, 64, 8), size=2):
    visualize_labeled_array3d((voxel_tensor.view(shape).float()).detach().cpu().numpy(), size=size, marker='box')