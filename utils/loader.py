from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import bqplot.scales
import ipyvolume as ipv
import bqplot.scales
import numpy as np
import ipywidgets as widgets
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt




SEMANTICKITTI_DIR = "/workspace/Dataset/dataset"
def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed

def voxel_to_coordinates(voxel_data, voxel_size=0.01, threshold = 0):
    # 获取体素数据的形状
    xx, yy, zz = voxel_data.shape

    # 初始化坐标和标签列表
    x_coords = []
    y_coords = []
    z_coords = []
    tags = []

    # 遍历体素数据
    for x in range(xx):
        for y in range(yy):
            for z in range(zz):
                tag = voxel_data[x, y, z]
                # 只记录非零标签的体素
                if tag > threshold:
                    x_coords.append(x * voxel_size)
                    y_coords.append(y * voxel_size)
                    z_coords.append(z * voxel_size)
                    tags.append(tag)

    return np.array(x_coords), np.array(y_coords), np.array(z_coords), np.array(tags)


class SemanticKITTIDataset(Dataset):
    def __init__(self, root_dir=SEMANTICKITTI_DIR, mode='train', sequences=['00'], split_ratio=0.8, downsample=False, get_vox_lidar=False, get_vox_invalid=False, get_vox_occluded=False):
        """
        Args:
            root_dir (string): Directory with all the images and corresponding voxel data.
            mode (string): 'train' or 'test' mode.
            sequences (list of strings): List of sequences to load, default is ['00'].
        """
        self.root_dir = root_dir
        self.mode = mode
        self.sequences = sequences
        self.image_names = []
        self.image2_dir = {}
        self.image3_dir = {}
        self.voxels_dir = {}
        self.downsample = downsample
        self.get_vox_lidar=False
        self.get_vox_invalid=False
        self.get_vox_occluded=False
        
        self.scene_size = [51.2, 51.2, 6.4] #unit m, 51.2m = 256 * 0.2m
        self.vox_origin = [0, -25.6, -2]#unit m
        self.voxel_size = 0.2  #m
        self.raw_class_labels= {
                0: "unlabeled",#
                1: "outlier",
                10: "car",#
                11: "bicycle",#
                13: "bus",
                15: "motorcycle",#
                16: "on-rails",
                18: "truck",#
                20: "other-vehicle",#
                30: "person",#
                31: "bicyclist",#
                32: "motorcyclist",#
                40: "road",#
                44: "parking",#
                48: "sidewalk",#
                49: "other-ground",#
                50: "building",#
                51: "fence",#
                52: "other-structure",
                60: "lane-marking",
                70: "vegetation",#
                71: "trunk",#
                72: "terrain",#
                80: "pole",#
                81: "traffic-sign",#
                99: "other-object",
                252: "moving-car",
                253: "moving-bicyclist",
                254: "moving-person",
                255: "moving-motorcyclist",
                256: "moving-on-rails",
                257: "moving-bus",
                258: "moving-truck",
                259: "moving-other-vehicle"
        }
        self.class_map = {
            0 : 0,     # "unlabeled"
            1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 1,     # "car"
            11: 2,     # "bicycle"
            13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 3,     # "motorcycle"
            16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 4,     # "truck"
            20: 5,     # "other-vehicle"
            30: 6,     # "person"
            31: 7,     # "bicyclist"
            32: 8,     # "motorcyclist"
            40: 9,     # "road"
            44: 10,    # "parking"
            48: 11,    # "sidewalk"
            49: 12,    # "other-ground"
            50: 13,    # "building"
            51: 14,    # "fence"
            52: 15,     # "other-structure" mapped to "unlabeled" ------------------mapped ####
            60: 9,     # "lane-marking" to "road" ---------------------------------mapped
            70: 15,    # "vegetation"
            71: 16,    # "trunk"
            72: 17,    # "terrain"
            80: 18,    # "pole"
            81: 19,    # "traffic-sign"
            99: 15,     # "other-object" to "unlabeled" ----------------------------mapped ####
            252: 1,    # "moving-car" to "car" ------------------------------------mapped
            253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 6,    # "moving-person" to "person" ------------------------------mapped
            255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 4,    # "moving-truck" to "truck" --------------------------------mapped
            259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }


        self.class_freqs = np.array(
            [
                5.41773033e09,
                1.57835390e07,
                1.25136000e05,
                1.18809000e05,
                6.46799000e05,
                8.21951000e05,
                2.62978000e05,
                2.83696000e05,
                2.04750000e05,
                6.16887030e07,
                4.50296100e06,
                4.48836500e07,
                2.26992300e06,
                5.68402180e07,
                1.57196520e07,
                1.58442623e08,
                2.06162300e06,
                3.69705220e07,
                1.15198800e06,
                3.34146000e05,
            ]
        )
        self.class_names = [
            "empty",
            "car",
            "bicycle",
            "motorcycle",
            "truck",
            "other-vehicle",
            "person",
            "bicyclist",
            "motorcyclist",
            "road",
            "parking",
            "sidewalk",
            "other-ground",
            "building",
            "fence",
            "vegetation",
            "trunk",
            "terrain",
            "pole",
            "traffic-sign",
        ]

        self.class_weights = torch.from_numpy(
            1 / np.log(self.class_freqs+ 0.001)
        )

        for seq in self.sequences:
            self.image2_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_2')
            self.image3_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_3')
            self.voxels_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'voxels')
            seq_image_names = [f.split('.')[0] for f in os.listdir(self.image2_dir[seq]) if f.endswith('.png')]
            self.image_names.extend([(seq, img_name) for img_name in seq_image_names])
        
        split_idx = int(len(self.image_names) * split_ratio )
        
        if self.mode == 'train':
            self.image_names = self.image_names[:split_idx]
        else:
            self.image_names = self.image_names[split_idx:]
        
        # Define transformations for standardization and conversion to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # Standard ImageNet std
        ])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        if self.get_vox_lidar:
            voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        if self.get_vox_invalid:
            invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        if self.get_vox_occluded:
            occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')
        
        
        # Load and transform images
        left_image = self.transform(Image.open(img2_name).convert('RGB'))
        right_image = self.transform(Image.open(img3_name).convert('RGB'))

        extra_data = []
        # Load voxel data and convert to tensor
        if self.get_vox_lidar:
            voxel_data = torch.tensor(unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
            extra_data.append(voxel_data)
        #  0 invalid 1-19 class and 255 invalid
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32).astype(np.float32)
        voxel_labels = torch.tensor(np.vectorize(self.class_map.get)(voxel_labels))
        #print(voxel_data.shape)
        #print(voxel_labels.shape)
        
        if self.get_vox_occluded:
            voxel_occluded = torch.tensor(unpack(np.fromfile(occluded_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
            extra_data.append(voxel_occluded)
        
        if self.get_vox_invalid:
            voxel_invalid = torch.tensor(unpack(np.fromfile(invalid_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
            extra_data.append(voxel_invalid)
        
        # if self.downsample:
        #     voxel_labels = self._downsample_label(voxel_labels, downscale=2)
        #     voxel_occluded = self._downsample_label(voxel_occluded, downscale=2)
        
        
        if len(extra_data) != 0:
            return left_image, right_image, voxel_labels, extra_data
        else:
            return left_image, right_image, voxel_labels
    
    
    def get_data(self, idx):
        return self.__getitem__(idx)
    



    
if __name__=="__main__":
    # Example usage
    dataset = SemanticKITTIDataset(root_dir='/workspace/Dataset/dataset', mode='train', sequences=['00'])
    print(len(dataset))
    print(dataset[0])  # Access the first sample in the dataset

    

def plot_tensor2d(img_tensor):
    tensor = img_tensor.permute(1, 2, 0)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    plt.imshow(tensor)

def visualize_3D_points_in_jupyter(points_3D, size=2, marker="sphere"):
    # Assuming points_3D is a N x 3 numpy array
    x = points_3D[:, 0]
    y = points_3D[:, 1]
    z = points_3D[:, 2]

    ipv.quickscatter(x, y, z, size=size, marker=marker)
    ipv.show()

def voxel_to_coordinates(voxel_data, voxel_size=0.01, threshold = 0):
    # 获取体素数据的形状
    xx, yy, zz = voxel_data.shape

    # 初始化坐标和标签列表
    x_coords = []
    y_coords = []
    z_coords = []
    tags = []

    # 遍历体素数据
    for x in range(xx):
        for y in range(yy):
            for z in range(zz):
                tag = voxel_data[x, y, z]
                # 只记录非零标签的体素
                if tag > threshold:
                    x_coords.append(x * voxel_size)
                    y_coords.append(y * voxel_size)
                    z_coords.append(z * voxel_size)
                    tags.append(tag)

    return np.array(x_coords), np.array(y_coords), np.array(z_coords), np.array(tags)




def visualize_voxels(voxel_data, key = 'voxel_labels',size = None, marker = None):

        max_idx = {
            'voxel_invalid':1,
            'voxel_occluded':1,
            'voxel':1,
            'mapped_label':21,
        }
            
        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = -1 if key in ['voxel_invalid','voxel_occluded'] else 0)

        # 创建颜色比例尺
        color_scale = bqplot.scales.ColorScale(min=0, max=max_idx.get(key, 255), colors=["#f00", "#0f0", "#00f"])

        fig = ipv.figure()

        # 确定tags中的唯一值
        unique_tags = np.unique(tags)

        # 为每个唯一的tag值创建一个scatter
        for tag in unique_tags:
            # 过滤出当前tag的坐标
            mask = tags == tag
            x_filtered, y_filtered, z_filtered, tags_f = x[mask], y[mask], z[mask], tags[mask]
            # 创建scatter
            if key=='voxel_labels':
                ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="{}, len({})={}".format(labels[tag],str(tag),x_filtered.shape[0]))
            elif key=='mapped_label':
                ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="{}, len({})={}".format(labels[learning_map_inv[tag]],str(tag),x_filtered.shape[0]))
            else:
                ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="len({})={}".format(str(tag),x_filtered.shape[0]))
        #ipv.scatter(1-y,x, z, color=tags, color_scale=color_scale, marker=marker or 'box', size=size or 0.1)
        ipv.xyzlabel('y','x','z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()

def visualize_labeled_array3d(voxel_data, num_classes=21, size = None, marker = None):
        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0])

        # 创建颜色比例尺
        color_scale = bqplot.scales.ColorScale(min=0, max=num_classes, colors=["#f00", "#0f0", "#00f"])

        fig = ipv.figure()

        # 确定tags中的唯一值
        unique_tags = np.unique(tags)

        # 为每个唯一的tag值创建一个scatter
        for tag in unique_tags:
            # 过滤出当前tag的坐标
            mask = tags == tag
            x_filtered, y_filtered, z_filtered, tags_f = x[mask], y[mask], z[mask], tags[mask]
            
            ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="len({})={}".format(str(tag),x_filtered.shape[0]))
        ipv.xyzlabel('y','x','z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()
        