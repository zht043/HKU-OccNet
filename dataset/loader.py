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

from dataset.utils import unpack
from dataset.config import SEMANTICKITTI_DIR
labels= {
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
learning_map_inv= {
        0: 0,
        1: 10,
        2: 11,
        3: 15,
        4: 18,
        5: 20,
        6: 30,
        7: 31,
        8: 32,
        9: 40,
        10: 44,
        11: 48,
        12: 49,
        13: 50,
        14: 51,
        15: 70,
        16: 71,
        17: 72,
        18: 80,
        19: 81
}
learning_map = {
    0:0,
    10:1,
    11:2,
    15:3,
    18:4,
    20:5,
    30:6,
    31:7,
    12:8,
    40:9,
    44:10,
    48:11,
    49:12,
    50:13,
    51:14,
    70:15,
    71:16,
    72:17,
    80:18,
    81:19
}

label_mapping_for_training = {
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

labels_lite = ['unlabeled', 'vehicle', 'two-wheeler', 'person', 'road', 'parking', 'sidewalk', 'building', 'natural', 'manmade']

label_mapping_lite_inv = {
    "unlabeled": 0,      # empty, outlier, other-structure, other-object, other-ground
    "vehicle": 1,        # car, moving-car, truck, moving-truck, other-vehicle, moving-other-vehicle, bus, on-rails, moving-bus, moving-on-rails
    "two-wheeler": 2,    # bicycle, moving-bicyclist, motorcycle, moving-motorcyclist
    "person": 3,         # person, moving-person
    "road": 4,           # road, lane-marking
    "parking": 5,        # parking
    "sidewalk": 6,       # sidewalk
    "building": 7,       # building
    "natural": 8,        # vegetation, trunk, terrain
    "manmade": 9,        # pole, traffic-sign, fence
}


label_mapping_lite_from_ori = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 1,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,     # "motorcycle"
    16: 1,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 1,     # "truck"
    20: 1,     # "other-vehicle"
    30: 3,     # "person"
    31: 3,     # "bicyclist"
    32: 3,     # "motorcyclist"
    40: 4,     # "road"
    44: 4,    # "parking"
    48: 5,    # "sidewalk"
    49: 5,    # "other-ground"
    50: 6,    # "building"
    51: 8,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped ####
    60: 4,     # "lane-marking" to "road" ---------------------------------mapped
    70: 7,    # "vegetation"
    71: 7,    # "trunk"
    72: 7,    # "terrain"
    80: 8,    # "pole"
    81: 8,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped ####
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 3,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 3,    # "moving-person" to "person" ------------------------------mapped
    255: 3,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 1,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 1,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 1,    # "moving-truck" to "truck" --------------------------------mapped
    259: 1,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
semantic_kitti_class_frequencies = np.array(
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
kitti_class_names = [
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

class_weights = torch.from_numpy(
    1 / np.log(semantic_kitti_class_frequencies + 0.001)
)

scene_size = torch.tensor([51.2, 51.2, 6.4], dtype=torch.float) #unit m, 51.2m = 256 * 0.2m
vox_origin = torch.tensor(np.array([0, -25.6, -2]))
voxel_size = torch.tensor([0.2])  # 0.2m

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
    def __init__(self, root_dir = SEMANTICKITTI_DIR, mode='train', sequences=['00'], downsample = False):
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
        
    
        for seq in self.sequences:
            self.image2_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_2')
            self.image3_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_3')
            self.voxels_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'voxels')
            seq_image_names = [f.split('.')[0] for f in os.listdir(self.image2_dir[seq]) if f.endswith('.png')]
            self.image_names.extend([(seq, img_name) for img_name in seq_image_names])
        
        #random.shuffle(self.image_names)
        split_idx = int(len(self.image_names) * 0.8 )
        
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

        # [Your existing code to get image paths]
        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')
        
        
        
        # Load and transform images
        image2 = self.transform(Image.open(img2_name).convert('RGB'))
        image3 = self.transform(Image.open(img3_name).convert('RGB'))

        # Load voxel data and convert to tensor
        voxel_data = torch.tensor(unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        #  0 invalid 1-19 class and 255 invalid
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32).astype(np.float32)
        voxel_labels = torch.tensor(np.vectorize(label_mapping_for_training.get)(voxel_labels))
        #print(voxel_data.shape)
        #print(voxel_labels.shape)
        
        
        voxel_occluded = torch.tensor(unpack(np.fromfile(occluded_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        
        voxel_invalid = torch.tensor(unpack(np.fromfile(invalid_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        
            
        return image2, image3, voxel_data, voxel_labels, voxel_occluded, voxel_invalid # Return voxel_occluded as well
    
    
    def get_meta_data(self):
        return kitti_class_names, class_weights, scene_size, vox_origin, voxel_size
    def get_data(self, idx):
        return self.__getitem__(idx)
    
    
    def _downsample_label(self, label, downscale=2):
        """
        Downsample the labeled data by a given factor.

        Args:
            label (numpy.ndarray): The original label data.
            downscale (int): The factor by which to downsample the label data.

        Returns:
            numpy.ndarray: The downsampled label data.
        """
        if downscale == 1:
            return label

        input_shape = label.shape
        output_shape = (input_shape[0] // downscale, input_shape[1] // downscale, input_shape[2] // downscale)
        label_downscaled = np.zeros(output_shape, dtype=label.dtype)

        for x in range(output_shape[0]):
            for y in range(output_shape[1]):
                for z in range(output_shape[2]):
                    # Extract the corresponding block from the input label
                    block = label[x*downscale:(x+1)*downscale, y*downscale:(y+1)*downscale, z*downscale:(z+1)*downscale]

                    # Find the most common label in the block, excluding labels 0 and 255
                    labels, counts = np.unique(block, return_counts=True)
                    mask = (labels != 0) & (labels != 255)
                    labels, counts = labels[mask], counts[mask]
                    if labels.size > 0:
                        label_downscaled[x, y, z] = labels[np.argmax(counts)]
                    else:
                        # If the block contains only labels 0 and 255, use the majority between them
                        label_downscaled[x, y, z] = 0 if np.sum(block == 0) > np.sum(block == 255) else 255

        return label_downscaled
    
    def get_dict(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')

        # Load images
        image2 = Image.open(img2_name).convert('RGB')
        image3 = Image.open(img3_name).convert('RGB')
        
        # 1241*376
        
        # Load voxel data
        voxel_data = unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32)
        #  0 invalid 1-19 class and 255 invalid
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32)
        voxel_invalid = unpack(np.fromfile(invalid_name, dtype=np.uint8)).reshape(256, 256, 32)
        voxel_occluded = unpack(np.fromfile(occluded_name, dtype=np.uint8)).reshape(256, 256, 32)

        sample = {
            'image2': image2, 
            'image3': image3, 
            'voxel': voxel_data, 
            'voxel_labels': voxel_labels,
            'voxel_invalid': voxel_invalid,
            'voxel_occluded': voxel_occluded
        }

        return sample
    
    def to_images_voxels(self, idx):
        sample = self.get_dict(idx)
        left, right, voxel = sample['image2'], sample['image3'], sample['voxel']

        # Define transformations for standardization and conversion to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # Standard ImageNet std
        ])

        # Apply transformations
        left = transform(left)
        right = transform(right)

        # Convert voxel data to tensor
        voxel = torch.tensor(voxel, dtype=torch.float)

        return left, right, voxel
    
    def plot(self, idx):
        import matplotlib.pyplot as plt
        import numpy as np

        sample = self.get_dict(idx)

        # 可视化图像和体素数据
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 现在是 4 个子图

        # 显示image2
        axes[0].imshow(sample['image2'])
        axes[0].set_title('Image 2')

        # 显示image3
        axes[1].imshow(sample['image3'])
        axes[1].set_title('Image 3')

        # 显示体素数据的一个切片
        voxel_bev = np.zeros(sample['voxel'].shape[:2], dtype=sample['voxel'].dtype)
        for i in range(sample['voxel'].shape[0]):
            for j in range(sample['voxel'].shape[1]):
                column = sample['voxel'][i, j, :]
                top_nonzero = np.max(column[np.nonzero(column)]) if np.any(column) else 0
                voxel_bev[i, j] = top_nonzero
        
        #voxel_slice = sample['voxel'][:, :, 15]  # 选择中间一个切片
        #flipped_voxel_bev = 
        axes[2].imshow(np.flip(np.flip(voxel_bev, 0), 1), cmap='gray')
        axes[2].set_title('BEV of Voxel')
        

#         # 生成并显示voxel_labels的BEV图
#         bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)
#         for i in range(sample['voxel_labels'].shape[0]):
#             for j in range(sample['voxel_labels'].shape[1]):
#                 column = sample['voxel_labels'][i, j, :]
#                 top_nonzero = np.max(column[np.nonzero(column)]) if np.any(column) else 0
#                 bev_image[i, j] = top_nonzero

#         axes[3].imshow(bev_image, cmap='rainbow')
#         axes[3].set_title('BEV of Voxel Labels')

        # 检查每个列是否被遮挡
        occluded = np.min(sample['voxel_occluded'], axis=2) == 1

        # 初始化BEV图像
        bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)

        # 对于未遮挡的列，找到最高非零值
        non_occluded_indices = np.where(~occluded)
        bev_image[non_occluded_indices] = np.max(np.where(sample['voxel_labels'][non_occluded_indices] > 0, sample['voxel_labels'][non_occluded_indices], 0), axis=1)

###### WIP
#         # 检查每个列是否被遮挡
#         occluded = sample['voxel_occluded'] == 0

#         # 初始化BEV图像
#         bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)

#         # 找到每个未遮挡列中最高的未遮挡层
#         for i in range(bev_image.shape[0]):
#             for j in range(bev_image.shape[1]):
#                 # 如果该列至少有一个未遮挡的体素
#                 if np.min(occluded[i, j, :]) == 0:
#                     # 翻转列以找到最高的未遮挡层索引
#                     flipped_column = np.flip(occluded[i, j, :])
#                     top_unoccluded_idx = len(flipped_column) - 1 - np.argmax(flipped_column)
#                     # 获取该点对应的label值
#                     bev_image[i, j] = sample['voxel_labels'][i, j, top_unoccluded_idx]
                    
        #flipped_bev_image = 
        axes[3].imshow(np.flip(np.flip(bev_image, 0), 1), cmap='jet')
        axes[3].set_title('BEV of Voxel Labels (voxel_occluded Filtered)')

    

    
    def visualize_voxels(self, idx, key = None, size = None, marker = None):


        sample = self.get_dict(idx) # sample['voxel_labels'] 是输入的体素数据

        voxel_data = sample[key or 'voxel_labels']
        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = -1 if key in ['voxel_invalid','voxel_occluded'] else 0)

        # 创建颜色比例尺
        color_scale = bqplot.scales.ColorScale(min=0, max=1 if key in ['voxel_invalid','voxel_occluded','voxel'] else 255, colors=["#f00", "#0f0", "#00f"])

        fig = ipv.figure()

        # 确定tags中的唯一值
        unique_tags = np.unique(tags)

        # 为每个唯一的tag值创建一个scatter
        for tag in unique_tags:
            # 过滤出当前tag的坐标
            mask = tags == tag
            x_filtered, y_filtered, z_filtered, tags_f = x[mask], y[mask], z[mask], tags[mask]
            # 创建scatter
            ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="{}, len({})={}".format(labels[tag],str(tag),x_filtered.shape[0]))

        ipv.xyzlabel('y','x','z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()

    

    def visualize_voxels_pred(self, voxel_data, size=None, marker=None, threshold=0.2, unit=0.005):
        """
        Visualize the predicted voxel data.

        Args:
            pred: Predicted voxel data.
            size: Size of the markers in the scatter plot.
            marker: Type of marker to use in scatter plot.
        """
        # Assuming 'pred' is a 3D numpy array (H, W, D) with values from 0 to 1

        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = threshold)

        # Categorize the predictions
        tags_categorized = (tags // unit).astype(int)  # Divide predictions into categories

        # Color scale: Create as many colors as categories
        color_scale = bqplot.scales.ColorScale(min=0, max=1, colors=["#0000", "#AFAF"])

        fig = ipv.figure()

        # Iterate over categories
        for tag_cat in np.unique(tags_categorized):
            # Filter out coordinates and predictions for the current category
            mask = tags_categorized == tag_cat
            x_filtered, y_filtered, z_filtered, tags_filtered = x[mask], y[mask], z[mask], tags[mask]

            # Create scatter plot for the current category
            ipv.scatter(1 - y_filtered, x_filtered, z_filtered, color=tags_filtered, 
                        color_scale=color_scale, description=str(tag_cat*unit) , marker=marker or 'box', size=size or 0.1)

        ipv.xyzlabel('y', 'x', 'z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()
    
if __name__=="__main__":
    # Example usage
    dataset = SemanticKITTIDataset(root_dir='/workspace/Dataset/dataset', mode='train', sequences=['00'])
    print(len(dataset))
    print(dataset[0])  # Access the first sample in the dataset

    

    

class SemanticKITTIDatasetTest(SemanticKITTIDataset):
    def __init__(self, root_dir = SEMANTICKITTI_DIR, mode='train', sequences=['00'], downsample = False):
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
        

        for seq in self.sequences:
            self.image2_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_2')
            self.image3_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_3')
            self.voxels_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'voxels')
            seq_image_names = [f.split('.')[0] for f in os.listdir(self.image2_dir[seq]) if f.endswith('.png')]
            self.image_names.extend([(seq, img_name) for img_name in seq_image_names])
        
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

        # [Your existing code to get image paths]
        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')
        
        # Load and transform images
        image2 = self.transform(Image.open(img2_name).convert('RGB'))
        image3 = self.transform(Image.open(img3_name).convert('RGB'))

        # Load voxel data and convert to tensor
        voxel_data = torch.tensor(unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        #  0 invalid 1-19 class and 255 invalid
        voxel_labels = torch.tensor(np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32).astype(np.float32))
        #print(voxel_data.shape)
        #print(voxel_labels.shape)
        
        #voxel_data = torch.tensor(np.fromfile(voxel_name, dtype=np.uint8).reshape(256, 256, 32), dtype=torch.float)
        
        voxel_occluded = torch.tensor(unpack(np.fromfile(occluded_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        if self.downsample:
            voxel_labels = self._downsample_label(voxel_labels, downscale=2)
            voxel_occluded = self._downsample_label(voxel_occluded, downscale=2)
            
        return image2, image3, voxel_labels, voxel_occluded  # Return voxel_occluded as well
    
    

class SemanticKITTIDataset2(Dataset):
    def __init__(self, root_dir = SEMANTICKITTI_DIR, mode='train', sequences=['00'], downsample = False):
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
        
    
        for seq in self.sequences:
            self.image2_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_2')
            self.image3_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_3')
            self.voxels_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'voxels')
            seq_image_names = [f.split('.')[0] for f in os.listdir(self.image2_dir[seq]) if f.endswith('.png')]
            self.image_names.extend([(seq, img_name) for img_name in seq_image_names])
        
        #random.shuffle(self.image_names)
        split_idx = int(len(self.image_names) * 0.8 )
        
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

        # [Your existing code to get image paths]
        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')
        
        
        
        # Load and transform images
        image2 = self.transform(Image.open(img2_name).convert('RGB'))
        image3 = self.transform(Image.open(img3_name).convert('RGB'))
        
        
        
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32).astype(np.float32)
        if self.downsample:
            voxel_labels = self._downsample_label(voxel_labels, downscale=2)
            #voxel_occluded = self._downsample_label(voxel_occluded, downscale=2)
        
        voxel_labels = torch.tensor(np.vectorize(label_mapping_for_training.get)(voxel_labels))
        
        # Load voxel data and convert to tensor
        #voxel_data = torch.tensor(unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        #  0 invalid 1-19 class and 255 invalid
        #print(voxel_data.shape)
        #print(voxel_labels.shape)
        
        image2 = F.pad(image2, (3, 2, 1, 1))  # (left, right, top, bottom)
        image3 = F.pad(image3, (3, 2, 1, 1))  # (left, right, top, bottom)
        
        return image2, image3, voxel_labels#, voxel_occluded, voxel_invalid # Return voxel_occluded as well
    
    
    def get_meta_data(self):
        return kitti_class_names, class_weights, scene_size, vox_origin, voxel_size
    def get_data(self, idx):
        return self.__getitem__(idx)
    
    
    def _downsample_label(self, label, downscale=2):
        """
        Downsample the labeled data by a given factor.

        Args:
            label (numpy.ndarray): The original label data.
            downscale (int): The factor by which to downsample the label data.

        Returns:
            numpy.ndarray: The downsampled label data.
        """
        if downscale == 1:
            return label

        input_shape = label.shape
        output_shape = (input_shape[0] // downscale, input_shape[1] // downscale, input_shape[2] // downscale)
        label_downscaled = np.zeros(output_shape, dtype=label.dtype)

        for x in range(output_shape[0]):
            for y in range(output_shape[1]):
                for z in range(output_shape[2]):
                    # Extract the corresponding block from the input label
                    block = label[x*downscale:(x+1)*downscale, y*downscale:(y+1)*downscale, z*downscale:(z+1)*downscale]

                    # Find the most common label in the block, excluding labels 0 and 255
                    labels, counts = np.unique(block, return_counts=True)
                    mask = (labels != 0) & (labels != 255)
                    labels, counts = labels[mask], counts[mask]
                    if labels.size > 0:
                        label_downscaled[x, y, z] = labels[np.argmax(counts)]
                    else:
                        # If the block contains only labels 0 and 255, use the majority between them
                        label_downscaled[x, y, z] = 0 if np.sum(block == 0) > np.sum(block == 255) else 255

        return label_downscaled
    
    def get_dict(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')

        # Load images
        image2 = Image.open(img2_name).convert('RGB')
        image3 = Image.open(img3_name).convert('RGB')
        
        # 1241*376
        
        # Load voxel data
        voxel_data = unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32)
        #  0 invalid 1-19 class and 255 invalid
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32)
        voxel_invalid = unpack(np.fromfile(invalid_name, dtype=np.uint8)).reshape(256, 256, 32)
        voxel_occluded = unpack(np.fromfile(occluded_name, dtype=np.uint8)).reshape(256, 256, 32)

        sample = {
            'image2': image2, 
            'image3': image3, 
            'voxel': voxel_data, 
            'voxel_labels': voxel_labels,
            'voxel_invalid': voxel_invalid,
            'voxel_occluded': voxel_occluded
        }

        return sample
    
    def to_images_voxels(self, idx):
        sample = self.get_dict(idx)
        left, right, voxel = sample['image2'], sample['image3'], sample['voxel']

        # Define transformations for standardization and conversion to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # Standard ImageNet std
        ])

        # Apply transformations
        left = transform(left)
        right = transform(right)

        # Convert voxel data to tensor
        voxel = torch.tensor(voxel, dtype=torch.float)

        return left, right, voxel
    
    def plot(self, idx):
        import matplotlib.pyplot as plt
        import numpy as np

        sample = self.get_dict(idx)

        # 可视化图像和体素数据
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 现在是 4 个子图

        # 显示image2
        axes[0].imshow(sample['image2'])
        axes[0].set_title('Image 2')

        # 显示image3
        axes[1].imshow(sample['image3'])
        axes[1].set_title('Image 3')

        # 显示体素数据的一个切片
        voxel_bev = np.zeros(sample['voxel'].shape[:2], dtype=sample['voxel'].dtype)
        for i in range(sample['voxel'].shape[0]):
            for j in range(sample['voxel'].shape[1]):
                column = sample['voxel'][i, j, :]
                top_nonzero = np.max(column[np.nonzero(column)]) if np.any(column) else 0
                voxel_bev[i, j] = top_nonzero
        
        #voxel_slice = sample['voxel'][:, :, 15]  # 选择中间一个切片
        #flipped_voxel_bev = 
        axes[2].imshow(np.flip(np.flip(voxel_bev, 0), 1), cmap='gray')
        axes[2].set_title('BEV of Voxel')
        

#         # 生成并显示voxel_labels的BEV图
#         bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)
#         for i in range(sample['voxel_labels'].shape[0]):
#             for j in range(sample['voxel_labels'].shape[1]):
#                 column = sample['voxel_labels'][i, j, :]
#                 top_nonzero = np.max(column[np.nonzero(column)]) if np.any(column) else 0
#                 bev_image[i, j] = top_nonzero

#         axes[3].imshow(bev_image, cmap='rainbow')
#         axes[3].set_title('BEV of Voxel Labels')

        # 检查每个列是否被遮挡
        occluded = np.min(sample['voxel_occluded'], axis=2) == 1

        # 初始化BEV图像
        bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)

        # 对于未遮挡的列，找到最高非零值
        non_occluded_indices = np.where(~occluded)
        bev_image[non_occluded_indices] = np.max(np.where(sample['voxel_labels'][non_occluded_indices] > 0, sample['voxel_labels'][non_occluded_indices], 0), axis=1)

###### WIP
#         # 检查每个列是否被遮挡
#         occluded = sample['voxel_occluded'] == 0

#         # 初始化BEV图像
#         bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)

#         # 找到每个未遮挡列中最高的未遮挡层
#         for i in range(bev_image.shape[0]):
#             for j in range(bev_image.shape[1]):
#                 # 如果该列至少有一个未遮挡的体素
#                 if np.min(occluded[i, j, :]) == 0:
#                     # 翻转列以找到最高的未遮挡层索引
#                     flipped_column = np.flip(occluded[i, j, :])
#                     top_unoccluded_idx = len(flipped_column) - 1 - np.argmax(flipped_column)
#                     # 获取该点对应的label值
#                     bev_image[i, j] = sample['voxel_labels'][i, j, top_unoccluded_idx]
                    
        #flipped_bev_image = 
        axes[3].imshow(np.flip(np.flip(bev_image, 0), 1), cmap='jet')
        axes[3].set_title('BEV of Voxel Labels (voxel_occluded Filtered)')

    

    
    def visualize_voxels(self, idx, key = None, size = None, marker = None):


        sample = self.get_dict(idx) # sample['voxel_labels'] 是输入的体素数据

        voxel_data = sample[key or 'voxel_labels']
        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = -1 if key in ['voxel_invalid','voxel_occluded'] else 0)

        # 创建颜色比例尺
        color_scale = bqplot.scales.ColorScale(min=0, max=1 if key in ['voxel_invalid','voxel_occluded','voxel'] else 255, colors=["#f00", "#0f0", "#00f"])

        fig = ipv.figure()

        # 确定tags中的唯一值
        unique_tags = np.unique(tags)

        # 为每个唯一的tag值创建一个scatter
        for tag in unique_tags:
            # 过滤出当前tag的坐标
            mask = tags == tag
            x_filtered, y_filtered, z_filtered, tags_f = x[mask], y[mask], z[mask], tags[mask]
            # 创建scatter
            ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="{}, len({})={}".format(labels[tag],str(tag),x_filtered.shape[0]))

        ipv.xyzlabel('y','x','z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()

    

    def visualize_voxels_pred(self, voxel_data, size=None, marker=None, threshold=0.2, unit=0.005):
        """
        Visualize the predicted voxel data.

        Args:
            pred: Predicted voxel data.
            size: Size of the markers in the scatter plot.
            marker: Type of marker to use in scatter plot.
        """
        # Assuming 'pred' is a 3D numpy array (H, W, D) with values from 0 to 1

        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = threshold)

        # Categorize the predictions
        tags_categorized = (tags // unit).astype(int)  # Divide predictions into categories

        # Color scale: Create as many colors as categories
        color_scale = bqplot.scales.ColorScale(min=0, max=1, colors=["#0000", "#AFAF"])

        fig = ipv.figure()

        # Iterate over categories
        for tag_cat in np.unique(tags_categorized):
            # Filter out coordinates and predictions for the current category
            mask = tags_categorized == tag_cat
            x_filtered, y_filtered, z_filtered, tags_filtered = x[mask], y[mask], z[mask], tags[mask]

            # Create scatter plot for the current category
            ipv.scatter(1 - y_filtered, x_filtered, z_filtered, color=tags_filtered, 
                        color_scale=color_scale, description=str(tag_cat*unit) , marker=marker or 'box', size=size or 0.1)

        ipv.xyzlabel('y', 'x', 'z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()
    
if __name__=="__main__":
    # Example usage
    dataset = SemanticKITTIDataset(root_dir='/workspace/Dataset/dataset', mode='train', sequences=['00'])
    print(len(dataset))
    print(dataset[0])  # Access the first sample in the dataset

    

    


class SemanticKITTIDatasetLite(Dataset):
    def __init__(self, root_dir = SEMANTICKITTI_DIR, mode='train', sequences=['00'], downsample = True):
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
        
    
        for seq in self.sequences:
            self.image2_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_2')
            self.image3_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'image_3')
            self.voxels_dir[seq] = os.path.join(root_dir, 'sequences', seq, 'voxels')
            seq_image_names = [f.split('.')[0] for f in os.listdir(self.image2_dir[seq]) if f.endswith('.png')]
            self.image_names.extend([(seq, img_name) for img_name in seq_image_names])
        
        #random.shuffle(self.image_names)
        split_idx = int(len(self.image_names) * 0.8 )
        
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

        # [Your existing code to get image paths]
        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')
        
        
        
        # Load and transform images
        image2 = self.transform(Image.open(img2_name).convert('RGB'))
        image3 = self.transform(Image.open(img3_name).convert('RGB'))
        
        
        
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32).astype(np.float32)
        if self.downsample:
            voxel_labels = voxel_labels[::2, ::2, ::2]
            #voxel_labels = self._downsample_label(voxel_labels, downscale=2)
            #voxel_occluded = self._downsample_label(voxel_occluded, downscale=2)
        
        voxel_labels = torch.tensor(np.vectorize(label_mapping_lite_from_ori.get)(voxel_labels))
        
        # Load voxel data and convert to tensor
        #voxel_data = torch.tensor(unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32).astype(np.float32))
        #  0 invalid 1-19 class and 255 invalid
        #print(voxel_data.shape)
        #print(voxel_labels.shape)
        
        image2 = F.pad(image2, (3, 2, 1, 1))  # (left, right, top, bottom)
        image3 = F.pad(image3, (3, 2, 1, 1))  # (left, right, top, bottom)
        
        return image2, image3, voxel_labels#, voxel_occluded, voxel_invalid # Return voxel_occluded as well
    
    
    def get_meta_data(self):
        return kitti_class_names, class_weights, scene_size, vox_origin, voxel_size
    def get_data(self, idx):
        return self.__getitem__(idx)
    
    
    def _downsample_label(self, label, downscale=2):
        """
        Downsample the labeled data by a given factor.

        Args:
            label (numpy.ndarray): The original label data.
            downscale (int): The factor by which to downsample the label data.

        Returns:
            numpy.ndarray: The downsampled label data.
        """
        if downscale == 1:
            return label

        input_shape = label.shape
        output_shape = (input_shape[0] // downscale, input_shape[1] // downscale, input_shape[2] // downscale)
        label_downscaled = np.zeros(output_shape, dtype=label.dtype)

        for x in range(output_shape[0]):
            for y in range(output_shape[1]):
                for z in range(output_shape[2]):
                    # Extract the corresponding block from the input label
                    block = label[x*downscale:(x+1)*downscale, y*downscale:(y+1)*downscale, z*downscale:(z+1)*downscale]

                    # Find the most common label in the block, excluding labels 0 and 255
                    labels, counts = np.unique(block, return_counts=True)
                    mask = (labels != 0) & (labels != 255)
                    labels, counts = labels[mask], counts[mask]
                    if labels.size > 0:
                        label_downscaled[x, y, z] = labels[np.argmax(counts)]
                    else:
                        # If the block contains only labels 0 and 255, use the majority between them
                        label_downscaled[x, y, z] = 0 if np.sum(block == 0) > np.sum(block == 255) else 255

        return label_downscaled
    
    def get_dict(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq, img_name = self.image_names[idx]
        img2_name = os.path.join(self.image2_dir[seq], img_name + '.png')
        img3_name = os.path.join(self.image3_dir[seq], img_name + '.png')
        voxel_name = os.path.join(self.voxels_dir[seq], img_name + '.bin')
        label_name = os.path.join(self.voxels_dir[seq], img_name + '.label')
        invalid_name = os.path.join(self.voxels_dir[seq], img_name + '.invalid')
        occluded_name = os.path.join(self.voxels_dir[seq], img_name + '.occluded')

        # Load images
        image2 = Image.open(img2_name).convert('RGB')
        image3 = Image.open(img3_name).convert('RGB')
        
        # 1241*376
        
        # Load voxel data
        voxel_data = unpack(np.fromfile(voxel_name, dtype=np.uint8)).reshape(256, 256, 32)
        #  0 invalid 1-19 class and 255 invalid
        voxel_labels = np.fromfile(label_name, dtype=np.uint16).reshape(256, 256, 32)
        voxel_invalid = unpack(np.fromfile(invalid_name, dtype=np.uint8)).reshape(256, 256, 32)
        voxel_occluded = unpack(np.fromfile(occluded_name, dtype=np.uint8)).reshape(256, 256, 32)

        sample = {
            'image2': image2, 
            'image3': image3, 
            'voxel': voxel_data, 
            'voxel_labels': voxel_labels,
            'voxel_invalid': voxel_invalid,
            'voxel_occluded': voxel_occluded
        }

        return sample
    
    def to_images_voxels(self, idx):
        sample = self.get_dict(idx)
        left, right, voxel = sample['image2'], sample['image3'], sample['voxel']

        # Define transformations for standardization and conversion to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # Standard ImageNet std
        ])

        # Apply transformations
        left = transform(left)
        right = transform(right)

        # Convert voxel data to tensor
        voxel = torch.tensor(voxel, dtype=torch.float)

        return left, right, voxel
    
    def plot(self, idx):
        import matplotlib.pyplot as plt
        import numpy as np

        sample = self.get_dict(idx)

        # 可视化图像和体素数据
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 现在是 4 个子图

        # 显示image2
        axes[0].imshow(sample['image2'])
        axes[0].set_title('Image 2')

        # 显示image3
        axes[1].imshow(sample['image3'])
        axes[1].set_title('Image 3')

        # 显示体素数据的一个切片
        voxel_bev = np.zeros(sample['voxel'].shape[:2], dtype=sample['voxel'].dtype)
        for i in range(sample['voxel'].shape[0]):
            for j in range(sample['voxel'].shape[1]):
                column = sample['voxel'][i, j, :]
                top_nonzero = np.max(column[np.nonzero(column)]) if np.any(column) else 0
                voxel_bev[i, j] = top_nonzero
        
        #voxel_slice = sample['voxel'][:, :, 15]  # 选择中间一个切片
        #flipped_voxel_bev = 
        axes[2].imshow(np.flip(np.flip(voxel_bev, 0), 1), cmap='gray')
        axes[2].set_title('BEV of Voxel')
        

#         # 生成并显示voxel_labels的BEV图
#         bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)
#         for i in range(sample['voxel_labels'].shape[0]):
#             for j in range(sample['voxel_labels'].shape[1]):
#                 column = sample['voxel_labels'][i, j, :]
#                 top_nonzero = np.max(column[np.nonzero(column)]) if np.any(column) else 0
#                 bev_image[i, j] = top_nonzero

#         axes[3].imshow(bev_image, cmap='rainbow')
#         axes[3].set_title('BEV of Voxel Labels')

        # 检查每个列是否被遮挡
        occluded = np.min(sample['voxel_occluded'], axis=2) == 1

        # 初始化BEV图像
        bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)

        # 对于未遮挡的列，找到最高非零值
        non_occluded_indices = np.where(~occluded)
        bev_image[non_occluded_indices] = np.max(np.where(sample['voxel_labels'][non_occluded_indices] > 0, sample['voxel_labels'][non_occluded_indices], 0), axis=1)

###### WIP
#         # 检查每个列是否被遮挡
#         occluded = sample['voxel_occluded'] == 0

#         # 初始化BEV图像
#         bev_image = np.zeros(sample['voxel_labels'].shape[:2], dtype=sample['voxel_labels'].dtype)

#         # 找到每个未遮挡列中最高的未遮挡层
#         for i in range(bev_image.shape[0]):
#             for j in range(bev_image.shape[1]):
#                 # 如果该列至少有一个未遮挡的体素
#                 if np.min(occluded[i, j, :]) == 0:
#                     # 翻转列以找到最高的未遮挡层索引
#                     flipped_column = np.flip(occluded[i, j, :])
#                     top_unoccluded_idx = len(flipped_column) - 1 - np.argmax(flipped_column)
#                     # 获取该点对应的label值
#                     bev_image[i, j] = sample['voxel_labels'][i, j, top_unoccluded_idx]
                    
        #flipped_bev_image = 
        axes[3].imshow(np.flip(np.flip(bev_image, 0), 1), cmap='jet')
        axes[3].set_title('BEV of Voxel Labels (voxel_occluded Filtered)')

    

    
    def visualize_voxels(self, idx, key = None, size = None, marker = None):


        sample = self.get_dict(idx) # sample['voxel_labels'] 是输入的体素数据

        voxel_data = sample[key or 'voxel_labels']
        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = -1 if key in ['voxel_invalid','voxel_occluded'] else 0)

        # 创建颜色比例尺
        color_scale = bqplot.scales.ColorScale(min=0, max=1 if key in ['voxel_invalid','voxel_occluded','voxel'] else 255, colors=["#f00", "#0f0", "#00f"])

        fig = ipv.figure()

        # 确定tags中的唯一值
        unique_tags = np.unique(tags)

        # 为每个唯一的tag值创建一个scatter
        for tag in unique_tags:
            # 过滤出当前tag的坐标
            mask = tags == tag
            x_filtered, y_filtered, z_filtered, tags_f = x[mask], y[mask], z[mask], tags[mask]
            # 创建scatter
            ipv.scatter(1-y_filtered,x_filtered, z_filtered, color=tags_f, color_scale=color_scale, marker=marker or 'box', size=size or 0.1, description="{}, len({})={}".format(labels[tag],str(tag),x_filtered.shape[0]))

        ipv.xyzlabel('y','x','z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()

    

    def visualize_voxels_pred(self, voxel_data, size=None, marker=None, threshold=0.2, unit=0.005):
        """
        Visualize the predicted voxel data.

        Args:
            pred: Predicted voxel data.
            size: Size of the markers in the scatter plot.
            marker: Type of marker to use in scatter plot.
        """
        # Assuming 'pred' is a 3D numpy array (H, W, D) with values from 0 to 1

        x, y, z, tags = voxel_to_coordinates(voxel_data, voxel_size = 1 / voxel_data.shape[0], threshold = threshold)

        # Categorize the predictions
        tags_categorized = (tags // unit).astype(int)  # Divide predictions into categories

        # Color scale: Create as many colors as categories
        color_scale = bqplot.scales.ColorScale(min=0, max=1, colors=["#0000", "#AFAF"])

        fig = ipv.figure()

        # Iterate over categories
        for tag_cat in np.unique(tags_categorized):
            # Filter out coordinates and predictions for the current category
            mask = tags_categorized == tag_cat
            x_filtered, y_filtered, z_filtered, tags_filtered = x[mask], y[mask], z[mask], tags[mask]

            # Create scatter plot for the current category
            ipv.scatter(1 - y_filtered, x_filtered, z_filtered, color=tags_filtered, 
                        color_scale=color_scale, description=str(tag_cat*unit) , marker=marker or 'box', size=size or 0.1)

        ipv.xyzlabel('y', 'x', 'z')
        ipv.view(0, -50, distance=2.5)
        ipv.show()
    
if __name__=="__main__":
    # Example usage
    dataset = SemanticKITTIDataset(root_dir='/workspace/Dataset/dataset', mode='train', sequences=['00'])
    print(len(dataset))
    print(dataset[0])  # Access the first sample in the dataset

    


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()
 
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
    
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
 
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target