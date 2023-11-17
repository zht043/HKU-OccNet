import torch
from torch import nn
import torch.nn.functional as F
class StereoToVoxelNet(nn.Module):
    def __init__(self):
        super(StereoToVoxelNet, self).__init__()
        # Branch for each image
        self.branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: [-1, 32, 621, 188]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: [-1, 64, 311, 94]
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Merging and producing voxel output
        self.merge_and_output = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Output: [-1, 64, 311, 94]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [-1, 32, 622, 188]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=4, padding=1, output_padding=1),  # Output: [-1, 32, 2488, 752]
            # Additional layers can be added to refine output and match the desired dimensions
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, image2, image3):
        branch_out_2 = self.branch(image2)
        branch_out_3 = self.branch(image3)

        merged = torch.cat([branch_out_2, branch_out_3], dim=1)
        voxel_output = self.merge_and_output(merged)

        # Adjust the output to the desired shape [batch_size, 256, 256, 32]
        # This might require resizing, cropping, or additional layers
        # Here's an example with adaptive average pooling to resize
        voxel_output = F.adaptive_avg_pool2d(voxel_output, (256, 256)).permute(0, 2, 3, 1)
        voxel_output = self.sigmoid(voxel_output)
        return voxel_output

    
if __name__=="__main__":
    pass