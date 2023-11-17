import torch
from torch import nn

    

class StereoToVoxelNetLoss(nn.Module):
    def __init__(self):
        super(StereoToVoxelNetLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets, voxel_occluded):
        """
        Compute the BCE loss only for non-occluded voxels.
        
        Args:
            inputs: Predicted voxel data. e.g. shape (-1,256,256,32) with value(0,1) for binary or shape (-1,256,256,32,X) with value(0,1) for muiti-semantic X classes
            targets: Ground truth voxel data. e.g. shape (-1,256,256,32) with value(0,1) for binary or shape (-1,256,256,32,X) with value(0,1) for muiti-semantic X classes
            voxel_occluded: Tensor indicating whether each voxel is occluded. e.g. shape (-1,256,256,32) with value(0,1) for the voxel is valid(0) or not(1) 
        
        Returns:
            Loss value computed only on non-occluded voxels.
        """
        # Compute BCE loss for all voxels
        loss = self.bce_loss(inputs, targets)

        # Apply mask to ignore occluded voxels in the loss computation
        mask = (voxel_occluded == 0)
        masked_loss = loss * mask

        # Average the loss over non-occluded voxels only
        return torch.sum(masked_loss) / torch.sum(mask)


# 使用方法    
