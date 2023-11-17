import sys
sttr_repo_path = '/workspace/stereo-transformer'
sys.path.insert(0, sttr_repo_path) 
from module.sttr import STTR
from utilities.misc import NestedTensor, batched_index_select
import torch
import torch.nn as nn
import torch.nn.functional as F

class NestedTensor(object):
    def __init__(self, left, right, disp=None, sampled_cols=None, sampled_rows=None, occ_mask=None,
                 occ_mask_right=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.occ_mask = occ_mask
        self.occ_mask_right = occ_mask_right
        self.sampled_cols = sampled_cols
        self.sampled_rows = sampled_rows

def batched_index_select(source, dim, index):
    views = [source.shape[0]] + [1 if i != dim else -1 for i in range(1, len(source.shape))]
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(source, dim, index)


def load_STTR_model(pretrained_weight_path):
    # Default parameters
    args = type('', (), {})() # create empty args
    args.channel_dim = 128
    args.position_encoding='sine1d_rel'
    args.num_attn_layers=6
    args.nheads=8
    args.regression_head='ot'
    args.context_adjustment_layer='cal'
    args.cal_num_blocks=8
    args.cal_feat_dim=16
    args.cal_expansion_ratio=4


    model = STTR(args).cuda().eval()

    # Load the pretrained model
    model_file_name = pretrained_weight_path
    checkpoint = torch.load(model_file_name)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict, strict=False) # prevent BN parameters from breaking the model loading
    print("Pre-trained model successfully loaded.")
    
    return model
