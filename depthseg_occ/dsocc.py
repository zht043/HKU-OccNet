import sys
import random
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from STTR_Loader import load_STTR_model
from STTR_Loader import NestedTensor, batched_index_select


from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math 


import torch
from deformable_attention import DeformableAttention3D

from loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss, one_hot_encoding

def inference(model, left_img, right_img, gt, device):
    left_img = left_img.to(device)
    right_img = right_img.to(device)
    gt = gt.to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_1h = model(left_img, right_img)
    
    pred = torch.argmax(pred_1h, dim=1)
    return pred


def instantiate_dsocc(sttr_weight_path, num_classes, device):
    
    sttr = STTR(sttr_weight_path, device)
    print("[STTR]Number of parameters (in millions):", sum(p.numel() for p in sttr.parameters()) / 1_000_000, 'M')
    
    segformer = SegFormer(device)
    print("[SegFormer]Number of parameters (in millions):", sum(p.numel() for p in segformer.parameters()) / 1_000_000, 'M')
    
    sttr = sttr.to(device)
    segformer = segformer.to(device)
    model = DepthSegLift_OCC(num_classes, sttr, segformer)
    model = model.to(device)
    print("[DSOCC]Number of parameters (in millions):", sum(p.numel() for p in model.parameters()) / 1_000_000, 'M')
    print("[DSOCC]Number of trainable parameters (in millions):", 
          sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000, 'M')

    return model


class DepthSegLift_OCC(nn.Module):
    def __init__(self, num_classes, sttr, segformer):
        super(DepthSegLift_OCC, self).__init__()

        self.sttr = sttr
        self.seg = segformer

        ch_in = 26
        ch_out = 64
        self.ConvResBlock1 = nn.Sequential(
            Conv2D_Block(ch_in=ch_in, ch_out=ch_out, k_size=3),
            ResNet2D_Block(ch=ch_out, k_size=3),
            nn.MaxPool2d(3, stride=4, padding=1)
        )
        ch_in = 64
        ch_out = 128
        self.ConvResBlock2 = nn.Sequential(
            Conv2D_Block(ch_in=ch_in, ch_out=ch_out, k_size=3),
            ResNet2D_Block(ch=ch_out, k_size=3),
            nn.MaxPool2d(3, stride=4, padding=1)
        )
        ch_in = 128
        ch_out = 256
        self.ConvResBlock3 = nn.Sequential(
            Conv2D_Block(ch_in=ch_in, ch_out=ch_out, k_size=3),
            ResNet2D_Block(ch=ch_out, k_size=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.hs_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.relu = nn.LeakyReLU()
        self.linear1 = nn.Linear(512*12*39, 128)
        self.linear2 = nn.Linear(128, 32*32*32*4)

        self.deform_attn3d = DeformableAttention3D(
            dim = 32,                          # feature dimensions
            dim_head = 32,                      # dimension per head
            heads = 4,                          # attention heads
            dropout = 0.5,                       # dropout
            downsample_factor = (2, 8, 8),      # downsample factor (r in paper)
            offset_scale = (2, 8, 8),           # scale of offset, maximum offset
            offset_kernel_size = (4, 10, 10),   # offset kernel size
        )

        self.bn2d = nn.BatchNorm2d(512)
        self.bn3d_att = nn.BatchNorm3d(32)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.header = Header(32, num_classes)
        
    def forward(self, left_img, right_img): 
        #disp, feat_left, feat_right, attn_weight = self.sttr(left_img, right_img)
        #disp = self.sttr(left_img, right_img)

        batch_size = left_img.size(0)
        disp_list = []
        for i in range(batch_size):
            single_left_img = left_img[i].unsqueeze(0)  # Add batch dimension
            single_right_img = right_img[i].unsqueeze(0)  # Add batch dimension
            single_disp = self.sttr(single_left_img, single_right_img)
            disp_list.append(single_disp)
        disp = torch.cat(disp_list, dim=0)

        logit, hidden_state = self.seg(left_img)
        x = torch.cat((disp.unsqueeze(1), logit, left_img, right_img), dim=1)
        x = self.ConvResBlock1(x)
        x = self.ConvResBlock2(x)
        x = self.ConvResBlock3(x)

        seg_hs = self.hs_conv(hidden_state)
        seg_hs = self.relu(seg_hs)
        seg_hs = F.adaptive_avg_pool2d(seg_hs, (12, 39))
        x = torch.cat((x, seg_hs), dim=1)
        x = self.bn2d(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = x.view(-1, 32, 32, 32, 4)
        x_att = x.permute(0, 1, 4, 2, 3)
        x_att = self.deform_attn3d(x_att)
        x_att = x_att.permute(0, 1, 3, 4, 2)
        x_att = self.bn3d_att(x_att)
        x_att = self.relu(x_att)
        
        x = x + x_att 
        
        logit = self.header(x)
        return logit
        
    def step(self, left_img, right_img, gt, class_weights):
        pred_1h = model(left_img, right_img)

        loss = sem_scal_loss(pred_1h, gt)
        loss += geo_scal_loss(pred_1h, gt)
        loss += CE_ssc_loss(pred_1h, gt, class_weights)
        return loss







class Conv3D_Block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1):
        super(Conv3D_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch_out),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class ResNet3D_Block(nn.Module):
    def __init__(self, ch, k_size, stride=1, p=1):
        super(ResNet3D_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p), 
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x) + x
        return out

# class ConvUpSample3D_Block(nn.Module):
#     def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
#         super(ConvUpSample3D_Block, self).__init__()
#         self.up = nn.Sequential(
#             nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
#             nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
#         )
#     def forward(self, x):
#         return self.up(x)

class Conv2D_Block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1):
        super(Conv2D_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ResNet2D_Block(nn.Module):
    def __init__(self, ch, k_size, stride=1, p=1):
        super(ResNet2D_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=k_size, stride=stride, padding=p),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(ch, ch, kernel_size=k_size, stride=stride, padding=p),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out




class Header(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(Header, self).__init__()
        self.C = input_channel
        self.num_classes = num_classes
        
        # Upsample layer to double the spatial dimensions
        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(self.C, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        # Final convolutional layer to produce logits for each class
        self.final_conv = nn.Conv3d(128, self.num_classes, kernel_size=1)

    def forward(self, x):
        # x is expected to have the shape [B, C, 32, 32, 4]
        
        # Upscale to double the spatial dimensions: [B, C, 64, 64, 8]
        x = self.up_scale_2(x)

        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Get num_classes for each voxel
        x = self.final_conv(x)

        # The output tensor shape will be [B, num_classes, 64, 64, 8]
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        # Cross-Attention
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        # Add & Norm (Residual Connection and Layer Normalization)
        query = self.norm1(query + attn_output)
        return query


class STTR_InputAdapterLayer(nn.Module):
    def __init__(self, device, downsample=3):
        super(STTR_InputAdapterLayer, self).__init__()
        self.downsample = downsample
        self.device = device

    def forward(self, left_imgs, right_imgs):
        bs, _, h, w = left_imgs.shape  # Extract batch size, height, and width

        col_offset = int(self.downsample / 2)
        row_offset = int(self.downsample / 2)
        sampled_cols = torch.arange(col_offset, w, self.downsample)[None,].expand(bs, -1).to(self.device)
        sampled_rows = torch.arange(row_offset, h, self.downsample)[None,].expand(bs, -1).to(self.device)


        # Create NestedTensor for the batch
        nested_tensor = NestedTensor(left_imgs, right_imgs,  
                                    sampled_cols=sampled_cols, sampled_rows=sampled_rows)

        return nested_tensor

class STTR(nn.Module):
    def __init__(self, pretrained_weight_path, device, requires_grad = False):
        super(STTR, self).__init__()
        self.sttr_adapter_layer = STTR_InputAdapterLayer(device, downsample=3)
        self.sttr_pt = load_STTR_model(pretrained_weight_path, device)

        #print(next(self.sttr_pt.parameters()).device)
        for param in self.sttr_pt.parameters():
            param.requires_grad = requires_grad
        
    def forward(self, left_img, right_img): 
        x = self.sttr_adapter_layer(left_img, right_img)

        # bs, _, h, w = x.left.size()
        # feat = self.sttr_pt.backbone(x)
        # tokens = self.sttr_pt.tokenizer(feat)
        # pos_enc = self.sttr_pt.pos_encoder(x)
        # # separate left and right
        # feat_left = tokens[:bs]
        # feat_right = tokens[bs:]  # NxCxHxW
        # # downsample
        # if x.sampled_cols is not None:
        #     feat_left = batched_index_select(feat_left, 3, x.sampled_cols)
        #     feat_right = batched_index_select(feat_right, 3, x.sampled_cols)
        # if x.sampled_rows is not None:
        #     feat_left = batched_index_select(feat_left, 2, x.sampled_rows)
        #     feat_right = batched_index_select(feat_right, 2, x.sampled_rows)
        # attn_weight = self.sttr_pt.transformer(feat_left, feat_right, pos_enc)
        # output = self.sttr_pt.regression_head(attn_weight, x)

        output = self.sttr_pt(x)
        disp_map = output['disp_pred']
        occ_map = output['occ_pred'] > 0.5
        disp_map[occ_map] = 0.0
        
        return disp_map #, feat_left, feat_right, attn_weight



class SegFormer(nn.Module):
    def __init__(self, device, requires_grad=False):
        super(SegFormer, self).__init__()
        
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        for param in self.segformer.parameters():
            param.requires_grad = requires_grad

        self.device = device
        
    def forward(self, img): 
        inputs = self.feature_extractor(images=img, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden_state = self.segformer.segformer(**inputs).last_hidden_state      
        logits = self.segformer(**inputs).logits
        logits = F.interpolate(logits, size=(376, 1241), mode='bilinear', align_corners=False)
        return logits, hidden_state





