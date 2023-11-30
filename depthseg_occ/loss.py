import sys
import random
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


#sem_scal_loss, geo_scal_loss, CE_ssc_loss

def geo_scal_loss(pred, ssc_target, epsilon=1e-6):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Check if prediction matches target exactly (pseudo-prediction case)
    if torch.all(pred.argmax(dim=1) == ssc_target):
        return torch.tensor(0.0).to(pred.device)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels (if necessary)
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + epsilon)
    recall = intersection / (nonempty_target.sum() + epsilon)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum() + epsilon)

    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, epsilon=1e-6):
    # Check for perfect match
    if torch.all(pred.argmax(dim=1) == ssc_target):
        return torch.tensor(0.0).to(pred.device)

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(n_classes):
        # Get probability of class i
        p = pred[:, i, :, :, :][mask]
        target = ssc_target[mask]

        completion_target = (target == i).float()

        if completion_target.sum() > 0:
            count += 1.0
            nominator = (p * completion_target).sum()

            # Precision
            precision = nominator / (p.sum() + epsilon)
            loss_precision = F.binary_cross_entropy(
                precision, torch.ones_like(precision)
            )

            # Recall
            recall = nominator / (completion_target.sum() + epsilon)
            loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

            # Specificity
            specificity = ((1 - p) * (1 - completion_target)).sum() / (
                (1 - completion_target).sum() + epsilon
            )
            loss_specificity = F.binary_cross_entropy(
                specificity, torch.ones_like(specificity)
            )

            loss_class = loss_precision + loss_recall + loss_specificity
            loss += loss_class

    return loss / max(count, epsilon)

def CE_ssc_loss(pred, target, class_weights):
    # Check for perfect match
    if torch.all(pred.argmax(dim=1) == target):
        return torch.tensor(0.0).to(pred.device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target != 255]
    return torch.mean(loss_valid)



def one_hot_encoding(labels, num_classes):
    # Assuming labels of shape [batch_size, depth, height, width]
    one_hot = F.one_hot(labels, num_classes)  # Convert to one-hot
    return one_hot.permute(0, 4, 1, 2, 3).float()


    