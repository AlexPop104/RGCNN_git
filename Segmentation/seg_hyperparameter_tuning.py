from collections import defaultdict
from datetime import datetime
from torch_geometric.loader import DenseDataLoader
import os
import ChebConv_rgcnn as conv
import torch as t
from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
import torch
from torch import float32, nn
import time
from torch.nn.functional import one_hot
from torch.nn.functional import relu
from sklearn.utils import class_weight
torch.manual_seed(0)
from torch_geometric.transforms import Compose
from GaussianNoiseTransform import GaussianNoiseTransform
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np

writer = SummaryWriter(comment='_sn_ww_025_drop_weight_decay', filename_suffix='_no_reg')


def IoU_accuracy(pred, target, n_classes=16):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)


def compute_loss(logits, y, x, L, criterion, s=1e-9):
    if not logits.device == y.device:
        y = y.to(logits.device)

    loss = criterion(logits, y)
    l=0
    for i in range(len(x)):
        l += (1/2) * t.linalg.norm(t.matmul(t.matmul(t.permute(x[i], (0, 2, 1)), L[i]), x[i]))**2
    l = l * s
    loss += l
    return loss