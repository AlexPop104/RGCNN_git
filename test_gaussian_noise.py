from operator import add
from turtle import shape
import open3d as o3d

from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

import random

import torch as t
import numpy as np

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch_geometric.datasets import ShapeNet
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DenseDataLoader
from torch_geometric.transforms import FixedPoints
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from typing import Optional, Union

from GaussianNoiseTransform import GaussianNoiseTransform

def classic_testing():
    num_points = 2048
    root = "/media/rambo/ssd2/Alex_data/RGCNN/ShapeNet/" 

    transforms1 = Compose([FixedPoints(num_points), GaussianNoiseTransform(mu=0, sigma=0.000001)])

    transforms = Compose([FixedPoints(num_points), GaussianNoiseTransform(mu=0, sigma=0.05)])

    dataset_original = ShapeNet(root=root, split="test", transform=transforms1)
    dataset_transformed = ShapeNet(root=root, split="test", transform=transforms)
    iter_original = iter(dataset_original)
    iter_transformed = iter(dataset_transformed)
    
    print("---" * 25)
    print(dataset_original[0])

    pcd_original = dataset_original[0]
    pcd_noisy    = dataset_transformed[0]

    colors_o = pcd_original.y.reshape([2048, 1]).clone().expand(-1, 3)
    colors_n = pcd_noisy.y.reshape([2048, 1]).clone().expand(-1, 3)

    writer.add_3d('original_pcd', 
    {
        'vertex_positions': pcd_original.pos, 
        'vertex_features': colors_o
    }, 0)
    
    writer.add_3d('noisy_pcd', 
    {
        'vertex_positions': pcd_noisy.pos ,
        'vertex_features': colors_n
    }, 1)

def add_to_tensorboard(name, writer, pos, labels, preds):

    colors_o = t.tensor(labels).reshape([2048, 1]).expand(-1, 3)
    colors_p = t.tensor(preds).reshape([2048, 1]).expand(-1, 3)
    writer.add_3d(name, 
    {
        'vertex_positions': pos,
        'vertex_labels': colors_o,
        'vertex_predictions': colors_p
    }, 0)



if __name__ == '__main__':
    root = '/home/victor/workspace/thesis_ws/github/RGCNN_git/'
    pos_n = np.load(root + "positions_noisy.npy")
    lab_n = np.load(root + "label_original_noisy.npy")
    pred_n = np.load(root + "label_predicted_noisy.npy")

    pos_o = np.load(root + "positions_original.npy")
    lab_o = np.load(root + "label_original_original.npy")
    pred_o = np.load(root + "label_predicted_original.npy")
    idx = 2
    
    add_to_tensorboard('original_pcd', writer, pos_o[idx], lab_o[idx], pred_o[idx])
    add_to_tensorboard('noisy_pcd', writer, pos_n[idx], lab_n[idx], pred_n[idx])

    print('ok')