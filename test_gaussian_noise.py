import open3d as o3d

from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

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

if __name__ == '__main__':
    if False:
        num_points = 2048
        root = "/media/rambo/ssd2/Alex_data/RGCNN/ShapeNet/" 
        transforms = Compose([FixedPoints(num_points), GaussianNoiseTransform(mu=0, sigma=0.01)])
        dataset_original = ShapeNet(root=root, split="test", transform=FixedPoints(num_points))
        dataset_transformed = ShapeNet(root=root, split="test", transform=transforms)
        iter_original = iter(dataset_original)
        iter_transformed = iter(dataset_transformed)

        print("---" * 25)

        pcd_original = iter_original.__next__()
        pcd_noisy    = iter_transformed.__next__()

        print(pcd_original.x - pcd_noisy.x)

        writer.add_3d('original_pcd', 
        {
            'vertex_positions': pcd_original.pos, 
            'vertex_features': pcd_original.x
        }, 0)
        
        writer.add_3d('noisy_pcd', 
        {
            'vertex_positions': pcd_noisy.pos ,
            'vertex_features': pcd_noisy.x
        }, 0)
    modelnet_num = 40
    num_points = 1024
    root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)+'/' 
    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale(), GaussianNoiseTransform(0, 0.01)])
    dataset = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)
    iter_data = iter(dataset)

    pcd = iter_data.__next__()
    print(pcd.pos)
    print(pcd.x)
    writer.add_3d('original_pcd', 
    {
        'vertex_positions': pcd.pos, 
        'vertex_features': pcd.x
    }, 0)

    print("ok")

