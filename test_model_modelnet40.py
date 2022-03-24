# import open3d as o3d

from collections import defaultdict

# from open3d.visualization.tensorboard_plugin import summary
# from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from cls_model_rambo import cls_model
from seg_model_rambo import seg_model
import torch as t
import numpy as np
from GaussianNoiseTransform import GaussianNoiseTransform

from torch.nn.functional import one_hot

from torch_geometric.datasets import ModelNet
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import Compose
from torch_geometric.loader import DenseDataLoader
from torch_geometric.loader import DataLoader

from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import FixedPoints

from torch_geometric.transforms import NormalizeScale

PATH = '/home/victor/workspace/thesis_ws/github/RGCNN_git/models/ModelNet/18_03_22_19:04:50/1024_40_chebconv_model50.pt'

num_points = 1024
modelnet_num = 40
batch_size = 2
dropout = 1
one_layer = False
reg_prior = True

mu = 0
sigma = 0

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

transforms_original = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])
transforms_noisy    = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale(), GaussianNoiseTransform(mu, sigma, recompute_normals=True)])

root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet" + str(modelnet_num)

dataset_original = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms_original)
dataset_noisy    = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms_noisy)

loader_original  = DenseDataLoader(dataset_original, batch_size=batch_size, shuffle=True, pin_memory=True)
loader_noisy     = DenseDataLoader(dataset_noisy, batch_size=batch_size, shuffle=True, pin_memory=True)

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, 50]

model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, one_layer=one_layer, reg_prior=reg_prior)
model = model.to(device)
model.load_state_dict(t.load(PATH))

print("Testing on " + str(device))

@t.no_grad()
def test_model(loader, model, noisy=True):
    model.eval()
    total_correct = 0
    
    for data in loader:        
        if data.pos.dtype != t.float32:
            data.pos = data.pos.float()
        if data.normal.dtype != t.float32:
            data.normal = data.normal.float()
            
        x = t.cat([data.pos, data.normal], dim=2)
        x = x.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred==data.y.to(device)).sum())

    return total_correct / len(loader.dataset)

# accuracy_original   = test_model(loader=loader_original, model=model)
accuracy_noisy      = test_model(loader=loader_noisy, model=model)

# print(f"Original:   {accuracy_original * 100}%")
print(f"Noisy:      {accuracy_noisy * 100}%")