import torch
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import FixedPoints
from torch_geometric.transforms import Compose
from torch_geometric.data import DenseDataLoader
import numpy as np
from sklearn.utils import class_weight
root = "/media/rambo/ssd2/Alex_data/RGCNN/ShapeNet/"

transforms = Compose([FixedPoints(num=2048)])

dataset = ShapeNet(root=root, split="test", transform=transforms)
loader  = DenseDataLoader(dataset, batch_size=1)
print(dataset)


def get_weights(dataset, sk=True):
    weights = torch.zeros(50)
    if sk:
        y = np.empty(len(dataset)*dataset[0].y.shape[0])
        print(y.shape)
        i = 0
        for data in dataset:
            y[i:2048+i] = data.y
            i+=2048
        weights = class_weight.compute_class_weight(
            'balanced', np.unique(y), y)
    else:
        for data in dataset:
            for l in torch.unique(data.y):
                weights[l] += torch.sum(data.y == l.item())
        weights = 1 - weights / len(dataset)
    print(weights)
    return weights

if __name__ == "__main__":
    weights = get_weights(dataset)

    print(weights)

