from sqlite3 import SQLITE_DROP_TEMP_INDEX
from tracemalloc import stop
from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.datasets import ShapeNet
from GaussianNoiseTransform import GaussianNoiseTransform
from torch_geometric.transforms import Compose
from torch_geometric.transforms import FixedPoints
import numpy as np
import time
from torch_geometric.data.data import Data
from torch_geometric.data import InMemoryDataset


root = "/home/domsa/workspace/data/ShapeNet/"

num_points = 512

transforms = Compose([FixedPoints(num_points), GaussianNoiseTransform(
    mu=0, sigma=0, recompute_normals=False)])

dataset = ShapeNet(root, split='test', transform=transforms)
dataloader = DenseDataLoader(dataset, batch_size=6, shuffle=True)

print(dataset[0])

cat = dataset.categories
print(cat)

def count_categories(dataset):
    cat = dataset.categories
    count_cat = np.zeros_like(cat, dtype=int)
    for i, data in enumerate(dataset):
        c = data.category.item()
        count_cat[c] += 1
    return count_cat

def reduce_dataset(dataset, reduce_factor=0.6):
    count_cat = count_categories(dataset)
    print(count_cat)
    new_count = np.zeros_like(count_cat, dtype=int)
    new_data = []
    for i, data in enumerate(dataset):
        c = data.category.item()
        new_count[c] += 1
        if(new_count[c] < int(count_cat[c] * reduce_factor) or count_cat[c] < 100):
            new_data.append(data)
    return new_data

if __name__=='__main__':
    data = reduce_dataset(dataset, 0.5)
    print(len(dataset))
    print(len(data))

    print(type(dataset))

    print(dataset[0])
    print(data[0])
    
    print(type(dataset))
    print(type(data))
    
    data_loader_reduced = DenseDataLoader(data, batch_size=6)

    for data in data_loader_reduced:
        print(data)
        break
    
    for data in dataloader:
        print(data)
        break
    