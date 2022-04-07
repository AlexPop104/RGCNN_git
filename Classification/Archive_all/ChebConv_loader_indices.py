from torch_geometric.datasets import ModelNet
from torch_geometric.datasets import GeometricShapes

from torch_geometric.transforms import Compose
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader

import torch

from torch.utils.data import Dataset, DataLoader

class Modelnet_with_indices(Dataset):
    def __init__(self,root,modelnet_num,train_bool,transforms):
        self.ModelNet=ModelNet(root=root, name=str(modelnet_num), train=train_bool, transform=transforms)
        
    def __getitem__(self, index):
        pos, y, normal = self.ModelNet[index]
        
        return pos, y, normal, index

    def __len__(self):
        return len(self.ModelNet)

class Geometric_with_indices(Dataset):
    def __init__(self,root,train_bool,transforms):
        self.GeometricShapes=GeometricShapes(root=root, train=train_bool, transform=transforms)
        
    def __getitem__(self, index):
        pos, y, normal = self.GeometricShapes[index]
        
        return pos, y, normal, index

    def __len__(self):
        return len(self.GeometricShapes)




# num_points = 1024
# batch_size = 64
# num_epochs = 200
# learning_rate = 1e-3
# modelnet_num = 40

# ##################################

# root="/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)

# transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

# dataset_train = Modelnet_with_indices(root=root,modelnet_num=modelnet_num,train_bool=True,transforms=transforms)
# dataset_test = Modelnet_with_indices(root=root,modelnet_num=modelnet_num,train_bool=False,transforms=transforms)


# train_loader = DataLoader(dataset_train,batch_size=batch_size, shuffle=True, pin_memory=True)
# test_loader= DataLoader(dataset_test,batch_size=batch_size)

# # for batch_idx, (pos, y, normal, idx) in enumerate(test_loader):
# #     # print('Batch idx {}, dataset index {}'.format(
# #     #     batch_idx, idx))
# #     print(pos[1].shape[0])

# print(len(test_loader))

# #####################################################
# #Geometric Shapes

# root="/media/rambo/ssd2/Alex_data/RGCNN/GeometricShapes"

# transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

# dataset_train = Geometric_with_indices(root=root,train_bool=True,transforms=transforms)
# dataset_test = Geometric_with_indices(root=root,train_bool=False,transforms=transforms)


# train_loader = DataLoader(dataset_train,batch_size=batch_size, shuffle=True, pin_memory=True)
# test_loader= DataLoader(dataset_test,batch_size=batch_size)

# # for batch_idx, (pos, y, normal, idx) in enumerate(train_loader):
# #     print('Batch idx {}, dataset index {}'.format(
# #         batch_idx, idx))
# #     print(pos[1].shape[0])

# print(len(test_loader))

# #########################################################3











