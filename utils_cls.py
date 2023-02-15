from torch_geometric.datasets import ModelNet
from torch_geometric.datasets import GeometricShapes
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