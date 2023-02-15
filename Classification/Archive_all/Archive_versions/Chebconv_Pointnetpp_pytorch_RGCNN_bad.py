from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

import torch
from torch_geometric.transforms import SamplePoints

import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

from torch_geometric.datasets import GeometricShapes

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader

import ChebConv_rgcnn_functions as conv


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.

class PointNet(torch.nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 128)
        self.conv2 = PointNetLayer(128, 128)

        self.RGCNN_conv1= conv.DenseChebConv(128, 512, 3)
        self.RGCNN_conv2 = conv.DenseChebConv_theta_nosum(512,1024, 3)

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

        self.classifier = Sequential(Linear(1024, 512),
                                     ReLU(),
                                     Linear(512, 128),
                                     ReLU(),
                                    Linear(128, num_classes))

        
        
    def forward(self, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=pos.to(device), pos=pos.to(device), edge_index=edge_index.to(device))
        h = h.relu()
        h = self.conv2(h=h.to(device), pos=pos.to(device), edge_index=edge_index.to(device))
        h = h.relu()

        batch_size=batch.unique().shape[0]
        num_points=int(pos.shape[0]/batch_size)

        h=torch.reshape(h,(batch_size,num_points,h.shape[1]))

        with torch.no_grad():
            L = conv.pairwise_distance(h) # W - weight matrix
            L = conv.get_laplacian(L)

        h = self.RGCNN_conv1(h, L)
        h = self.relu1(h)

        with torch.no_grad():
            L = conv.pairwise_distance(h) # W - weight matrix
            L = conv.get_laplacian(L)


        h = self.RGCNN_conv2(h, L)
        h = self.relu2(h)

        h=torch.reshape(h,(batch_size*num_points,h.shape[2]))

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        
        # 5. Classifier.
        return self.classifier(h)

def train(model, optimizer, loader):
    model.train()
    
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos.to(device), data.batch.to(device))  # Forward pass.
        loss = criterion(logits, data.y.to(device))  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos.to(device), data.batch.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_correct / len(loader.dataset)



modelnet_num = 40
num_points= 512
batch_size=16


torch.manual_seed(42)


root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)

#root="/media/rambo/ssd2/Alex_data/RGCNN/GeometricShapes"

transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    
# train_dataset = GeometricShapes(root=root, train=True,
#                                 transform=transforms)
# test_dataset = GeometricShapes(root=root, train=False,
#                                transform=transforms)

train_dataset = ModelNet(root=root, train=True,
                                transform=transforms)
test_dataset = ModelNet(root=root, train=False,
                               transform=transforms)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = PointNet(num_classes=modelnet_num)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

for epoch in range(1, 251):
    loss = train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')



