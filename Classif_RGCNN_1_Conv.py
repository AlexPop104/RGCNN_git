import torch_geometric
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
import torch_geometric.nn.conv as conv
from torch_geometric.transforms import Compose

import numpy as np
import matplotlib.pyplot as plt


num_points = 1024
batch_size = 64
modelnet_num = 10
nr_epochs=40

transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

root = "data/ModelNet"+str(modelnet_num)
print(root)
dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

# Verification...
print(f"Train dataset shape: {dataset_train}")
print(f"Test dataset shape:  {dataset_test}")

print(dataset_train[0])

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test  = DataLoader(dataset_test, batch_size=batch_size)

import torch.nn as nn
class GetGraph(nn.Module):
    def __init__(self):
        """
        Creates the weighted adjacency matrix 'W'
        Taked directly from RGCNN
        """

        super(GetGraph, self).__init__()

    def forward(self, point_cloud, batch):
        point_cloud = point_cloud.reshape(batch_size, -1, 6)
        point_cloud_transpose = point_cloud.permute(0, 2, 1)
        point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2 * point_cloud_inner
        point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
        point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
        adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
        adj_matrix = torch.exp(-adj_matrix)
        edge_index, edge_weight = utils.dense_to_sparse(adj_matrix)
        return edge_index, edge_weight

def get_graph(point_cloud, batch):
    point_cloud = point_cloud.reshape(batch.unique().shape[0], -1, 6)
    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    adj_matrix = torch.exp(-adj_matrix)
    edge_index, edge_weight = utils.dense_to_sparse(adj_matrix)

    return edge_index, edge_weight

def get_graph_v2(point_cloud, batch):
    point_cloud = point_cloud.reshape(batch.unique().shape[0], -1, 6)
    adj_matrix = torch.exp(-(torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True) - 2 * torch.matmul(point_cloud, point_cloud.permute(0, 2, 1)) + torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True).permute(0, 2, 1)))

    return utils.dense_to_sparse(adj_matrix)

import torch


from torch import nn
from torch_geometric.nn.conv import ChebConv
from torch.nn import Linear
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

class RGCNN_model(nn.Module):
    def __init__(self):
        super(RGCNN_model, self).__init__()
        self.conv1  = ChebConv(6, 128, 6)
        self.fc1    = Linear(128, modelnet_num)
        self.relu   = nn.ReLU()
        #self.get_graph = GetGraph()

    def forward(self, x, batch):
        edge_index, edge_weight = get_graph(x, batch=batch)
        out = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        out = self.relu(out)
        out = global_max_pool(out, batch)
        out = self.fc1(out)
        return out

model = RGCNN_model()
print(model)

import torch
device = "cuda"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        x = torch.cat([data.pos, data.normal], dim=1)
        logits  = model(x.to(device),  data.batch.to(device))

        pred = logits.argmax(dim=-1)

        
        loss    = criterion(logits, data.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    
    return total_loss / len(loader.dataset) 

@torch.no_grad()
def test(model, loader,modelnet_num,num_points):
    confusion_matrix=np.zeros((modelnet_num,modelnet_num))

    model.eval()

    total_correct = 0
    for data in loader:
        x = torch.cat([data.pos, data.normal], dim=1)
        logits = model(x.to(device), data.batch.to(device))
        pred = logits.argmax(dim=-1)

        iteration_batch_size=int(data.pos.shape[0]/num_points)

        for j in range(iteration_batch_size):
            confusion_matrix[data.y[j]][ pred[j]]+=1



        total_correct += int((pred == data.y.to(device)).sum())
    
    confusion_matrix=confusion_matrix/len(loader.dataset)

    return total_correct / len(loader.dataset) , confusion_matrix

all_losses=np.array([])
test_accuracy=np.array([])
confusion_matrix_collection = np.zeros((1,modelnet_num))

for epoch in range(1, nr_epochs):
    

    loss = train(model, optimizer, dataloader_train)
    all_losses=np.append(all_losses, loss)
    test_acc, confusion_matrix = test(model, dataloader_test,modelnet_num,num_points)
    test_accuracy=np.append(test_accuracy, test_acc)
    confusion_matrix_collection=np.append(confusion_matrix_collection,confusion_matrix,axis=0)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')

    trace_confusion_matrix = np.trace(confusion_matrix)
    print(trace_confusion_matrix==test_acc)

    if(epoch%5==0):
        np.save('/home/alex/Alex_documents/RGCNN_git/data/conf_matrix.npy', confusion_matrix_collection)
        np.save('/home/alex/Alex_documents/RGCNN_git/data/losses.npy', all_losses)
        np.save('/home/alex/Alex_documents/RGCNN_git/data/test_accuracy.npy', test_accuracy)

        print(confusion_matrix)



iterations=range(1,nr_epochs)
np.save('/home/alex/Alex_documents/RGCNN_git/data/losses.npy', all_losses)
plt.plot(iterations, all_losses, '-b', label='Training loss')
plt.pyplot.title('Training loss', fontdict=None, loc='center', pad=None, **kwargs)
plt.show()

np.save('/home/alex/Alex_documents/RGCNN_git/data/test_accuracy.npy', test_accuracy)
plt.plot(iterations, test_accuracy, '-r', label='Test accuracy')
plt.pyplot.title('Test accuracy', fontdict=None, loc='center', pad=None, **kwargs)
plt.show()

print(confusion_matrix_collection[1+(nr_epochs-1)*modelnet_num:1+(nr_epochs-1)*modelnet_num])
