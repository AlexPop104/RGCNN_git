import torch
import torch_geometric
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose
from torch_geometric.transforms import LinearTransformation
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
import sys

import torch.nn as nn
import torch_geometric.utils as utils
import torch_geometric.nn.conv as conv

import os
from datetime import datetime
now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
parent_directory = "/home/victor/workspace/thesis_ws/models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)

## NOTE:

# ----------------------------------------------------------------
# Hyper parameters:
num_points = 512
batch_size_nr = 2   # not yet used
num_epochs = 100
learning_rate = 0.001
modelnet_num = 10    # 10 or 40

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, modelnet_num]


# ----------------------------------------------------------------
# Choosing device:
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

####################################################################################

transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

root = "/home/victor/workspace/thesis_ws/datasets/Modelnet" + str(modelnet_num)
print(root)
dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

# Shuffle Data
dataset_train = dataset_train.shuffle()
dataset_test = dataset_test.shuffle()

# Verification...
print(f"Train dataset shape: {dataset_train}")
print(f"Test dataset shape:  {dataset_test}")

print(dataset_train[0])

#################################################################################

class GetGraph(nn.Module):
    def __init__(self):
        """
        Creates the weighted adjacency matrix 'W'
        Taked directly from RGCNN
        """

        super(GetGraph, self).__init__()

    def forward(self, point_cloud):
        point_cloud_transpose = point_cloud.permute(0, 2, 1)
        point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2 * point_cloud_inner
        point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
        point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
        adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
        adj_matrix = torch.exp(-adj_matrix)
        return adj_matrix


class GetLaplacian(nn.Module):
    def __init__(self, normalize=True):
        """
        Computes the Graph Laplacian from a Weighted Graph
        Taken directly from RGCNN - currently not used - might need to find alternatives in PyG for loss function
        """
        super(GetLaplacian, self).__init__()
        self.normalize = normalize

        def diag(self, mat):
        # input is batch x vertices
            d = []
            for vec in mat:
                d.append(torch.diag(vec))
            return torch.stack(d)

    def forward(self, adj_matrix):
        if self.normalize:
            D = torch.sum(adj_matrix, dim=1)
            eye = torch.ones_like(D)
            eye = self.diag(eye)
            D = 1 / torch.sqrt(D)
            D = self.diag(D)
            L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
        else:
            D = torch.sum(adj_matrix, dim=1)
            D = torch.diag(D)
            L = D - adj_matrix
        return L


class RGCNN_model(nn.Module):
    def __init__(self, vertice, F, K, M, regularization = 0, dropout = 0):
        # verify the consistency w.r.t. the number of layers
        assert len(F) == len(K)
        super(RGCNN_model, self).__init__()
        '''
        F := List of Convolutional Layers dimensions
        K := List of Chebyshev polynomial degrees
        M := List of Fully Connected Layers dimenstions
        
        Currently the dimensions are 'hardcoded'
        '''

        self.F = F
        self.K = K
        self.M = M

        self.vertice = vertice
        self.regularization = regularization    # gamma from the paper: 10^-9
        self.dropout = dropout
        self.regularizers = []

        # initialize the model layers
        self.get_graph = GetGraph()
        # self.get_laplacian = GetLaplacian(normalize=True)
        self.pool = nn.MaxPool1d(self.vertice)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.dropout)

        ###################################################################
        #                               Hardcoded Values for Conv filters
        self.conv1 = conv.ChebConv(6, 128, 6)
        self.conv2 = conv.ChebConv(128, 512, 5)
        self.conv3 = conv.ChebConv(512, 1024, 3)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, modelnet_num)
        ###################################################################


    def forward(self, x, batch, batch_size, nr_points):

        with torch.no_grad():
            out_reshaped_graph = torch.reshape(x, (batch_size * nr_points, 6))
            self.regularizers = []
            W   = self.get_graph(x)  # we don't want to compute gradients when building the graph
            edge_index, edge_weight = utils.dense_to_sparse(W)

        #out = self.conv1(x, edge_index, edge_weight,batch)
        out = self.conv1(out_reshaped_graph, edge_index, edge_weight, batch)
        out = self.relu(out)
        
        with torch.no_grad():
            edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
            L_edge_index, L_edge_weight = torch_geometric.utils.get_laplacian(edge_index, edge_weight, normalization="sym")
            L = torch_geometric.utils.to_dense_adj(edge_index = L_edge_index, edge_attr = L_edge_weight)     

            out=out.unsqueeze(0)
            self.regularizers.append(torch.linalg.norm(torch.matmul(torch.matmul(torch.Tensor.permute(out, [0, 2, 1]), L), out)))
            out=out.squeeze(0)
            out_reshaped_graph=torch.reshape(out, (batch_size, nr_points, 128))
            W   = self.get_graph(out_reshaped_graph)
            edge_index, edge_weight = utils.dense_to_sparse(W)

        #out = self.conv2(out, edge_index, edge_weight)

        out = self.conv2(out, edge_index, edge_weight, batch)
        out = self.relu(out)

        with torch.no_grad():
            edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
            L_edge_index, L_edge_weight = torch_geometric.utils.get_laplacian(edge_index, edge_weight, normalization="sym")
            L = torch_geometric.utils.to_dense_adj(edge_index=L_edge_index, edge_attr=L_edge_weight)

            out = out.unsqueeze(0)

            self.regularizers.append(torch.linalg.norm(torch.matmul(torch.matmul(torch.Tensor.permute(out, [0, 2, 1]), L), out)))

            out = out.squeeze(0)

            out_reshaped_graph=torch.reshape(out,(batch_size,nr_points, 512))

            W   = self.get_graph(out_reshaped_graph)
            edge_index, edge_weight = utils.dense_to_sparse(W)

            #out = self.conv3(out, edge_index, edge_weight)

        out = self.conv3(out, edge_index, edge_weight, batch)
        out = self.relu(out)

        with torch.no_grad():
            edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
            L_edge_index, L_edge_weight = torch_geometric.utils.get_laplacian(edge_index, edge_weight, normalization="sym")
            L = torch_geometric.utils.to_dense_adj(edge_index=L_edge_index, edge_attr=L_edge_weight)

            out=out.unsqueeze(0)

            self.regularizers.append(torch.linalg.norm(torch.matmul(torch.matmul(torch.Tensor.permute(out, [0, 2, 1]), L), out)))

            out=out.squeeze(0)

            out_reshaped_graph=torch.reshape(out,(batch_size,nr_points, 1024))

            #out = out.permute(0, 2, 1) # Transpose

            out=out_reshaped_graph.permute(0, 2, 1)

        out = self.pool(out)
        out.squeeze_(2)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        

        '''
        for param in self.fc1.parameters():
            self.regularizers.append(torch.linalg.norm(param))
        
        for param in self.fc2.parameters():
            self.regularizers.append(torch.linalg.norm(param))

        for param in self.fc3.parameters():
            self.regularizers.append(torch.linalg.norm(param))
        '''

        return out, self.regularizers


# Training

# PATH = "/home/alex/Alex_pyt_geom/Models/model"

#model_number = 5                # Change this acording to the model you want to load
# model.load_state_dict(torch.load(path + '/model' + str(model_number) + '.pt'))

train_loader = DataLoader(dataset_train, batch_size=batch_size_nr, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size_nr)

model = RGCNN_model(num_points, F, K, M, dropout=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()

def get_loss(y, labels, regularization, regularizers):
    cross_entropy_loss = loss(y, labels)
    s = torch.sum(torch.as_tensor(regularizers))
    regularization *= s
    l = cross_entropy_loss + regularization
    return l

import time
import torch.nn.functional as F

model.train()
for epoch in range(1, 51):
    i=0
    total_loss = 0
    accuracy = 0
    now_time = time.time()
    for data in train_loader:
        i= i + 1
        # print(i)
        optimizer.zero_grad()  # Clear gradients.

        pos = data.pos.cuda()        # (num_points * 3)   
        normals = data.normal.cuda() # (num_points * 3)

        batch=data.batch

        nr_points=int(pos.shape[0]/batch_size_nr)
        
        x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
        x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
        
        x=torch.reshape(x, (batch_size_nr, nr_points,6))

        x = x.type(torch.float32)  # other types of data may be unstable

        y = data.y              # (1)
        y = y.type(torch.long)  # required by the loss function

        x = x.to(device)      # to CUDA if available
        y = y.to(device)

        batch=batch.to(device)
        
        logits, regularizers = model(x,batch, batch_size_nr, nr_points)  # Forward pass.
        

        #loss = loss_criterion(logits, y)  # Loss computation.

        l = get_loss(logits, y, regularization=1e-9, regularizers=regularizers) 
        
        l.backward()  # Backward pass.
        
        optimizer.step()  # Update model parameters.
        
        total_loss += l.item() * data.num_graphs

        acc = sum(logits.max(axis=1).indices == y) / logits.shape[0]
        accuracy += acc

        if (i % 100) == 0:
            print(f"Iter: {i} - Loss: {total_loss/i} - Acc - {acc}")

    epoch_loss = total_loss / i
    
    print(f"--- Epoch: {epoch} --- Loss: {epoch_loss} --- Time: {time.time() - now_time}")
    print(f"--- Accuracy: {accuracy/i}")

'''
def train(model, optimizer, loader, batch_size):
    model.train()
    i=0
    
    total_loss = 0
    for data in loader:
        i=i+1
        # print(i)
        optimizer.zero_grad()  # Clear gradients.

        pos = data.pos.cuda()        # (num_points * 3)   
        normals = data.normal.cuda() # (num_points * 3)

        batch=data.batch

        nr_points=int(pos.shape[0]/batch_size)
        
        
        x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
        x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
        
        x=torch.reshape(x, (batch_size, nr_points,6))

        x = x.type(torch.float32)  # other types of data may be unstable

        y = data.y              # (1)
        y = y.type(torch.long)  # required by the loss function

        x = x.to(device)      # to CUDA if available
        y = y.to(device)

        batch=batch.to(device)
        

        logits,regularizers = model(x,batch,batch_size,nr_points)  # Forward pass.
        

        #loss = loss_criterion(logits, y)  # Loss computation.

        l = get_loss(logits, y, regularization=1e-9, regularizers=regularizers) 
        l.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += l.item() * data.num_graphs
        if (i % 101) == 0:
            print(f"Iter: {i} - Loss: {total_loss/i} ")

    return total_loss / len(train_loader.dataset)


for epoch in range(1, 51):
    loss = train(model, optimizer, train_loader, batch_size_nr)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
'''


##########################################################

#Previous iteration, working without batch data


# correct_percentage_list = []
# loss = torch.nn.CrossEntropyLoss()
# model.train()
# for epoch in range(num_epochs):

#     correct = 0
#     for i, data in enumerate(dataset_train):
#         # make sure the gradients are empty
#         optimizer.zero_grad()
        
#         # Data preparation 
#         pos = data.pos        # (num_points * 3)   
#         normals = data.normal # (num_points * 3)
#         x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
#         x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
#         x = x.type(torch.float32)  # other types of data may be unstable

#         y = data.y              # (1)
#         y = y.type(torch.long)  # required by the loss function
        
#         x = x.to(device)      # to CUDA if available
#         y = y.to(device)
     
#         # Forward pass
#         y_pred, regularizers = model(x)     # (1 * 40)
        
#         class_pred = torch.argmax(y_pred.squeeze(0))  # (1)  
#         correct += int((class_pred == y).sum())       # to compute the accuracy for each epoch
        

#         # loss and backward
#         ###################################################################################
#         #                           CrossEntropyLoss
#         # This WORKS but I am testing the other way...
#         l = loss(y_pred, y)   # one value
#         # l.backward()          # update gradients
#         ###################################################################################
       
#         #l = get_loss(y_pred, y, regularization=1e-9, regularizers=regularizers)
#         l.backward()

#         # optimisation
#         optimizer.step()
        
            
#         if i%100==0:
#             print(f"Epoch: {epoch}, Sample: {i}, Loss:{l} - Predicted class vs Real Cass: {class_pred} <-> {y.item()}")
#             # print(torch.sum(torch.as_tensor(regularizers)))
#         if epoch%5==0:
#             torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')
#     print(f"~~~~~~~~~ CORRECT: {correct / len(dataset_train)} ~~~~~~~~~~~")
#     correct_percentage_list.append(correct / len(dataset_train))
# print(correct_percentage_list)

# torch.save(model.state_dict(), "/home/alex/Alex_pyt_geom/Models/final_model.pt")

# with torch.no_grad():
#     model.eval()
#     correct = 0
#     for data in dataset_test:
#         pos = data.pos        # (num_points * 3)   
#         normals = data.normal # (num_points * 3)
#         x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
#         x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
#         x = x.type(torch.float32)  # other types of data may be unstable

#         y = data.y              # (1)
#         y = y.type(torch.long)  # required by the loss function
        
#         x = x.to(device)      # to CUDA if available
#         y = y.to(device)
     
#         # Forward pass
#         y_pred, _ = model(x)     # (1 * 40)

#         class_pred = torch.argmax(y_pred)
#         correct += int((class_pred == y).sum())

#     print(f"Correct percentage : {correct / len(dataset_test)}")