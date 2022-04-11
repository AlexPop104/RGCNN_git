import torch
from torch import nn

import torch as t
import torch_geometric as tg

import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch.nn import Parameter

from torch import float32, nn

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader

from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (add_self_loops, get_laplacian,
                                   remove_self_loops)

from torch_geometric.utils import get_laplacian as get_laplacian_pyg

#from noise_transform import GaussianNoiseTransform
#import ChebConv_rgcnn_functions as conv
import os


from datetime import datetime
from torch.nn import MSELoss
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(1, '/home/alex/Alex_documents/RGCNN_git/')

from utils import GaussianNoiseTransform
import utils as util_functions

from torch.nn.functional import relu


import random
random.seed(0)



class cls_model(nn.Module):
    def __init__(self, vertice ,F, K, M, class_num, regularization=0,  dropout=1, reg_prior:bool=True, b2relu=True,input_dim=6, fc_bias=True,recompute_L=False):
        assert len(F) == len(K)
        super(cls_model, self).__init__()

        self.F = F
        self.K = K
        self.M = M

        
        
        self.reg_prior = reg_prior
        self.vertice = vertice
        self.regularization = regularization    # gamma from the paper: 10^-9
        self.dropout = dropout
        self.regularizers = []
        self.recompute_L = recompute_L

        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.relus = self.F + self.M

        self.max_pool = nn.MaxPool1d(self.vertice)

        if b2relu:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, vertice, i))) for i in self.relus
            ])
        else:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, 1, i))) for i in self.relus
                ])

        self.conv  = nn.ModuleList([
            util_functions.DenseChebConvV2(input_dim, self.F[i], self.K[i]) 
            if i==0 
            else util_functions.DenseChebConvV2(self.F[i-1], self.F[i], self.K[i]) for i in range(len(K))
        ])

        self.fc  = nn.ModuleList([])
        for i in range(len(M)):
            if i==0:
                self.fc.append(nn.Linear(self.F[-1], self.M[i], fc_bias))
            else:
                self.fc.append(nn.Linear(self.M[i-1], self.M[i], fc_bias))

        self.L = []
        self.x = []

    def b1relu(self, x, bias):
        return relu(x + bias)
    
    def brelu(self, x, bias):
        return relu(x + bias)

    def get_laplacian(self, x):
        with torch.no_grad():
            return util_functions.get_laplacian(util_functions.pairwise_distance(x))
        
    @torch.no_grad()
    def append_regularization_terms(self, x, L):
        if self.reg_prior:
            self.L.append(L)
            self.x.append(x)

    
    @torch.no_grad()
    def reset_regularization_terms(self):
        self.L = []
        self.x = []

        
    def forward(self, x):
        self.reset_regularization_terms()

        x1 = 0  # cache for layer 1

        L = self.get_laplacian(x)
        out =x

        for i in range(len(self.K)):
            out = self.conv[i](out, L)
            self.append_regularization_terms(out, L)
            out = self.dropout(out)
            out = self.brelu(out, self.bias_relus[i])
            if self.recompute_L:
                L = self.get_laplacian(out)

        out, _ = t.max(out, 1)

        for i in range(len(self.M)):
            out = self.fc[i](out)
            self.append_regularization_terms(out, L)
            out = self.dropout(out)
            out = self.b1relu(out, self.bias_relus[i + len(self.K)])
        
        return out, self.x, self.L

#criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader, regularization, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x = torch.cat([data.pos, data.normal], dim=1)   
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        

        logits,  out, L  = model(x=x.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

        logits=logits.squeeze(0)

        #logits = logits.permute([1, 0])


        #logits = logits.permute([0, 2, 1])

        y = data.y.type(torch.LongTensor)
        loss = util_functions.compute_loss(logits, y, out, L, criterion, s=regularization)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 

@torch.no_grad()
def test(model, loader, criterion):
    model.eval()

    total_loss = 0
    total_correct = 0
    for data in loader:
       
        x = torch.cat([data.pos, data.normal], dim=1)   
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        logits,  out, L  = model(x=x.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

        # logits = logits.permute([0, 2, 1])

        # loss = util_functions.compute_loss(logits, data.y.to(device), out, L, criterion, s=regularization)

        # total_loss += loss.item() * data.num_graphs
       
        

    return 0 , total_correct / len(loader.dataset) 
    #return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 


now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)

num_points = 512
batch_size = 16
num_epochs = 250
learning_rate = 1e-3
modelnet_num = 40
learning_rate = 1e-3
decay_rate = 0.8
weight_decay = 1e-8
dropout=0.25

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, modelnet_num]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")

root = "/mnt/ssd1/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
#root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
#print(root)

#model = cls_model(num_points, F, K, M, modelnet_num, dropout=1, reg_prior=True)

model = cls_model(num_points, F, K, M,
                      class_num= modelnet_num,
                      dropout=dropout, 
                      reg_prior=True, 
                      recompute_L=True,  
                      b2relu=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.




my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
regularization = 1e-9

torch.manual_seed(0)
#################################################################33

# transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

# train_dataset = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
# test_dataset = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

###################################################################

mu=0
sigma=0.
#transforms_noisy = Compose([SamplePoints(num_points),NormalizeScale(),GaussianNoiseTransform(mu, sigma,recompute_normals=True)])

rot_x=30
rot_y=30
rot_z=30

torch.manual_seed(0)

#################################


random_rotate_0 = Compose([
    RandomRotate(degrees=0, axis=0),
    RandomRotate(degrees=0, axis=1),
    RandomRotate(degrees=0, axis=2),
    ])

test_transform_0 = Compose([
    random_rotate_0,
    SamplePoints(num_points, include_normals=True),
    NormalizeScale(),
    #GaussianNoiseTransform(mu, 0.,recompute_normals=True)
    ])

# random_rotate_10 = Compose([
#     RandomRotate(degrees=0, axis=0),
#     RandomRotate(degrees=0, axis=1),
#     RandomRotate(degrees=0, axis=2),
#     ])

# test_transform_10 = Compose([
#     random_rotate_10,
#     SamplePoints(num_points, include_normals=True),
#     NormalizeScale(),
#     GaussianNoiseTransform(mu, 0.05,recompute_normals=True)
#     ])

# random_rotate_20 = Compose([
#     RandomRotate(degrees=0, axis=0),
#     RandomRotate(degrees=0, axis=1),
#     RandomRotate(degrees=0, axis=2),
#     ])

# test_transform_20 = Compose([
#     random_rotate_20,
#     SamplePoints(num_points, include_normals=True),
#     NormalizeScale(),
#     GaussianNoiseTransform(mu, 0.08,recompute_normals=True)
#     ])


train_dataset_0 = ModelNet(root=root, name=str(modelnet_num), train=True, transform=test_transform_0)
test_dataset_0 = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform_0)

# train_dataset_10 = ModelNet(root=root, name=str(modelnet_num), train=True, transform=test_transform_10)
# test_dataset_10 = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform_10)

# train_dataset_20 = ModelNet(root=root, name=str(modelnet_num), train=True, transform=test_transform_20)
# test_dataset_20 = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform_20)



###############################################################################

train_loader_0 = DataLoader(train_dataset_0, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader_0  = DataLoader(test_dataset_0, batch_size=batch_size)

# train_loader_10 = DataLoader(train_dataset_10, batch_size=batch_size, shuffle=True, pin_memory=True)
# test_loader_10  = DataLoader(test_dataset_10, batch_size=batch_size)

# train_loader_20 = DataLoader(train_dataset_20, batch_size=batch_size, shuffle=True, pin_memory=True)
# test_loader_20  = DataLoader(test_dataset_20, batch_size=batch_size)

###############################################################################

weights = util_functions.get_weights(train_dataset_0,num_points=num_points,nr_classes=modelnet_num)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=float32).to('cuda'))  # Define loss criterion.

for epoch in range(1, num_epochs+1):

    #program_name="RGCNN-recomputed-normals"
    # conv.view_pcd(model=model,loader=test_loader,num_points=num_points,device=device,program_name=program_name)

    loss_tr=0
    loss_t=0
    acc_t=0
    acc_tr=0

    train_start_time = time.time()
    train_loss,train_acc = train(model, optimizer, train_loader_0, regularization=regularization,criterion=criterion)
    train_stop_time = time.time()

    loss_tr=loss_tr+train_loss
    acc_tr=acc_tr+train_acc

    # train_start_time = time.time()
    # train_loss,train_acc = train(model, optimizer, train_loader_10, regularization=regularization,criterion=criterion)
    # train_stop_time = time.time()

    # loss_tr=loss_tr+train_loss
    # acc_tr=acc_tr+train_acc

    # train_start_time = time.time()
    # train_loss,train_acc = train(model, optimizer, train_loader_20, regularization=regularization,criterion=criterion)
    # train_stop_time = time.time()

    # loss_tr=loss_tr+train_loss
    # acc_tr=acc_tr+train_acc

    test_start_time = time.time()
    test_loss,test_acc = test(model, test_loader_0,criterion=criterion)
    test_stop_time = time.time()

    loss_t=loss_t+test_loss
    acc_t=acc_t+test_acc

    # test_start_time = time.time()
    # test_loss,test_acc = test(model, test_loader_10,criterion=criterion)
    # test_stop_time = time.time()

    # loss_t=loss_t+test_loss
    # acc_t=acc_t+test_acc

    # test_start_time = time.time()
    # test_loss,test_acc = test(model, test_loader_20,criterion=criterion)
    # test_stop_time = time.time()

    # loss_t=loss_t+test_loss
    # acc_t=acc_t+test_acc

    # train_loss=loss_tr/3
    # test_loss=loss_t/3
    # test_acc=acc_t/3
    # train_acc=acc_tr/3
 

    # train_start_time = time.time()
    # train_loss,train_acc = train(model, optimizer, train_loader, regularization=regularization)
    # train_stop_time = time.time()

    # test_start_time = time.time()
    # test_loss,test_acc = test(model, test_loader)
    # test_stop_time = time.time()

    #conv.test_pcd_pred(model=model,loader=train_loader,num_points=num_points,device=device)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Acc/train", train_acc, epoch)
    writer.add_scalar("Acc/test", test_acc, epoch)

    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
    Test Time: \t{test_stop_time - test_start_time }')

    # writer.add_figure("Confusion matrix", createConfusionMatrix(model,test_loader), epoch)

    if(epoch%5==0):
        torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

    my_lr_scheduler.step()


torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


