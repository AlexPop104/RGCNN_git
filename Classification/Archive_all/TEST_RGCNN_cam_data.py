import torch
from torch import nn

import torch as t
import torch_geometric as tg

import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch.nn import Parameter

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

from path import Path

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

import dataset_loader as cam_loader


import random
random.seed(0)



class cls_model(nn.Module):
    def __init__(self, vertice ,F, K, M, class_num, regularization=0,  dropout=0, reg_prior:bool=True):
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

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)


        self.conv1 = util_functions.DenseChebConvV2(6, 128, 3)
        self.conv2 = util_functions.DenseChebConvV2(128,512, 3)
        self.conv3 = util_functions.DenseChebConvV2(512,1024, 3)

        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.max_pool = nn.MaxPool1d(self.vertice)

        self.regularizer = 0
        self.regularization = []             

    def forward(self, x):
    #def forward(self, x,x2):
        self.regularizers = []
        with torch.no_grad():
            L = util_functions.pairwise_distance(x) # W - weight matrix
            L = util_functions.get_laplacian(L)

        out = self.conv1(x, L)
        out = self.relu1(out)

        # if self.reg_prior:

        #     # self.L.append(L)
        #     # self.x.append(out)

        #     self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)
        
       
        with torch.no_grad():
            L = util_functions.pairwise_distance(out) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv2(out, L)
        out = self.relu2(out)

        # if self.reg_prior:
        #     # self.L.append(L)
        #     # self.x.append(out)

        #     self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)

        with torch.no_grad():
            L = util_functions.pairwise_distance(out) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv3(out, L)
        out = self.relu3(out)
        
        # if self.reg_prior:
        #     # self.L.append(L)
        #     # self.x.append(x)

        #     self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)

        out, _ = t.max(out, 1)
       
        # ~~~~ Fully Connected ~~~~
        
        out = self.fc1(out)

        # if self.reg_prior:
        #     self.regularizers.append(t.linalg.norm(self.fc1.weight.data[0]) ** 2)
        #     self.regularizers.append(t.linalg.norm(self.fc1.bias.data[0]) ** 2)

        out = self.dropout(out)
        out = self.relu4(out)
        #out = self.dropout(out)

        out = self.fc2(out)
        # if self.reg_prior:
        #     self.regularizers.append(t.linalg.norm(self.fc2.weight.data[0]) ** 2)
        #     self.regularizers.append(t.linalg.norm(self.fc2.bias.data[0]) ** 2)
        out = self.dropout(out)
        out = self.relu5(out)
        #out = self.dropout(out)

        out = self.fc3(out)
        # if self.reg_prior:
        #     self.regularizers.append(t.linalg.norm(self.fc3.weight.data[0]) ** 2)
        #     self.regularizers.append(t.linalg.norm(self.fc3.bias.data[0]) ** 2)
        
        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer,num_points,criterion, loader, regularization,device):
    model.train()
    total_loss = 0
    total_correct = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x = torch.cat([data.pos, data.normal], dim=1)   
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        x =x.float()
        logits, regularizers  = model(x=x.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())
        
        loss    = criterion(logits, data.y.to(device))
       
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 

@torch.no_grad()
def test(model, loader,num_points,criterion,device):
    model.eval()

    # Dict from labels to names

    total_loss = 0
    total_correct = 0
    for data in loader:
       
        x = torch.cat([data.pos, data.normal], dim=1)   
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        x=x.float()

        logits, regularizers = model(x=x.to(device))
        loss    = criterion(logits, data.y.to(device))

        total_loss += loss.item() * data.num_graphs
        
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 

def createConfusionMatrix(model,loader):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for  data in loader:
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        # x=data.pos
        # x=x.reshape(data.batch.unique().shape[0], num_points, 3)
        logits, _ = model(x.to(device))
        pred = logits.argmax(dim=-1)
        
        output = pred.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = data.y.cpu().numpy()
        y_true.extend(labels)  # save ground truth

   
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred,normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(40)],
                         columns=[i for i in range(40)])
    plt.figure(figsize=(50, 50))    
    return sn.heatmap(df_cm, annot=True).get_figure()


now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)

num_points = 1024
batch_size = 16
num_epochs = 250
learning_rate = 1e-3
modelnet_num = 5
dropout=0.25

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, modelnet_num]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")

#root = "/mnt/ssd1/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
#root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
#print(root)




model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
regularization = 1e-9

torch.manual_seed(0)
path_saved_model="/home/alex/Alex_documents/RGCNN_git/Classification/Modele_selectate/Camera/model3.pt"
model.load_state_dict(torch.load(path_saved_model))
print(model.parameters)
model = model.to(device)

torch.manual_seed(0)

root = Path("/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/Dataset")
test_dataset_0 = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test')


###############################################################################


test_loader  = DataLoader(test_dataset_0, batch_size=batch_size)

################################################
test_start_time = time.time()
test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
test_stop_time = time.time()

print(f'{test_acc:.4f}')

#print(f'\Test Time: \t{test_stop_time - test_start_time }')
    #print(f' Test Accuracy: {test_acc:.4f}')

    