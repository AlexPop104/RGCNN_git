import torch
from torch import nn

import torch as t
import torch_geometric as tg

import time

from noise_transform import GaussianNoiseTransform



# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

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


import ChebConv_rgcnn_functions as conv
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

from noise_transform import GaussianNoiseTransform

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

        self.conv1 = conv.DenseChebConv(6, 128, 3)
        self.conv2 = conv.DenseChebConv(128,512, 3)
        self.conv3 = conv.DenseChebConv(512,1024, 3)


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
            L = conv.pairwise_distance(x) # W - weight matrix
            #L = conv.get_laplacian(L)

            # for it_pcd in range(1):
            #     viz_points_2=x[it_pcd,:,:]
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,1)
                


            L = conv.get_laplacian(L)

        

        out = self.conv1(x, L)
        out = self.relu1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)
        
        with torch.no_grad():
            L = conv.pairwise_distance(out) # W - weight matrix
            #L = conv.get_laplacian(L)
            # for it_pcd in range(1):
            #     viz_points_2=x[it_pcd,:,:]
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,2)
            L = conv.get_laplacian(L)
        
        out = self.conv2(out, L)
        out = self.relu2(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)

        with torch.no_grad():
            L = conv.pairwise_distance(out) # W - weight matrix
            #L = conv.get_laplacian(L)
            # for it_pcd in range(1):
            #     viz_points_2=x[it_pcd,:,:]
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,3)
            L = conv.get_laplacian(L)
        
        # plt.show()

        


        out = self.conv3(out, L)
        out = self.relu3(out)

        #conv.tsne_features(x=x,out=out,batch_size=batch_size)
        
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)

        out, _ = t.max(out, 1)
        #################################################################################
        
        out = self.fc1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(self.fc1.weight.data[0]) ** 2)
            self.regularizers.append(t.linalg.norm(self.fc1.bias.data[0]) ** 2)

        out = self.relu4(out)
        #out = self.dropout(out)

        out = self.fc2(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(self.fc2.weight.data[0]) ** 2)
            self.regularizers.append(t.linalg.norm(self.fc2.bias.data[0]) ** 2)
        out = self.relu5(out)
        #out = self.dropout(out)

        out = self.fc3(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(self.fc3.weight.data[0]) ** 2)
            self.regularizers.append(t.linalg.norm(self.fc3.bias.data[0]) ** 2)
        
        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader, regularization):
    model.train()
    total_loss = 0
    total_correct = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x = torch.cat([data.pos, data.normal], dim=1)   
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        
        logits, regularizers  = model(x=x.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())
        
        loss    = criterion(logits, data.y.to(device))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_loss = 0
    total_correct = 0
    for data in loader:
       
        x = torch.cat([data.pos, data.normal], dim=1)   
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)

        logits, regularizers  = model(x=x.to(device))
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




num_points = 512
batch_size = 16
num_epochs = 250
learning_rate = 1e-3
modelnet_num = 40

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, modelnet_num]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")

root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
print(root)


model = cls_model(num_points, F, K, M, modelnet_num, dropout=1, reg_prior=True)
path_saved_model="/home/alex/Alex_documents/RGCNN_git/Modele_selectate/Normals_recomputed/Augmented_rotation/RGCNN_aug_rotation.pt"
model.load_state_dict(torch.load(path_saved_model))
print(model.parameters)
model = model.to(device)

rot_x=1
rot_y=1
rot_z=1

sigma=[0, 0.01,0.03,0.05,0.08,0.1,0.15]

ceva=0
ceva2=0

torch.manual_seed(0)

for ceva in range(0,4):
#for ceva2 in range(0,len(sigma)):

    mu=0
    
    random_rotate = Compose([
    RandomRotate(degrees=rot_x*ceva*10, axis=0),
    RandomRotate(degrees=rot_y*ceva*10, axis=1),
    RandomRotate(degrees=rot_z*ceva*10, axis=2),
    ])

    test_transform = Compose([
    random_rotate,
    SamplePoints(num_points, include_normals=True),
    NormalizeScale(),
    GaussianNoiseTransform(mu, sigma[ceva2],recompute_normals=True)
    ])

    test_dataset = ModelNet(root=root, train=False,
                            transform=test_transform)


    program_name="RGCNN_rot_x"+str(10*rot_x*ceva)+"_rot_y"+str(10*rot_y*ceva)+"_rot_z"+str(10*rot_z*ceva)


######################################################################33

    # transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    # test_dataset = ModelNet(root=root, train=False,
    #                                transform=transforms)

    # program_name="RGCNN"

################################################################33

    

    # transforms_noisy = Compose([SamplePoints(num_points),NormalizeScale(),GaussianNoiseTransform(mu, sigma,recompute_normals=True)])
   

    # test_dataset = ModelNet(root=root, train=False,
    #                         transform=transforms_noisy)

    #program_name="RGCNN_noise_mu"+str(mu)+"_sigma_"+str(sigma)
    

    

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #conv.view_pcd(model=model,loader=test_loader,num_points=num_points,device=device,program_name=program_name)


    #conv.test_pcd_pred(model=model,loader=test_loader,num_points=num_points,device=device)
        
    test_start_time = time.time()
    test_loss, test_acc = test(model, test_loader)
    test_stop_time = time.time()

    print(f'{test_acc:.4f}')

    #print(f'\Test Time: \t{test_stop_time - test_start_time }')
    #print(f' Test Accuracy: {test_acc:.4f}')

    