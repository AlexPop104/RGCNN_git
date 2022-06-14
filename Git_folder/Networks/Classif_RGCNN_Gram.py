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


import numpy as np


from utils import GaussianNoiseTransform
import utils as util_functions

import dataset_loader_noise as cam_loader


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


        self.conv1 = util_functions.DenseChebConvV2(self.K[0], 128, 3)
        self.conv2 = util_functions.DenseChebConvV2(128,512, 3)
        self.conv3 = util_functions.DenseChebConvV2(512,1024, 3)

        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.max_pool = nn.MaxPool1d(self.vertice)

        self.regularizer = 0
        self.regularization = []             

    def forward(self, x,gram):
        self.regularizers = []

        
        with torch.no_grad():

            L = util_functions.pairwise_distance(x,normalize=True) # W - weight matrix
            L = util_functions.get_laplacian(L)

        out = self.conv1(gram, L)
        out = self.relu1(out)

        
       
        with torch.no_grad():
            L = util_functions.pairwise_distance(out,normalize=True) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv2(out, L)
        out = self.relu2(out)

  

        with torch.no_grad():
            L = util_functions.pairwise_distance(out,normalize=True) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv3(out, L)
        out = self.relu3(out)
        

        out, _ = t.max(out, 1)
       
        # ~~~~ Fully Connected ~~~~
        
        out = self.fc1(out)


        out = self.dropout(out)
        out = self.relu4(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu5(out)

        out = self.fc3(out)
        
        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer,num_points,criterion, loader, regularization,device):
    model.train()
    total_loss = 0
    total_correct = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        if (model.F[0]==6):
            x = torch.cat([data.pos, data.normal], dim=1)   
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()

        if (model.F[0]==3):
            x = data.pos
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()  

        x_transpose = x.permute(0, 2, 1)
        gram = t.matmul(x, x_transpose)

        gram=gram.to(device)

        gram,indices=torch.sort(gram)

        

        logits, regularizers  = model(x=x.to(device),gram=gram.to(device))
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
       
        if (model.F[0]==6):
            x = torch.cat([data.pos, data.normal], dim=1)   
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()

        if (model.F[0]==3):
            x = data.pos
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()  

        x=x.to(device)

        x_transpose = x.permute(0, 2, 1)
        gram = t.matmul(x, x_transpose)

        gram=gram.to(device)

     


        gram,indices=torch.sort(gram)

        
        
        logits, regularizers = model(x=x.to(device),gram=gram.to(device))
        loss    = criterion(logits, data.y.to(device))

        total_loss += loss.item() * data.num_graphs
        
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 




now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
directory="RGCNN_"+directory
parent_directory = "/media/rambo/ssd2/Alex_data/RGCNN/data/logs/Trained_Models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)

num_points = 512
batch_size = 16
num_epochs = 250
learning_rate = 1e-3
modelnet_num = 36
dropout=0.25
input_feature_selection=3
input_feature_size=6


F = [input_feature_selection, 512, 1024]  # Outputs size of convolutional filter.
K = [input_feature_size, 5, 3]         # Polynomial orders.
M = [1024, 128, modelnet_num]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")


model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
regularization = 1e-9

torch.manual_seed(0)
#################################################################33

print("Select type of training  (1 - no noise, 2 - Rotation noise , 3- Position noise)")
selection=int(input())

if(selection==1):
    root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/Modelnet40_512/")
    train_dataset_0 = cam_loader.PcdDataset(root_dir=root, points=num_points)
    test_dataset_0 = cam_loader.PcdDataset(root_dir=root, folder='test',points=num_points)

    ###############################################################################

    train_loader_0 = DataLoader(train_dataset_0, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader_0  = DataLoader(test_dataset_0, batch_size=batch_size)

    ###############################################################################

    for epoch in range(0, num_epochs+1):

        loss_tr=0
        loss_t=0
        acc_t=0
        acc_tr=0

        train_start_time = time.time()
        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_0, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
        train_stop_time = time.time()

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        

        test_start_time = time.time()
        test_loss,test_acc = test(model=model, loader=test_loader_0,num_points=num_points,criterion=criterion,device=device)
        test_stop_time = time.time()

        loss_t=loss_t+test_loss
        acc_t=acc_t+test_acc

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%3==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        my_lr_scheduler.step()

    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


elif(selection==2):
    
    root_train_10 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_r_10/")
    root_train_20 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_r_20/")
    root_train_30 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_r_30/")
    root_train_40 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_r_40/")

    root_test = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024/")

    train_dataset_10 = cam_loader.PcdDataset(root_dir=root_train_10, points=num_points)
    train_dataset_20 = cam_loader.PcdDataset(root_dir=root_train_10, points=num_points)
    train_dataset_30 = cam_loader.PcdDataset(root_dir=root_train_10, points=num_points)
    train_dataset_40 = cam_loader.PcdDataset(root_dir=root_train_10, points=num_points)

    test_dataset = cam_loader.PcdDataset(root_dir=root_test, folder='test',points=num_points)

    ###############################################################################

    train_loader_10 = DataLoader(train_dataset_10, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_20 = DataLoader(train_dataset_20, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_30 = DataLoader(train_dataset_30, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_40 = DataLoader(train_dataset_40, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    ###############################################################################

    for epoch in range(0, num_epochs+1):

        loss_tr=0
        loss_t=0
        acc_t=0
        acc_tr=0

        train_start_time = time.time()
        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_10, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
       

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_20, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
   

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc


        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_30, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
      

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc


        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_40, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
        train_stop_time = time.time()

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        loss_tr=loss_tr/4
        acc_tr=acc_tr/4

        test_start_time = time.time()
        test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
        test_stop_time = time.time()

        loss_t=loss_t+test_loss
        acc_t=acc_t+test_acc

        writer.add_scalar("Loss/train", loss_tr, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/train", acc_tr, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%3==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        my_lr_scheduler.step()

    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

elif(selection==3):
    
    root_train_002 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_n_002/")
    root_train_005 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_n_005/")
    root_train_008 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_n_008/")
    root_train_010 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024_n_010/")

    root_test = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_1024/")

    train_dataset_002 = cam_loader.PcdDataset(root_dir=root_train_002, points=num_points)
    train_dataset_005 = cam_loader.PcdDataset(root_dir=root_train_005, points=num_points)
    train_dataset_008 = cam_loader.PcdDataset(root_dir=root_train_008, points=num_points)
    train_dataset_010 = cam_loader.PcdDataset(root_dir=root_train_010, points=num_points)

    test_dataset = cam_loader.PcdDataset(root_dir=root_test, folder='test',points=num_points)

    ###############################################################################

    train_loader_002 = DataLoader(train_dataset_002, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_005 = DataLoader(train_dataset_005, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_008 = DataLoader(train_dataset_008, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_010 = DataLoader(train_dataset_010, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    ###############################################################################

    for epoch in range(0, num_epochs+1):

        loss_tr=0
        loss_t=0
        acc_t=0
        acc_tr=0

        train_start_time = time.time()
        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_002, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
        

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

    
        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_005, regularization=regularization,num_points=num_points,criterion=criterion, device=device)


        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

   
        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_008, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
        

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        
        train_loss,train_acc = train(model=model, optimizer=optimizer, loader=train_loader_010, regularization=regularization,num_points=num_points,criterion=criterion, device=device)
        train_stop_time = time.time()

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        loss_tr=loss_tr/4
        acc_tr=acc_tr/4

        test_start_time = time.time()
        test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
        test_stop_time = time.time()

        loss_t=loss_t+test_loss
        acc_t=acc_t+test_acc

        writer.add_scalar("Loss/train", loss_tr, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/train", acc_tr, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {loss_t:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%3==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        my_lr_scheduler.step()

    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


