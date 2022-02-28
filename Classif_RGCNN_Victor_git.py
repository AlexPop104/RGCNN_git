import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch import nn
import torch
from torch.nn import Parameter


from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints


from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (add_self_loops, get_laplacian,
                                   remove_self_loops)
import torch as t
import torch_geometric as tg

from torch_geometric.utils import get_laplacian as get_laplacian_pyg
from torch_geometric.transforms import Compose

import ChebConv_rgcnn as conv
import os
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
from datetime import datetime
from torch.nn import MSELoss
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np



class cls_model(nn.Module):
    def __init__(self, vertice ,F, K, M, class_num, regularization=0, one_layer=True, dropout=0, reg_prior:bool=True):
        assert len(F) == len(K)
        super(cls_model, self).__init__()

        self.F = F
        self.K = K
        self.M = M

        self.one_layer = one_layer

        self.reg_prior = reg_prior
        self.vertice = vertice
        self.regularization = regularization    # gamma from the paper: 10^-9
        self.dropout = dropout
        self.regularizers = []

        # self.get_laplacian = GetLaplacian(normalize=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.conv1 = conv.DenseChebConv(6, 128, 6)
        self.conv2 = conv.DenseChebConv(128, 512, 5)
        self.conv3 = conv.DenseChebConv(512, 1024, 3)
        
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.fc_t = nn.Linear(128, class_num)

        self.max_pool = nn.MaxPool1d(self.vertice)

        if one_layer == True:
            self.fc = nn.Linear(128, class_num)

        self.regularizer = 0
        self.regularization = []


    def forward(self, x):
        self.regularizers = []
        with torch.no_grad():
            L = conv.pairwise_distance(x) # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv1(x, L)
        out = self.relu1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        
        if self.one_layer == False:
            with torch.no_grad():
                L = conv.pairwise_distance(out) # W - weight matrix
                L = conv.get_laplacian(L)
            
            out = self.conv2(out, L)
            out = self.relu2(out)
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
    
            with torch.no_grad():
                L = conv.pairwise_distance(out) # W - weight matrix
                L = conv.get_laplacian(L)
            
            out = self.conv3(out, L)
            out = self.relu3(out)
            
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
    
            out, _ = t.max(out, 1)

            # ~~~~ Fully Connected ~~~~
            
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
        else:
            out, _ = t.max(out, 1)
            out = self.fc(out)

        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader, regularization):
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)
        logits, regularizers  = model(x.to(device))
        loss    = criterion(logits, data.y.to(device))
        s = t.sum(t.as_tensor(regularizers))
        loss = loss + regularization * s
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        #if i%100 == 0:
            #print(f"{i}: curr loss: {loss}")
            #$print(f"{data.y} --- {logits.argmax(dim=1)}")
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)
        logits, _ = model(x.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_correct / len(loader.dataset)

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

    # constant for classes
    # classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred,normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(40)],
                         columns=[i for i in range(40)])
    plt.figure(figsize=(50, 50))    
    return sn.heatmap(df_cm, annot=True).get_figure()



if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_points = 1024
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    modelnet_num = 40

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, modelnet_num]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

        
    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    root = "data/ModelNet"+str(modelnet_num)
    print(root)
    dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
    dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)


    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")


    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader  = DataLoader(dataset_test, batch_size=batch_size)
    
    model = cls_model(num_points, F, K, M, modelnet_num, dropout=1, one_layer=False, reg_prior=True)
    path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models/28_02_22_10:10:19/model50.pt"
    model.load_state_dict(torch.load(path_saved_model))
    model = model.to(device)

    test_start_time = time.time()
    test_acc = test(model, test_loader)
    test_stop_time = time.time()
    print(f'Test Accuracy: {test_acc:.4f}')

    writer.add_figure("Confusion matrix", createConfusionMatrix(model,test_loader), 1)

    
    print(model.parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    regularization = 1e-9
    # for epoch in range(1, 51):
    #     train_start_time = time.time()
    #     loss = train(model, optimizer, train_loader, regularization=regularization)
    #     train_stop_time = time.time()

    #     writer.add_scalar("Loss/train", loss, epoch)
        
    #     test_start_time = time.time()
    #     test_acc = test(model, test_loader)
    #     test_stop_time = time.time()



    #     writer.add_scalar("Acc/test", test_acc, epoch)
    #     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
    #     print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
    #     Test Time: \t{test_stop_time - test_start_time }')

    #     writer.add_figure("Confusion matrix", createConfusionMatrix(model,test_loader), epoch)

    #     if(epoch%5==0):
    #         torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

    #     my_lr_scheduler.step()

    
    #torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

    

    