from datetime import datetime
from typing import Optional
from venv import create
#from more_itertools import one
from torch_geometric.loader import DenseDataLoader
import os
import ChebConv_rgcnn as conv
import torch as t
from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
import torch
from torch import nn
import time
from torch.nn.functional import one_hot
from Classif_RGCNN_n_DenseConv_functions import DenseChebConv as DenseChebConvPyG
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
from torch.optim import lr_scheduler

import numpy as np

def create_batched_dataset(class_num, dataset, batch_size):
    data = []
    for d in dataset:
        if d.category == class_num:
            data.append(d)

    batched_y = []
    batched_pos = []
    batched_x = []

    batch_y = torch.zeros((batch_size,   2048))
    batch_pos = torch.zeros((batch_size, 2048, 3))
    batch_x = torch.zeros((batch_size,   2048, 3))
    i = 0
    for d in data:
        if i<batch_size:
            batch_y[i,:] = d.y
            batch_pos[i,:,:] = d.pos
            batch_x[i,:,:] = d.x
        else:
            i = 0
            batched_x.append(batch_x)
            batched_y.append(batch_y)
            batched_pos.append(batch_pos)

            batch_y = torch.zeros((batch_size,   2048))
            batch_pos = torch.zeros((batch_size, 2048, 3))
            batch_x = torch.zeros((batch_size,   2048, 3))

        i += 1
    print(len(data))
    return [batched_pos, batched_x, batched_y]


class seg_model(nn.Module):
    def __init__(self, vertice, F, K, M, input_dim=22 ,regularization=0, one_layer=True, dropout=0, reg_prior: bool = True):
        assert len(F) == len(K)
        super(seg_model, self).__init__()

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
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.bias_relu1 = nn.Parameter(torch.zeros((1, 1, 128)), requires_grad=True)
        self.bias_relu2 = nn.Parameter(torch.zeros((1, 1, 512)), requires_grad=True)
        self.bias_relu3 = nn.Parameter(torch.zeros((1, 1, 1024)), requires_grad=True)
        self.bias_relu4 = nn.Parameter(torch.zeros((1, 1, 512)), requires_grad=True)
        self.bias_relu5 = nn.Parameter(torch.zeros((1, 1, 128)), requires_grad=True)
        self.bias_relu6 = nn.Parameter(torch.zeros((1, 1, 50)), requires_grad=True)

        self.conv1 = conv.DenseChebConv(input_dim, 128, 6)
        self.conv2 = conv.DenseChebConv(128, 512, 5)
        self.conv3 = conv.DenseChebConv(512, 1024, 3)

        '''
        self.conv4 = conv.DenseChebConv(1152, 512, 1)
        self.conv5 = conv.DenseChebConv(512, 128, 1)
        self.conv6 = conv.DenseChebConv(128, 50, 1)
        '''
        self.recompute_L = True

        bias=True
        self.fc1 = t.nn.Linear(1024, 512, bias=bias)
        self.fc2 = t.nn.Linear(512 + 512, 128, bias=bias)
        self.fc3 = t.nn.Linear(128, 50, bias=bias)

        self.regularizer = 0
        self.regularization = []

    
    def b1relu(self, x, bias, relu):
        return relu(x + bias)

    def get_laplacian(self, x):
        with torch.no_grad():
            return conv.get_laplacian(conv.pairwise_distance(x))

    def forward(self, x):
        self.regularizers = []
        x1 = 0  # cache for layer1

        L = self.get_laplacian(x)

        out = self.conv1(x, L)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        out = self.b1relu(out, self.bias_relu1, self.relu1)


        if self.recompute_L:
            L = self.get_laplacian(out)
        
        out = self.conv2(out, L)
        
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        out = self.b1relu(out, self.bias_relu2, self.relu2)

        x1 = out

        if self.recompute_L:
            L = self.get_laplacian(out)

        out = self.conv3(out, L)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        out = self.b1relu(out, self.bias_relu3, self.relu3)


        out = self.fc1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        out = self.b1relu(out, self.bias_relu4, self.relu4)

        out = t.concat([out, x1], dim=2)

        out = self.fc2(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        out = self.b1relu(out, self.bias_relu5, self.relu5)


        out = self.fc3(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), x))**2)
        
        out = self.b1relu(out, self.bias_relu6, self.relu6)

        return out, self.regularizers


criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.


def train(model, optimizer, loader, regularization):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        cat = one_hot(data.category, num_classes=16)
        cat = torch.tile(cat, [1, num_points, 1]) 
        x = torch.cat([data.pos, data.x, cat], dim=2)  ### Pass this to the model
        y = data.y.type(torch.LongTensor)

        logits, regularizers = model(x.to(device))
        logits = logits.permute([0, 2, 1])

        loss = criterion(logits, y.to(device))
        s = t.sum(t.as_tensor(regularizers))
        loss = t.mean(loss)
        loss = loss + regularization * s
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i%100 == 0:
            print(f"{i}: curr loss: {loss}")
            #print(f"{data.y} --- {logits.argmax(dim=1)}")
    return total_loss * batch_size / len(dataset_train)


@torch.no_grad()
def test(model, loader):
    model.eval()
    size = len(dataset_test)
    predictions = np.empty((size, num_points))
    labels = np.empty((size, num_points))
    total_correct = 0

    for i, data in enumerate(loader):
        cat = one_hot(data.category, num_classes=16)
        cat = torch.tile(cat, [1, num_points, 1]) 
        x = torch.cat([data.pos, data.x, cat], dim=2)  ### Pass this to the model
        y = data.y
        logits, _ = model(x.to(device))
        logits = logits.to('cpu')
        pred = logits.argmax(dim=2)
        # print(pred)
        # print(f"TEST: {int((pred == data.y.to(device)).sum())}")
        total_correct += int((pred == y).sum())
        start = i * batch_size
        stop  = start + batch_size
        predictions[start:stop] = pred
        lab = data.y
        labels[start:stop] = lab.reshape([-1, num_points])
    ncorrects = np.sum(predictions == labels)
    accuracy  = ncorrects * 100 / (len(dataset_test) * num_points)
    print(f"Accuracy: {accuracy}, ncorrect: {ncorrects} / {len(dataset_test) * num_points}")
    return accuracy

def start_training(model, train_loader, test_loader, optimizer, epochs,path, learning_rate=1e-3, regularization=1e-9, decay_rate=0.95):
    print(model.parameters)
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f"\nTraining on {device}")
    model.to(device)
    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    for epoch in range(1, epochs+1):
        train_start_time = time.time()
        loss = train(model, optimizer, train_loader, regularization=regularization)
        train_stop_time = time.time()

        writer.add_scalar('loss/train', loss, epoch)

        test_start_time = time.time()
        test_acc = test(model, test_loader)
        test_stop_time = time.time()

        writer.add_scalar("accuracy/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
            Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%5==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        my_lr_scheduler.step()


    print(f"Training finished")
    print(model.parameters())


if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_points = 2048

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    root = "/home/alex/Alex_documents/RGCNN_git/data/ShapeNet"
    print(root)
    dataset_train = ShapeNet(root=root, split="train", transform=FixedPoints(num_points))
    dataset_test = ShapeNet(root=root, split="test", transform=FixedPoints(num_points))
   

    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-3
    decay_rate = 0.95
    decay_steps = len(dataset_train) / batch_size

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]
    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")

    train_loader = DenseDataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DenseDataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = seg_model(num_points, F, K, M,
                      dropout=1, one_layer=False, reg_prior=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    start_training(model, train_loader, test_loader, optimizer, epochs=num_epochs,path=path)