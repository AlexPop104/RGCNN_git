from collections import defaultdict
from datetime import datetime
from typing import Optional
from venv import create
from torch_geometric.loader import DenseDataLoader
import os
import ChebConv_rgcnn as conv
import torch as t
from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
import torch
from torch import float32, nn
import time
from torch.nn.functional import one_hot
from torch.nn.functional import relu

from sklearn.utils import class_weight

torch.manual_seed(0)

from torch_geometric.transforms import Compose

from GaussianNoiseTransform import GaussianNoiseTransform

from Classif_RGCNN_n_DenseConv_functions import DenseChebConv as DenseChebConvPyG
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='_sn_ww_025_drop_weight_decay', filename_suffix='_no_reg')
import numpy as np
from torch.optim import lr_scheduler

import numpy as np


def get_weights(dataset, sk=True):
    weights = torch.zeros(50)
    if sk:
        y = np.empty(len(dataset)*dataset[0].y.shape[0])
        i = 0
        for data in dataset:
            y[i:2048+i] = data.y
            i+=2048
        weights = class_weight.compute_class_weight(
            'balanced', np.unique(y), y)
    else:
        for data in dataset:
            for l in torch.unique(data.y):
                weights[l] += torch.sum(data.y == l.item())
        weights = 1 - weights / len(dataset)
    return weights



seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                    'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                    'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                    'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                    'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

label_to_cat = {}
for key in seg_classes.keys():
    for label in seg_classes[key]:
        label_to_cat[label] = key



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
    def __init__(self, vertice, F, K, M, input_dim=22 , one_layer=False, dropout=1, reg_prior: bool = True, b2relu=False, relus=[128, 512, 1024, 512, 128, 50], recompute_L = False):
        assert len(F) == len(K)
        super(seg_model, self).__init__()

        self.F = F
        self.K = K
        self.M = M

        self.one_layer = one_layer

        self.reg_prior = reg_prior
        self.vertice = vertice
        self.dropout = dropout
        self.regularizers = []

        # self.get_laplacian = GetLaplacian(normalize=True)

        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.relus = relus

        if b2relu:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, num_points, i))) for i in self.relus
            ])
        else:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, 1, i))) for i in self.relus
                ])
        
        '''
        self.bias_relu1 = nn.Parameter(torch.zeros((1, 1, 128)), requires_grad=True)
        self.bias_relu2 = nn.Parameter(torch.zeros((1, 1, 512)), requires_grad=True)
        self.bias_relu3 = nn.Parameter(torch.zeros((1, 1, 1024)), requires_grad=True)
        self.bias_relu4 = nn.Parameter(torch.zeros((1, 1, 512)), requires_grad=True)
        self.bias_relu5 = nn.Parameter(torch.zeros((1, 1, 128)), requires_grad=True)
        self.bias_relu6 = nn.Parameter(torch.zeros((1, 1, 50)), requires_grad=True)
        '''

        self.conv1 = conv.DenseChebConvV2(input_dim, 128, 6)
        self.conv2 = conv.DenseChebConvV2(128, 512, 5)
        self.conv3 = conv.DenseChebConvV2(512, 1024, 3)

        '''
        self.conv4 = conv.DenseChebConv(1152, 512, 1)
        self.conv5 = conv.DenseChebConv(512, 128, 1)
        self.conv6 = conv.DenseChebConv(128, 50, 1)
        '''
        self.recompute_L = recompute_L

        bias=True
        self.fc1 = t.nn.Linear(1024, 512, bias=bias)
        self.fc2 = t.nn.Linear(512 + 512, 128, bias=bias)
        self.fc3 = t.nn.Linear(128, 50, bias=bias)

        self.regularizer = 0
        self.regularization = []
        self.L = []
        self.x = []

    
    def b1relu(self, x, bias):
        return relu(x + bias)

    def get_laplacian(self, x):
        with torch.no_grad():
            return conv.get_laplacian(conv.pairwise_distance(x))
    
    @torch.no_grad()
    def append_regularization_terms(self, x, L):
        if self.reg_prior:
            self.L.append(L)
            self.x.append(x)
    

    def reset_regularization_terms(self):
        self.L = []
        self.x = []

    def forward(self, x, cat):
        self.reset_regularization_terms()

        x1 = 0  # cache for layer 1

        L = self.get_laplacian(x)

        cat = one_hot(cat, num_classes=16)
        cat = torch.tile(cat, [1, self.vertice, 1]) 
        x = torch.cat([x, cat], dim=2)  ### Pass this to the model
        out = self.conv1(x, L)
        self.append_regularization_terms(out, L)
        out = self.dropout(out)
        out = self.b1relu(out, self.bias_relus[0])
        
        if self.recompute_L:
            L = self.get_laplacian(out)
        out = self.conv2(out, L)
        self.append_regularization_terms(out, L)
        out = self.dropout(out)

        out = self.b1relu(out, self.bias_relus[1])
        x1 = out

        if self.recompute_L:
            L = self.get_laplacian(out)
        out = self.conv3(out, L)
        self.append_regularization_terms(out, L)
        out = self.dropout(out)
        out = self.b1relu(out, self.bias_relus[2])

        out = self.fc1(out)
        self.append_regularization_terms(out, L)
        out = self.dropout(out)
        out = self.b1relu(out, self.bias_relus[3])

        out = t.concat([out, x1], dim=2)

        out = self.fc2(out)
        self.append_regularization_terms(out, L)
        out = self.dropout(out)

        out = self.b1relu(out, self.bias_relus[4])

        out = self.fc3(out)

        self.append_regularization_terms(out, L)
        out = self.dropout(out)
        out = self.b1relu(out, self.bias_relus[5])

        return out, self.x, self.L

    
num_points = 2048
root = "/media/rambo/ssd2/Alex_data/RGCNN/ShapeNet/"

transforms = Compose([FixedPoints(num_points), GaussianNoiseTransform(mu=0, sigma=0, recompute_normals=False)])
dataset_train = ShapeNet(root=root, split="train", transform=transforms)
dataset_test = ShapeNet(root=root, split="test", transform=transforms)

weights = get_weights(dataset_train)

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=float32).to('cuda'))  # Define loss criterion.

def compute_loss(logits, y, x, L, s=1e-9):
    if not logits.device == y.device:
        y = y.to(logits.device)

    loss = criterion(logits, y)
    l=0
    for i in range(len(x)):
        l += (1/2) * t.linalg.norm(t.matmul(t.matmul(t.permute(x[i], (0, 2, 1)), L[i]), x[i]))**2
    l = l * s
    loss += l
    return loss

def train(model, optimizer, loader, regularization):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        cat = data.category
        y = data.y.type(torch.LongTensor)
        x = t.cat([data.pos.type(torch.float32), data.x.type(torch.float32)], dim=2)
        logits, out, L = model(x.to(device), cat.to(device)) # out, L are for regularization
        logits = logits.permute([0, 2, 1])

        loss = compute_loss(logits, y, out, L, s=regularization)

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
        cat = data.category
        x = t.cat([data.pos.type(torch.float32), data.x.type(torch.float32)], dim=2)
        y = data.y
        logits, _, _= model(x.to(device), cat.to(device))
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
        
    tot_iou = []
    cat_iou = defaultdict(list)
    for i in range(predictions.shape[0]):
        segp = predictions[i, :]
        segl = labels[i, :]
        cat = label_to_cat[segl[0]]
        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

        for l in seg_classes[cat]:
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        cat_iou[cat].append(np.mean(part_ious))
        tot_iou.append(np.mean(part_ious))

    # print(tot_iou)
    # accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
    # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')

    ncorrects = np.sum(predictions == labels)
    accuracy  = ncorrects * 100 / (len(dataset_test) * num_points)
    # print(f"\tAccuracy: {accuracy}, ncorrect: {ncorrects} / {len(dataset_test) * num_points}")
    # print(f"\tIoU: \t{np.mean(tot_iou)*100}")
    return accuracy, cat_iou, tot_iou, ncorrects

def start_training(model, train_loader, test_loader, optimizer, epochs=50, learning_rate=1e-3, regularization=1e-9, decay_rate=0.95):
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
        test_acc, cat_iou, tot_iou, ncorrects = test(model, test_loader)
        test_stop_time = time.time()

        for key, value in cat_iou.items():
            print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
            writer.add_scalar(key + '/test', np.mean(value), epoch)

        writer.add_scalar("IoU/test", np.mean(tot_iou) * 100, epoch)
        writer.add_scalar("accuracy/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}%, IoU: {np.mean(tot_iou)*100:.4f}%')
        print(f'ncorrect: {ncorrects} / {len(dataset_test) * num_points}')
        print(f'Train Time: \t{train_stop_time - train_start_time} \nTest Time: \t{test_stop_time - test_start_time }')
        print("~~~" * 30)

        my_lr_scheduler.step()

        # Save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), path + '/2048p_model_v2' + str(epoch) + '.pt')

    print(f"Training finished")


def IoU_accuracy(pred, target, n_classes=16):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)


if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    parent_directory = "/home/victor/workspace/thesis_ws/github/RGCNN_git/models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_points = 2048

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    root = "/media/rambo/ssd2/Alex_data/RGCNN/ShapeNet/"
    print(root)

    batch_size = 3
    num_epochs = 100
    learning_rate = 1e-3
    decay_rate = 0.8
    decay_steps = len(dataset_train) / batch_size
    weight_decay = 1e-8
    dropout=0.25

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]

    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")

    train_loader = DenseDataLoader(
        dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DenseDataLoader(
        dataset_test, batch_size=batch_size, shuffle=True)

    model = seg_model(num_points, F, K, M,
                      dropout=dropout, one_layer=False, reg_prior=True, recompute_L=False, relus=[128, 512, 1024, 512, 128, 50], b2relu=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    start_training(model, train_loader, test_loader, optimizer, epochs=num_epochs)