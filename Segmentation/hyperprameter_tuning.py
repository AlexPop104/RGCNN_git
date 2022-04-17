from collections import defaultdict
import sys
from matplotlib import transforms
sys.path.append('/home/domsa/workspace/git/RGCNN_git')

from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DenseDataLoader
from torch.nn.functional import one_hot, relu
from torch import float32, nn
import torch as t
import torch
import time
import os
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose
from utils import DenseChebConvV2 as conv
from utils import GaussianNoiseTransform
from utils import compute_loss
from utils import get_weights
from utils import IoU_accuracy
from datetime import datetime

torch.manual_seed(0)

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
               'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
               'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
               'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

label_to_cat = {}
for key in seg_classes.keys():
    for label in seg_classes[key]:
        label_to_cat[label] = key

class seg_model(nn.Module):
    def __init__(self, parameters):
        self.F =  parameters["F"]
        self.K =  parameters["K"]
        self.M =  parameters["M"]

        assert len(self.F) == len(self.K)
        super(seg_model, self).__init__()

        self.one_layer = parameters["one_layer"]
        self.reg_prior = parameters["reg_prior"]
        self.num_points = parameters["num_points"]
        self.dropout = parameters["dropout"]
        self.fc_bias = parameters["fc_bias"]
        self.regularizers = []
        self.recompute_L = parameters["recompute_L"]
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.relus = self.F + self.M



        if parameters["b2relu"]:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, self.num_points, i))) for i in self.relus
            ])
        else:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, 1, i))) for i in self.relus
            ])

        self.conv = nn.ModuleList([
            conv.DenseChebConvV2(22, self.F[i], self.K[i]) if i == 0 else conv.DenseChebConvV2(self.F[i-1], self.F[i], self.K[i]) for i in range(len(self.K))
        ])

        self.fc = nn.ModuleList([])
        for i in range(len(self.M)):
            if i == 0:
                self.fc.append(nn.Linear(self.F[-1], self.M[i], self.fc_bias))
            elif i == 1:
                self.fc.append(
                    nn.Linear(self.M[i-1]+self.M[i-1], self.M[i], self.fc_bias))
            else:
                self.fc.append(nn.Linear(self.M[i-1], self.M[i], self.fc_bias))

        self.L = []
        self.x = []

    def b1relu(self, x, bias):
        return relu(x + bias)

    def brelu(self, x, bias):
        return relu(x + bias)

    def get_laplacian(self, x):
        with torch.no_grad():
            return conv.get_laplacian(conv.pairwise_distance(x))

    @torch.no_grad()
    def append_regularization_terms(self, x, L):
        if self.reg_prior:
            self.L.append(L)
            self.x.append(x)

    @torch.no_grad()
    def reset_regularization_terms(self):
        self.L = []
        self.x = []

    def forward(self, x, cat):
        self.reset_regularization_terms()

        x1 = 0  # cache for layer 1

        L = self.get_laplacian(x)

        cat = one_hot(cat, num_classes=16)
        cat = torch.tile(cat, [1, self.vertice, 1])
        out = torch.cat([x, cat], dim=2)  # Pass this to the model

        for i in range(len(self.K)):
            out = self.conv[i](out, L)
            self.append_regularization_terms(out, L)
            if self.recompute_L:
                L = self.get_laplacian(out)
            out = self.dropout(out)
            out = self.brelu(out, self.bias_relus[i])
            if i == 1:
                x1 = out

        for i in range(len(self.M)):
            if i == 1:
                out = t.concat([out, x1], dim=2)
            out = self.fc[i](out)
            self.append_regularization_terms(out, L)
            out = self.dropout(out)
            out = self.b1relu(out, self.bias_relus[i + len(self.K)])

        return out, self.x, self.L

def fit(model, optimizer, loader, criterion, regularization):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        cat = data.category
        y = data.y.type(torch.LongTensor)
        x = t.cat([data.pos.type(torch.float32),
                  data.x.type(torch.float32)], dim=2)
        # out, L are for regularization
        logits, out, L = model(x.to(model.device), cat.to(model.device))
        logits = logits.permute([0, 2, 1])

        loss = compute_loss(logits, y, out, L, criterion, s=regularization)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"{i}: curr loss: {loss}")
    return total_loss * model.batch_size / len(loader.dataset)


def test(model, loader):
    model.eval()
    size = len(loader.dataset)
    predictions = np.empty((size, model.num_points))
    labels = np.empty((size, model.num_points))
    total_correct = 0

    for i, data in enumerate(loader):
        cat = data.category
        x = t.cat([data.pos.type(torch.float32),
                  data.x.type(torch.float32)], dim=2)
        y = data.y
        logits, _, _ = model(x.to(model.device), cat.to(model.device))
        logits = logits.to('cpu')
        pred = logits.argmax(dim=2)

        total_correct += int((pred == y).sum())
        start = i * model.batch_size
        stop = start + model.batch_size
        predictions[start:stop] = pred
        lab = data.y
        labels[start:stop] = lab.reshape([-1, model.num_points])

    tot_iou = []
    cat_iou = defaultdict(list)
    for i in range(predictions.shape[0]):
        segp = predictions[i, :]
        segl = labels[i, :]
        cat = label_to_cat[segl[0]]
        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

        for l in seg_classes[cat]:
            # part is not present, no prediction as well
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                part_ious[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        cat_iou[cat].append(np.mean(part_ious))
        tot_iou.append(np.mean(part_ious))

    ncorrects = np.sum(predictions == labels)
    accuracy = ncorrects * 100 / (len(loader.dataset) * model.num_points)

    return accuracy, cat_iou, tot_iou, ncorrects


def train(parameters, root):
    
    transforms = Compose([FixedPoints(parameters["num_points"]), GaussianNoiseTransform(
        mu=0, sigma=0, recompute_normals=False)])

    dataset_train = ShapeNet(root=root, split="train", transform=transforms)
    dataset_test  = ShapeNet(root=root, split="test", transform=transforms)
    
    train_loader = DenseDataLoader(
        dataset_train, batch_size=parameters["batch_size"],
        shuffle=True, pin_memory=True)

    test_loader = DenseDataLoader(
        dataset_test, batch_size=parameters["batch_size"],
        shuffle=True)


    decay_steps   = len(dataset_train) / parameters["batch_size"]
    weights       = get_weights(dataset_train, parameters["num_points"])

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=float32).to('cuda'))

    model = seg_model(parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"])
    
    for epoch in range(50):    
        loss = fit(model, optimizer, train_loader, criterion, parameters['regularization'])
        acc = test(model, test_loader)

        if epoch % 5 == 0:
             torch.save(model.state_dict(), path + '/' +
                       str(parameters["num_points"]) + 'p_model_v2_' + str(epoch) + '.pt')


now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
parent_directory = "/home/domsa/workspace/git/RGCNN_git/Segmentation/models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)


if __name__ == "__main__":
    root = "/home/domsa/workspace/data/ShapeNet/"

    num_epochs = 50

    parameters = {
        "num_points": 2048,
        "dropout": 0.25,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "decay_rate": 0.7,
        "weight_decay": 1e-9,
        "one_layer": False,
        "reg_prior": False,
        "recompute_L": False,
        "b2relu": False,
        "fc_bias": True,
        "F": [128, 512, 1024],  # Outputs size of convolutional filter.
        "K": [6, 5, 3],         # Polynomial orders.
        "M": [512, 128, 50],
        "regularization": 1e-9,
    }

    train(parameters)
