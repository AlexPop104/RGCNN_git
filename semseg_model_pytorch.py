from collections import defaultdict
from datetime import datetime
from venv import create
from torch_geometric.loader import DenseDataLoader
import os
import ChebConv_rgcnn as conv
import torch as t
from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
from torch_geometric.datasets import S3DIS
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
        x = torch.cat([data.pos, data.x], dim=2)  ### Pass this to the model
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
        x = torch.cat([data.pos, data.x], dim=2)  ### Pass this to the model
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

    '''  
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

    print("~~~" * 50)
    for key, value in cat_iou.items():
        print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
    # print(tot_iou)
    # accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
    # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
    '''    
    ncorrects = np.sum(predictions == labels)
    accuracy  = ncorrects * 100 / (len(dataset_test) * num_points)
    print("~~~" * 30)
    print(f"\tAccuracy: {accuracy}, ncorrect: {ncorrects} / {len(dataset_test) * num_points}")
    # print(f"\tIoU: \t{np.mean(tot_iou)*100}")
    return accuracy

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
        test_acc = test(model, test_loader)
        test_stop_time = time.time()

        writer.add_scalar("accuracy/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
            Test Time: \t{test_stop_time - test_start_time }')

        my_lr_scheduler.step()

    print(f"Training finished")
    print(model.parameters())

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
    parent_directory = "/home/victor/workspace/thesis_ws/models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_points = 4092

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    root = "/home/victor/workspace/thesis_ws/datasets/S3DIS"
    print(root)
    dataset_train = S3DIS(root=root, test_area=6, train=True)
    dataset_test = S3DIS(root=root, test_area=6, train=False)
   

    batch_size = 2
    num_epochs = 50
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
                      dropout=1, one_layer=False, reg_prior=True, input_dim=9)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    start_training(model, train_loader, test_loader, optimizer, epochs=50)