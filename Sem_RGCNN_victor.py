from datetime import datetime

import numpy as np

from torch_geometric.loader import DenseDataLoader
import os
import Classif_RGCNN_n_DenseConv_functions as conv
import torch as t
from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
import torch
from torch import nn
import time
from torch.nn.functional import one_hot

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


class seg_model(nn.Module):
    def __init__(self, vertice, F, K, M, regularization=0, one_layer=True, dropout=0, reg_prior: bool = True):
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
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.conv1 = conv.DenseChebConv(22, 128, 6)
        self.conv2 = conv.DenseChebConv(128, 512, 5)
        self.conv3 = conv.DenseChebConv(512, 1024, 3)
        self.conv4 = conv.DenseChebConv(1152, 512, 1)
        self.conv5 = conv.DenseChebConv(512, 128, 1)
        self.conv6 = conv.DenseChebConv(128, 50, 1)

        self.regularizer = 0
        self.regularization = []

    def forward(self, x):
        self.regularizers = []
        x1 = 0  # cache for layer1

        with torch.no_grad():
            L = conv.pairwise_distance(x)  # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv1(x, L)
        out = self.relu1(out)

        x1 = out

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(out.permute(0, 2, 1), L), x))**2)

        with torch.no_grad():
            L = conv.pairwise_distance(out)  # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv2(out, L)
        out = self.relu2(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(out.permute(0, 2, 1), L), x))**2)

        with torch.no_grad():
            L = conv.pairwise_distance(out)  # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv3(out, L)
        out = self.relu3(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(out.permute(0, 2, 1), L), x))**2)

        out = torch.cat([out, x1], dim=2)

        with torch.no_grad():
            L = conv.pairwise_distance(out)  # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv4(out, L)
        out = self.relu4(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(out.permute(0, 2, 1), L), x))**2)

        with torch.no_grad():
            L = conv.pairwise_distance(out)  # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv5(out, L)
        out = self.relu5(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(out.permute(0, 2, 1), L), x))**2)

        with torch.no_grad():
            L = conv.pairwise_distance(out)  # W - weight matrix
            L = conv.get_laplacian(L)

        out = self.conv6(out, L)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(
                t.matmul(t.matmul(out.permute(0, 2, 1), L), x))**2)

        return out, self.regularizers


criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.


def train(model, optimizer, loader, regularization):
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        cat = one_hot(data.category, num_classes=16)
        cat = torch.tile(cat, [1, 2048, 1]) 
        x = torch.cat([data.pos, data.x, cat], dim=2)  ### Pass this to the model
        y = one_hot(data.y, num_classes=50)
        y = y.type(t.FloatTensor)
        logits, regularizers = model(x.to(device))
        loss = criterion(logits, y.to(device))
        s = t.sum(t.as_tensor(regularizers))
        loss = loss + regularization * s
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        # if i%100 == 0:
        # print(f"{i}: curr loss: {loss}")
        # $print(f"{data.y} --- {logits.argmax(dim=1)}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        cat = one_hot(data.category, num_classes=16)
        cat = torch.tile(cat, [1, 2048, 1]) 
        x = torch.cat([data.pos, data.x, cat], dim=2)  ### Pass this to the model 
        logits, _ = model(x.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_correct / len(loader.dataset)


if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    log_folder_path="/home/alex/Alex_documents/RGCNN_git/data/logs/Network_performances/"
   

    loss_log_path=log_folder_path+"Type_"+directory+"_losses.npy"
    accuracy_log_path=log_folder_path+"Type_"+directory+"test_accuracy.npy"

    num_points = 2048
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-3

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    root = "/home/alex/Alex_documents/RGCNN_git/data/ShapeNet"
    print(root)
    dataset_train = ShapeNet(root=root, split="train", transform=FixedPoints(2048))
    dataset_test = ShapeNet(root=root, split="test", transform=FixedPoints(2048))

    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")

    train_loader = DenseDataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DenseDataLoader(dataset_test, batch_size=batch_size)

    model = seg_model(num_points, F, K, M,
                      dropout=1, one_layer=False, reg_prior=True)
    model = model.to(device)

    print(model.parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    all_losses=np.array([])
    test_accuracy=np.array([])

    regularization = 1e-9
    for epoch in range(1, num_epochs):
        train_start_time = time.time()
        loss = train(model, optimizer, train_loader,
                     regularization=regularization)
        train_stop_time = time.time()

        # writer.add_scalar("Loss/train", loss, epoch)

        all_losses=np.append(all_losses, loss)

        test_start_time = time.time()
        test_acc = test(model, test_loader)
        test_stop_time = time.time()

        test_accuracy=np.append(test_accuracy, test_acc)

        if(epoch%5==0):
            np.save(loss_log_path, all_losses)
            np.save(accuracy_log_path, test_accuracy)

            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        # writer.add_scalar("Acc/test", test_acc, epoch)
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')
    
    np.save(loss_log_path, all_losses)
    np.save(accuracy_log_path, test_accuracy)