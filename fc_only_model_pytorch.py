import time
from sklearn.model_selection import learning_curve
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch import dropout, nn
import torch as t
import torch_geometric as tg


from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose

import ChebConv_rgcnn as conv

import os
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DenseDataLoader
from datetime import datetime
from torch.optim import lr_scheduler



class fc_model(nn.Module):
    def __init__(self, vertice, class_num, in_features=6, regularization=0, reg_prior=True, dropout=0):
        super(fc_model, self).__init__()

        self.reg_prior = reg_prior
        self.vertice = vertice
        self.regularization = regularization

        self.relu = nn.ReLU()
        
        self.dropout = t.nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(in_features, 128, bias=True)
        self.fc2 = nn.Linear(128, 512, bias=True)
        self.fc3 = nn.Linear(512, 1024, bias=True)

        #self.fc1 = conv.DenseChebConv(in_features, 128, 1)
        #self.fc2 = conv.DenseChebConv(128, 512, 1)
        #self.fc3 = conv.DenseChebConv(512, 1024, 1)
        
        self.fc4 = nn.Linear(1024, 512, bias=True)
        self.fc5 = nn.Linear(512, 128, bias=True)
        self.fc6 = nn.Linear(128, class_num, bias=True)
        
        self.max_pool = nn.MaxPool1d(self.vertice)

        self.regularizers = []
    
    def forward(self, x):
        with t.no_grad():
            L = conv.pairwise_distance(x) # W - weight matrix
            L = conv.get_laplacian(L)

        x = self.fc1(x)
        x = self.relu(x)
        
        with t.no_grad():
            L = conv.pairwise_distance(x) # W - weight matrix
            L = conv.get_laplacian(L)
        
        x = self.fc2(x)
        x = self.relu(x)

        with t.no_grad():
            L = conv.pairwise_distance(x) # W - weight matrix
            L = conv.get_laplacian(L)
        
        x = self.fc3(x)
        x = self.relu(x)

        x, _ = t.max(x, 1)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)
        x = self.relu(x)

        x = self.fc6(x)

        return x, self.regularizers


criterion = t.nn.CrossEntropyLoss()  # Define loss criterion.


def train(model, optimizer, loader, regularization, device='cuda'):
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        x = t.cat([data.pos, data.normal], dim=2)

        y = data.y.to(device)
        y = t.squeeze(y,1)
        logits, regularizers = model(x.to(device))
        
        loss = criterion(logits, y)

        # loss += regularization * t.sum(t.as_tensor(regularizers))
        loss.backward()

        optimizer.step()

        total_loss += loss.item() 

    return total_loss * batch_size  / len(dataset_train)


def test(model, loader, device='cuda'):
    with t.no_grad():
        model.eval()

        total_correct = 0
        for data in loader:
            x = t.cat([data.pos, data.normal], dim=2)  ### Pass this to the model
            y = data.y

            logits, _ = model(x.to(device))
            pred = logits.argmax(dim=-1).to('cpu')
            pred = pred.reshape([batch_size])
            y = y.reshape([batch_size])
            total_correct += int((pred == y).sum())

        return total_correct / len(loader.dataset)
 
if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    parent_directory = "/home/victor/workspace/thesis_ws/github/RGCNN_git/models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)
    num_points = 1024
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-3
    class_num = 40
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet" + str(class_num)
    print(root)

    dataset_train = ModelNet(root=root, name=str(class_num), train=True, transform=transforms)
    dataset_test = ModelNet(root=root, name=str(class_num), train=False, transform=transforms)

    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")

    train_loader = DenseDataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader  = DenseDataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = fc_model(num_points, class_num, dropout=1)
    model.to(device)

    print(model.parameters)

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    regularization = 1e-9
    for epoch in range(1, 51):
        train_start_time = time.time()
        loss = train(model, optimizer, train_loader, regularization=regularization)
        train_stop_time = time.time()

        writer.add_scalar("Loss/train", loss, epoch)
        
        test_start_time = time.time()
        test_acc = test(model, test_loader)
        test_stop_time = time.time()

        writer.add_scalar("Acc/test", test_acc, epoch)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        my_lr_scheduler.step()