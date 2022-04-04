
import time

from datetime import datetime

import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch_geometric.transforms import SamplePoints

import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

from torch_geometric.datasets import GeometricShapes

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader

from noise_transform import GaussianNoiseTransform

import ChebConv_rgcnn_functions as conv

from torch.optim import lr_scheduler

import os

import random
random.seed(0)


class Tnet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self,k_transform):
        super().__init__()
        self.input_transform = Tnet(k=k_transform)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(k_transform,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        # self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        # xb = self.bn3(self.conv3(xb))

        
        # xb = nn.MaxPool1d(xb.size(-1))(xb)
        # output = nn.Flatten(1)(xb)

        return xb, matrix3x3, matrix64x64
        #return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, num_classes ,nr_features):
        super().__init__()
        self.transform = Transform(k_transform=nr_features)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.RGCNN_conv=conv.DenseChebConv(128,1024,3)
        self.relu_rgcnn=torch.nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)

        xb=torch.permute(xb,(0,2,1))

        with torch.no_grad():
            L = conv.pairwise_distance(xb) # W - weight matrix
            # for it_pcd in range(1):
            #     viz_points_2=input[it_pcd,:,:]
            #     viz_points_2=torch.permute(viz_points_2,(1,0))
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,1)
            # plt.show()
            L = conv.get_laplacian(L)

        xb = self.RGCNN_conv(xb, L)
        xb = self.relu_rgcnn(xb)


        xb=torch.permute(xb,(0,2,1))

        xb = nn.MaxPool1d(xb.size(-1))(xb)
        xb = nn.Flatten(1)(xb)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

def pointnetloss(outputs, labels, m3x3, m64x64,k, alpha = 0.0001,):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    #id3x3 = torch.eye(6, requires_grad=True).repeat(bs,1,1)
    id3x3 = torch.eye(k, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)      
        
   

def train(model, optimizer, loader,nr_points):
    model.train()
    
    total_loss = 0
    #for data in loader:
    for i, data in enumerate(loader, 0):
        optimizer.zero_grad()
        
        batch_size=int(data.y.shape[0])

        x = torch.cat([data.pos, data.normal], dim=1)   
        x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))

        #x=torch.reshape(data.pos,(batch_size,nr_points,data.pos.shape[1]))

        k=x.shape[2]
        

        x=x.to(device)
        labels=data.y.to(device)
       

        outputs, m3x3, m64x64 = model(x.transpose(1,2))
        #logits = model(data.pos.to(device).transpose(1,2))  # Forward pass.

        #loss = criterion(outputs, labels)  # Loss computation.
        loss = pointnetloss(outputs, labels, m3x3, m64x64,k=k)
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader,nr_points):
    model.eval()

    correct = total = 0
    total_correct=0
    for i, data in enumerate(loader, 0):

        batch_size=int(data.y.shape[0])

        x = torch.cat([data.pos, data.normal], dim=1)   
        x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))

        #x=torch.reshape(data.pos,(batch_size,nr_points,data.pos.shape[1]))

        x=x.to(device)
        labels=data.y.to(device)
       

        #logits = model(data.pos.to(device))
        outputs, __, __ = model(x.transpose(1,2))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        

    val_acc = 1. *correct / total
    

    #print(val_acc)
   
        

    #return total_correct / len(loader.dataset)
    return val_acc

now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)


modelnet_num = 40
num_points= 512
batch_size=16
nr_features=6


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")

root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)

model = PointNet(num_classes=modelnet_num,nr_features=nr_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)


#################################################################33

# transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

# train_dataset = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
# test_dataset = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

###################################################################

mu=0
sigma=0.
#transforms_noisy = Compose([SamplePoints(num_points),NormalizeScale(),GaussianNoiseTransform(mu, sigma,recompute_normals=True)])

rot_x=30
rot_y=30
rot_z=30

torch.manual_seed(0)

#################################


random_rotate_0 = Compose([
    RandomRotate(degrees=0, axis=0),
    RandomRotate(degrees=0, axis=1),
    RandomRotate(degrees=0, axis=2),
    ])

test_transform_0 = Compose([
    random_rotate_0,
    SamplePoints(num_points, include_normals=True),
    NormalizeScale(),
    GaussianNoiseTransform(mu, sigma,recompute_normals=True)
    ])

random_rotate_10 = Compose([
    RandomRotate(degrees=10, axis=0),
    RandomRotate(degrees=10, axis=1),
    RandomRotate(degrees=10, axis=2),
    ])

test_transform_10 = Compose([
    random_rotate_10,
    SamplePoints(num_points, include_normals=True),
    NormalizeScale(),
    GaussianNoiseTransform(mu, sigma,recompute_normals=True)
    ])

random_rotate_20 = Compose([
    RandomRotate(degrees=20, axis=0),
    RandomRotate(degrees=20, axis=1),
    RandomRotate(degrees=20, axis=2),
    ])

test_transform_20 = Compose([
    random_rotate_20,
    SamplePoints(num_points, include_normals=True),
    NormalizeScale(),
    GaussianNoiseTransform(mu, sigma,recompute_normals=True)
    ])


train_dataset_0 = ModelNet(root=root, name=str(modelnet_num), train=True, transform=test_transform_0)
test_dataset_0 = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform_0)

train_dataset_10 = ModelNet(root=root, name=str(modelnet_num), train=True, transform=test_transform_0)
test_dataset_10 = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform_0)

train_dataset_20 = ModelNet(root=root, name=str(modelnet_num), train=True, transform=test_transform_0)
test_dataset_20 = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform_0)



###############################################################################

train_loader_0 = DataLoader(train_dataset_0, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader_0  = DataLoader(test_dataset_0, batch_size=batch_size)

train_loader_10 = DataLoader(train_dataset_10, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader_10  = DataLoader(test_dataset_10, batch_size=batch_size)

train_loader_20 = DataLoader(train_dataset_20, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader_20  = DataLoader(test_dataset_20, batch_size=batch_size)

# program_name="Pointnet"
# conv.view_pcd(model=model,loader=test_loader,num_points=num_points,device=device,program_name=program_name)

for epoch in range(1, 251):

    loss_t=0
    acc_t=0

    train_start_time = time.time()
    train_loss = train(model, optimizer, train_loader_0,nr_points=num_points)
    train_stop_time = time.time()

    loss_t=loss_t+train_loss

    train_start_time = time.time()
    train_loss = train(model, optimizer, train_loader_10,nr_points=num_points)
    train_stop_time = time.time()

    loss_t=loss_t+train_loss

    train_start_time = time.time()
    train_loss = train(model, optimizer, train_loader_20,nr_points=num_points)
    train_stop_time = time.time()

    loss_t=loss_t+train_loss


    test_start_time = time.time()
    test_acc = test(model, test_loader_0,nr_points=num_points)
    test_stop_time = time.time()

    acc_t=acc_t+test_acc

    test_start_time = time.time()
    test_acc = test(model, test_loader_10,nr_points=num_points)
    test_stop_time = time.time()

    acc_t=acc_t+test_acc

    test_start_time = time.time()
    test_acc = test(model, test_loader_20,nr_points=num_points)
    test_stop_time = time.time()

    acc_t=acc_t+test_acc

    train_loss=loss_t/3
    test_acc=acc_t/3

    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    writer.add_scalar("Loss/train", train_loss, epoch)
    
    writer.add_scalar("Acc/test", test_acc, epoch)

    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
    Test Time: \t{test_stop_time - test_start_time }')

    # writer.add_figure("Confusion matrix", createConfusionMatrix(model,test_loader), epoch)

    if(epoch%5==0):
        torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

    my_lr_scheduler.step()


torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


