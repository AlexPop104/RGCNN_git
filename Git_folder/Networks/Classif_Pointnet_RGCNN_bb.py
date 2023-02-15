
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

# from noise_transform import GaussianNoiseTransform

# import ChebConv_rgcnn_functions as conv

from torch.optim import lr_scheduler

import os

# import sys
# sys.path.insert(1, '/home/alex/Alex_documents/RGCNN_git/')

from utils import GaussianNoiseTransform
import utils as util_functions

import random
random.seed(0)

import dataset_loader_noise_rot_center as cam_loader

from path import Path


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
        
        self.RGCNN_conv=util_functions.DenseChebConvV2(128,1024,3)
        self.relu_rgcnn=torch.nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)

        xb=torch.permute(xb,(0,2,1))

        with torch.no_grad():
            L = util_functions.pairwise_distance(xb) # W - weight matrix
            # for it_pcd in range(1):
            #     viz_points_2=input[it_pcd,:,:]
            #     viz_points_2=torch.permute(viz_points_2,(1,0))
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,1)
            # plt.show()
            L = util_functions.get_laplacian(L)

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
    total_correct = 0
    #for data in loader:
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        
        batch_size=int(data.y.shape[0])

        if (model.transform.input_transform.k==6):

            x = torch.cat([data.pos, data.normal], dim=1)   
            x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))

        if (model.transform.input_transform.k==3):
            x = data.pos
            x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))

        x=x.float()

        #x=torch.reshape(data.pos,(batch_size,nr_points,data.pos.shape[1]))

        k=x.shape[2]
        

        x=x.to(device)
        labels=data.y.to(device)
       

        outputs, m3x3, m64x64 = model(x.transpose(1,2))
        #logits = model(data.pos.to(device).transpose(1,2))  # Forward pass.

        pred = outputs.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

        loss = criterion(outputs, labels)  # Loss computation.
        #loss = pointnetloss(outputs, labels, m3x3, m64x64,k=k)
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset),total_correct / len(loader.dataset) 


@torch.no_grad()
def test(model, loader,nr_points):
    model.eval()

    correct = total = 0
    total_correct=0

    for i, data in enumerate(loader):

        batch_size=int(data.y.shape[0])

        if (model.transform.input_transform.k==6):

            x = torch.cat([data.pos, data.normal], dim=1)   
            x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))

        if (model.transform.input_transform.k==3):
            x = data.pos
            x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))

        x=x.float()

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
directory="Pointnet_RGCNN"+directory
parent_directory = "/media/rambo/ssd2/Alex_data/RGCNN/data/logs/Trained_Models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)


modelnet_num = 36
num_points= 512
batch_size=16
nr_features=3
num_epochs=200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")

root = "/mnt/ssd1/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
#root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)

model = PointNet(num_classes=modelnet_num,nr_features=nr_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)




###################################################################
print("Select type of training  (1 - no noise, 2 - Rotation noise , 3- Position noise)")
selection=int(input())

torch.manual_seed(0)

#################################


if(selection==1):
    root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/Modelnet40_512/")
    train_dataset_0 = cam_loader.PcdDataset(root_dir=root, points=num_points)
    test_dataset_0 = cam_loader.PcdDataset(root_dir=root, folder='test',points=num_points)


    ###############################################################################

    train_loader_0 = DataLoader(train_dataset_0, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader_0  = DataLoader(test_dataset_0, batch_size=batch_size)


    ###############################################################################

    # program_name="Pointnet"
    # conv.view_pcd(model=model,loader=test_loader,num_points=num_points,device=device,program_name=program_name)

    for epoch in range(1, num_epochs+1):


        train_start_time = time.time()
        train_loss,train_acc = train(model, optimizer, train_loader_0,nr_points=num_points)
        train_stop_time = time.time()

        

        test_start_time = time.time()
        test_acc = test(model, test_loader_0,nr_points=num_points)
        test_stop_time = time.time()

        
        writer.add_scalar("Loss/train", train_loss, epoch)
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
    root_train_10 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_r_10/")
    root_train_20 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_r_20/")
    root_train_30 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_r_30/")
    root_train_40 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_r_40/")

    root_test = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"/")

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
    for epoch in range(1, num_epochs+1):
        loss_tr=0
        loss_t=0
        acc_t=0
        acc_tr=0

        train_start_time = time.time()
        train_loss,train_acc = train(model, optimizer, train_loader_10,nr_points=num_points)

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model, optimizer, train_loader_20,nr_points=num_points)

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model, optimizer, train_loader_30,nr_points=num_points)

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model, optimizer, train_loader_40,nr_points=num_points)
        train_stop_time = time.time()

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        loss_tr=loss_tr/4
        acc_tr=acc_tr/4

        test_start_time = time.time()
        test_acc = test(model, test_loader,nr_points=num_points)
        test_stop_time = time.time()




        writer.add_scalar("Loss/train", loss_tr, epoch)
        writer.add_scalar("Acc/train", acc_tr, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {loss_tr:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%3==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


        my_lr_scheduler.step()


    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

elif(selection==3):
    root_train_002 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_n_002/")
    root_train_005 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_n_005/")
    root_train_008 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_n_008/")
    root_train_010 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"_n_010/")

    root_test = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_"+str(num_points)+"/")

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

    for epoch in range(1, num_epochs+1):
        loss_tr=0
        loss_t=0
        acc_t=0
        acc_tr=0

        train_start_time = time.time()
        train_loss,train_acc = train(model, optimizer, train_loader_002,nr_points=num_points)

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model, optimizer, train_loader_005,nr_points=num_points)

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model, optimizer, train_loader_008,nr_points=num_points)

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        train_loss,train_acc = train(model, optimizer, train_loader_010,nr_points=num_points)
        train_stop_time = time.time()

        loss_tr=loss_tr+train_loss
        acc_tr=acc_tr+train_acc

        loss_tr=loss_tr/4
        acc_tr=acc_tr/4

        test_start_time = time.time()
        test_acc = test(model, test_loader,nr_points=num_points)
        test_stop_time = time.time()




        writer.add_scalar("Loss/train", loss_tr, epoch)
        writer.add_scalar("Acc/train", acc_tr, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

        print(f'Epoch: {epoch:02d}, Loss: {loss_tr:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%3==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


        my_lr_scheduler.step()


    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')