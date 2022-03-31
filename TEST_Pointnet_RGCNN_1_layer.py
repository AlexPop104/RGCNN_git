import time



import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn

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


from torch.optim import lr_scheduler

import ChebConv_rgcnn_functions as conv

import matplotlib.pyplot as plt

from datetime import datetime
import os

from noise_transform import GaussianNoiseTransform

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
        #self.conv3 = nn.Conv1d(128,256,1) # 1024 initial


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(256) #1024 initial

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        #xb = self.bn3(self.conv3(xb))

        
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
        
criterion = torch.nn.CrossEntropyLoss()  

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
num_epochs=250
nr_features=6





root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)

#root="/media/rambo/ssd2/Alex_data/RGCNN/GeometricShapes"






    





model = PointNet(num_classes=modelnet_num,nr_features=nr_features)
path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/logs/Modele_selectate/Pointnet_RGCNN/model75.pt"
model.load_state_dict(torch.load(path_saved_model))
#print(model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

rot_x=0
rot_y=0
rot_z=0

mu=0
sigma=0

torch.manual_seed(0)

for ceva in range(0,1):
    
    # random_rotate = Compose([
    # RandomRotate(degrees=rot_x*10, axis=0),
    # RandomRotate(degrees=rot_y*10, axis=1),
    # RandomRotate(degrees=rot_z*10, axis=2),
    # ])

    # test_transform = Compose([
    # random_rotate,
    # SamplePoints(num_points, include_normals=True),
    # NormalizeScale()
    # ])

    # test_dataset = ModelNet(root=root, train=False,
    #                         transform=test_transform)

# program_name="Pointnet_RGCNN_rot_x"+str(10*rot_x)+"_rot_y"+str(10*rot_y)+"_rot_z"+str(10*rot_z)


######################################################################33

    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    test_dataset = ModelNet(root=root, train=False,
                                   transform=transforms)

    program_name="Pointnet_RGCNN"

################################################################33

    # mu=0
    # sigma=0.05

    # transforms_noisy = Compose([SamplePoints(num_points),NormalizeScale(),GaussianNoiseTransform(mu, sigma,recompute_normals=True)])
   

    # test_dataset = ModelNet(root=root, train=False,
    #                         transform=transforms_noisy)

    #program_name="Pointnet_RGCNN_noise_mu"+str(mu)+"_sigma_"+str(sigma)
    

    

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    conv.view_pcd(model=model,loader=test_loader,num_points=num_points,device=device,program_name=program_name)


    #conv.test_pcd_pred(model=model,loader=test_loader,num_points=num_points,device=device)
        
    test_start_time = time.time()
    test_acc = test(model, test_loader,nr_points=num_points)
    test_stop_time = time.time()

    print(f'{test_acc:.4f}')

    #print(f'\Test Time: \t{test_stop_time - test_start_time }')
    #print(f' Test Accuracy: {test_acc:.4f}')

    
