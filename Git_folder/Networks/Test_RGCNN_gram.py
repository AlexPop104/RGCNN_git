import torch
from torch import nn

import torch as t
import torch_geometric as tg

import time



from torch.nn import Parameter

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader

from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (add_self_loops, get_laplacian,
                                   remove_self_loops)

from torch_geometric.utils import get_laplacian as get_laplacian_pyg

from path import Path

#from noise_transform import GaussianNoiseTransform
#import ChebConv_rgcnn_functions as conv
import os


from datetime import datetime
from torch.nn import MSELoss
from torch.optim import lr_scheduler


import numpy as np


from utils import GaussianNoiseTransform
import utils as util_functions

import dataset_loader_noise as cam_loader


import random
random.seed(0)

def get_n_params(model):
    pp=0 
    for p in list(model.parameters()):
        nn=1 
        for s in list(p.size()):
            nn = nn*s
            pp += nn
    return pp


class cls_model(nn.Module):
    def __init__(self, vertice ,F, K, M, class_num, regularization=0,  dropout=0, reg_prior:bool=True):
        assert len(F) == len(K)
        super(cls_model, self).__init__()

        self.F = F
        self.K = K
        self.M = M

        
        self.reg_prior = reg_prior
        self.vertice = vertice
        self.regularization = regularization    # gamma from the paper: 10^-9
        self.dropout = dropout
        self.regularizers = []

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)


        self.conv1 = util_functions.DenseChebConvV2(self.K[0], 128, 3)
        self.conv2 = util_functions.DenseChebConvV2(128,512, 3)
        self.conv3 = util_functions.DenseChebConvV2(512,1024, 3)

        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.max_pool = nn.MaxPool1d(self.vertice)

        self.regularizer = 0
        self.regularization = []             

    def forward(self, x,gram):
        self.regularizers = []

        
        with torch.no_grad():

            L = util_functions.pairwise_distance(x,normalize=True) # W - weight matrix
            L = util_functions.get_laplacian(L)

        out = self.conv1(gram, L)
        out = self.relu1(out)

        
       
        with torch.no_grad():
            L = util_functions.pairwise_distance(out,normalize=True) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv2(out, L)
        out = self.relu2(out)

  

        with torch.no_grad():
            L = util_functions.pairwise_distance(out,normalize=True) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv3(out, L)
        out = self.relu3(out)
        

        out, _ = t.max(out, 1)
       
        # ~~~~ Fully Connected ~~~~
        
        out = self.fc1(out)


        out = self.dropout(out)
        out = self.relu4(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu5(out)

        out = self.fc3(out)
        
        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer,num_points,criterion, loader, regularization,device):
    model.train()
    total_loss = 0
    total_correct = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        if (model.F[0]==6):
            x = torch.cat([data.pos, data.normal], dim=1)   
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()

        if (model.F[0]==3):
            x = data.pos
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()  

        x_transpose = x.permute(0, 2, 1)
        gram = t.matmul(x, x_transpose)

        gram=gram.to(device)

        gram,indices=torch.sort(gram)

        

        logits, regularizers  = model(x=x.to(device),gram=gram.to(device))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())
        
        loss    = criterion(logits, data.y.to(device))
       
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 

@torch.no_grad()
def test(model, loader,num_points,criterion,device):
    model.eval()

    # Dict from labels to names

    total_loss = 0
    total_correct = 0
    for data in loader:
       
        if (model.F[0]==6):
            x = torch.cat([data.pos, data.normal], dim=1)   
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()

        if (model.F[0]==3):
            x = data.pos
            x = x.reshape(data.batch.unique().shape[0], num_points, model.F[0])

            x =x.float()  

        x=x.to(device)

        for j in range(10000):

            start = time.time()

            x_transpose = x.permute(0, 2, 1)
            gram = t.matmul(x, x_transpose)

            # gram=gram.to(device)

            gram,indices=torch.sort(gram)
        
            stop = time.time()

            f = open("log_RGCNN_gram_pre.txt", "a")
            f.write(str(stop-start)+"\n")
            f.close()    
        
        

        # gram=gram.to(device)

        x_transpose = x.permute(0, 2, 1)
        gram = t.matmul(x, x_transpose)

        gram,indices=torch.sort(gram)

        # for j in range(10000):

        #     start = time.time()

        #     logits, regularizers = model(x=x.to(device),gram=gram.to(device))
        
        #     stop = time.time()

        #     f = open("log_RGCNN_gram_net.txt", "a")
        #     f.write(str(stop-start)+"\n")
        #     f.close()   


        
        logits, regularizers = model(x=x.to(device),gram=gram.to(device))
        loss    = criterion(logits, data.y.to(device))

        total_loss += loss.item() * data.num_graphs
        
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 





# print("Model nr of points")
# num_points=int(input())

batch_size = 1
num_epochs = 250
learning_rate = 1e-3
modelnet_num = 36
dropout=0.25
input_feature_selection=3



device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}")

data_select=-500

print("Select type of training  (1 - No noise training, 2 - Noisy data trainng)")
train_select=int(input())

if(train_select==2):
    print("Select category augmented data (1- Rotation Noise, 2-Position Noise, 3-Occlusion Noise")
    data_select=int(input())

if (train_select==1):
    #array_points=[64,256,512]
    array_points=[512]
else:
    array_points=[512]



# print("Select type of testing  (1 Position noise, 2 - Rotation noise , 3-Occlusion Noise, 4-Nr_points)")
# selection=int(input())

# print("Dataset nr of points")
# num_points_dataset=int(input())


torch.manual_seed(0)

list_final=[]



for j in range(1,4):
    selection=j
    if(selection==1):
        print(selection)
        list_final.append("Noise test sigma 0-0.02.05-0.08-0.10")
        for i in range(len(array_points)):
            num_points=int(array_points[i]) 
            #print(num_points)  
            list_final.append(num_points)

            input_feature_size=num_points


            F = [input_feature_selection, 512, 1024]  # Outputs size of convolutional filter.
            K = [input_feature_size, 5, 3]         # Polynomial orders.
            M = [1024, 128, modelnet_num]

            model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

            if (train_select!=1)and (data_select==1):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Occlusion/RGCNN_gram_Occlusion512_105.pt"
            elif (data_select==2):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Position/RGCNN_gram_Pos_noise512_132.pt"
            elif (data_select==3):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Rotation/RGCNN_gram_Rotation512_132.pt"
            
            else:
               path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/"+str(num_points)+"/RGCNN_"+str(num_points)+"_gram.pt"


            


            model.load_state_dict(torch.load(path_saved_model))
            #print(model.parameters)
            model = model.to(device) 

            nr_parameters = get_n_params(model)

            print(nr_parameters)
        

            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # ##############################################################################################

            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_002/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))
            # ##############################################################################################

            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_005/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))
            # ##############################################################################################

            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_008/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))
            # ##############################################################################################

            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_010/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))


    ###############################################################################################
    elif(selection==2):
        
        print(selection)
        list_final.append("Rotation invariance test 0-10-20-30-40")
        for i in range(len(array_points)):
            
            num_points=int(array_points[i])   
            #print(num_points)  
            list_final.append(num_points)

            input_feature_size=num_points


            F = [input_feature_selection, 512, 1024]  # Outputs size of convolutional filter.
            K = [input_feature_size, 5, 3]         # Polynomial orders.
            M = [1024, 128, modelnet_num]

            model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

            if (train_select!=1)and (data_select==1):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Occlusion/RGCNN_gram_Occlusion512_105.pt"
            elif (data_select==2):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Position/RGCNN_gram_Pos_noise512_132.pt"
            elif (data_select==3):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Rotation/RGCNN_gram_Rotation512_132.pt"
            
            else:
               path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/"+str(num_points)+"/RGCNN_"+str(num_points)+"_gram.pt"


            model.load_state_dict(torch.load(path_saved_model))
            #print(model.parameters)
            model = model.to(device) 
        

            
            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))



            # ##############################################################################################
            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_10/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # ###############################################################################################
            #print("Modelnet40 2048 points rotation 20 accuracy")
            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_20/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # ###############################################################################################
            # #print("Modelnet40 2048 points rotation 30 accuracy")
            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_30/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # ###############################################################################################
            # #print("Modelnet40 2048 points rotation 40 accuracy")
            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_40/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))


    elif(selection==3):
      
        print(selection)
        list_final.append("Occlusion noise, R=0.25")
        for i in range(len(array_points)):
            num_points=int(array_points[i])   
            #print(num_points)  
            list_final.append(num_points)

            input_feature_size=num_points


            F = [input_feature_selection, 512, 1024]  # Outputs size of convolutional filter.
            K = [input_feature_size, 5, 3]         # Polynomial orders.
            M = [1024, 128, modelnet_num]

            model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

            if (train_select!=1)and (data_select==1):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Occlusion/RGCNN_gram_Occlusion512_105.pt"
            elif (data_select==2):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Position/RGCNN_gram_Pos_noise512_132.pt"
            elif (data_select==3):
                path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/Noise_train_512/Rotation/RGCNN_gram_Rotation512_132.pt"
            
            else:
               path_saved_model="/media/cuda/ssd/Alex/RGCNN/Trained+models/"+str(num_points)+"/RGCNN_"+str(num_points)+"_gram.pt"


            model.load_state_dict(torch.load(path_saved_model))
            #print(model.parameters)
            model = model.to(device) 
        
            # #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Noise/Modelnet40_ocl_014/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Noise/Modelnet40_ocl_020/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            #num_points=512
            root = Path("/media/cuda/ssd/Alex/RGCNN/PCD_DATA/Normals/Occlusion/Modelnet40_occlusion_"+str(num_points)+"_025/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))



for t in range(len(list_final)):
    print(list_final[t])
