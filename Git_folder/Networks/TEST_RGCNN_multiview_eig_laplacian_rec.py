import torch
from torch import nn

import torch as t
import torch_geometric as tg

import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

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

import dataset_loader_noise_rot_matrix_eigen as cam_loader


import random
random.seed(0)



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

    def forward(self, x,Rotation):
        self.regularizers = []

        x_batch_size=x.shape[0]
        x_nr_points=x.shape[1]
        x_feature_size=x.shape[2]

        rotation_size=Rotation.shape[2]

        

        # Rotation = Rotation.reshape(Rotation.shape[0], Rotation.shape[1]* Rotation.shape[2])
        # Rotation=Rotation.tile((4))
        # Rotation =Rotation.reshape(x_batch_size, rotation_size*4 *rotation_size)
        # Rotation =Rotation.reshape(x_batch_size*4, rotation_size*rotation_size)
        # Rotation =Rotation.reshape(x_batch_size*4, rotation_size,rotation_size)



        # sign_matrix=[[1 ,1 ,1],
        #              [1 ,1 ,-1], 
        #              [1 ,-1 ,1],
        #              [1 ,-1 ,-1],
        #              [-1 ,1 ,1],
        #              [-1 ,1 ,-1],
        #              [-1 ,-1 ,1],
        #              [-1 ,-1 ,-1]]


    
        
        # sign_matrix=torch.Tensor(sign_matrix)

        # sign_matrix=sign_matrix.to("cuda")

        # sign_matrix=sign_matrix.tile(rotation_size)
        # sign_matrix=sign_matrix.reshape(8,rotation_size,rotation_size)

        # sign_matrix=sign_matrix.reshape(8,rotation_size*rotation_size)
        # sign_matrix=sign_matrix.reshape(8*rotation_size*rotation_size)
        # sign_matrix=sign_matrix.tile(x_batch_size)

        # sign_matrix=sign_matrix.reshape(8*x_batch_size,rotation_size*rotation_size)
        # sign_matrix=sign_matrix.reshape(8*x_batch_size,rotation_size,rotation_size)

        

        Rotation_matrix=torch.Tensor([[[1.,0. ,0.],[0.,1. ,0.],[0.,0. ,1.]],
                                     [[1.,0. ,0.],[0.,-1. ,0.],[0.,0. ,-1.]],
                                     [[-1.,0. ,0.],[0.,1. ,0.],[0.,0. ,-1.]],
                                     [[-1.,0. ,0.],[0.,-1. ,0.],[0.,0. ,1.]],])
        Rotation_matrix=Rotation_matrix.to(device)

        Rotation_matrix=  Rotation_matrix.reshape(4,rotation_size*rotation_size)    
        Rotation_matrix=  Rotation_matrix.reshape(4*rotation_size*rotation_size)   
        Rotation_matrix=  Rotation_matrix.tile(x_batch_size)  

        Rotation_matrix=Rotation_matrix.reshape(x_batch_size,4*rotation_size*rotation_size)
        Rotation_matrix=Rotation_matrix.reshape(4*x_batch_size,rotation_size*rotation_size)
        Rotation_matrix=Rotation_matrix.reshape(4*x_batch_size,rotation_size,rotation_size)                  

            

        #Rotation=torch.bmm(Rotation,Rotation_matrix)
        Rotation=Rotation_matrix
        
        with torch.no_grad():
            L = util_functions.pairwise_distance(x) # W - weight matrix
            L = util_functions.get_laplacian(L)

        x = x.reshape(x.shape[0], x.shape[1]* x.shape[2])
        x=x.tile((4))
        x =x.reshape(x_batch_size, x_nr_points*4 *x_feature_size)
        x =x.reshape(x_batch_size*4, x_nr_points*x_feature_size)
        x =x.reshape(x_batch_size*4, x_nr_points,x_feature_size)

        x=torch.bmm(x,Rotation)

        L = L.reshape(L.shape[0], L.shape[1]* L.shape[2])
        L=L.tile((4))
        L =L.reshape(x_batch_size, x_nr_points*4 *x_nr_points)
        L =L.reshape(x_batch_size*4, x_nr_points *x_nr_points)
        L =L.reshape(x_batch_size*4, x_nr_points,x_nr_points)

        out = self.conv1(x, L)
        out = self.relu1(out)

        
       
        with torch.no_grad():
            L = util_functions.pairwise_distance(out) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv2(out, L)
        out = self.relu2(out)

  

        with torch.no_grad():
            L = util_functions.pairwise_distance(out) # W - weight matrix
            L = util_functions.get_laplacian(L)
        
        out = self.conv3(out, L)
        out = self.relu3(out)
        
        

        out, _ = t.max(out, 1)

        out =out.reshape(x_batch_size,4,1024)

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



@torch.no_grad()
def test(model, loader,num_points,criterion,device):
    model.eval()

    # Dict from labels to names

    total_loss = 0
    total_correct = 0
    for data in loader:

        Rotation=data.Rotation
        Rotation=Rotation.to(device)
        Rotation=Rotation.float()

        Rotation = Rotation.reshape(data.batch.unique().shape[0], Rotation.shape[1], Rotation.shape[1])
       
        if (model.conv1.in_channels==6):
            x = torch.cat([data.pos, data.normal], dim=1)   
            x = x.reshape(data.batch.unique().shape[0], num_points, model.conv1.in_channels)

            x =x.float()

        if (model.conv1.in_channels==3):
            x = data.pos
            x = x.reshape(data.batch.unique().shape[0], num_points, model.conv1.in_channels)

            x =x.float()  

        logits, regularizers = model(x=x.to(device),Rotation=Rotation.to(device))
        loss    = criterion(logits, data.y.to(device))

        total_loss += loss.item() * data.num_graphs
        
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 





# print("Model nr of points")
# num_points=int(input())

batch_size = 16
num_epochs = 250
learning_rate = 1e-3
modelnet_num = 36
dropout=0.25
input_feature_size=3

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [input_feature_size, 5, 3]         # Polynomial orders.
M = [512, 128, modelnet_num]


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

            model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

            if (train_select!=1)and (data_select==1):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Rotation/RGCNN_mview_eig_lr_Rotation512_102.pt"
            elif (data_select==2):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Position/RGCNN_mview_eig_lr_Pos_noise512_102.pt"
            elif (data_select==3):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Occlusion/RGCNN_mview_eig_lr_Occlusion512_105.pt"
            
            else:
               path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/"+str(num_points)+"/RGCNN_"+str(num_points)+"_multiview_eig_lap_rec.pt"

            


            model.load_state_dict(torch.load(path_saved_model))
            #print(model.parameters)
            model = model.to(device) 
        

            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # ##############################################################################################

            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_002/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))
            # ##############################################################################################

            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_005/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))
            # ##############################################################################################

            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_008/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))
            # ##############################################################################################

            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Noise/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_n_010/")
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
            model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

            if (train_select!=1)and (data_select==1):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Rotation/RGCNN_mview_eig_lr_Rotation512_102.pt"
            elif (data_select==2):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Position/RGCNN_mview_eig_lr_Pos_noise512_102.pt"
            elif (data_select==3):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Occlusion/RGCNN_mview_eig_lr_Occlusion512_105.pt"
            
            else:
               path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/"+str(num_points)+"/RGCNN_"+str(num_points)+"_multiview_eig_lap_rec.pt"


            model.load_state_dict(torch.load(path_saved_model))
            #print(model.parameters)
            model = model.to(device) 
        

            
            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))



            # ##############################################################################################
            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_10/")
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
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_20/")
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
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_30/")
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
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Test_rotation_invariant/"+str(num_points)+"/Modelnet40_"+str(num_points)+"_r_40/")
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
            model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)

            if (train_select!=1)and (data_select==1):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Rotation/RGCNN_mview_eig_lr_Rotation512_102.pt"
            elif (data_select==2):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Position/RGCNN_mview_eig_lr_Pos_noise512_102.pt"
            elif (data_select==3):
                path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/Noise_train_512/Occlusion/RGCNN_mview_eig_lr_Occlusion512_105.pt"
            
            else:
               path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/Trained+models/"+str(num_points)+"/RGCNN_"+str(num_points)+"_multiview_eig_lap_rec.pt"


            model.load_state_dict(torch.load(path_saved_model))
            #print(model.parameters)
            model = model.to(device) 
        
            # #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Noise/Modelnet40_ocl_014/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            # #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Noise/Modelnet40_ocl_020/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))

            #num_points=512
            root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Occlusion/Modelnet40_occlusion_"+str(num_points)+"_025/")
            test_dataset = cam_loader.PcdDataset(root_dir=root, valid=True, folder='test',points=num_points)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size)

            test_start_time = time.time()
            test_loss,test_acc = test(model=model, loader=test_loader,num_points=num_points,criterion=criterion,device=device)
            test_stop_time = time.time()

            #print(f'{test_acc:.4f}')
            list_final.append(str(test_acc))


for t in range(len(list_final)):
    print(list_final[t])
