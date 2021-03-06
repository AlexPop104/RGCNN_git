import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch import nn
import torch
from torch.nn import Parameter

from torch_geometric.nn import fps,radius_graph,nearest


from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import RandomRotate


from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (add_self_loops, get_laplacian,
                                   remove_self_loops)
import torch as t
import torch_geometric as tg

from torch_geometric.utils import get_laplacian as get_laplacian_pyg
from torch_geometric.transforms import Compose

import ChebConv_rgcnn_functions as conv
import os
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
from datetime import datetime
from torch.nn import MSELoss
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np



class cls_model(nn.Module):
    def __init__(self, vertice ,F, K, M, class_num, regularization=0, one_layer=True, dropout=0, reg_prior:bool=True):
        assert len(F) == len(K)
        super(cls_model, self).__init__()

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
        self.relu_Reeb = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.conv1 = conv.DenseChebConv(6, 128, 6)
        self.conv2 = conv.DenseChebConv(128, 512, 5)
        self.conv_Reeb = conv.DenseChebConv(128, 512, 5)
        
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.fc_t = nn.Linear(128, class_num)

        self.max_pool = nn.MaxPool1d(self.vertice)

        if one_layer == True:
            self.fc = nn.Linear(128, class_num)

        self.regularizer = 0
        self.regularization = []


    def forward(self, x,k,batch_size,num_points,nr_points_fps):
        self.regularizers = []
        with torch.no_grad():
            L = conv.pairwise_distance(x) # W - weight matrix
            #L = conv.get_one_matrix_knn(L, k,batch_size,num_points)
            L = conv.get_laplacian(L)

        out = self.conv1(x, L)
        out = self.relu1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)
        
        if self.one_layer == False:


            with torch.no_grad():

                # nr_points_fps=55

                x2=x
                x2=torch.reshape(x2,(batch_size*num_points,x2.shape[2]))

                nr_points_batch=int(num_points)
                out_2=torch.reshape(out,(batch_size*num_points,out.shape[2]))
                sccs_batch, pcd_fps_points=conv.get_fps_matrix_2(point_cloud=out_2,original_cloud=x2,batch_size=batch_size,nr_points=num_points,nr_points_fps=nr_points_fps)
                sccs_batch=sccs_batch.long()
                sccs_batch=torch.reshape(sccs_batch,(batch_size,nr_points_fps,nr_points_batch))
                Vertices_final_FPS=torch.zeros([batch_size,sccs_batch.shape[1],nr_points_batch, out_2.shape[1]], dtype=torch.float32,device='cuda')
                
                for batch_iter in range(batch_size):   
                    for i in range(sccs_batch.shape[1]):
                        Vertices_pool_FPS=torch.zeros([sccs_batch[batch_iter,i].shape[0],out_2.shape[1]], dtype=torch.float32,device='cuda')
                        
                        ceva=torch.unique(sccs_batch[batch_iter,i],dim=0)
                        
                        Vertices_pool_FPS=out[batch_iter,sccs_batch[batch_iter,i]]

                        vertices_ceva=out[batch_iter,ceva]

                        vertices_test=torch.cat((Vertices_pool_FPS,vertices_ceva),dim=0)

                        
                        Vertices_final_FPS[batch_iter,i]=Vertices_pool_FPS

                Vertices_final_FPS=Vertices_final_FPS.reshape([batch_size*sccs_batch.shape[1],nr_points_batch, out_2.shape[1]])   

            Vertices_final_FPS=torch.unique(Vertices_final_FPS,dim=2)     

            with torch.no_grad():

                L = conv.pairwise_distance(Vertices_final_FPS) # W - weight matrix
                #L = conv.get_one_matrix_knn(L, 40,batch_size,L.shape[2])
                L = conv.get_laplacian(L)

            out_FPS=self.conv_Reeb(Vertices_final_FPS,L)
            out_FPS=self.relu_Reeb(out_FPS)

            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out_FPS, (0, 2, 1)), L), out_FPS))**2)

            out_final_FPS=torch.zeros([batch_size,sccs_batch.shape[1], out_FPS.shape[2]], dtype=torch.float32,device='cuda')  

            for batch_iter in range(batch_size):   
                    for i in range(sccs_batch.shape[1]):
                        out_final_FPS[batch_iter,i],_ =t.max(out_FPS, 0)
      

            with torch.no_grad():
                L = conv.pairwise_distance(out) # W - weight matrix
                #L = conv.get_one_matrix_knn(L, 40,batch_size,L.shape[2])
                L = conv.get_laplacian(out)


            out = self.conv2(out, L)
            out = self.relu2(out)
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)

            
            
            
    
            out, _ = t.max(out, 1)
            out_FPS, _ = t.max(out_FPS, 1)

            out=torch.cat((out_FPS,out),1)

            # ~~~~ Fully Connected ~~~~
            
            out = self.fc1(out)

            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(self.fc1.weight.data[0]) ** 2)
                self.regularizers.append(t.linalg.norm(self.fc1.bias.data[0]) ** 2)

            out = self.relu4(out)
            #out = self.dropout(out)

            out = self.fc2(out)
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(self.fc2.weight.data[0]) ** 2)
                self.regularizers.append(t.linalg.norm(self.fc2.bias.data[0]) ** 2)
            out = self.relu5(out)
            #out = self.dropout(out)

            out = self.fc3(out)
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(self.fc3.weight.data[0]) ** 2)
                self.regularizers.append(t.linalg.norm(self.fc3.bias.data[0]) ** 2)
        else:
            out, _ = t.max(out, 1)
            out = self.fc(out)

        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader,k,num_points, regularization,nr_points_fps):
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)
        

        logits, regularizers  = model(x.to(device),k=k,batch_size=data.batch.unique().shape[0],num_points=num_points,nr_points_fps=nr_points_fps)
        loss    = criterion(logits, data.y.to(device))
        s = t.sum(t.as_tensor(regularizers))
        loss = loss + regularization * s
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        #if i%100 == 0:
            #print(f"{i}: curr loss: {loss}")
            #$print(f"{data.y} --- {logits.argmax(dim=1)}")
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader,k,num_points,nr_points_fps):
    model.eval()

    total_correct = 0
    for i,data in enumerate(loader):
        
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)
        

        logits, regularizers  = model(x.to(device),k=k,batch_size=data.batch.unique().shape[0],num_points=num_points,nr_points_fps=nr_points_fps)
        
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y.to(device)).sum())

    return total_correct / len(loader.dataset)

def createConfusionMatrix(model,loader,k,num_points):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for  i,data in enumerate(loader):
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)
        

        logits, regularizers  = model(x.to(device),k=k,batch_size=data.batch.unique().shape[0],num_points=num_points)

        pred = logits.argmax(dim=-1)
        
        output = pred.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = data.y.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    # classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred,normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(40)],
                         columns=[i for i in range(40)])
    plt.figure(figsize=(50, 50))    
    return sn.heatmap(df_cm, annot=True).get_figure()



if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_points = 1024
    batch_size = 32
    num_epochs = 260
    learning_rate = 1e-3
    modelnet_num = 40
    k_KNN=5
    nr_points_fps=30

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, modelnet_num]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

        
    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
    print(root)
    dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
    dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)


    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")


    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader  = DataLoader(dataset_test, batch_size=batch_size)
    
    model = cls_model(num_points, F, K, M, modelnet_num, dropout=1, one_layer=False, reg_prior=True)
    # path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models/28_02_22_10:10:19/model50.pt"
    # model.load_state_dict(torch.load(path_saved_model))
    model = model.to(device)

    
    print(model.parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)


    
    ###############################


    # all_sccs_test, all_reeb_laplacian_test= conv.Create_Reeb_from_Dataset_batched(loader=test_loader,sccs_path=sccs_path_test,reeb_laplacian_path=reeb_laplacian_path_test,time_execution=timp_test,knn=knn_REEB,ns=ns,tau=tau,reeb_nodes_num=reeb_nodes_num,reeb_sim_margin=reeb_sim_margin,pointNumber=pointNumber)
    # all_sccs_train, all_reeb_laplacian_train=conv.Create_Reeb_from_Dataset_batched(loader=train_loader,sccs_path=sccs_path_train,reeb_laplacian_path=reeb_laplacian_path_train,time_execution=timp_train,knn=knn_REEB,ns=ns,tau=tau,reeb_nodes_num=reeb_nodes_num,reeb_sim_margin=reeb_sim_margin,pointNumber=pointNumber)


    regularization = 1e-9
    for epoch in range(1, num_epochs+1):
        train_start_time = time.time()
        loss = train(model, optimizer,loader=train_loader,k=k_KNN,num_points=num_points,regularization=regularization,nr_points_fps=nr_points_fps)
        
        train_stop_time = time.time()

        writer.add_scalar("Loss/train", loss, epoch)
        
        test_start_time = time.time()
        test_acc = test(model, loader=test_loader,k=k_KNN,num_points=num_points,nr_points_fps=nr_points_fps)
        test_stop_time = time.time()



        writer.add_scalar("Acc/test", test_acc, epoch)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{test_stop_time - test_start_time }')

        # writer.add_figure("Confusion matrix", createConfusionMatrix(model,test_loader,k=k_KNN,num_points=num_points), epoch)

        if(epoch%5==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        my_lr_scheduler.step()

    
    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


       ###################################################################################################3
    #Testing the model

#     timp_train=0
#     timp_test=0


#     knn_REEB = 20
#     ns = 20
#     tau = 2
#     reeb_nodes_num=20
#     reeb_sim_margin=20
#     pointNumber=200

#     path_logs="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/"
#     sccs_path_test=path_logs+directory+"sccs_test.npy"
#     reeb_laplacian_path_test=path_logs+directory+"reeb_laplacian_test.npy"

#     random_rotate = Compose([
#     RandomRotate(degrees=180, axis=0),
#     RandomRotate(degrees=180, axis=1),
#     RandomRotate(degrees=180, axis=2),
# ])

#     test_transform = Compose([
#     random_rotate,
#     SamplePoints(num_points, include_normals=True),
#     NormalizeScale()
# ])
#     dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
#     dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=test_transform)

#     train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
#     test_loader  = DataLoader(dataset_test, batch_size=batch_size)

#     all_sccs_test, all_reeb_laplacian_test= conv.Create_Reeb_from_Dataset_batched(loader=test_loader,sccs_path=sccs_path_test,reeb_laplacian_path=reeb_laplacian_path_test,time_execution=timp_test,knn=knn_REEB,ns=ns,tau=tau,reeb_nodes_num=reeb_nodes_num,reeb_sim_margin=reeb_sim_margin,pointNumber=pointNumber)
    
    
#     model = cls_model(num_points, F, K, M, modelnet_num, dropout=1, one_layer=False, reg_prior=True)
#     path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models/28_02_22_21:52:37/model260.pt"
#     model.load_state_dict(torch.load(path_saved_model))
#     model = model.to(device)

#     test_start_time = time.time()
#     test_acc = test(model, loader=test_loader,all_sccs=all_sccs_test,all_Reeb_laplacian=all_reeb_laplacian_test,k=k_KNN,num_points=num_points)
#     test_stop_time = time.time()
#     print(f'Test Accuracy: {test_acc:.4f}')


    

    