import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch import nn
import torch
torch.manual_seed(0)
from torch.nn import Parameter


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
import ChebConv_rgcnn_functions_reeb as conv_reeb
import ChebConv_loader_indices as index_dataset
import ChebConv_test_reeb as test_reeb

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

import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

import torch_geometric.utils 

#from ChebConv_loader_indices import Modelnet_with_indices



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

        # self.get_laplacian = GetLaplacian(normalize=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu_Reeb = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.conv1 = conv.DenseChebConv(6, 128, 3)
        self.conv2 = conv.DenseChebConv(128, 512, 3)
        self.conv_Reeb = conv.DenseChebConv(128, 512,3)
        
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.fc_t = nn.Linear(128, class_num)

        self.max_pool = nn.MaxPool1d(self.vertice)

        self.regularizer = 0
        self.regularization = []


    def forward(self, x,k,batch_size,num_points,laplacian_Reeb,sccs,vertices_reeb,edges_reeb,labels):
        
        position=0
        self.regularizers = []
        with torch.no_grad():
            L = conv.pairwise_distance(x) # W - weight matrix
            L = conv.get_one_matrix_knn(L, k,batch_size,num_points)

            # for it_pcd in range(1):
            #     viz_points_2=x[it_pcd,:,:]
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,1)
            L = conv.get_laplacian(L)


        out = self.conv1(x, L)
        out = self.relu1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)

        

        with torch.no_grad():
            Vertices_final_Reeb=torch.zeros([batch_size,laplacian_Reeb.shape[2], out.shape[2]], dtype=torch.float32,device='cuda')

            for batch_iter in range(batch_size):   
                for i in range(laplacian_Reeb.shape[1]):
                    Vertices_pool_Reeb=torch.zeros([sccs[batch_iter,i].shape[0],out.shape[2]], dtype=torch.float32,device='cuda')
                    Vertices_pool_Reeb=out[batch_iter,sccs[batch_iter,i]]
                    Vertices_final_Reeb[batch_iter,i],_ =t.max(Vertices_pool_Reeb, 0)

            laplacian_Reeb_final= torch.tensor(laplacian_Reeb, dtype=torch.float32,device='cuda')

           
            
            
            
        with torch.no_grad():         
            L = conv.pairwise_distance(out) # W - weight matrix
            L = conv.get_one_matrix_knn(L, k,batch_size,num_points)
            # for it_pcd in range(1):
            #     viz_points_2=x[it_pcd,:,:]
            #     distances=L[it_pcd,:,:]
            #     threshold=0.3
            #     conv.view_graph(viz_points_2,distances,threshold,4)
            L = conv.get_laplacian(L)

        #plt.show()
        
        out = self.conv2(out, L)
        out = self.relu3(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out, (0, 2, 1)), L), out))**2)


        conv.tsne_features(x=x,out=out,batch_size=batch_size,labels=labels,position=position)



        out_Reeb=self.conv_Reeb(Vertices_final_Reeb,laplacian_Reeb_final)
        out_Reeb=self.relu_Reeb(out_Reeb)

        vertices_Reeb_torch=torch.tensor(vertices_reeb).to(device)
        x_reeb=torch.reshape(vertices_Reeb_torch,(batch_size,laplacian_Reeb.shape[2],vertices_reeb.shape[1]))

        

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(t.permute(out_Reeb, (0, 2, 1)), laplacian_Reeb_final), out_Reeb))**2)



        #conv.tsne_features(x=x_reeb,out=out,batch_size=batch_size,labels=labels,position=position)

        out, _ = t.max(out, 1)
        out_Reeb, _ = t.max(out_Reeb, 1)

        out=torch.cat((out_Reeb,out),1)

        # ~~~~ Fully Connected ~~~~
        
        out = self.fc1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(self.fc1.weight.data[0]) ** 2)
            self.regularizers.append(t.linalg.norm(self.fc1.bias.data[0]) ** 2)

        out = self.relu4(out)
        ######out = self.dropout(out)

        out = self.fc2(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(self.fc2.weight.data[0]) ** 2)
            self.regularizers.append(t.linalg.norm(self.fc2.bias.data[0]) ** 2)
        out = self.relu5(out)
        #######out = self.dropout(out)

        out = self.fc3(out)
        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(self.fc3.weight.data[0]) ** 2)
            self.regularizers.append(t.linalg.norm(self.fc3.bias.data[0]) ** 2)
        

        return out, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

@torch.no_grad()
def test(model, loader,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points):
    model.eval()
    total_loss = 0
    total_correct = 0
    for i,(pos, y, normal, idx) in enumerate(loader):

        batch_size=pos[1].shape[0]
        ground_truth_labels=y[1].squeeze(1)
        num_vertices_reeb=all_Reeb_laplacian.shape[1]
        edge_dim=edges.shape[1]

        #test_reeb.Test_reeb_iteration(i, pos, y, normal, idx,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points)

        position=6
        #test_reeb.Test_reeb_iteration_labels(i, pos, y, normal, idx,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points,position=position)

        ceva=torch.tile(idx.unsqueeze(1).to(device)*num_vertices_reeb,(1,num_vertices_reeb))
        ceva=torch.reshape(ceva,[idx.shape[0]*num_vertices_reeb])
        ceva=torch.reshape(ceva,(idx.shape[0],num_vertices_reeb))

        ceva2=torch.arange(0,num_vertices_reeb,device='cuda')
        ceva3=torch.tile(ceva2,[1,idx.shape[0]])
        ceva3=torch.reshape(ceva3,(idx.shape[0],num_vertices_reeb))

        ceva4_batch=torch.add(ceva,ceva3)
        ceva4=torch.reshape(ceva4_batch,[idx.shape[0]*num_vertices_reeb])

        ceva4=ceva4.to('cpu')

        sccs_batch=all_sccs[ceva4]
        reeb_laplace_batch=all_Reeb_laplacian[ceva4]
        vertices_batch=vertices[ceva4]
        edges_batch=edges[ceva4]

        sccs_batch=sccs_batch.astype(int)
        sccs_batch=np.reshape(sccs_batch,(batch_size,all_Reeb_laplacian.shape[1],all_sccs.shape[1]))
        reeb_laplace_batch=np.reshape(reeb_laplace_batch,(batch_size,all_Reeb_laplacian.shape[1],all_Reeb_laplacian.shape[1]))

        x = torch.cat([pos[1], normal[1]], dim=2)
        #x=pos[1]

        logits, regularizers = model(x.to(device),k=k,batch_size=batch_size,num_points=num_points,laplacian_Reeb=reeb_laplace_batch,sccs=sccs_batch,vertices_reeb=vertices_batch,edges_reeb=edges_batch,labels=ground_truth_labels)
        
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == ground_truth_labels.to(device)).sum())

        loss    = criterion(logits, ground_truth_labels.to(device))
        # s = t.sum(t.as_tensor(regularizers))
        # loss = loss + regularization * s
        total_loss += loss.item() * batch_size

    return total_loss / len(loader.dataset) , total_correct / len(loader.dataset) 

def createConfusionMatrix(model, loader,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for i,(pos, y, normal, idx) in enumerate(loader):
        batch_size=pos[1].shape[0]
        ground_truth_labels=y[1].squeeze(1)
        num_vertices_reeb=all_Reeb_laplacian.shape[1]
        edge_dim=edges.shape[1]

        # test_reeb.Test_reeb_iteration(i, pos, y, normal, idx,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points)

        ceva=torch.tile(idx.unsqueeze(1).to(device)*num_vertices_reeb,(1,num_vertices_reeb))
        ceva=torch.reshape(ceva,[idx.shape[0]*num_vertices_reeb])
        ceva=torch.reshape(ceva,(idx.shape[0],num_vertices_reeb))

        ceva2=torch.arange(0,num_vertices_reeb,device='cuda')
        ceva3=torch.tile(ceva2,[1,idx.shape[0]])
        ceva3=torch.reshape(ceva3,(idx.shape[0],num_vertices_reeb))

        ceva4_batch=torch.add(ceva,ceva3)
        ceva4=torch.reshape(ceva4_batch,[idx.shape[0]*num_vertices_reeb])

        ceva4=ceva4.to('cpu')

        sccs_batch=all_sccs[ceva4]
        reeb_laplace_batch=all_Reeb_laplacian[ceva4]
        vertices_batch=vertices[ceva4]
        edges_batch=edges[ceva4]

        sccs_batch=sccs_batch.astype(int)
        sccs_batch=np.reshape(sccs_batch,(batch_size,all_Reeb_laplacian.shape[1],all_sccs.shape[1]))
        reeb_laplace_batch=np.reshape(reeb_laplace_batch,(batch_size,all_Reeb_laplacian.shape[1],all_Reeb_laplacian.shape[1]))

        x = torch.cat([pos[1], normal[1]], dim=2)

        logits, _  = model(x.to(device),k=k,batch_size=batch_size,num_points=num_points,laplacian_Reeb=reeb_laplace_batch,sccs=sccs_batch,labels=y[1])
        pred = logits.argmax(dim=-1)
        
        output = pred.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = y[1].cpu().numpy()
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

    num_points = 512
    batch_size = 16
    num_epochs = 260
    learning_rate = 1e-3
    modelnet_num = 40
    k_KNN=5

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, modelnet_num]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    ############################################
    #Modelnet test
    
    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

    root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
    print(root)

    dataset_test = index_dataset.Modelnet_with_indices(root=root,modelnet_num=modelnet_num,train_bool=False,transforms=transforms)

    
    model = cls_model(num_points, F, K, M, modelnet_num, dropout=1,  reg_prior=True)
    path_saved_model="/home/alex/Alex_documents/RGCNN_git/data/logs/Modele_selectate/Reeb/Reeb_512_2layers/model250.pt"
    model.load_state_dict(torch.load(path_saved_model))
    model = model.to(device)

    print(model.parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    test_loader= DataLoader(dataset_test,batch_size=batch_size)

   
    #############################################################
    #Load Reeb_graphs from file



    
    path_Reeb_laplacian_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/Rb_data/Modelnet40_unshuffled/512/_512_test_reeb_laplacian.npy"

    path_sccs_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/Rb_data/Modelnet40_unshuffled/512/_512_test_sccs.npy"

   
    path_vertices_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/Rb_data/Modelnet40_unshuffled/512/_512_test_vertices.npy"

    
    path_edges_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/Rb_data/Modelnet40_unshuffled/512/_512_test_edge_matrix.npy"

    all_sccs_test=np.load(path_sccs_test)
    all_reeb_laplacian_test=np.load(path_Reeb_laplacian_test)
    vertices_test=np.load(path_vertices_test)
    edges_test=np.load(path_edges_test)

    #conv.test_pcd_with_index(model=model,loader=train_loader,num_points=num_points,device=device)
#     ################################
    

    test_start_time = time.time()
    test_loss,test_acc = test(model, loader=test_loader,all_sccs=all_sccs_test,all_Reeb_laplacian=all_reeb_laplacian_test,edges=edges_test,vertices=vertices_test,k=k_KNN,num_points=num_points)
    test_stop_time = time.time()


    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'\n \Test Time: \t{test_stop_time - test_start_time }')

    # writer.add_figure("Confusion matrix", createConfusionMatrix(model,test_loader,all_sccs=all_sccs_test,all_Reeb_laplacian=all_reeb_laplacian_test,k=k_KNN,num_points=num_points), epoch)



    



    

    