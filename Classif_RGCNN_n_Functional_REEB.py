import time
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

from torch import Tensor, normal, tensor
from torch import nn
import torch
from torch.nn import Parameter

from typing import Optional
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (add_self_loops, get_laplacian,
                                   remove_self_loops)
import torch as t
import torch_geometric as tg

from torch_geometric.utils import get_laplacian as get_laplacian_pyg
from torch_geometric.transforms import Compose

import Classif_RGCNN_n_DenseConv_functions_REEB as conv
import os

from torch_geometric.transforms import LinearTransformation
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, batch
from datetime import datetime
from torch_geometric.nn import global_max_pool

from torch.nn import MSELoss

import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import heapq
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances_argmin





class cls_model(nn.Module):
    def __init__(self, vertice ,F, K, M, class_num, batch_size,regularization=0, one_layer=True, dropout=0, reg_prior:bool=True):
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
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu_REEB = nn.ReLU()

        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.conv1 = conv.DenseChebConv(6, 1000, 3)
        self.conv2 = conv.DenseChebConv(1000, 1000, 6)


        self.conv3 = conv.DenseChebConv(512, 1024, 3)

        self.conv_Reeb = conv.DenseChebConv(1000, 1000, 3)

        
        self.fc1 = nn.Linear(2000, 512, bias=True)
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.fc3 = nn.Linear(128, class_num, bias=True)
        
        self.fc_t = nn.Linear(128, class_num)

        self.max_pool = nn.MaxPool1d(self.vertice)

        if one_layer == True:
            self.fc = nn.Linear(128, class_num)

        self.regularizer = 0
        self.regularization = []

    def forward(self, x,k,batch_size,num_points,vertices_Reeb,laplacian_Reeb,sccs):
       
        self.regularizers = []
        with torch.no_grad():

            L = conv.pairwise_distance(x) # W - weight matrix
            L = conv.get_one_matrix_knn(L, k,batch_size,num_points)
            L = conv.get_laplacian(L)

        out = self.conv1(x, L)
        out = self.relu1(out)

        if self.reg_prior:
            self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(out.permute(0, 2, 1), L), x)))
        
        if self.one_layer == False:

            Vertices_final_Reeb=torch.zeros([20, out.shape[2]], dtype=torch.float32,device='cuda')

            for i in range(20):
                Vertices_pool_Reeb=torch.zeros([sccs[i].shape[0],out.shape[2]], dtype=torch.float32,device='cuda')
                for j in range(sccs[i].shape[0]):
                    Vertices_pool_Reeb[j]=out[0,sccs[i][j]]
                
                Vertices_final_Reeb[i], _ =t.max(Vertices_pool_Reeb, 0)
                
            Vertices_final_Reeb=Vertices_final_Reeb.unsqueeze(0)    
            laplacian_Reeb_final= torch.tensor(laplacian_Reeb, dtype=torch.float32,device='cuda')
            laplacian_Reeb_final=laplacian_Reeb_final.unsqueeze(0) 


            with torch.no_grad():
                L = conv.pairwise_distance(out) # W - weight matrix
                L = conv.get_one_matrix_knn(L, k,batch_size,num_points)
                L = conv.get_laplacian(L)
            
            out = self.conv2(out, L)
            out = self.relu2(out)

            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(t.matmul(t.matmul(out.permute(0, 2, 1), L), x)))


            out_Reeb=self.conv_Reeb(Vertices_final_Reeb,laplacian_Reeb_final)
            out_Reeb=self.relu_REEB(out_Reeb)


           
            
            out, _ = t.max(out, 1)
            out_Reeb, _ = t.max(out_Reeb, 1)

            out_final=torch.cat((out_Reeb,out),1)

            # ~~~~ Fully Connected ~~~~
            
            out_final = self.fc1(out_final)
            out_final = self.relu4(out_final)

            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(self.fc1.weight.data[0]) ** 2)
                self.regularizers.append(t.linalg.norm(self.fc1.bias.data[0]) ** 2)
            #out = self.dropout(out)

            out_final = self.fc2(out_final)
            out_final = self.relu5(out_final)
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(self.fc2.weight.data[0]) ** 2)
                self.regularizers.append(t.linalg.norm(self.fc2.bias.data[0]) ** 2)
            #out = self.dropout(out)

            out_final = self.fc3(out_final)
            if self.reg_prior:
                self.regularizers.append(t.linalg.norm(self.fc3.weight.data[0]) ** 2)
                self.regularizers.append(t.linalg.norm(self.fc3.bias.data[0]) ** 2)
            
        else:
            out_final, _ = t.max(out, 1)
            out_final = self.fc(out)

        return out_final, self.regularizers

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader,k,batch_size,num_points,regularization):
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6) 
        knn = 20
        ns = 20
        tau = 2
        reeb_nodes_num=20
        reeb_sim_margin=20
        pointNumber=200

        point_cloud=np.asarray(x[0,:,0:3])
        Reeb_Graph_start_time = time.time()
        vertices_Reeb, laplacian_Reeb, sccs = conv.extract_reeb_graph(point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
        Reeb_Graph_end_time = time.time()

        print(Reeb_Graph_end_time-Reeb_Graph_start_time)

        logits, regularizers  = model(x.to(device),k,batch_size=data.batch.unique().shape[0],num_points=num_points,vertices_Reeb=vertices_Reeb,laplacian_Reeb=laplacian_Reeb,sccs=sccs)
        
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
def test(model, loader,modelnet_num,num_points,batch_size):
    confusion_matrix=np.zeros((modelnet_num,modelnet_num))
    category_counters=np.zeros(modelnet_num)

    

    model.eval()

    total_correct = 0
    for data in loader:
        x = torch.cat([data.pos, data.normal], dim=1)
        x = x.reshape(data.batch.unique().shape[0], num_points, 6)
        logits, _ = model(x.to(device),k_KNN,batch_size=data.batch.unique().shape[0],num_points=num_points)
        pred = logits.argmax(dim=-1)

        iteration_batch_size=int(data.pos.shape[0]/num_points)

        for j in range(iteration_batch_size):
            confusion_matrix[pred[j]][data.y[j]]+=1
            category_counters[pred[j]]+=1


        total_correct += int((pred == data.y.to(device)).sum())

    for j in range(modelnet_num):
            confusion_matrix[j,:] =confusion_matrix[j,:]/ category_counters[j]

    return total_correct / len(loader.dataset) , confusion_matrix

if __name__ == '__main__':
    
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")

    log_folder_path="/home/alex/Alex_documents/RGCNN_git/data/logs/Network_performances/"
   

    conf_matrix_path=log_folder_path+"Type_"+directory+"_conf_matrix.npy"
    loss_log_path=log_folder_path+"Type_"+directory+"_losses.npy"
    accuracy_log_path=log_folder_path+"Type_"+directory+"test_accuracy.npy"

    parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_points = 1024
    batch_size = 1
    num_epochs = 100
    learning_rate = 1e-3
    modelnet_num = 40

    k_KNN=55

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, modelnet_num]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

        
    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])
    root = "data/ModelNet"+str(modelnet_num)
    print(root)
    dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
    dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")


    train_loader = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)
    
    model = cls_model(num_points, F, K, M, modelnet_num, dropout=1, one_layer=False,batch_size=batch_size, reg_prior=True)
    model = model.to(device)
    
    print(model.parameters)

    all_sccs=[]
    all_reeb_laplacians = np.zeros((2,1))

    for i, data in enumerate(train_loader):

        
        
        knn = 20
        ns = 20
        tau = 2
        reeb_nodes_num=20
        reeb_sim_margin=20
        pointNumber=200

        point_cloud=np.asarray(data.pos)
        Reeb_Graph_start_time = time.time()
        vertices, laplacian_Reeb, sccs ,edges= conv.extract_reeb_graph(point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
        Reeb_Graph_end_time = time.time()

        print(Reeb_Graph_end_time-Reeb_Graph_start_time)

        all_sccs=np.append(all_sccs,np.asarray(sccs))

        ceva=np.asarray(laplacian_Reeb)
    
        if (laplacian_Reeb.shape[1]>ceva.shape[1]):
                ceva2=ceva[:][ceva.shape[1]-1]
        else:
            ceva=all_reeb_laplacians[:][all_reeb_laplacians.shape[1]-1]
        print(ceva)
        #all_reeb_laplacians=np.append(all_reeb_laplacians,laplacian_Reeb,axis=0)

        

        # fig = matplotlib.pyplot.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()
        # for e in edges:
        #     ax.plot([vertices[e[0]][0], vertices[e[1]][0]], [vertices[e[0]][1], vertices[e[1]][1]], [vertices[e[0]][2], vertices[e[1]][2]], color='b')
        # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, color='r')
        # matplotlib.pyplot.show()

        



    

#     regularization = 1e-9

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#     all_losses=np.array([])
#     test_accuracy=np.array([])
#     confusion_matrix_collection = np.zeros((1,modelnet_num))


#     for epoch in range(1, num_epochs):
#         train_start_time = time.time()
#         loss = train(model, optimizer, train_loader,k_KNN,batch_size,num_points,regularization=regularization)
#         train_stop_time = time.time()

#         all_losses=np.append(all_losses, loss)

#         # writer.add_scalar("Loss/train", loss, epoch)
        
#         test_start_time = time.time()
#         test_acc, confusion_matrix = test(model, test_loader,modelnet_num,num_points,batch_size)
#         test_stop_time = time.time()
        
#         test_accuracy=np.append(test_accuracy, test_acc)
#         print(test_accuracy.shape)
#         confusion_matrix_collection=np.append(confusion_matrix_collection,confusion_matrix,axis=0)
#         # writer.add_scalar("Acc/test", test_acc, epoch)
#         print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
#         print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
#         \tTest Time: \t{test_stop_time - test_start_time }')

#         if(epoch%5==0):
#             np.save(conf_matrix_path, confusion_matrix_collection)
#             np.save(loss_log_path, all_losses)
#             np.save(accuracy_log_path, test_accuracy)

#             torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

#             print(confusion_matrix)

# np.save(loss_log_path, all_losses)
# np.save(accuracy_log_path, test_accuracy)
# np.save(conf_matrix_path, confusion_matrix_collection)