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






def Create_Reeb_from_Dataset_batched(loader,sccs_path,reeb_laplacian_path,time_execution):
    knn = 20
    ns = 20
    tau = 2
    reeb_nodes_num=20
    reeb_sim_margin=20
    pointNumber=200
    
    all_sccs=np.eye(3)
    all_reeb_laplacians = np.zeros((3,reeb_nodes_num))

    

    
    for i, data in enumerate(loader):

        print(i+1)
        
        batch_size=int(data.batch.unique().shape[0])
        nr_points=int (data.pos.shape[0]/batch_size)


        point_cloud=np.asarray(data.pos)

        point_cloud=np.reshape(point_cloud,(batch_size,nr_points,data.pos.shape[1]))
        

        

        for k in range(batch_size):
            Reeb_Graph_start_time = time.time()
            vertices, laplacian_Reeb, sccs ,edges= conv.extract_reeb_graph(point_cloud[k], knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
            Reeb_Graph_end_time = time.time()
            print(Reeb_Graph_end_time-Reeb_Graph_start_time)
            time_execution +=Reeb_Graph_end_time-Reeb_Graph_start_time
            np_sccs_batch=np.asarray(sccs)
            np_reeb_laplacian=np.asarray(laplacian_Reeb)
            nr_columns_batch= np_sccs_batch.shape[1]
            nr_columns_all=all_sccs.shape[1]

            nr_lines_batch=np_sccs_batch.shape[0]
            nr_lines_all=all_sccs.shape[0]

        
        
            if (nr_columns_batch>nr_columns_all):
                ceva=all_sccs[:,nr_columns_all-1]
                ceva=ceva.reshape((nr_lines_all,1))
                ceva=np.tile(ceva,(nr_columns_batch-nr_columns_all))
                all_sccs=np.concatenate((all_sccs,ceva),1)


            else:
                ceva=np_sccs_batch[:,nr_columns_batch-1]
                ceva=ceva.reshape((nr_lines_batch,1))
                ceva=np.tile(ceva,(nr_columns_all-nr_columns_batch))
                np_sccs_batch=np.concatenate((np_sccs_batch,ceva),1)

            all_sccs=np.concatenate((all_sccs,np_sccs_batch),0)
            all_reeb_laplacians=np.concatenate((all_reeb_laplacians,np_reeb_laplacian),0)
        

        
        print(all_sccs.shape)
        print(all_reeb_laplacians.shape)

        if((i+1)%5==0):
            np.save(sccs_path, all_sccs)
            np.save(reeb_laplacian_path, all_reeb_laplacians)

    np.save(sccs_path, all_sccs)
    np.save(reeb_laplacian_path, all_reeb_laplacians)

    return all_sccs,all_reeb_laplacians

def Create_Reeb_from_Dataset(loader,sccs_path,reeb_laplacian_path,time_execution):
    
    knn = 20
    ns = 20
    tau = 2
    reeb_nodes_num=20
    reeb_sim_margin=20
    pointNumber=200

    
    all_sccs=np.eye(3)
    all_reeb_laplacians = np.zeros((3,reeb_nodes_num))
    
    for i, data in enumerate(loader):

        print(i+1)
        


        point_cloud=np.asarray(data.pos)
        Reeb_Graph_start_time = time.time()
        vertices, laplacian_Reeb, sccs ,edges= conv.extract_reeb_graph(point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
        Reeb_Graph_end_time = time.time()

        print(Reeb_Graph_end_time-Reeb_Graph_start_time)
        time_execution +=Reeb_Graph_end_time-Reeb_Graph_start_time


        np_sccs_batch=np.asarray(sccs)
        np_reeb_laplacian=np.asarray(laplacian_Reeb)

        

        nr_columns_batch= np_sccs_batch.shape[1]
        nr_columns_all=all_sccs.shape[1]

        nr_lines_batch=np_sccs_batch.shape[0]
        nr_lines_all=all_sccs.shape[0]

       
    
        if (nr_columns_batch>nr_columns_all):
            ceva=all_sccs[:,nr_columns_all-1]
            ceva=ceva.reshape((nr_lines_all,1))
            ceva=np.tile(ceva,(nr_columns_batch-nr_columns_all))
            all_sccs=np.concatenate((all_sccs,ceva),1)


        else:
            ceva=np_sccs_batch[:,nr_columns_batch-1]
            ceva=ceva.reshape((nr_lines_batch,1))
            ceva=np.tile(ceva,(nr_columns_all-nr_columns_batch))
            np_sccs_batch=np.concatenate((np_sccs_batch,ceva),1)

        all_sccs=np.concatenate((all_sccs,np_sccs_batch),0)
        all_reeb_laplacians=np.concatenate((all_reeb_laplacians,np_reeb_laplacian),0)

        
        print(all_sccs.shape)
        print(all_reeb_laplacians.shape)

        if((i+1)%5==0):
            np.save(sccs_path, all_sccs)
            np.save(reeb_laplacian_path, all_reeb_laplacians)

    np.save(sccs_path, all_sccs)
    np.save(reeb_laplacian_path, all_reeb_laplacians)

    return all_sccs,all_reeb_laplacians


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
    batch_size = 8
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


    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)
    
    
    

    

    path_logs="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/"

    

    now = datetime.now()
    time_stamp = now.strftime("%d_%m_%y_%H:%M:%S")

    sccs_path_train=path_logs+time_stamp+"sccs_train.npy"
    reeb_laplacian_path_train=path_logs+time_stamp+"reeb_laplacian_train.npy"

    sccs_path_test=path_logs+time_stamp+"sccs_test.npy"
    reeb_laplacian_path_test=path_logs+time_stamp+"reeb_laplacian_test.npy"

    timp_train=0
    timp_test=0

    all_sccs_test, all_reeb_laplacians_test= Create_Reeb_from_Dataset_batched(loader=test_loader,sccs_path=sccs_path_test,reeb_laplacian_path=reeb_laplacian_path_test,time_execution=timp_test)
    all_sccs_train, all_reeb_laplacians_train= Create_Reeb_from_Dataset_batched(loader=train_loader,sccs_path=sccs_path_train,reeb_laplacian_path=reeb_laplacian_path_train,time_execution=timp_train)
    


    

    # all_sccs=np.eye(3)
    # all_reeb_laplacians = np.zeros((3,20))

    # timp_test=0

    # for i, data in enumerate(test_loader):

    #     print(i+1)
    #     knn = 20
    #     ns = 20
    #     tau = 2
    #     reeb_nodes_num=20
    #     reeb_sim_margin=20
    #     pointNumber=200



    #     point_cloud=np.asarray(data.pos)
    #     Reeb_Graph_start_time = time.time()
    #     vertices, laplacian_Reeb, sccs ,edges= conv.extract_reeb_graph(point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
    #     Reeb_Graph_end_time = time.time()

    #     print(Reeb_Graph_end_time-Reeb_Graph_start_time)
    #     timp_test +=Reeb_Graph_end_time-Reeb_Graph_start_time


    #     np_sccs_batch=np.asarray(sccs)
    #     np_reeb_laplacian=np.asarray(laplacian_Reeb)

        

    #     nr_columns_batch= np_sccs_batch.shape[1]
    #     nr_columns_all=all_sccs.shape[1]

    #     nr_lines_batch=np_sccs_batch.shape[0]
    #     nr_lines_all=all_sccs.shape[0]

       
    
    #     if (nr_columns_batch>nr_columns_all):
    #         ceva=all_sccs[:,nr_columns_all-1]
    #         ceva=ceva.reshape((nr_lines_all,1))
    #         ceva=np.tile(ceva,(nr_columns_batch-nr_columns_all))
    #         all_sccs=np.concatenate((all_sccs,ceva),1)


    #     else:
    #         ceva=np_sccs_batch[:,nr_columns_batch-1]
    #         ceva=ceva.reshape((nr_lines_batch,1))
    #         ceva=np.tile(ceva,(nr_columns_all-nr_columns_batch))
    #         np_sccs_batch=np.concatenate((np_sccs_batch,ceva),1)

    #     all_sccs=np.concatenate((all_sccs,np_sccs_batch),0)
    #     all_reeb_laplacians=np.concatenate((all_reeb_laplacians,np_reeb_laplacian),0)

        
    #     print(all_sccs.shape)
    #     print(all_reeb_laplacians.shape)

    #     if((i+1)%5==0):
    #         np.save(sccs_path_test, all_sccs)
    #         np.save(reeb_laplacian_path_test, all_reeb_laplacians)

    # np.save(sccs_path_test, all_sccs)
    # np.save(reeb_laplacian_path_test, all_reeb_laplacians)

    print("Train Reeb Computation time:")
    print(timp_train)
    # print("Test Reeb Computation time:")
    # print(timp_test)

        # fig = matplotlib.pyplot.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()
        # for e in edges:
        #     ax.plot([vertices[e[0]][0], vertices[e[1]][0]], [vertices[e[0]][1], vertices[e[1]][1]], [vertices[e[0]][2], vertices[e[1]][2]], color='b')
        # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, color='r')
        # matplotlib.pyplot.show()

        
   

    

