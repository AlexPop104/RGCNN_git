import time

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

from torch import nn
import torch
torch.manual_seed(0)
from torch.nn import Parameter

import random

random.seed(0)


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
np.random.seed(0)


def Test_reeb(loader,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points):
    for i, (pos, y, normal, idx) in enumerate(loader):

        torch.manual_seed(0)

        Test_reeb_iteration(i, pos, y, normal, idx,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points)

        # batch_size=pos[1].shape[0]
        # ground_truth_labels=y[1].squeeze(1)
        # num_vertices_reeb=all_Reeb_laplacian.shape[1]
        # edge_dim=edges.shape[1]

        
        # ceva=torch.tile(idx.unsqueeze(1).to(device)*num_vertices_reeb,(1,num_vertices_reeb))
        # ceva=torch.reshape(ceva,[idx.shape[0]*num_vertices_reeb])
        # ceva=torch.reshape(ceva,(idx.shape[0],num_vertices_reeb))

        # ceva2=torch.arange(0,num_vertices_reeb,device='cuda')
        # ceva3=torch.tile(ceva2,[1,idx.shape[0]])
        # ceva3=torch.reshape(ceva3,(idx.shape[0],num_vertices_reeb))

        # ceva4_batch=torch.add(ceva,ceva3)
        # ceva4=torch.reshape(ceva4_batch,[idx.shape[0]*num_vertices_reeb])

        # ceva4=ceva4.to('cpu')

        # sccs_batch=all_sccs[ceva4]
        # reeb_laplace_batch=all_Reeb_laplacian[ceva4]
        # vertices_batch=vertices[ceva4]
        # edges_batch=edges[ceva4]
        
        

        # for iter_pcd in range(batch_size):

        #     points_pcd=pos[1][iter_pcd]

            

        #     sccs_pcd=sccs_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb]
        #     reeb_laplace_pcd=reeb_laplace_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb,0:num_vertices_reeb]
        #     vertices_batch_pcd=vertices_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb]
        #     matrix_edges_batch_pcd=edges_batch[iter_pcd*all_Reeb_laplacian.shape[1]:(iter_pcd+1)*edge_dim]

        #     t_matrix_edges_batch=torch.tensor(matrix_edges_batch_pcd)
        #     t_matrix_edges_2=t_matrix_edges_batch.unsqueeze(0)
        #     New_edge_indices, New_edge_values=torch_geometric.utils.dense_to_sparse(t_matrix_edges_2)
        #     New_edge_indices_cpu=New_edge_indices.to('cpu')

           
        #    ##############################
        #    ###############Computing Reeb graph on the spot 
           
        #     point_pcd_np=np.asarray(points_pcd)
        #     knn = 20
        #     ns = 20
        #     tau = 2
        #     reeb_nodes_num=20
        #     reeb_sim_margin=20
        #     pointNumber=200
        #     vertices_aux, laplacian_Reeb_aux, sccs_aux ,edges_aux= conv_reeb.extract_reeb_graph(point_pcd_np, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)

        #     # fig = matplotlib.pyplot.figure()
        #     # ax = fig.add_subplot(111, projection='3d')
        #     # ax.set_axis_off()
        #     # for e in edges_aux:
        #     #     ax.plot([vertices_aux[e[0]][0], vertices_aux[e[1]][0]], [vertices_aux[e[0]][1], vertices_aux[e[1]][1]], [vertices_aux[e[0]][2], vertices_aux[e[1]][2]], color='b')
        #     # ax.scatter(point_pcd_np[:, 0], point_pcd_np[:, 1], point_pcd_np[:, 2], s=1, color='r')   
        #     # matplotlib.pyplot.show()
            
        #     ########################################
        #     #Visualizing both Reeb graphs 

            # fig = matplotlib.pyplot.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_axis_off()
            # for e in edges_aux:
            #     ax.plot([vertices_aux[e[0]][0], vertices_aux[e[1]][0]], [vertices_aux[e[0]][1], vertices_aux[e[1]][1]], [vertices_aux[e[0]][2], vertices_aux[e[1]][2]], color='g')
            # for test_iter in range(New_edge_indices_cpu.shape[1]):
            #     ax.plot([vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][0], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][0]], [vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][1], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][1]], [vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][2], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][2]], color='b')
            # ax.scatter(points_pcd[:, 0], points_pcd[:, 1], points_pcd[:, 2], s=1, color='r')   
            # matplotlib.pyplot.show()

def Test_reeb_iteration(i, pos, y, normal, idx,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points):
    
    torch.manual_seed(0)
    batch_size=pos[1].shape[0]
    ground_truth_labels=y[1].squeeze(1)
    num_vertices_reeb=all_Reeb_laplacian.shape[1]
    edge_dim=edges.shape[1]

    
    ceva=torch.tile(idx.unsqueeze(1).to('cuda')*num_vertices_reeb,(1,num_vertices_reeb))
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
    
    

    for iter_pcd in range(batch_size):

        points_pcd=pos[1][iter_pcd]

        

        sccs_pcd=sccs_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb]
        reeb_laplace_pcd=reeb_laplace_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb,0:num_vertices_reeb]
        vertices_batch_pcd=vertices_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb]
        matrix_edges_batch_pcd=edges_batch[iter_pcd*all_Reeb_laplacian.shape[1]:(iter_pcd+1)*edge_dim]

        t_matrix_edges_batch=torch.tensor(matrix_edges_batch_pcd)
        t_matrix_edges_2=t_matrix_edges_batch.unsqueeze(0)
        New_edge_indices, New_edge_values=torch_geometric.utils.dense_to_sparse(t_matrix_edges_2)
        New_edge_indices_cpu=New_edge_indices.to('cpu')

        
        ##############################
        ###############Computing Reeb graph on the spot 
        
        point_pcd_np=np.asarray(points_pcd)
        knn = 55
        ns = 40
        tau = 3
        reeb_nodes_num=20
        reeb_sim_margin=40
        pointNumber=400
        vertices_aux, laplacian_Reeb_aux, sccs_aux ,edges_aux= conv_reeb.extract_reeb_graph(point_pcd_np, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)

        # fig = matplotlib.pyplot.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()
        # for e in edges_aux:
        #     ax.plot([vertices_aux[e[0]][0], vertices_aux[e[1]][0]], [vertices_aux[e[0]][1], vertices_aux[e[1]][1]], [vertices_aux[e[0]][2], vertices_aux[e[1]][2]], color='b')
        # ax.scatter(point_pcd_np[:, 0], point_pcd_np[:, 1], point_pcd_np[:, 2], s=1, color='r')   
        # matplotlib.pyplot.show()
        
        ########################################
        #Visualizing both Reeb graphs 

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        for i in range(sccs_pcd.shape[0]):
            pcd_region=points_pcd[sccs_aux[i]]
            colour_red=random.uniform(0, 1)
            colour_green=random.uniform(0, 1)
            colour_blue=random.uniform(0, 1)
            ax.scatter(pcd_region[:, 0], pcd_region[:, 1], pcd_region[:, 2], s=5, color=(colour_red,colour_green,colour_blue))
            ax.scatter(vertices_aux[i,0],vertices_aux[i,1],vertices_aux[i,2],s=20,color=(colour_red,colour_green,colour_blue))

            

        for e in edges_aux:
            ax.plot([vertices_aux[e[0]][0], vertices_aux[e[1]][0]], [vertices_aux[e[0]][1], vertices_aux[e[1]][1]], [vertices_aux[e[0]][2], vertices_aux[e[1]][2]], color='b')
        # for test_iter in range(New_edge_indices_cpu.shape[1]):
        #     ax.plot([vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][0], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][0]], [vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][1], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][1]], [vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][2], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][2]], color='b')
        #ax.scatter(points_pcd[:, 0], points_pcd[:, 1], points_pcd[:, 2], s=1, color='r')
        #ax.scatter(vertices_batch_pcd[:, 0], vertices_batch_pcd[:, 1], vertices_batch_pcd[:, 2], s=1, color='r')
        matplotlib.pyplot.show()


def Test_reeb_iteration_labels(i, pos, y, normal, idx,all_sccs,all_Reeb_laplacian,edges,vertices,k,num_points,position):
    
    label_to_names = {0: 'airplane',
                                1: 'bathtub',
                                2: 'bed',
                                3: 'bench',
                                4: 'bookshelf',
                                5: 'bottle',
                                6: 'bowl',
                                7: 'car',
                                8: 'chair',
                                9: 'cone',
                                10: 'cup',
                                11: 'curtain',
                                12: 'desk',
                                13: 'door',
                                14: 'dresser',
                                15: 'flower_pot',
                                16: 'glass_box',
                                17: 'guitar',
                                18: 'keyboard',
                                19: 'lamp',
                                20: 'laptop',
                                21: 'mantel',
                                22: 'monitor',
                                23: 'night_stand',
                                24: 'person',
                                25: 'piano',
                                26: 'plant',
                                27: 'radio',
                                28: 'range_hood',
                                29: 'sink',
                                30: 'sofa',
                                31: 'stairs',
                                32: 'stool',
                                33: 'table',
                                34: 'tent',
                                35: 'toilet',
                                36: 'tv_stand',
                                37: 'vase',
                                38: 'wardrobe',
                                39: 'xbox'}

    torch.manual_seed(0)
    batch_size=pos[1].shape[0]
    ground_truth_labels=y[1].squeeze(1)
    num_vertices_reeb=all_Reeb_laplacian.shape[1]
    edge_dim=edges.shape[1]

    
    ceva=torch.tile(idx.unsqueeze(1).to('cuda')*num_vertices_reeb,(1,num_vertices_reeb))
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
    

    for iter_pcd in range(batch_size):

        if(position==y[1][iter_pcd].item()):

            points_pcd=pos[1][iter_pcd]

            

            sccs_pcd=sccs_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb]
            reeb_laplace_pcd=reeb_laplace_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb,0:num_vertices_reeb]
            vertices_batch_pcd=vertices_batch[iter_pcd*num_vertices_reeb:(iter_pcd+1)*num_vertices_reeb]
            matrix_edges_batch_pcd=edges_batch[iter_pcd*all_Reeb_laplacian.shape[1]:(iter_pcd+1)*edge_dim]

            t_matrix_edges_batch=torch.tensor(matrix_edges_batch_pcd)
            t_matrix_edges_2=t_matrix_edges_batch.unsqueeze(0)
            New_edge_indices, New_edge_values=torch_geometric.utils.dense_to_sparse(t_matrix_edges_2)
            New_edge_indices_cpu=New_edge_indices.to('cpu')

            
            ##############################
            ###############Computing Reeb graph on the spot 
            
            point_pcd_np=np.asarray(points_pcd)
            knn = 55
            ns = 40
            tau = 3
            reeb_nodes_num=20
            reeb_sim_margin=40
            pointNumber=400
            vertices_aux, laplacian_Reeb_aux, sccs_aux ,edges_aux= conv_reeb.extract_reeb_graph(point_pcd_np, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)

            # fig = matplotlib.pyplot.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_axis_off()
            # for e in edges_aux:
            #     ax.plot([vertices_aux[e[0]][0], vertices_aux[e[1]][0]], [vertices_aux[e[0]][1], vertices_aux[e[1]][1]], [vertices_aux[e[0]][2], vertices_aux[e[1]][2]], color='b')
            # ax.scatter(point_pcd_np[:, 0], point_pcd_np[:, 1], point_pcd_np[:, 2], s=1, color='r')   
            # matplotlib.pyplot.show()
            
            ########################################
            #Visualizing both Reeb graphs 

            fig = matplotlib.pyplot.figure()
            fig.suptitle(label_to_names[y[1][iter_pcd].item()], fontsize=16)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_axis_off()

            for i in range(sccs_pcd.shape[0]):
                pcd_region=points_pcd[sccs_aux[i]]
                colour_red=random.uniform(0, 1)
                colour_green=random.uniform(0, 1)
                colour_blue=random.uniform(0, 1)
                ax.scatter(pcd_region[:, 0], pcd_region[:, 1], pcd_region[:, 2], s=5, color=(colour_red,colour_green,colour_blue))
                ax.scatter(vertices_aux[i,0],vertices_aux[i,1],vertices_aux[i,2],s=20,color=(colour_red,colour_green,colour_blue))

                

            for e in edges_aux:
                ax.plot([vertices_aux[e[0]][0], vertices_aux[e[1]][0]], [vertices_aux[e[0]][1], vertices_aux[e[1]][1]], [vertices_aux[e[0]][2], vertices_aux[e[1]][2]], color='b')
            # for test_iter in range(New_edge_indices_cpu.shape[1]):
            #     ax.plot([vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][0], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][0]], [vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][1], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][1]], [vertices_batch_pcd[New_edge_indices_cpu[0][test_iter]][2], vertices_batch_pcd[New_edge_indices_cpu[1][test_iter]][2]], color='b')
            #ax.scatter(points_pcd[:, 0], points_pcd[:, 1], points_pcd[:, 2], s=1, color='r')
            #ax.scatter(vertices_batch_pcd[:, 0], vertices_batch_pcd[:, 1], vertices_batch_pcd[:, 2], s=1, color='r')
            matplotlib.pyplot.show()

        

        

# if __name__ == '__main__':
#     now = datetime.now()
#     directory = now.strftime("%d_%m_%y_%H:%M:%S")
#     parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
#     path = os.path.join(parent_directory, directory)
#     os.mkdir(path)

#     num_points = 1024
#     batch_size = 20
#     num_epochs = 260
#     learning_rate = 1e-3
#     modelnet_num = 40
#     k_KNN=30

#     F = [128, 512, 1024]  # Outputs size of convolutional filter.
#     K = [6, 5, 3]         # Polynomial orders.
#     M = [512, 128, modelnet_num]

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     print(f"Training on {device}")

        
#     transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

#     # root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)
#     # print(root)


#     # dataset_train =index_dataset.Modelnet_with_indices(root=root,modelnet_num=modelnet_num,train_bool=True,transforms=transforms)
#     # dataset_test = index_dataset.Modelnet_with_indices(root=root,modelnet_num=modelnet_num,train_bool=False,transforms=transforms)


#     # # Verification...
#     # print(f"Train dataset shape: {dataset_train}")
#     # print(f"Test dataset shape:  {dataset_test}")


#     # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
#     # test_loader  = DataLoader(dataset_test, batch_size=batch_size)


#     ###################################################################

#     root="/media/rambo/ssd2/Alex_data/RGCNN/GeometricShapes"

#     transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

#     dataset_train = index_dataset.Geometric_with_indices(root=root,train_bool=True,transforms=transforms)
#     dataset_test = index_dataset.Geometric_with_indices(root=root,train_bool=False,transforms=transforms)


#     train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True, pin_memory=True)
#     test_loader= DataLoader(dataset_test,batch_size=batch_size)


#     path_logs="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/"

#     # sccs_path_train=path_logs+directory+"train_sccs.npy"
#     # reeb_laplacian_path_train=path_logs+directory+"train_reeb_laplacian.npy"
#     # edge_matrix_path_train=path_logs+directory+"train_edge_matrix.npy"
#     # vertices_path_train=path_logs+directory+"train_vertices.npy"

#     # sccs_path_test=path_logs+directory+"test_sccs.npy"
#     # reeb_laplacian_path_test=path_logs+directory+"test_reeb_laplacian.npy"
#     # edge_matrix_path_test=path_logs+directory+"test_edge_matrix.npy"
#     # vertices_path_test=path_logs+directory+"test_vertices.npy"

#     sccs_path_train=path_logs+"train_sccs.npy"
#     reeb_laplacian_path_train=path_logs+"train_reeb_laplacian.npy"
#     edge_matrix_path_train=path_logs+"train_edge_matrix.npy"
#     vertices_path_train=path_logs+"train_vertices.npy"

#     sccs_path_test=path_logs+"test_sccs.npy"
#     reeb_laplacian_path_test=path_logs+"test_reeb_laplacian.npy"
#     edge_matrix_path_test=path_logs+"test_edge_matrix.npy"
#     vertices_path_test=path_logs+"test_vertices.npy"


#     timp_train=0
#     timp_test=0


#     knn_REEB = 20
#     ns = 20
#     tau = 2
#     reeb_nodes_num=20
#     reeb_sim_margin=20
#     pointNumber=200

#     # all_sccs_test, all_reeb_laplacian_test,edges_test,vertices_test= conv_reeb.Create_Reeb_custom_loader_batched(loader=test_loader,sccs_path=sccs_path_test,reeb_laplacian_path=reeb_laplacian_path_test,edge_matrix_path=edge_matrix_path_test,vertices_path=vertices_path_test,time_execution=timp_test,knn=knn_REEB,ns=ns,tau=tau,reeb_nodes_num=reeb_nodes_num,reeb_sim_margin=reeb_sim_margin,pointNumber=pointNumber)
#     # all_sccs_train, all_reeb_laplacian_train,edges_train,vertices_train=conv_reeb.Create_Reeb_custom_loader_batched(loader=train_loader,sccs_path=sccs_path_train,reeb_laplacian_path=reeb_laplacian_path_train,edge_matrix_path=edge_matrix_path_train,vertices_path=vertices_path_train,time_execution=timp_train,knn=knn_REEB,ns=ns,tau=tau,reeb_nodes_num=reeb_nodes_num,reeb_sim_margin=reeb_sim_margin,pointNumber=pointNumber)



#     #############################################################
#     #Load Reeb_graphs from file

#     path_Reeb_laplacian_train="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/train_reeb_laplacian.npy"
#     path_Reeb_laplacian_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/test_reeb_laplacian.npy"

#     path_sccs_train="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/train_sccs.npy"
#     path_sccs_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/test_sccs.npy"

#     path_vertices_train="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/train_vertices.npy"
#     path_vertices_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/test_vertices.npy"

#     path_edges_train="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/train_edge_matrix.npy"
#     path_edges_test="/home/alex/Alex_documents/RGCNN_git/data/logs/Reeb_data/test_edge_matrix.npy"

#     all_sccs_train=np.load(path_sccs_train)
#     all_sccs_test=np.load(path_sccs_test)

#     all_reeb_laplacian_train=np.load(path_Reeb_laplacian_train)
#     all_reeb_laplacian_test=np.load(path_Reeb_laplacian_test)

#     vertices_train=np.load(path_vertices_train)
#     vertices_test=np.load(path_vertices_test)

#     edges_train=np.load(path_edges_train)
#     edges_test=np.load(path_edges_test)

    

#     ################################
    
#     Test_reeb(loader=train_loader,all_sccs=all_sccs_train,all_Reeb_laplacian=all_reeb_laplacian_train,edges=edges_train,vertices=vertices_train,k=k_KNN,num_points=num_points) 
#     Test_reeb(loader=test_loader,all_sccs=all_sccs_test,all_Reeb_laplacian=all_reeb_laplacian_test,edges=edges_test,vertices=vertices_test,k=k_KNN,num_points=num_points)
        

    
    

 