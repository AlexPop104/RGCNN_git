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

        

    

 