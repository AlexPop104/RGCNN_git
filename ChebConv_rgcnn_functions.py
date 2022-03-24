from typing import Optional

from time import time
import torch
import torch as t
import torch_geometric as tg
from torch import Tensor, nn
from torch_geometric.nn import fps,radius_graph,nearest
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian
import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D






def get_laplacian(adj_matrix, normalize=True):
    if normalize:
        D = t.sum(adj_matrix, dim=1)
        eye = t.ones_like(D)
        eye = t.diag_embed(eye)
        D = 1 / t.sqrt(D)
        D = t.diag_embed(D)
        L = eye - t.matmul(t.matmul(D, adj_matrix), D)


        #################################3
        #Trying to not have small values / failed because laplacian has almost 1 on diagonal and negativ in rest
        # nr_points=adj_matrix.shape[1]

        # maximum_values=torch.max(adj_matrix,dim=2)
        # minimum_values=torch.min(adj_matrix,dim=2)

        # interval=torch.subtract(maximum_values[0],minimum_values[0])
        # interval=torch.tile(interval,[nr_points])
        # interval=torch.reshape(interval,(adj_matrix.shape[0],adj_matrix.shape[1],adj_matrix.shape[1]))
        # L=torch.div(L,interval)

        #######################

    else:
        D = t.sum(adj_matrix, dim=1)
        D = t.diag(D)
        L = D - adj_matrix

    return L


def pairwise_distance(point_cloud):
    """Compute the pairwise distance of a point cloud.

    Args: 
        point_cloud: tensof (batch_size, num_points, num_points)

    Returns: 
        pairwise distance: (batch_size, num_points, num_points)
    """

    nr_points=point_cloud.shape[1]

    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

    maximum_values=torch.max(adj_matrix,dim=2)
    minimum_values=torch.min(adj_matrix,dim=2)

    interval=torch.subtract(maximum_values[0],minimum_values[0])

    interval=torch.tile(interval,[nr_points])

    interval=torch.reshape(interval,(point_cloud.shape[0],point_cloud.shape[1],point_cloud.shape[1]))

    adj_matrix=torch.div(adj_matrix,interval)
    adj_matrix = torch.exp(-adj_matrix)

    return adj_matrix

def get_one_matrix_knn(matrix, k,batch_size,nr_points):

    values,indices = torch.topk(matrix, k,sorted=False)
    
    batch_correction=torch.arange(0,batch_size,device='cuda')*nr_points
    batch_correction=torch.reshape(batch_correction,[batch_size,1])
    batch_correction=torch.tile(batch_correction,(1,nr_points*k))
    batch_correction=torch.reshape(batch_correction,(batch_size,nr_points,k))
       
    my_range=torch.unsqueeze(torch.arange(0,indices.shape[1],device='cuda'),1)
    my_range_repeated=torch.tile(my_range,[1,k])
    my_range_repeated=torch.unsqueeze(my_range_repeated,0)
    my_range_repeated_2=torch.tile(my_range_repeated,[batch_size,1,1])

    indices=indices+batch_correction
    my_range_repeated_2=my_range_repeated_2+batch_correction
    
    edge_indices=torch.cat((torch.unsqueeze(my_range_repeated_2,2),torch.unsqueeze(indices,2)),axis=2)
    edge_indices=torch.transpose(edge_indices,2,3)
    edge_indices=torch.reshape(edge_indices,(batch_size,nr_points*k,2))
    edge_indices=torch.reshape(edge_indices,(batch_size*nr_points*k,2))
    edge_indices=torch.transpose(edge_indices,0,1)
    edge_indices=edge_indices.long()

    edge_weights=torch.reshape(values,[-1])

    batch_indexes=torch.arange(0,batch_size,device='cuda')
    batch_indexes=torch.reshape(batch_indexes,[batch_size,1])
    batch_indexes=torch.tile(batch_indexes,(1,nr_points))
    batch_indexes=torch.reshape(batch_indexes,[batch_size*nr_points])
    batch_indexes=batch_indexes.long()

    knn_weight_matrix=tg.utils.to_dense_adj(edge_indices,batch_indexes,edge_weights)

    return knn_weight_matrix

def get_fps_matrix(point_cloud,data,nr_points_fps):
    nr_points_batch=int(data.batch.shape[0]/data.batch.unique().shape[0])
        
    index = fps(point_cloud, data.batch, ratio=float(nr_points_fps*data.batch.unique().shape[0]/data.pos.shape[0]) , random_start=True)

    fps_point_cloud=point_cloud[index]
    fps_batch=data.batch[index]

    cluster = nearest(point_cloud, fps_point_cloud, data.batch, fps_batch)


    batch_correction=torch.arange(0,data.batch.unique().shape[0],device='cuda')
    batch_correction=torch.reshape(batch_correction,[data.batch.unique().shape[0],1])
    batch_correction=torch.tile(batch_correction,(1,nr_points_batch))
    batch_correction=torch.reshape(batch_correction,[data.batch.unique().shape[0]*nr_points_batch])

    Batch_indexes=batch_correction

    batch_correction_num_points=batch_correction*nr_points_batch
    batch_correction=batch_correction*nr_points_fps
    
    cluster_new=torch.subtract(cluster,batch_correction)
   
    edge_index_1=torch.arange(0,data.batch.unique().shape[0]*nr_points_batch,device='cuda')
    edge_index_2=cluster_new

    edge_index_final=torch.cat((torch.unsqueeze(edge_index_1,1),torch.unsqueeze(edge_index_2,1)),axis=1)
    edge_index_final=torch.transpose(edge_index_final,0,1)
    
    edge_weight=torch.ones([data.batch.unique().shape[0]*nr_points_batch],device='cuda')

    Matrix_near=tg.utils.to_dense_adj(edge_index_final,Batch_indexes,edge_weight)
    Matrix_near=Matrix_near[:,:,0:nr_points_fps]
    Matrix_near= Matrix_near.permute(0, 2, 1)
    Matrix_near_2,Matrix_near_indices= torch.sort(Matrix_near,dim=2,descending=True)

    Matrix_near_3=torch.multiply(Matrix_near_2,Matrix_near_indices)
    Matrix_near_3,_=torch.sort(Matrix_near_3,dim=2,descending=True)
    Matrix_near_4=torch.reshape(Matrix_near_3,(data.batch.unique().shape[0]*nr_points_fps,nr_points_batch))

    for i in range(nr_points_fps*data.batch.unique().shape[0]):
        Matrix_near_4[i]=torch.where(Matrix_near_4[i] > 0, Matrix_near_4[i], Matrix_near_4[i,0])
    return Matrix_near_4


def get_fps_matrix_2(point_cloud,batch_size,nr_points,nr_points_fps):
    nr_points_batch=nr_points

    Batch_indexes=torch.arange(0,batch_size,device='cuda')
    Batch_indexes=torch.reshape(Batch_indexes,[batch_size,1])
    Batch_indexes=torch.tile(Batch_indexes,(1,nr_points_batch))
    Batch_indexes=torch.reshape(Batch_indexes,[batch_size*nr_points_batch])

    index = fps(point_cloud, Batch_indexes, ratio=float(nr_points_fps/nr_points) , random_start=True)

    fps_point_cloud=point_cloud[index]
    fps_batch=Batch_indexes[index]

    cluster = nearest(point_cloud, fps_point_cloud, Batch_indexes, fps_batch)

    batch_correction=Batch_indexes

    batch_correction_num_points=batch_correction*nr_points_batch
    batch_correction=batch_correction*nr_points_fps
    
    cluster_new=torch.subtract(cluster,batch_correction)
   
    edge_index_1=torch.arange(0,batch_size*nr_points_batch,device='cuda')
    edge_index_2=cluster_new

    edge_index_final=torch.cat((torch.unsqueeze(edge_index_1,1),torch.unsqueeze(edge_index_2,1)),axis=1)
    edge_index_final=torch.transpose(edge_index_final,0,1)
    
    edge_weight=torch.ones([batch_size*nr_points_batch],device='cuda')

    Matrix_near=tg.utils.to_dense_adj(edge_index_final,Batch_indexes,edge_weight)
    Matrix_near=Matrix_near[:,:,0:nr_points_fps]
    Matrix_near= Matrix_near.permute(0, 2, 1)
    Matrix_near_2,Matrix_near_indices= torch.sort(Matrix_near,dim=2,descending=True)

    Matrix_near_3=torch.multiply(Matrix_near_2,Matrix_near_indices)
    Matrix_near_3,_=torch.sort(Matrix_near_3,dim=2,descending=True)
    Matrix_near_4=torch.reshape(Matrix_near_3,(batch_size*nr_points_fps,nr_points_batch))

    for i in range(nr_points_fps*batch_size):
        Matrix_near_4[i]=torch.where(Matrix_near_4[i] > 0, Matrix_near_4[i], Matrix_near_4[i,0])
    return Matrix_near_4

def get_fps_matrix_topk(point_cloud,batch_size,nr_points,nr_points_fps):
    nr_points_batch=nr_points

    Batch_indexes=torch.arange(0,batch_size,device='cuda')
    Batch_indexes=torch.reshape(Batch_indexes,[batch_size,1])
    Batch_indexes=torch.tile(Batch_indexes,(1,nr_points_batch))
    Batch_indexes=torch.reshape(Batch_indexes,[batch_size*nr_points_batch])

    index = fps(point_cloud, Batch_indexes, ratio=float(nr_points_fps/nr_points) , random_start=True)

    fps_point_cloud=point_cloud[index]
    fps_batch=Batch_indexes[index]

    fps_point_cloud_2=torch.tile(fps_point_cloud,(1,nr_points))

    point_cloud_2=torch.reshape(point_cloud,(batch_size,nr_points,point_cloud.shape[1]))

    point_cloud_3=torch.reshape(point_cloud_2,(batch_size,nr_points*point_cloud.shape[1]))

    point_cloud_4=torch.tile(point_cloud_3,(1,nr_points_batch))

    ##Work in progress
    
    return fps_batch

def get_centroid(point_cloud,num_points):

    nr_coordinates=point_cloud.shape[2]
    batch_size=point_cloud.shape[0]
    
    centroid=torch.sum(point_cloud,1)
    centroid=centroid/num_points

    centroid=torch.tile(centroid,(1,num_points))
    centroid=torch.reshape(centroid,(batch_size,num_points,nr_coordinates))

    point_cloud_2=torch.subtract(point_cloud,centroid)

    Distances=torch.linalg.norm(point_cloud_2,dim=2)

    Distances=torch.unsqueeze(Distances,2)

    return Distances

def get_RotationInvariantFeatures(point_cloud,num_points):

    nr_coordinates=point_cloud.shape[2]
    batch_size=point_cloud.shape[0]
    
    centroid=torch.sum(point_cloud,1)
    centroid=centroid/num_points

    centroid_pos=torch.tile(centroid[:,0:3],(1,num_points))
    centroid_pos=torch.reshape(centroid_pos,(batch_size,num_points,3))

    centroid_norm=torch.tile(centroid[:,3:6],(1,num_points))
    centroid_norm=torch.reshape(centroid_pos,(batch_size,num_points,3))


    Pos_dif=torch.subtract(point_cloud[:,:,0:3],centroid_pos)
    

    Distances=torch.linalg.norm(Pos_dif,dim=2)
    Distances=torch.unsqueeze(Distances,2)

    cosine_sim=torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    output_1 = cosine_sim(centroid_norm, point_cloud[:,:,3:6])
    output_1=torch.unsqueeze(output_1,2)
    output_2 = cosine_sim(centroid_norm, Pos_dif)
    output_2=torch.unsqueeze(output_2,2)
    output_3=cosine_sim(point_cloud[:,:,3:6],Pos_dif)
    output_3=torch.unsqueeze(output_3,2)
    
    PPF_features=torch.cat((Distances,output_1,output_2,output_3),dim=2)

    

    
    


    return PPF_features

def test_pcd_pred(model, loader,num_points,device):
    with torch.no_grad():
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
        for data in loader:
                # x=data.pos
                # x=x.reshape(data.batch.unique().shape[0], num_points, 3)
                # x2=get_centroid(point_cloud=x,num_points=num_points)

                x = torch.cat([data.pos, data.normal], dim=1)   
                x = x.reshape(data.batch.unique().shape[0], num_points, 6)

                L = pairwise_distance(x)
                #L = get_laplacian(L)

                # x=torch.cat([x,x2],dim=2)
        
                #logits, regularizers  = model(x=x.to(device),x2=x2.to(device))
                logits, regularizers  = model(x=x.to(device))


                viz_points=data.pos
                viz_points=viz_points.reshape(data.batch.unique().shape[0], num_points, 3)

                pred = logits.argmax(dim=-1)

                ground_truth=data.y.to(device)
                
                for it_pcd in range(data.batch.unique().shape[0]):
                    # if(ground_truth[it_pcd]!=pred[it_pcd]):

                        print("Actual label:")
                        print(label_to_names[ground_truth[it_pcd].item()])
                        print("Predicted label:")
                        print(label_to_names[pred[it_pcd].item()])

                        viz_points_2=viz_points[it_pcd,:,:]
                        distances=L[it_pcd,:,:]

                        threshold=0.7

                        view_graph(viz_points_2,distances,threshold,it_pcd)

                        # fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        # ax.set_axis_off()
                        # for i in range(viz_points.shape[1]):
                        #     ax.scatter(viz_points[it_pcd,i,0].item(),viz_points[it_pcd,i, 1].item(), viz_points[it_pcd,i,2].item(),color='r')

                        # #distances=distances-distances.min()
                        
                        # distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype))

                        # for i in range(viz_points.shape[1]):
                        #     for j in range(viz_points.shape[1]):
                        #         if (distances[i,j].item()>0.4):
                        #             ax.plot([viz_points[it_pcd,i,0].item(),viz_points[it_pcd,j,0].item()],[viz_points[it_pcd,i, 1].item(),viz_points[it_pcd,j, 1].item()], [viz_points[it_pcd,i,2].item(),viz_points[it_pcd,j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))
                            
                        # plt.show()

                plt.show()

def view_graph(viz_points,distances,threshold,nr):

    distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype,device='cuda'))

    fig = plt.figure(nr)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    for i in range(viz_points.shape[0]):
        ax.scatter(viz_points[i,0].item(),viz_points[i, 1].item(), viz_points[i,2].item(),color='r')

    for i in range(viz_points.shape[0]):
        for j in range(viz_points.shape[0]):
            if (distances[i,j].item()>threshold):
                ax.plot([viz_points[i,0].item(),viz_points[j,0].item()],[viz_points[i, 1].item(),viz_points[j, 1].item()], [viz_points[i,2].item(),viz_points[j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))

def view_graph_Reeb(viz_points,Reeb_points,distances,threshold,nr):

    distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype,device='cuda'))

    fig = plt.figure(nr)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    

    for t in range(Reeb_points.shape[0]):
        ax.scatter(Reeb_points[t,0].item(),Reeb_points[t, 1].item(), Reeb_points[t,2].item(),color='g')

    for k in range(viz_points.shape[0]):
        ax.scatter(viz_points[k,0].item(),viz_points[k, 1].item(), viz_points[k,2].item(),color='r')    

    for i in range(Reeb_points.shape[0]):
        for j in range(Reeb_points.shape[0]):
            if (distances[i,j].item()>threshold):
                ax.plot([Reeb_points[i,0].item(),Reeb_points[j,0].item()],[Reeb_points[i, 1].item(),Reeb_points[j, 1].item()], [Reeb_points[i,2].item(),Reeb_points[j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))
                            
    


def test_pcd_with_index(model,loader,num_points,device):
    with torch.no_grad():
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
        for i,(pos, y, normal, idx) in enumerate(loader):
               
                viz_points=pos[1]
                viz_points=viz_points.reshape(pos[1].shape[0], num_points, 3)

                
                for it_pcd in range(pos[1].shape[0]):
                     
                        print("PCD nr:")
                        print(idx[it_pcd])
                        print("PCD label")
                        print(label_to_names[y[1][it_pcd].item()])
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_axis_off()
                        ax.scatter(viz_points[it_pcd,:,0], viz_points[it_pcd,:, 1], viz_points[it_pcd,:,2], s=1, color='r')   
                        plt.show()




class DenseChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        
        self.reset_parameters()


    def reset_parameters(self):        
        self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        x = x.unsqueeze if x.dim() == 2 else x
        L = L.unsqueeze if L.dim() == 2 else L

        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin
        x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            x = concat(x, x1)

        for _ in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2

        x = x.permute([1,2,3,0])
        x = x.reshape([N * M, Fin * self.K])

        x = self.lin(x)
        x = x.reshape([N, M, self.out_channels])
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')


class DenseChebConv_theta_and_sum(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv_theta_and_sum, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels , out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(1, 1, bias=False, 
                weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):        
        for lin in self.lins:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, 0, 0.1)
        self.lin.reset_parameters()
        
        # self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        #x = x.unsqueeze if x.dim() == 2 else x
        #L = L.unsqueeze if L.dim() == 2 else L

        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin



        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,[x.shape[0]*x.shape[1]*x.shape[2]])
        x0=x0.unsqueeze(0)

        x0=torch.permute(x0,(1,0))


        # x = x0.unsqueeze(0)
        out = self.lins[0](x0)

        x0=x0.unsqueeze(1)
        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,(x.shape[0],x.shape[1],x.shape[2]))

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)

            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,[x.shape[0]*x.shape[1]*x.shape[2]])

            x1=x1.unsqueeze(0)
            x1=torch.permute(x1,(1,0))

            out = out + self.lins[1](x1)

            x1=x1.unsqueeze(1)
            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,(x.shape[0],x.shape[1],x.shape[2]))
            # x = concat(x, x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0

            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,[x.shape[0]*x.shape[1]*x.shape[2]])

            x2=x2.unsqueeze(0)
            x2=torch.permute(x2,(1,0))

            out += self.lins[i](x2)

            x2=x2.unsqueeze(1)
            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,(x.shape[0],x.shape[1],x.shape[2]))
            # x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        out = out.reshape([N * M, Fin])

        out = self.lin(out)
        out = out.reshape([N, M, self.out_channels])

        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')

class DenseChebConv_small_linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv_small_linear, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels, out_channels, bias=bias)
        
        self.reset_parameters()


    def reset_parameters(self):        
        self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        x = x.unsqueeze if x.dim() == 2 else x
        L = L.unsqueeze if L.dim() == 2 else L

        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin
        #x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            x = torch.add(x, x1)
            #x = concat(x, x1)

        for _ in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            x = torch.add(x, x2)
            #x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        x = x.reshape([N * M, Fin ])

        x = self.lin(x)
        x = x.reshape([N, M, self.out_channels])
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')



class DenseChebConvV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConvV2, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False, 
                weight_initializer='glorot') for _ in range(K)
        ])
        
        if bias:
            self.bias = Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):        
        for lin in self.lins:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, 0, 0.1)
        
        
        # self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        #x = x.unsqueeze if x.dim() == 2 else x
        #L = L.unsqueeze if L.dim() == 2 else L

        #N, M, Fin = x.shape
        #N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin
        # x = x0.unsqueeze(0)
        out = self.lins[0](x0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            out = out + self.lins[1](x1)
            # x = concat(x, x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            out += self.lins[i](x2)
            # x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        # x = x.reshape([N * M, Fin * self.K])

        # x = self.lin(x)
        # x = x.reshape([N, M, self.out_channels])

        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')

class DenseChebConv_theta_nosum(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv_theta_nosum, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False, 
                weight_initializer='glorot') for _ in range(K)
        ])
        self.lins_theta = t.nn.ModuleList([
            Linear(1, 1, bias=False, 
                weight_initializer='glorot') for _ in range(K)
        ])


        if bias:
            self.bias = Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):        
        for lin in self.lins:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, 0, 0.1)
        for lin in self.lins_theta:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, 0, 0.1)
        
        # self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        #x = x.unsqueeze if x.dim() == 2 else x
        #L = L.unsqueeze if L.dim() == 2 else L

        #N, M, Fin = x.shape
        #N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin

        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,[x.shape[0]*x.shape[1]*x.shape[2]])
        x0=x0.unsqueeze(0)

        x0=torch.permute(x0,(1,0))


        # x = x0.unsqueeze(0)
        out_theta = self.lins_theta[0](x0)

        x0=x0.unsqueeze(1)
        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,(x.shape[0],x.shape[1],x.shape[2]))

        out_theta=out_theta.unsqueeze(1)
        out_theta=torch.reshape(out_theta,(x.shape[0],x.shape[1],x.shape[2]))

        # x = x0.unsqueeze(0)
        out = self.lins[0](out_theta)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)

            x1 = t.matmul(L, x0)

            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,[x.shape[0]*x.shape[1]*x.shape[2]])

            x1=x1.unsqueeze(0)
            x1=torch.permute(x1,(1,0))

            out_theta = self.lins_theta[1](x1)

            x1=x1.unsqueeze(1)
            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,(x.shape[0],x.shape[1],x.shape[2]))

            out_theta=out_theta.unsqueeze(1)
            out_theta=torch.reshape(out_theta,(x.shape[0],x.shape[1],x.shape[2]))

            out = out + self.lins[1](out_theta)
            # x = concat(x, x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0

            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,[x.shape[0]*x.shape[1]*x.shape[2]])

            x2=x2.unsqueeze(0)
            x2=torch.permute(x2,(1,0))

            out_theta = self.lins_theta[i](x2)

            x2=x2.unsqueeze(1)
            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,(x.shape[0],x.shape[1],x.shape[2]))

            out_theta=out_theta.unsqueeze(1)
            out_theta=torch.reshape(out_theta,(x.shape[0],x.shape[1],x.shape[2]))

            out =out + self.lins[i](out_theta)
            # x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        # x = x.reshape([N * M, Fin * self.K])

        # x = self.lin(x)
        # x = x.reshape([N, M, self.out_channels])

        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')


if __name__ == "__main__":
    device = "cuda"
    A = t.rand(1024, 6)
    A = A.float()
    A = t.cat([A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A])
    A = A.reshape([16, 1024, 6])
    print(A.shape)

    if False:
        A_tf = tf.convert_to_tensor(A, dtype=float)
        L_A_tf = get_laplacian_tf(A_tf, normalize=True)
        with tf.Session() as sess:
            start_time = time()
            L_A_tf.eval()
            print(L_A_tf.shape)
            print(f"Tf Time: {time() -start_time}")
        print("~~" * 30)

    A = A.to(device)
    
    if False:
        start_time = time()
        edge_index, edge_weight = tg.utils.dense_to_sparse(A)
        L_index, L_weight = tg.utils.get_laplacian(edge_index, edge_weight, normalization='sym')
        L_pyg = tg.utils.to_dense_adj(L_index, edge_attr=L_weight) 
        print(f"PyG Time: \t\t{time() -start_time}")
        print(L_pyg.shape)
        print(f"Memory allocated: \t{torch.cuda.memory_allocated(0)}")
        del(L_pyg)
        del(edge_index, edge_weight, L_index, L_weight)
        torch.cuda.empty_cache()

        print("~~" * 30)
        # print(torch.cuda.memory_allocated(0))

    if False:
        start_time = time()
        L_A = get_laplacian(A, normalize=True)
        print(f"Remake Time: \t\t{time() -start_time}")
        print(L_A.shape)
        #print(L_A)
        print(f"Memory allocated: \t{torch.cuda.memory_allocated(0)}")



    #print(L_pyg)
    conv_dense = DenseChebConv(6, 128, 6).to('cuda').float()
    max_pool = nn.MaxPool2d(128, 128)
    fc = nn.Linear(128, 10).to(device)

    L = pairwise_distance(A)
    L = get_laplacian(L)
    out = conv_dense(A, L)
    print("OK")
    
   

  
    