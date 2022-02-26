
import numpy
from numpy import dtype
from numpy import ones

from torch import Tensor, double, normal, tensor
from torch import nn
import torch
from torch.nn import Parameter

from typing import Optional

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (add_self_loops, get_laplacian,
                                   remove_self_loops)
import torch as t
import torch_geometric as tg

from torch_geometric.utils import get_laplacian as get_laplacian_pyg
import os
# import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

def get_laplacian(adj_matrix, normalize=True):
    if normalize:
        D = t.sum(adj_matrix, dim=1)
        eye = t.ones_like(D)
        eye = t.diag_embed(eye)
        D = 1 / t.sqrt(D)
        D = t.diag_embed(D)
        L = eye - t.matmul(t.matmul(D, adj_matrix), D)
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

    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    adj_matrix = torch.exp(-adj_matrix)
    return adj_matrix


def get_one_matrix_knn(matrix, k,batch_size,nr_points):


    values,indices = torch.topk(matrix, k,sorted=False)
    
    batch_correction=torch.range(0,batch_size-1,device='cuda')*nr_points
    batch_correction=torch.reshape(batch_correction,[batch_size,1])
    batch_correction=torch.tile(batch_correction,(1,nr_points*k))
    batch_correction=torch.reshape(batch_correction,(batch_size,1024,k))
       
    my_range=torch.unsqueeze(torch.range(0,indices.shape[1]-1,device='cuda'),1)
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

    batch_indexes=torch.range(0,batch_size-1,device='cuda')
    batch_indexes=torch.reshape(batch_indexes,[batch_size,1])
    batch_indexes=torch.tile(batch_indexes,(1,nr_points))
    batch_indexes=torch.reshape(batch_indexes,[batch_size*1024])
    batch_indexes=batch_indexes.long()

    knn_weight_matrix=tg.utils.to_dense_adj(edge_indices,batch_indexes,edge_weights)

    return knn_weight_matrix


class DenseChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=True, **kwargs):
        assert K > 0
        super(DenseChebConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        
        
        self.lins = nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])

        self.lin = Linear(in_channels * K, out_channels, bias=False)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
        lin.reset_parameters()
        zeros(self.bias)

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

        for lin in self.lins[2:]:
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
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')

# class ChebConv_rgcnn(MessagePassing):
#     r"""ChebConv spectral convolutional operator 
    
#     """

#     def __init__(self, in_channels: int, out_channels: int, K: int, 
#                  normalization: Optional[str]='sym', bias: bool=True, **kwargs):
#         assert K > 0
#         assert normalization in [None, 'sym', 'rw'], 'Invalid normalization type'

#         self.K = K
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalization = normalization
#         self.lins = nn.ModuleList([
#             Linear(in_channels, out_channels, bias=False,
#             weight_initializer='glorot') for _ in range(K)
#         ])

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()
#         zeros(self.bias)
    
#     def __norm__(self, adj_matrix, num_nodes: Optional[int],
#                  normalization: Optional[bool],
#                  lambda_max, dtype: Optional[int] = None):
#         if normalization:
#             L = get_laplacian(adj_matrix=adj_matrix, normalize=True)
#         else: 
#             L = get_laplacian(adj_matrix=adj_matrix, normalize=False)
        
#         return L
    
#     def forward(self, x: Tensor, L: Tensor):
#         N, M, Fin = x.shape
#         N, M, Fin = int(N), int(M), int(Fin)

#         x0 = x # N x M x Fin
#         x = x.expand(0)

#         def concat(x, x_):
#             x_ = x_.expand(0) # 1 x M x Fin*N
#             return t.cat([x, x_], dim=0) # K x M x Fin*N
        
#         if self.K > 1:
#             x1 = t.matmul(L, x0)
#             x = concat(x, x1)
#         for k in range(2, K):
#             x2 = 2 * t.matmul(L, x1) - x0
#             x = concat(x, x2)
#             x0, x1 = x1, x2

#         # K x N x M x Fin
#         x = x.permute([1, 2, 3, 0]) # N x M x Fin X K
#         x = x.reshape([N * M, Fin * self.K]) 
    
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j


    