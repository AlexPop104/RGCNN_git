import torch.nn as nn
import torch as t
from typing import Optional
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter


def get_laplacian(adj_matrix, normalize=True):
    """ 
    Function to compute the Laplacian of an adjacency matrix

    Args:
        adj_matrix: tensor (batch_size, num_points, num_points)
        normlaize:  boolean
    Returns: 
        L:          tensor (batch_size, num_points, num_points)
    """

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

def pairwise_distance(point_cloud,normalize=False):
    """
    Compute the pairwise distance of a point cloud.

    Args: 
        point_cloud: tensor (batch_size, num_points, num_features)

    Returns: 
        pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = t.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = t.sum(t.mul(point_cloud, point_cloud), dim=2, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

    if(normalize):
        maximum_values=torch.max(adj_matrix,dim=2)
        minimum_values=torch.min(adj_matrix,dim=2)

        interval=torch.subtract(maximum_values[0],minimum_values[0])

        interval=torch.tile(interval,[nr_points])

        interval=torch.reshape(interval,(point_cloud.shape[0],point_cloud.shape[1],point_cloud.shape[1]))

        adj_matrix=torch.div(adj_matrix,interval)

    adj_matrix = t.exp(-adj_matrix)
    return adj_matrix



class DenseChebConvV2(nn.Module):
    '''
    Convolutional Module implementing ChebConv. The input to the forward method needs to be
    a tensor 'x' of size (batch_size, num_points, num_features) and a tensor 'L' of size
    (batch_size, num_points, num_points).
    '''
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConvV2, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(in_channels, out_channels, bias=True, 
                weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):        
        for lin in self.lins:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, mean=0, std=0.2)
            lin.bias      = t.nn.init.normal_(lin.bias, mean=0, std=0.2)


    def forward(self, x, L):
        x0 = x # N x M x Fin
        out = self.lins[0](x0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            out = out + self.lins[1](x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            out += self.lins[i](x2)
            x0, x1 = x1, x2

        if self.bias is not None:
            out += self.bias
        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')


