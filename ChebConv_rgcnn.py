from re import T
from typing import Optional

from time import time

import torch
import torch as t
import torch_geometric as tg
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian


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

def pairwise_distance(point_cloud):
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
    adj_matrix = t.exp(-adj_matrix)
    return adj_matrix


class DenseChebConv(nn.Module):

    '''
    Convolutional Module implementing ChebConv. The input to the forward method needs to be
    a tensor 'x' of size (batch_size, num_points, num_features) and a tensor 'L' of size
    (batch_size, num_points, num_points).

    !!! Warning !!!
    This aggregates the features from each 'T' and optimizes one big Weight. Not optimal. 
    You should use DenseChebConvV2! 
    '''

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
        self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L):
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
    '''
    test_tf = tf.reduce_max(out.cpu().detach().numpy(), 1)
    with tf.Session() as sess:
        print(test_tf.eval())
        print(out.shape)
    '''
    values, indices = t.max(out, 1)
    #print(f"Indices: {indices} ")
    print(f"Vals:    {values} ")

    
    out = fc(values)
    print(f"FC:       {out.shape}")
    