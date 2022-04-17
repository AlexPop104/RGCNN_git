import sys

sys.path.append('/home/domsa/workspace/git/RGCNN_git')

from re import T
from typing import Optional

from time import time

from utils import get_laplacian
from utils import pairwise_distance

import torch
import torch as t
import torch_geometric as tg
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


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

