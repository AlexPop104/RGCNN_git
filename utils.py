import torch.nn as nn
import torch as t
from typing import Optional, Union
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
import open3d as o3d


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
        nr_points=point_cloud.shape[1]
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



def get_weights(dataset, num_points=2048, nr_classes=40):
    from sklearn.utils import class_weight

    '''
    If sk=True the weights are computed using Scikit-learn. Otherwise, a 'custom' implementation will
    be used. It is recommended to use the sk=True.
    '''

    weights = torch.zeros(nr_classes)
    
    y = np.empty(len(dataset)*dataset[0].y.shape[0])
    i = 0
    for data in dataset:
        y[i:num_points+i] = data.y
        i += num_points
    weights = class_weight.compute_class_weight(
        class_weight = "balanced",
        classes = np.unique(y),
        y =  y 
    )


    #weights = dict(zip(np.unique(y), weights))

    return weights

def get_weights_clasif(dataset, num_points=2048, nr_classes=40):
    from sklearn.utils import class_weight

    '''
    If sk=True the weights are computed using Scikit-learn. Otherwise, a 'custom' implementation will
    be used. It is recommended to use the sk=True.
    '''

    weights = torch.zeros(nr_classes)
    
    y = np.empty(len(dataset))
    i = 0
    for data in dataset:
        y[i] = data.y
        i += 1
    weights = class_weight.compute_class_weight(
        class_weight = "balanced",
        classes = np.unique(y),
        y =  y 
    )


    #weights = dict(zip(np.unique(y), weights))

    return weights


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


def compute_loss(logits, y, x, L, criterion, s=1e-9):
    if not logits.device == y.device:
        y = y.to(logits.device)

    loss = criterion(logits, y)
    l=0
    for i in range(len(x)):
        l += (1/2) * t.linalg.norm(t.matmul(t.matmul(t.permute(x[i], (0, 2, 1)), L[i]), x[i]))**2
    l = l * s
    loss += l
    return loss

class GaussianNoiseTransform(BaseTransform):

    def __init__(self, mu: Optional[float] = 0, sigma: Optional[float] = 0.1, recompute_normals : bool = True):
        torch.manual_seed(0)
        np.random.seed(0)
        self.mu = mu
        self.sigma = sigma
        self.recompute_normals = recompute_normals

    def __call__(self, data: Union[Data, HeteroData]):
        noise = np.random.normal(self.mu, self.sigma, data.pos.shape)
        data.pos += noise
        data.pos = data.pos.float()
        if self.recompute_normals:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(data.pos)
            pcd_o3d.estimate_normals(fast_normal_computation=False)
            pcd_o3d.normalize_normals()
            if hasattr(data, 'normal'):
                data.normal = np.asarray(pcd_o3d.normals)
                data.normal = torch.tensor(data.normal, dtype=torch.float32)
            else:
                data.normal = np.asarray(pcd_o3d.normals)
                data.normal = torch.tensor(data.normal, dtype=torch.float32)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'