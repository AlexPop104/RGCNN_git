from typing import Optional

from time import time
# import tensorflow as tf
import torch
import torch as t
import torch_geometric as tg
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian

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

# def get_laplacian_tf(adj_matrix, normalize=True):
#     if normalize:
#         D = tf.reduce_sum(adj_matrix, axis=1)  # (batch_size,num_points)
#         eye = tf.ones_like(D)
#         eye = tf.matrix_diag(eye)
#         D = 1 / tf.sqrt(D)
#         D = tf.matrix_diag(D)
#         L = eye - tf.matmul(tf.matmul(D, adj_matrix), D)
#     else:
#         D = tf.reduce_sum(adj_matrix, axis=1)  # (batch_size,num_points)
#         # eye = tf.ones_like(D)
#         # eye = tf.matrix_diag(eye)
#         # D = 1 / tf.sqrt(D)
#         D = tf.matrix_diag(D)
#         L = D - adj_matrix
#     return L


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

        self.lin = Linear(in_channels * K, out_channels, bias=False)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):        
        self.lin.reset_parameters()
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
    
    test_tf = tf.reduce_max(out.cpu().detach().numpy(), 1)
    with tf.Session() as sess:
        print(test_tf.eval())
        print(out.shape)

    values, indices = t.max(out, 1)
    #print(f"Indices: {indices} ")
    print(f"Vals:    {values} ")

    
    out = fc(values)
    print(f"FC:       {out.shape}")
    