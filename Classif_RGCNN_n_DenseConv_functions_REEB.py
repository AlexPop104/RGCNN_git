
import numpy as np
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


import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import heapq
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances_argmin

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


def filter_out(vertices, edges, sccs):
    dist = np.zeros([vertices.shape[0], vertices.shape[0]])
    for e in edges:
        dist[e[0]][e[1]] = dist[e[1]][e[0]] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])

    idx2remove = np.where(np.sum(dist, axis=1, keepdims=False) == 0)[0]
    arr = np.ones(vertices.shape[0], np.int)
    arr[idx2remove] = 0
    idx_mapping = np.cumsum(arr)
    edges = [[idx_mapping[e[0]] - 1, idx_mapping[e[1]] - 1] for e in edges]
    sccs = [scc for i, scc in enumerate(sccs) if i not in idx2remove]
    return np.delete(vertices, idx2remove, 0), edges, sccs


def similarity(scc1, scc2, SIM_MARGIN):
    dist = distance_matrix(scc1, scc2)
    return (np.sum(np.max(np.min(dist, 0) - SIM_MARGIN, 0)) + np.sum(np.max(np.min(dist, 1) - SIM_MARGIN, 0))) / (scc1.shape[0] + scc2.shape[1])


def adjacency_reeb(vertices, edges, sccs, point_cloud, SIM_MARGIN):
    dist1 = np.zeros([vertices.shape[0], vertices.shape[0]])
    dist2 = np.zeros_like(dist1)
    for e in edges:
        dist1[e[0]][e[1]] = dist1[e[1]][e[0]] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])
        dist2[e[0]][e[1]] = dist2[e[1]][e[0]] = similarity(point_cloud[sccs[e[0]]], point_cloud[sccs[e[1]]], SIM_MARGIN)

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma1 = np.sum(dist1, axis=1, keepdims=True) / (np.count_nonzero(dist1, axis=1)[:, None])
        sigma2 = np.sum(dist2, axis=1, keepdims=True) / (np.count_nonzero(dist2, axis=1)[:, None])
    # sigma1 = 0.5 / NODES_NUM
    # sigma2 = 0.01

    dist = np.exp(- dist1 ** 2 / sigma1 ** 2 - dist2 ** 2 / sigma2 ** 2) * (dist1 > 0)
    dist[np.isnan(dist) | np.isinf(dist)] = 0

    idx2remove = np.where(np.sum(dist1, axis=1) == 0)[0]

    idx_mapping = np.ones(vertices.shape[0])
    idx_mapping[idx2remove] = 0
    idx_mapping = (idx_mapping.cumsum() - 1).astype(np.int)
    vertices = np.delete(vertices, idx2remove, 0)
    for e in edges:
        e[0] = idx_mapping[e[0]]
        e[1] = idx_mapping[e[1]]

    dist = np.delete(dist, idx2remove, 0)
    dist = np.delete(dist, idx2remove, 1)
    sccs = [scc for i, scc in enumerate(sccs) if i not in idx2remove]

    # dist[dist == np.inf] = 100
    return vertices, edges, dist, sccs


def normalize_reeb(vertices, edges, sccs, point_cloud, NODES_NUM):
    if vertices.shape[0] == NODES_NUM:
        return vertices, edges, sccs
    elif vertices.shape[0] > NODES_NUM:  # merge nodes
        while vertices.shape[0] > NODES_NUM:
            tomerge = min(edges, key=lambda e: np.linalg.norm(vertices[e[0]] - vertices[e[1]]))
            newidx = min(tomerge[0], tomerge[1])
            toremove = max(tomerge[0], tomerge[1])
            vertices[newidx] = (vertices[tomerge[0]] + vertices[tomerge[1]]) / 2
            # vertices[newidx] = (vertices[tomerge[0]] * len(sccs[tomerge[0]]) + vertices[tomerge[1]] * len(sccs[tomerge[1]])) / (len(sccs[tomerge[0]]) + len(sccs[tomerge[1]]))
            sccs[newidx] = np.concatenate([sccs[tomerge[0]], sccs[tomerge[1]]])
            if toremove + 1 < vertices.shape[0]:
                vertices[toremove:-1] = vertices[toremove + 1:]
                sccs[toremove:-1] = sccs[toremove + 1:]
            vertices = vertices[:-1]
            sccs = sccs[:-1]
            # hyper edge
            while [tomerge[0], tomerge[1]] in edges:
                edges.remove([tomerge[0], tomerge[1]])
            while [tomerge[1], tomerge[0]] in edges:
                edges.remove([tomerge[1], tomerge[0]])

            for i, e in enumerate(edges):
                e0 = e[0]
                e1 = e[1]
                if e0 == toremove:
                    e0 = newidx
                elif e0 > toremove:
                    e0 -= 1
                if e1 == toremove:
                    e1 = newidx
                elif e1 > toremove:
                    e1 -= 1
                edges[i] = [e0, e1]
        for e in edges:
            assert e[0] < NODES_NUM and e[1] < NODES_NUM
    elif vertices.shape[0] < NODES_NUM:
        while vertices.shape[0] < NODES_NUM:
            tosplit = min(edges, key=lambda e: -np.linalg.norm(vertices[e[0]] - vertices[e[1]]))
            node = (vertices[tosplit[0]] + vertices[tosplit[1]]) / 2
            # node = np.zeros([3])
            edges.remove([tosplit[0], tosplit[1]])
            edges.append([tosplit[0], vertices.shape[0]])
            edges.append([tosplit[1], vertices.shape[0]])
            # dummy scc
            # sccs.append([-1])
            sccs.append(np.concatenate([sccs[tosplit[0]], sccs[tosplit[1]]]))
            # node = (vertices[tosplit[0]] * len(sccs[tosplit[0]]) + vertices[tosplit[1]] * len(sccs[tosplit[1]])) / (len(sccs[tosplit[0]]) + len(sccs[tosplit[1]]))

            # idx1 = np.argpartition(np.linalg.norm(point_cloud[sccs[tosplit[0]]] - node[None, ...], axis=1),
            #                        len(sccs[tosplit[0]]) // 2)
            # idx2 = np.argpartition(np.linalg.norm(point_cloud[sccs[tosplit[1]]] - node[None, ...], axis=1),
            #                        len(sccs[tosplit[1]]) // 2)
            # sccs.append(np.concatenate([sccs[tosplit[0]][idx1[:max(len(sccs[tosplit[0]]) // 2, 1)]],
            #                             sccs[tosplit[1]][idx2[:max(len(sccs[tosplit[1]]) // 2, 1)]]]))
            # sccs[tosplit[0]] = np.delete(sccs[tosplit[0]], idx1[:len(sccs[tosplit[0]]) // 2])
            # sccs[tosplit[1]] = np.delete(sccs[tosplit[1]], idx1[:len(sccs[tosplit[1]]) // 2])
            vertices = np.append(vertices, [node], 0)
        for e in edges:
            assert e[0] < NODES_NUM and e[1] < NODES_NUM
    return vertices, edges, sccs
    # dist = np.zeros([vertices.shape[0], vertices.shape[0]])
    # for e in edges:
    #     dist[e[0]][e[1]] = dist[e[1]][e[0]] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])


def expand(x, k, visited, nbrs, valid_idxs, scc, tau, knn, cnt=0):
    if visited[k]:
        return

    visited[k] = True
    scc.append(k)
    distances, indices = nbrs.kneighbors(x[k][None, :])
    idx_in_range = (0 < distances[0]) & (distances[0] < tau[0])
    if np.count_nonzero(idx_in_range) == knn:
        tau[0] = (np.max(distances) + cnt * tau[0]) / (1 + cnt)
    elif np.count_nonzero(idx_in_range) > 0:
        tau[0] = (2 * np.max(distances[0, idx_in_range]) + cnt * tau[0]) / (1 + cnt)
    cnt += 1
    for i in indices[0, idx_in_range]:
        if valid_idxs[i]:
            expand(x, i, visited, nbrs, valid_idxs, scc, tau, knn, cnt)
    # marked[indices[0, distances[0] < tau]] = True




def extract_reeb_graph(point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber):  

    nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm='kd_tree').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)

    marked = np.zeros([point_cloud.shape[0]], np.bool)
    # calculate f
    # r = point_cloud[:, 2]
    
    r = np.linalg.norm(point_cloud, axis=-1)


    # mean_x=np.mean(point_cloud[:,0])
    # mean_y=np.mean(point_cloud[:,1])
    # mean_z=np.mean(point_cloud[:,2])

    # r=np.sqrt( (point_cloud[:, 0]-mean_x)*(point_cloud[:, 0]-mean_x) + (point_cloud[:, 1]-mean_y)*(point_cloud[:, 1]-mean_y) +(point_cloud[:, 2]-mean_z)*(point_cloud[:, 2]-mean_z) )


    
    


    r_min=np.amin(r)
    r_max=np.amax(r)

   
    sccs = []
    scc2idx = dict()
    vertices = []
    edges = []
    # np.random.seed(0)
    for i in range(ns):
        scc_level = []

        #idx = (-1 + i * 2. / ns < r) & (r <= -1 + (i + 1) * 2. / ns)
        idx = ( r_min+(r_max-r_min)*(i* 1./(ns+1) ) < r) & (r<= r_min+(r_max-r_min)*((i+1)* 1./(ns+1) ))
        #print(i)

        while not np.all(marked[idx]):
            scc = []
            # random choose a point
            valid_idx = np.where(~marked & idx)[0]
            rnd_idx = valid_idx[np.random.randint(valid_idx.shape[0])]

            # 5 tau_p
            tau = np.max(nbrs.kneighbors(point_cloud[rnd_idx][None, :])[0]) * 5
            unprocessed_idx = [rnd_idx]
            cnt = 0
            while unprocessed_idx:
                k = unprocessed_idx.pop(0)
                if marked[k]:
                    continue

                marked[k] = True
                scc.append(k)
                distances, indices = nbrs.kneighbors(point_cloud[k][None, :])
                idx_in_range = (0 < distances[0]) & (distances[0] < tau)
                if np.count_nonzero(idx_in_range) == knn:
                    tau = (np.max(distances) + cnt * tau) / (1 + cnt)
                elif np.count_nonzero(idx_in_range) > 0:
                    tau = (2 * np.max(distances[0, idx_in_range]) + cnt * tau) / (1 + cnt)
                cnt += 1
                for j in indices[0, idx_in_range]:
                    if idx[j]:
                        unprocessed_idx.append(j)
            if not scc:
                continue
            scc = np.asarray(scc)

            # append
            scc_level.append(scc)
            scc2idx[id(scc)] = len(vertices)
            vertices.append(np.mean(point_cloud[scc], axis=0))

            #print(tau)
            # connect edges
            if i > 0:
                for prev_scc in sccs[-1]:
                    if np.min(np.linalg.norm(point_cloud[prev_scc][None, :, :] - point_cloud[scc][:, None, :], axis=-1)) < 2 * tau:
                        edges.append([scc2idx[id(prev_scc)], scc2idx[id(scc)]])
        sccs.append(scc_level)

    # pad scc to the same shape
    sccs = [x for xs in sccs for x in xs]

    if len(vertices) == 1:
        sccs = [sccs[0][:len(sccs[0]) // 2], sccs[0][len(sccs[0]) // 2:]]
        vertices = np.stack([np.mean(point_cloud[sccs[0]], 0), np.mean(point_cloud[sccs[1]], 0)])
        edges.append([0, 1])
    vertices, edges, sccs = filter_out(np.asarray(vertices), edges, sccs)
    vertices, edges, sccs = normalize_reeb(np.asarray(vertices), edges, sccs, point_cloud, reeb_nodes_num)
    vertices, edges, laplacian, sccs = adjacency_reeb(vertices, edges, sccs, point_cloud, reeb_sim_margin)
    # # laplacian = np.delete(laplacian, idx2remove, 0)
    # # laplacian = np.delete(laplacian, idx2remove, 1)
    # # sccs = np.delete(sccs, idx2remove, 0)
    while vertices.shape[0] != reeb_nodes_num:
        # print(vertices.shape[0])
        vertices, edges, sccs = normalize_reeb(np.asarray(vertices), edges, sccs, point_cloud, reeb_nodes_num)
        vertices, edges, laplacian, sccs = adjacency_reeb(vertices, edges, sccs, point_cloud, reeb_sim_margin)
        # laplacian = np.delete(laplacian, idx2remove, 0)
        # laplacian = np.delete(laplacian, idx2remove, 1)
        # sccs = np.delete(sccs, idx2remove, 0)
    # # print(laplacian)
    # pad
    largest_dim = max([len(x) for x in sccs])
    # largest_dim = pointNumber
    sccs = np.asarray([np.pad(x, (0, largest_dim - len(x)), 'edge') for x in sccs])
    # assert np.all(np.isfinite(laplacian)) and np.all(np.isfinite(sccs))
    # print(vertices.shape, laplacian.shape)
    #print(np.shape(vertices))
    return vertices, laplacian, list(sccs) , edges
    #return vertices, list(sccs)


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



    