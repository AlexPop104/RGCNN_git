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
import torch_geometric.utils 
import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import heapq
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances_argmin
from zmq import device


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


def Create_Reeb_from_Dataset_batched(loader,sccs_path,reeb_laplacian_path,edge_matrix_path,time_execution,knn,ns,tau,reeb_nodes_num,reeb_sim_margin,pointNumber):
    # knn = 20
    # ns = 20
    # tau = 2
    # reeb_nodes_num=20
    # reeb_sim_margin=20
    # pointNumber=200
    
    all_sccs=np.eye(3)
    all_reeb_laplacians = np.zeros((3,reeb_nodes_num))
    all_reeb_edge_matrix = np.zeros((3,reeb_nodes_num))

    for i, data in enumerate(loader):

        print(i+1)
        
        batch_size=int(data.batch.unique().shape[0])
        nr_points=int (data.pos.shape[0]/batch_size)

        batch_edge_indices = torch.tensor([[1., 2.], [4., 5.]])
        batch_batch_indices_reeb = torch.tensor([1.,2. ])


        point_cloud=np.asarray(data.pos)
        point_cloud=np.reshape(point_cloud,(batch_size,nr_points,data.pos.shape[1]))
        

        for k in range(batch_size):
            
            vertices, laplacian_Reeb, sccs ,edges= extract_reeb_graph(point_cloud[k], knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
            
            np_sccs_batch=np.asarray(sccs)
            np_reeb_laplacian=np.asarray(laplacian_Reeb)
            np_reeb_edges=np.asarray(edges)

            nr_columns_batch= np_sccs_batch.shape[1]
            nr_columns_all=all_sccs.shape[1]

            nr_lines_batch=np_sccs_batch.shape[0]
            nr_lines_all=all_sccs.shape[0]

            edge_indices_iteration = torch.tensor(edges,device='cuda')
            edge_indices_iteration_2=torch.transpose(edge_indices_iteration,0,1)

            batch_values_iteration=(torch.zeros(edge_indices_iteration_2.shape[1])).to('cuda')
            batch_values_iteration=batch_values_iteration.long()

            edge_values_iteration=torch.ones(edge_indices_iteration_2.shape[1]).to('cuda')
            edge_values_iteration=edge_values_iteration.long()

            point_cloud_pcd=point_cloud[k]
            
            Matrix_edges=torch_geometric.utils.to_dense_adj(edge_index=edge_indices_iteration_2,batch=None,edge_attr=None,max_num_nodes=reeb_nodes_num)

            x_condition=torch.ones(Matrix_edges.shape[0],Matrix_edges.shape[1]).to('cuda')
            y_condition=torch.zeros(Matrix_edges.shape[0],Matrix_edges.shape[1]).to('cuda')
            
            Matrix_edges=torch.where(Matrix_edges > 0, x_condition, y_condition)
            New_edge_indices, New_edge_values=torch_geometric.utils.dense_to_sparse(Matrix_edges)

            New_edge_indices_cpu=New_edge_indices.to('cpu')

            Matrix_edges=Matrix_edges.squeeze(0)

            np_Matrix_edges=np.asarray(Matrix_edges.to('cpu'))
            
            # fig = matplotlib.pyplot.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_axis_off()
            # for e in edges:
            #     ax.plot([vertices[e[0]][0], vertices[e[1]][0]], [vertices[e[0]][1], vertices[e[1]][1]], [vertices[e[0]][2], vertices[e[1]][2]], color='b')
            # ax.scatter(point_cloud_pcd[:, 0], point_cloud_pcd[:, 1], point_cloud_pcd[:, 2], s=1, color='r')   
            # matplotlib.pyplot.show()

            # fig = matplotlib.pyplot.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_axis_off()
            # for test_iter in range(New_edge_indices_cpu.shape[1]):
            #     ax.plot([vertices[New_edge_indices_cpu[0][test_iter]][0], vertices[New_edge_indices_cpu[1][test_iter]][0]], [vertices[New_edge_indices_cpu[0][test_iter]][1], vertices[New_edge_indices_cpu[1][test_iter]][1]], [vertices[New_edge_indices_cpu[0][test_iter]][2], vertices[New_edge_indices_cpu[1][test_iter]][2]], color='b')
            # ax.scatter(point_cloud_pcd[:, 0], point_cloud_pcd[:, 1], point_cloud_pcd[:, 2], s=1, color='r')   
            # matplotlib.pyplot.show()

        
            if (nr_columns_batch>nr_columns_all):
                ceva=all_sccs[:,nr_columns_all-1]
                ceva=ceva.reshape((nr_lines_all,1))
                ceva=np.tile(ceva,(nr_columns_batch-nr_columns_all))
                all_sccs=np.concatenate((all_sccs,ceva),1)


            else:
                ceva=np_sccs_batch[:,nr_columns_batch-1]
                ceva=ceva.reshape((nr_lines_batch,1))
                ceva=np.tile(ceva,(nr_columns_all-nr_columns_batch))
                np_sccs_batch=np.concatenate((np_sccs_batch,ceva),1)

            all_sccs=np.concatenate((all_sccs,np_sccs_batch),0)
            all_reeb_laplacians=np.concatenate((all_reeb_laplacians,np_reeb_laplacian),0)
            all_reeb_edge_matrix=np.concatenate((all_reeb_edge_matrix,np_Matrix_edges),0)

        
        print(all_sccs.shape)
        print(all_reeb_laplacians.shape)
        print(all_reeb_edge_matrix.shape)
       
    all_scc=np.delete(all_sccs,[0,1,2],0)
    all_reeb_laplacians=np.delete(all_reeb_laplacians,[0,1,2],0)
    all_reeb_edge_matrix=np.delete(all_reeb_edge_matrix,[0,1,2],0) 

    np.save(sccs_path, all_sccs)
    np.save(reeb_laplacian_path, all_reeb_laplacians)
    np.save(edge_matrix_path, all_reeb_edge_matrix)

    

    return all_sccs,all_reeb_laplacians,all_reeb_edge_matrix


def Create_Reeb_custom_loader_batched(loader,sccs_path,reeb_laplacian_path,edge_matrix_path,vertices_path,time_execution,knn,ns,tau,reeb_nodes_num,reeb_sim_margin,pointNumber):
    # knn = 20
    # ns = 20
    # tau = 2
    # reeb_nodes_num=20
    # reeb_sim_margin=20
    # pointNumber=200
    
    all_sccs=np.eye(3)
    all_reeb_laplacians = np.zeros((3,reeb_nodes_num))
    all_reeb_edge_matrix = np.zeros((3,reeb_nodes_num))
    all_positions = np.zeros((3,3))

    for i, (pos, y, normal, idx) in enumerate(loader):

        print(i+1)

        batch_size=pos[1].shape[0]
        nr_points=pos[1].shape[1]
        feature_dim=pos[1].shape[2]

        batch_edge_indices = torch.tensor([[1., 2.], [4., 5.]])
        batch_batch_indices_reeb = torch.tensor([1.,2. ])


        point_cloud=np.asarray(pos[1])
        #point_cloud=np.reshape(point_cloud,(batch_size,nr_points,feature_dim)
        

        for k in range(batch_size):
            
            vertices, laplacian_Reeb, sccs ,edges= extract_reeb_graph(point_cloud[k], knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
            
            np_sccs_batch=np.asarray(sccs)
            np_reeb_laplacian=np.asarray(laplacian_Reeb)
            np_reeb_edges=np.asarray(edges)

            nr_columns_batch= np_sccs_batch.shape[1]
            nr_columns_all=all_sccs.shape[1]

            nr_lines_batch=np_sccs_batch.shape[0]
            nr_lines_all=all_sccs.shape[0]

            edge_indices_iteration = torch.tensor(edges,device='cuda')
            edge_indices_iteration_2=torch.transpose(edge_indices_iteration,0,1)

            batch_values_iteration=(torch.zeros(edge_indices_iteration_2.shape[1])).to('cuda')
            batch_values_iteration=batch_values_iteration.long()

            edge_values_iteration=torch.ones(edge_indices_iteration_2.shape[1]).to('cuda')
            edge_values_iteration=edge_values_iteration.long()

            point_cloud_pcd=point_cloud[k]
            
            Matrix_edges=torch_geometric.utils.to_dense_adj(edge_index=edge_indices_iteration_2,batch=None,edge_attr=None,max_num_nodes=reeb_nodes_num)

            x_condition=torch.ones(Matrix_edges.shape[0],Matrix_edges.shape[1]).to('cuda')
            y_condition=torch.zeros(Matrix_edges.shape[0],Matrix_edges.shape[1]).to('cuda')
            
            Matrix_edges=torch.where(Matrix_edges > 0, x_condition, y_condition)
            New_edge_indices, New_edge_values=torch_geometric.utils.dense_to_sparse(Matrix_edges)

            New_edge_indices_cpu=New_edge_indices.to('cpu')

            Matrix_edges=Matrix_edges.squeeze(0)

            np_Matrix_edges=np.asarray(Matrix_edges.to('cpu'))
            
            # fig = matplotlib.pyplot.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_axis_off()
            # for e in edges:
            #     ax.plot([vertices[e[0]][0], vertices[e[1]][0]], [vertices[e[0]][1], vertices[e[1]][1]], [vertices[e[0]][2], vertices[e[1]][2]], color='b')
            # ax.scatter(point_cloud_pcd[:, 0], point_cloud_pcd[:, 1], point_cloud_pcd[:, 2], s=1, color='r')   
            # matplotlib.pyplot.show()

            # fig = matplotlib.pyplot.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_axis_off()
            # for test_iter in range(New_edge_indices_cpu.shape[1]):
            #     ax.plot([vertices[New_edge_indices_cpu[0][test_iter]][0], vertices[New_edge_indices_cpu[1][test_iter]][0]], [vertices[New_edge_indices_cpu[0][test_iter]][1], vertices[New_edge_indices_cpu[1][test_iter]][1]], [vertices[New_edge_indices_cpu[0][test_iter]][2], vertices[New_edge_indices_cpu[1][test_iter]][2]], color='b')
            # ax.scatter(point_cloud_pcd[:, 0], point_cloud_pcd[:, 1], point_cloud_pcd[:, 2], s=1, color='r')   
            # matplotlib.pyplot.show()

        
            if (nr_columns_batch>nr_columns_all):
                ceva=all_sccs[:,nr_columns_all-1]
                ceva=ceva.reshape((nr_lines_all,1))
                ceva=np.tile(ceva,(nr_columns_batch-nr_columns_all))
                all_sccs=np.concatenate((all_sccs,ceva),1)


            else:
                ceva=np_sccs_batch[:,nr_columns_batch-1]
                ceva=ceva.reshape((nr_lines_batch,1))
                ceva=np.tile(ceva,(nr_columns_all-nr_columns_batch))
                np_sccs_batch=np.concatenate((np_sccs_batch,ceva),1)

            all_sccs=np.concatenate((all_sccs,np_sccs_batch),0)
            all_reeb_laplacians=np.concatenate((all_reeb_laplacians,np_reeb_laplacian),0)
            all_reeb_edge_matrix=np.concatenate((all_reeb_edge_matrix,np_Matrix_edges),0)
            all_positions=np.concatenate((all_positions,vertices),0)

        
        print(all_sccs.shape)
        print(all_reeb_laplacians.shape)
        print(all_reeb_edge_matrix.shape)
        print(all_positions.shape)
       
    all_sccs=np.delete(all_sccs,[0,1,2],0)
    all_reeb_laplacians=np.delete(all_reeb_laplacians,[0,1,2],0)
    all_reeb_edge_matrix=np.delete(all_reeb_edge_matrix,[0,1,2],0) 
    all_positions=np.delete(all_positions,[0,1,2],0)

    np.save(sccs_path, all_sccs)
    np.save(reeb_laplacian_path, all_reeb_laplacians)
    np.save(edge_matrix_path, all_reeb_edge_matrix)
    np.save(vertices_path, all_positions)

    

    return all_sccs,all_reeb_laplacians,all_reeb_edge_matrix,all_positions

def Create_Reeb_from_Dataset(loader,sccs_path,reeb_laplacian_path,time_execution):
    
    knn = 20
    ns = 20
    tau = 2
    reeb_nodes_num=20
    reeb_sim_margin=20
    pointNumber=200

    
    all_sccs=np.eye(3)
    all_reeb_laplacians = np.zeros((3,reeb_nodes_num))
    
    for i, data in enumerate(loader):

        print(i+1)
        
        point_cloud=np.asarray(data.pos)
        Reeb_Graph_start_time = time.time()
        vertices, laplacian_Reeb, sccs ,edges= extract_reeb_graph(point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)
        Reeb_Graph_end_time = time.time()

        print(Reeb_Graph_end_time-Reeb_Graph_start_time)
        time_execution +=Reeb_Graph_end_time-Reeb_Graph_start_time

        np_sccs_batch=np.asarray(sccs)
        np_reeb_laplacian=np.asarray(laplacian_Reeb)

        nr_columns_batch= np_sccs_batch.shape[1]
        nr_columns_all=all_sccs.shape[1]

        nr_lines_batch=np_sccs_batch.shape[0]
        nr_lines_all=all_sccs.shape[0]

        if (nr_columns_batch>nr_columns_all):
            ceva=all_sccs[:,nr_columns_all-1]
            ceva=ceva.reshape((nr_lines_all,1))
            ceva=np.tile(ceva,(nr_columns_batch-nr_columns_all))
            all_sccs=np.concatenate((all_sccs,ceva),1)

        else:
            ceva=np_sccs_batch[:,nr_columns_batch-1]
            ceva=ceva.reshape((nr_lines_batch,1))
            ceva=np.tile(ceva,(nr_columns_all-nr_columns_batch))
            np_sccs_batch=np.concatenate((np_sccs_batch,ceva),1)

        all_sccs=np.concatenate((all_sccs,np_sccs_batch),0)
        all_reeb_laplacians=np.concatenate((all_reeb_laplacians,np_reeb_laplacian),0)

        print(all_sccs.shape)
        print(all_reeb_laplacians.shape)

        # if((i+1)%5==0):
            # np.save(sccs_path, all_sccs)
            # np.save(reeb_laplacian_path, all_reeb_laplacians)

    # np.save(sccs_path, all_sccs)
    # np.save(reeb_laplacian_path, all_reeb_laplacians)

    return all_sccs,all_reeb_laplacians





  
    