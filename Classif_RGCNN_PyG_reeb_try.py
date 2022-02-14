import torch
import torch_geometric
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose
from torch_geometric.transforms import LinearTransformation
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
import sys

import torch.nn as nn
import torch_geometric.utils as utils
import torch_geometric.nn.conv as conv

import os
from datetime import datetime


import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import heapq
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances_argmin



now = datetime.now()
directory = now.strftime("%d_%m_%y_%H:%M:%S")
parent_directory = "/home/alex/Alex_pyt_geom/Models"
path = os.path.join(parent_directory, directory)
os.mkdir(path)



## NOTE:

# ----------------------------------------------------------------
# Hyper parameters:
num_points = 1024    
batch_size_nr = 1     # not yet used
num_epochs = 100
learning_rate = 0.001
modelnet_num = 40    # 10 or 40

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, modelnet_num]


# ----------------------------------------------------------------
# Choosing device:
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


####################################################################################


transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

root = "data/ModelNet"+str(modelnet_num)
dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

# Shuffle Data
dataset_train = dataset_train.shuffle()
dataset_test = dataset_test.shuffle()

# Verification...
print(f"Train dataset shape: {dataset_train}")
print(f"Test dataset shape:  {dataset_test}")

print(dataset_train[0])

######################################################3

class GetReebGraph(nn.Module):
    def __init__(self):
        """
        Creates the weighted adjacency matrix 'W'
        Taked directly from RGCNN
        """
        super(GetReebGraph, self).__init__()

    def filter_out(self,vertices, edges, sccs):
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


    def similarity(self,scc1, scc2, SIM_MARGIN):
        dist = distance_matrix(scc1, scc2)
        return (np.sum(np.max(np.min(dist, 0) - SIM_MARGIN, 0)) + np.sum(np.max(np.min(dist, 1) - SIM_MARGIN, 0))) / (scc1.shape[0] + scc2.shape[1])


    def adjacency_reeb(self,vertices, edges, sccs, point_cloud, SIM_MARGIN):
        dist1 = np.zeros([vertices.shape[0], vertices.shape[0]])
        dist2 = np.zeros_like(dist1)
        for e in edges:
            dist1[e[0]][e[1]] = dist1[e[1]][e[0]] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])
            dist2[e[0]][e[1]] = dist2[e[1]][e[0]] = self.similarity(point_cloud[sccs[e[0]]], point_cloud[sccs[e[1]]], SIM_MARGIN)

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


    def normalize_reeb(self,vertices, edges, sccs, point_cloud, NODES_NUM):
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
                edges.remove([tosplit[0], tosplit[1]])
                edges.append([tosplit[0], vertices.shape[0]])
                edges.append([tosplit[1], vertices.shape[0]])
                sccs.append(np.concatenate([sccs[tosplit[0]], sccs[tosplit[1]]]))
               
                vertices = np.append(vertices, [node], 0)
            for e in edges:
                assert e[0] < NODES_NUM and e[1] < NODES_NUM
        return vertices, edges, sccs
        

    def expand(self,x, k, visited, nbrs, valid_idxs, scc, tau, knn, cnt=0):
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




    def extract_reeb_graph(self,point_cloud, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber):  

        

        nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm='kd_tree').fit(point_cloud)
        distances, indices = nbrs.kneighbors(point_cloud)

        marked = np.zeros([point_cloud.shape[0]], np.bool)
        # calculate f
        r = point_cloud[:, 2]

        mean_x=np.mean(point_cloud[:,0])
        mean_y=np.mean(point_cloud[:,1])
        mean_z=np.mean(point_cloud[:,2])

        r=np.sqrt( (point_cloud[:, 0]-mean_x)*(point_cloud[:, 0]-mean_x) + (point_cloud[:, 1]-mean_y)*(point_cloud[:, 1]-mean_y) +(point_cloud[:, 2]-mean_z)*(point_cloud[:, 2]-mean_z) )

        r_min=np.amin(r)
        r_max=np.amax(r)
        #r = np.linalg.norm(point_cloud, axis=-1)
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
        vertices, edges, sccs = self.filter_out(np.asarray(vertices), edges, sccs)
        vertices, edges, sccs = self.normalize_reeb(np.asarray(vertices), edges, sccs, point_cloud, reeb_nodes_num)
        vertices, edges, laplacian, sccs = self.adjacency_reeb(vertices, edges, sccs, point_cloud, reeb_sim_margin)
        # laplacian = np.delete(laplacian, idx2remove, 0)
        # laplacian = np.delete(laplacian, idx2remove, 1)
        # sccs = np.delete(sccs, idx2remove, 0)
        while vertices.shape[0] != reeb_nodes_num:
            # print(vertices.shape[0])
            vertices, edges, sccs = self.normalize_reeb(np.asarray(vertices), edges, sccs, point_cloud, reeb_nodes_num)
            vertices, edges, laplacian, sccs = self.adjacency_reeb(vertices, edges, sccs, point_cloud, reeb_sim_margin)
            # laplacian = np.delete(laplacian, idx2remove, 0)
            # laplacian = np.delete(laplacian, idx2remove, 1)
            # sccs = np.delete(sccs, idx2remove, 0)
        # print(laplacian)
        #pad
        largest_dim = max([len(x) for x in sccs])
        # largest_dim = pointNumber
        sccs = np.asarray([np.pad(x, (0, largest_dim - len(x)), 'edge') for x in sccs])
        # assert np.all(np.isfinite(laplacian)) and np.all(np.isfinite(sccs))
        # print(vertices.shape, laplacian.shape)
        print(np.shape(vertices))
        return vertices, laplacian, list(sccs), edges
        #return vertices, list(sccs), edges

    def forward(self, point_cloud):

        knn = 20
        ns = 20
        tau = 2
        reeb_nodes_num=20
        reeb_sim_margin=20
        pointNumber=200

        point_cloud2=point_cloud[0,:,:].detach().cpu().numpy()


        vertices,laplacian, sccs,edges =self.extract_reeb_graph(point_cloud2, knn, ns, reeb_nodes_num, reeb_sim_margin,pointNumber)

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        for e in edges:
            ax.plot([vertices[e[0]][0], vertices[e[1]][0]], [vertices[e[0]][1], vertices[e[1]][1]], [vertices[e[0]][2], vertices[e[1]][2]], color='b')
        ax.scatter(point_cloud2[:, 0], point_cloud2[:, 1], point_cloud2[:, 2], s=1, color='r')
        matplotlib.pyplot.show()

      
        return vertices,laplacian,sccs



class GetGraph(nn.Module):
    def __init__(self):
        """
        Creates the weighted adjacency matrix 'W'
        Taked directly from RGCNN
        """
        super(GetGraph, self).__init__()
        

    def forward(self, point_cloud):
        point_cloud_transpose = point_cloud.permute(0, 2, 1)
        point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2 * point_cloud_inner
        point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
        point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
        adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
        adj_matrix = torch.exp(-adj_matrix)
        return adj_matrix


class GetLaplacian(nn.Module):
    def __init__(self, normalize=True):
        """
        Computes the Graph Laplacian from a Weighted Graph
        Taken directly from RGCNN - currently not used - might need to find alternatives in PyG for loss function
        """
        super(GetLaplacian, self).__init__()
        self.normalize = normalize

        def diag(self, mat):
        # input is batch x vertices
            d = []
            for vec in mat:
                d.append(torch.diag(vec))
            return torch.stack(d)

    def forward(self, adj_matrix):
        if self.normalize:
            D = torch.sum(adj_matrix, dim=1)
            eye = torch.ones_like(D)
            eye = self.diag(eye)
            D = 1 / torch.sqrt(D)
            D = self.diag(D)
            L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
        else:
            D = torch.sum(adj_matrix, dim=1)
            D = torch.diag(D)
            L = D - adj_matrix
        return L


class RGCNN_model(nn.Module):
    def __init__(self, vertice, F, K, M, regularization = 0, dropout = 0):
        # verify the consistency w.r.t. the number of layers
        assert len(F) == len(K)
        super(RGCNN_model, self).__init__()
        '''
        F := List of Convolutional Layers dimensions
        K := List of Chebyshev polynomial degrees
        M := List of Fully Connected Layers dimenstions
        
        Currently the dimensions are 'hardcoded'
        '''
        self.F = F
        self.K = K
        self.M = M

        self.vertice = vertice
        self.regularization = regularization    # gamma from the paper: 10^-9
        self.dropout = dropout
        self.regularizers = []

        # initialize the model layers
        self.get_graph = GetGraph()
        self.get_reeb_graph = GetReebGraph()
        # self.get_laplacian = GetLaplacian(normalize=True)
        self.pool = nn.MaxPool1d(self.vertice)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.dropout)

        ###################################################################
        #                               Hardcoded Values for Conv filters
        self.conv1 = conv.ChebConv(6, 128, 6)
        self.conv2 = conv.ChebConv(128, 512, 5)
        self.conv3 = conv.ChebConv(512, 1024, 3)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, modelnet_num)
        ###################################################################


    def forward(self, x,batch,batch_size,nr_points):

        out_reshaped_graph=torch.reshape(x.detach(),(batch_size*nr_points,6))

        self.regularizers = []
        # forward pass

        vertices,laplacian,sccs=self.get_reeb_graph(x.detach())

        W   = self.get_graph(x.detach())  # we don't want to compute gradients when building the graph
        edge_index, edge_weight = utils.dense_to_sparse(W)

        #out = self.conv1(x, edge_index, edge_weight,batch)
        out = self.conv1(out_reshaped_graph, edge_index, edge_weight,batch)


        out = self.relu(out)
        
        
        # edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
        # L_edge_index, L_edge_weight = torch_geometric.utils.get_laplacian(edge_index.detach(), edge_weight.detach(), normalization="sym")
        # L = torch_geometric.utils.to_dense_adj(edge_index=L_edge_index, edge_attr=L_edge_weight)

       

        out=out.unsqueeze(0)

        #self.regularizers.append(torch.linalg.norm(torch.matmul(torch.matmul(torch.Tensor.permute(out.detach(), [0, 2, 1]), L), out.detach())))

        out=out.squeeze(0)

        out_reshaped_graph=torch.reshape(out.detach(),(batch_size,nr_points,128))

        
        W   = self.get_graph(out_reshaped_graph.detach())
        edge_index, edge_weight = utils.dense_to_sparse(W)

        #out = self.conv2(out, edge_index, edge_weight)

        out = self.conv2(out, edge_index, edge_weight,batch)
        out = self.relu(out)

        # edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
        # L_edge_index, L_edge_weight = torch_geometric.utils.get_laplacian(edge_index.detach(), edge_weight.detach(), normalization="sym")
        # L = torch_geometric.utils.to_dense_adj(edge_index=L_edge_index, edge_attr=L_edge_weight)

        out=out.unsqueeze(0)

        #self.regularizers.append(torch.linalg.norm(torch.matmul(torch.matmul(torch.Tensor.permute(out.detach(), [0, 2, 1]), L), out.detach())))

        out=out.squeeze(0)

        out_reshaped_graph=torch.reshape(out.detach(),(batch_size,nr_points,512))


        W   = self.get_graph(out_reshaped_graph.detach())
        edge_index, edge_weight = utils.dense_to_sparse(W)

        #out = self.conv3(out, edge_index, edge_weight)
        
        out = self.conv3(out, edge_index, edge_weight,batch)
        out = self.relu(out)

        # edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
        # L_edge_index, L_edge_weight = torch_geometric.utils.get_laplacian(edge_index.detach(), edge_weight.detach(), normalization="sym")
        # L = torch_geometric.utils.to_dense_adj(edge_index=L_edge_index, edge_attr=L_edge_weight)

        out=out.unsqueeze(0)

        #self.regularizers.append(torch.linalg.norm(torch.matmul(torch.matmul(torch.Tensor.permute(out.detach(), [0, 2, 1]), L), out.detach())))

        out=out.squeeze(0)

        out_reshaped_graph=torch.reshape(out.detach(),(batch_size,nr_points,1024))

        #out = out.permute(0, 2, 1) # Transpose

        out=out_reshaped_graph.permute(0, 2, 1)

        out = self.pool(out)
        out.squeeze_(2)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        for param in self.fc1.parameters():
            self.regularizers.append(torch.linalg.norm(param))

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        for param in self.fc1.parameters():
            self.regularizers.append(torch.linalg.norm(param))
        out = self.fc3(out)
        for param in self.fc1.parameters():
            self.regularizers.append(torch.linalg.norm(param))
        

        return out, self.regularizers



def get_loss(y, labels, regularization, regularizers):
    cross_entropy_loss = loss(y, labels)
    s = torch.sum(torch.as_tensor(regularizers))
    regularization *= s
    l = cross_entropy_loss + regularization
    return l

#
# PATH = "/home/alex/Alex_pyt_geom/Models/model"
#model_number = 5                # Change this acording to the model you want to load
# model.load_state_dict(torch.load(path + '/model' + str(model_number) + '.pt'))

train_loader = DataLoader(dataset_train, batch_size=batch_size_nr, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size_nr)






model = RGCNN_model(num_points, F, K, M, dropout=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()



def train(model, optimizer, loader, batch_size):
    model.train()
    
    i=0
    total_loss = 0
    for data in loader:


        
        print(i)
        i=i+1
        optimizer.zero_grad()  # Clear gradients.

        pos = data.pos.cuda()        # (num_points * 3)   
        normals = data.normal.cuda() # (num_points * 3)

        batch=data.batch

        nr_points=int(pos.shape[0]/batch_size)
        
        

        
        
        x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
        x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
        
        x=torch.reshape(x, (batch_size, nr_points,6))

        x = x.type(torch.float32)  # other types of data may be unstable

        y = data.y              # (1)
        y = y.type(torch.long)  # required by the loss function

        x = x.to(device)      # to CUDA if available
        y = y.to(device)

        batch=batch.to(device)
        

        logits,regularizers = model(x,batch,batch_size,nr_points)  # Forward pass.
        

        l = loss(logits, y)  # Loss computation.

        #l=get_loss(logits, y, regularization=1e-9, regularizers=regularizers) 
        l.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += l.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


for epoch in range(1, 100):
    loss = train(model, optimizer, train_loader,batch_size_nr)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')



##########################################################

#Previous iteration, working without batch data


# correct_percentage_list = []
# loss = torch.nn.CrossEntropyLoss()
# model.train()
# for epoch in range(num_epochs):

#     correct = 0
#     for i, data in enumerate(dataset_train):
#         # make sure the gradients are empty
#         optimizer.zero_grad()
        
#         # Data preparation 
#         pos = data.pos        # (num_points * 3)   
#         normals = data.normal # (num_points * 3)
#         x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
#         x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
#         x = x.type(torch.float32)  # other types of data may be unstable

#         y = data.y              # (1)
#         y = y.type(torch.long)  # required by the loss function
        
#         x = x.to(device)      # to CUDA if available
#         y = y.to(device)
     
#         # Forward pass
#         y_pred, regularizers = model(x)     # (1 * 40)
        
#         class_pred = torch.argmax(y_pred.squeeze(0))  # (1)  
#         correct += int((class_pred == y).sum())       # to compute the accuracy for each epoch
        

#         # loss and backward
#         ###################################################################################
#         #                           CrossEntropyLoss
#         # This WORKS but I am testing the other way...
#         l = loss(y_pred, y)   # one value
#         # l.backward()          # update gradients
#         ###################################################################################
       
#         #l = get_loss(y_pred, y, regularization=1e-9, regularizers=regularizers)
#         l.backward()

#         # optimisation
#         optimizer.step()
        
            
#         if i%100==0:
#             print(f"Epoch: {epoch}, Sample: {i}, Loss:{l} - Predicted class vs Real Cass: {class_pred} <-> {y.item()}")
#             # print(torch.sum(torch.as_tensor(regularizers)))
#         if epoch%5==0:
#             torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')
#     print(f"~~~~~~~~~ CORRECT: {correct / len(dataset_train)} ~~~~~~~~~~~")
#     correct_percentage_list.append(correct / len(dataset_train))
# print(correct_percentage_list)

# torch.save(model.state_dict(), "/home/alex/Alex_pyt_geom/Models/final_model.pt")

# with torch.no_grad():
#     model.eval()
#     correct = 0
#     for data in dataset_test:
#         pos = data.pos        # (num_points * 3)   
#         normals = data.normal # (num_points * 3)
#         x = torch.cat([pos, normals], dim=1)   # (num_points * 6)
#         x = x.unsqueeze(0)    # (1 * num_points * 6)     the first dimension may be used for batching?
#         x = x.type(torch.float32)  # other types of data may be unstable

#         y = data.y              # (1)
#         y = y.type(torch.long)  # required by the loss function
        
#         x = x.to(device)      # to CUDA if available
#         y = y.to(device)
     
#         # Forward pass
#         y_pred, _ = model(x)     # (1 * 40)

#         class_pred = torch.argmax(y_pred)
#         correct += int((class_pred == y).sum())

#     print(f"Correct percentage : {correct / len(dataset_test)}")