import torch_geometric
import torch.nn as nn
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import NormalizeScale
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
import torch_geometric.nn.conv as conv
from torch_geometric.transforms import Compose


import torch
from torch_geometric.nn.conv import ChebConv
from torch.nn import Linear
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

import numpy as np
import matplotlib.pyplot as plt

import time

import os
from datetime import datetime


select_net_archi=1


now = datetime.now()
time_now_string = now.strftime("%d_%m_%y_%H:%M:%S")

log_folder_path="/home/alex/Alex_documents/RGCNN_git/data/logs/Network_performances/"
model_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"

conf_matrix_path=log_folder_path+"Type_"+str(select_net_archi)+"_"+time_now_string+"_conf_matrix.npy"
loss_log_path=log_folder_path+"Type_"+str(select_net_archi)+"_"+time_now_string+"_losses.npy"
accuracy_log_path=log_folder_path+"Type_"+str(select_net_archi)+"_"+time_now_string+"test_accuracy.npy"

########################################################################################




num_points = 1024
batch_size = 32
modelnet_num = 40
nr_epochs=100


path = os.path.join(model_directory, time_now_string)
os.mkdir(path)

transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])

root = "data/ModelNet"+str(modelnet_num)
print(root)
dataset_train = ModelNet(root=root, name=str(modelnet_num), train=True, transform=transforms)
dataset_test = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms)

# Verification...
print(f"Train dataset shape: {dataset_train}")
print(f"Test dataset shape:  {dataset_test}")

print(dataset_train[0])

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test  = DataLoader(dataset_test, batch_size=batch_size)


def get_graph(point_cloud, batch):
    point_cloud = point_cloud.reshape(batch.unique().shape[0], num_points, -1)
    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    adj_matrix = torch.exp(-adj_matrix)
    edge_index, edge_weight = utils.dense_to_sparse(adj_matrix)
    
    return edge_index, edge_weight ,adj_matrix

def get_one_matrix_knn(matrix, k,batch_size,batch):

    nr_points=int(batch.size(0)/batch_size)

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
    
    full_indices=torch.cat((torch.unsqueeze(my_range_repeated_2,2),torch.unsqueeze(indices,2)),axis=2)
    full_indices=torch.transpose(full_indices,2,3)
    full_indices=torch.reshape(full_indices,(batch_size,nr_points*k,2))
    full_indices=torch.reshape(full_indices,(batch_size*nr_points*k,2))
    full_indices=torch.transpose(full_indices,0,1)
    full_indices=full_indices.long()

    edge_weights=torch.reshape(values,[-1])

   
    
    return full_indices ,edge_weights


class RGCNN_model(nn.Module):
    def __init__(self):
        super(RGCNN_model, self).__init__()
        self.conv1  = ChebConv(6, 128, 6)


        self.relu   = nn.ReLU()

     
        self.fc1_output    = Linear(128, modelnet_num)




    def forward(self, x, batch,num_points,select_archi):
        # time_start = time.time()
        batch_size=int(batch.size(0)/num_points)

        if(select_net_archi==1):
            with torch.no_grad():
                edge_index, edge_weight,adj_matrix = get_graph(x, batch=batch)
            out = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            out = self.relu(out)
            out = global_max_pool(out, batch)
            out = self.fc1_output(out)
        
        else:
            with torch.no_grad():
                edge_index, edge_weight,adj_matrix = get_graph(x, batch=batch)
                edge_index_knn, edge_weight_knn =get_one_matrix_knn(matrix=adj_matrix,k=30,batch_size=batch_size,batch=batch)
            out = self.conv1(x=x, edge_index=edge_index_knn, edge_weight=edge_weight_knn, batch=batch)
            out = self.relu(out)
            out = global_max_pool(out, batch)
            out = self.fc1_output(out)
        

        return out

model = RGCNN_model()
print(model)

import torch
device = "cuda"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.


def train(model, optimizer, loader,num_points,select_archi):
    model.train()
    total_loss = 0
    
    time_start = time.time()
    for data in loader:
        
        optimizer.zero_grad()
        x = torch.cat([data.pos, data.normal], dim=1)
        
        logits  = model(x.to(device),  data.batch.to(device),num_points,select_archi=select_net_archi)
        pred = logits.argmax(dim=-1)


        loss    = criterion(logits, data.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    time_end=time.time()
    print("Time train layer")
    print(time_end-time_start)
    return total_loss / len(loader.dataset) 

@torch.no_grad()
def test(model, loader,modelnet_num,num_points,select_archi):
    confusion_matrix=np.zeros((modelnet_num,modelnet_num))
    category_counters=np.zeros(modelnet_num)

    model.eval()

    total_correct = 0
    for data in loader:
        x = torch.cat([data.pos, data.normal], dim=1)
        logits = model(x.to(device), data.batch.to(device),num_points,select_archi=select_net_archi)
        pred = logits.argmax(dim=-1)

        iteration_batch_size=int(data.pos.shape[0]/num_points)

        for j in range(iteration_batch_size):
            confusion_matrix[pred[j]][data.y[j]]+=1
            category_counters[pred[j]]+=1



        total_correct += int((pred == data.y.to(device)).sum())

    for j in range(modelnet_num):
            confusion_matrix[j,:] =confusion_matrix[j,:]/ category_counters[j]
    
    #confusion_matrix=confusion_matrix/len(loader.dataset)

    return total_correct / len(loader.dataset) , confusion_matrix

all_losses=np.array([])
test_accuracy=np.array([])
confusion_matrix_collection = np.zeros((1,modelnet_num))

for epoch in range(nr_epochs):
    
    train_time_start = time.time()
    loss = train(model, optimizer, dataloader_train,num_points=num_points,select_archi=select_net_archi)
    train_time_end=time.time()
    
    all_losses=np.append(all_losses, loss)

    eval_time_start = time.time()
    test_acc, confusion_matrix = test(model, dataloader_test,modelnet_num,num_points,select_archi=select_net_archi)
    eval_time_end=time.time()
    test_accuracy=np.append(test_accuracy, test_acc)
    confusion_matrix_collection=np.append(confusion_matrix_collection,confusion_matrix,axis=0)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f} --- Train Time: {train_time_end - train_time_start} --- Eval Time: {eval_time_end - eval_time_start }')

    

    if(epoch%5==0):
        np.save(conf_matrix_path, confusion_matrix_collection)
        np.save(loss_log_path, all_losses)
        np.save(accuracy_log_path, test_accuracy)

        torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')

        print(confusion_matrix)




np.save(loss_log_path, all_losses)
np.save(accuracy_log_path, test_accuracy)
np.save(conf_matrix_path, confusion_matrix_collection)


plt.plot(all_losses, '-b', label='Training loss')
plt.title('Training loss', fontdict=None, loc='center', pad=None)
plt.show()

plt.plot(test_accuracy, '-r', label='Test accuracy')
plt.title('Test accuracy', fontdict=None, loc='center', pad=None)
plt.show()

