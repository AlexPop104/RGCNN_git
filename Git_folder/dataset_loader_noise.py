from nis import cat
from unicodedata import category
import numpy as np
import os

from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader
from torchvision import transforms, utils

from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.data.dataset import Dataset
from pathlib import Path
import scipy.spatial.distance

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from torch_geometric.nn import fps
from math import floor 

import torch_geometric.transforms

import open3d as o3d
import random
import time

from torch_geometric.transforms import Compose

from utils import GaussianNoiseTransform
from utils import Sphere_Occlusion_Transform
from torch_geometric.transforms import RandomRotate

def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])

class PcdDataset(Dataset):
    def __init__(self, root_dir, points=512, valid=False, folder="train", transform=default_transforms(), save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        self.save_path = None
        self.folder = folder
        self.points = points
        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                
            for category in self.classes.keys():
                save_dir = save_path/Path(category)/folder
                os.makedirs(save_dir)

        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.pcd'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, idx):
        pcd = o3d.io.read_point_cloud(file)
        #print(file)
        points = np.asarray(pcd.points)
        points = torch.tensor(points)

        normals = []

      
            
           

            #print(len(points))
        if self.save_path is not None:

            pcd.estimate_normals(fast_normal_computation=False)
            pcd.normalize_normals()
            #pcd.orient_normals_consistent_tangent_plane(100)

            normals=np.asarray(pcd.normals)



            if len(points) < self.points:
                alpha = 0.03
                rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha)


                #o3d.visualization.draw_geometries([pcd, rec_mesh])

                num_points_sample = self.points

                pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 

                points = pcd_sampled.points
                points = torch.tensor(points)
                points=points.float()
                normals = np.asarray(pcd_sampled.normals)

                # print(len(points))
                # print(len(normals))
            else:

                nr_points_fps=self.points
                nr_points=points.shape[0]

                index_fps = fps(points, ratio=float(nr_points_fps/nr_points) , random_start=True)

                index_fps=index_fps[0:nr_points_fps]

                fps_points=points[index_fps]
                fps_normals=normals[index_fps]

                points=fps_points
                normals = fps_normals
        else:
            normals=np.asarray(pcd.normals)
        normals = torch.Tensor(normals)
        normals=normals.float()

        pointcloud = torch_geometric.data.Data(normal=normals, pos=points, y=self.classes[self.files[idx]['category']])

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f.name.strip(), idx)
            if self.save_path is not None:
                # name = str(time.time())
                # name = name.replace('.', '')
                # name = str(name) + ".pcd"
                splits = f.name.strip().split("/")
                cat = splits[len(splits) - 3]

                name = splits[len(splits) - 1]

                total_path = self.save_path/cat/self.folder/name
                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(pointcloud.pos)
                pcd_save.normals =  o3d.utility.Vector3dVector(pointcloud.normal)
                o3d.io.write_point_cloud(str(total_path), pcd_save, write_ascii=True)
        return pointcloud

def process_dataset(root, save_path, transform=None, num_points=512):
    dataset_train = PcdDataset(root, folder="train", transform=transform, save_path=save_path, points=num_points)
    dataset_test =  PcdDataset(root, folder="test", transform=transform, save_path=save_path, points=num_points)

    print("Processing train data: ")
    for i in tqdm(range(len(dataset_train))):
        _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]

if __name__ == '__main__':


    ########################3
    ######New tests

    mu=0
    sigma=0.1

    rot_x=1
    rot_y=1
    rot_z=1

    ceva=2

    radius=0.03
    percentage=0.25

    random_rotate = Compose([
    RandomRotate(degrees=rot_x*ceva*10, axis=0),
    RandomRotate(degrees=rot_y*ceva*10, axis=1),
    RandomRotate(degrees=rot_z*ceva*10, axis=2),
    ])

    test_transform = Compose([
                    #random_rotate,
                    #GaussianNoiseTransform(mu=mu,sigma=sigma),
                    #Sphere_Occlusion_Transform(percentage=percentage)
                    ])

    ##################################################3
   
    print("Type value for selected choice---------1 for processing dataset-----all else for loading dataset")
    print("Choice=")
    choice=int(input())

    num_points = 1024
    root = Path("/home/alex/Alex_documents/RGCNN_git/Git_folder/data/Dataset/")
    root_noise = Path("/home/alex/Alex_documents/RGCNN_git/Git_folder/data/dataset_resampled_1024/")

    if(choice==1):
    ####Processing the datasets
         process_dataset(root=root, save_path=root_noise,  num_points=num_points,transform=test_transform)

    else:
    ##################################################################333333333

    ##Loading the processed dataset

    

        num_points = 1024

        train_dataset = PcdDataset(root,valid=False,points=num_points)
        test_dataset = PcdDataset(root,folder="test",points=num_points)

        train_dataset_noise = PcdDataset(root_noise,valid=False,points=num_points)
        test_dataset_noise = PcdDataset(root_noise,folder="test",points=num_points)

        


        for i in range(100):
        
            pcd_sampled = o3d.geometry.PointCloud()
            pcd_noise = o3d.geometry.PointCloud()

            print("PCD sampled")
            pcd_sampled.points=o3d.utility.Vector3dVector(train_dataset[i].pos)
            #pcd_sampled.normals=o3d.utility.Vector3dVector(train_dataset[i].normal)

            pcd_sampled.paint_uniform_color([0, 0, 1])

            print("PCD noise")
            pcd_noise.points=o3d.utility.Vector3dVector(train_dataset_noise[i].pos)
            #pcd_noise.normals=o3d.utility.Vector3dVector(train_dataset_noise[i].normal)
            pcd_noise.paint_uniform_color([1, 0, 0])

            o3d.visualization.draw_geometries([pcd_sampled, pcd_noise])
  