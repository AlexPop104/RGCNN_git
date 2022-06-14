from asyncio import FastChildWatcher
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
from utils import Rotate_transform
from torch_geometric.transforms import RandomRotate

def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])


def rotate_pcd(pcd, angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis == 0:
        """rot on x"""
        R = np.array([[1 ,0 ,0],[0 ,c ,-s],[0 ,s ,c]])
    elif axis == 1:
        """rot on y"""
        R = np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]])
    elif axis == 2:
        """rot on z"""
        R = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])

    return pcd.rotate(R)

class PcdDataset(Dataset):
    def __init__(self, root_dir, points=512, valid=False, folder="train", transform=default_transforms(), save_path=None,rotation=[0.,0.,0.],random_rotation_limits=[[0,0],[0,0],[0,0]]):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        self.save_path = None
        self.folder = folder
        self.points = points
        self.rotation=rotation
        self.random_rotation_limits=random_rotation_limits
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

        # if((np.array_equal(self.rotation,[0.,0.,0.]))==False ):
        #     pcd=rotate_pcd(pcd,self.rotation[0],0)
        #     pcd=rotate_pcd(pcd,self.rotation[1],1)
        #     pcd=rotate_pcd(pcd,self.rotation[2],2)

        # elif((np.array_equal(self.random_rotation_limits,[0.,0.,0.,0.,0.,0.]))==False):

        #     angle_x = np.random.uniform(low = self.random_rotation_limits[0][0], high=self.random_rotation_limits[0][1], size=(1,))
        #     angle_y = np.random.uniform(low = self.random_rotation_limits[1][0], high=self.random_rotation_limits[1][1], size=(2,))
        #     angle_z = np.random.uniform(low = self.random_rotation_limits[2][0], high=self.random_rotation_limits[2][1], size=(3,))

        #     pcd=rotate_pcd(pcd,angle_x[0],0)
        #     pcd=rotate_pcd(pcd,angle_y[0],1)
        #     pcd=rotate_pcd(pcd,angle_y[0],2)
            
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

def process_dataset(root, save_path, transform=None, num_points=512,rotation=[0.,0.,0.],random_rotation_limits=[[0,0],[0,0],[0,0]]):
    dataset_train = PcdDataset(root, folder="train", transform=transform, save_path=save_path, points=num_points,rotation=rotation,random_rotation_limits=random_rotation_limits)
    dataset_test =  PcdDataset(root, folder="test", transform=transform, save_path=save_path, points=num_points,rotation=rotation,random_rotation_limits=random_rotation_limits)

    print("Processing train data: ")
    for i in tqdm(range(len(dataset_train))):
        _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]

if __name__ == '__main__':


    ########################3
    ######New tests

    print("Type value for selected choice---------1 for processing dataset-----all else for loading dataset")
    print("Choice=")
    choice=int(input())

    print("Type value for rotation choice---------1 for fixed rotation-----2 for random rotation-----all else for no rotation")
    print("rotation Choice=")
    choice_rotation=int(input())

    ceva=4

    rotation=[0.,0.,0.]
    random_rotation_limits=[[0,0],[0,0],[0,0]]

    if(choice_rotation==1):
        angle_x=np.pi/180*90
        angle_y=np.pi/180*90
        angle_z=np.pi/180*90

        rotation=[angle_x,angle_y,angle_z]

    elif(choice_rotation==2):
        angle_limits_x_low=-np.pi/180*ceva*10
        angle_limits_x_high=np.pi/180*ceva*10
        angle_limits_y_low=-np.pi/180*ceva*10
        angle_limits_y_high=np.pi/180*ceva*10
        angle_limits_z_low=-np.pi/180*ceva*10
        angle_limits_z_high=np.pi/180*ceva*10

        random_rotation_limits=[[angle_limits_x_low,angle_limits_x_high],[angle_limits_y_low,angle_limits_y_high],[angle_limits_z_low,angle_limits_z_high]]


    mu=0
    sigma=0.0

    radius=0.25
    percentage=0.25
    
    # random_rotate = Compose([
    # RandomRotate(degrees=rot_x*ceva*10, axis=0),
    # RandomRotate(degrees=rot_y*ceva*10, axis=1),
    # RandomRotate(degrees=rot_z*ceva*10, axis=2),
    # ])



    test_transform = Compose([
                    Rotate_transform(rotation=rotation,random_rotation_limits=random_rotation_limits)
                    #GaussianNoiseTransform(mu=mu,sigma=sigma)
                    #Sphere_Occlusion_Transform(radius=radius, percentage=percentage,num_points=1024)
                    ])

    ##################################################3
   
   

    num_points = 512
    root = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Normals_2048/Modelnet40_2048/")

    root_noise_1 = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Normals/Modelnet40_"+str(num_points)+"_r_10"+"/")

    if(choice==1):
    ####Processing the datasets
         process_dataset(root=root, save_path=root_noise_1,  num_points=num_points,transform=test_transform)

    else:
    ##################################################################333333333

    ##Loading the processed dataset

        num_points_original=2048
        num_points_noise=512

        
        train_dataset = PcdDataset(root,valid=False,points=num_points_original)
        train_dataset_noise_1 = PcdDataset(root_noise_1,valid=False,points=num_points_noise)
        

        for i in range(0,200):
        
            pcd_sampled = o3d.geometry.PointCloud()
            pcd_noise_1 = o3d.geometry.PointCloud()
           
            

            print("PCD sampled")
            pcd_sampled.points=o3d.utility.Vector3dVector(train_dataset[i].pos)
            #pcd_sampled.normals=o3d.utility.Vector3dVector(train_dataset[i].normal)

            pcd_sampled.paint_uniform_color([0, 0, 1])

            # print("PCD noise")
            pcd_noise_1.points=o3d.utility.Vector3dVector(train_dataset_noise_1[i].pos)
            #pcd_noise.normals=o3d.utility.Vector3dVector(train_dataset_noise[i].normal)
            pcd_noise_1.paint_uniform_color([1, 0, 0])

            o3d.visualization.draw_geometries([pcd_sampled, pcd_noise_1])
            #o3d.visualization.draw_geometries([pcd_sampled])
            #o3d.visualization.draw_geometries([ pcd_noise_1])
  