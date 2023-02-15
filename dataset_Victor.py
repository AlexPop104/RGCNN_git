#!/home/victor/anaconda3/envs/thesis_env/bin/python3

from concurrent.futures import process
from nis import cat
from unicodedata import category
import numpy as np
import os
import copy
from numpy import save
from pyrsistent import PClass

from tqdm import tqdm

import torch
# from torch.utils.data import DataLoader
from torchvision import transforms, utils

from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.data.dataset import Dataset
from pathlib import Path
import scipy.spatial.distance
import probreg

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch_geometric.transforms

import open3d as o3d
import random
import time

def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])



class pcd_registration():
    def __init__(self) -> None:
        self.source = None
        self.target = None
        self.result = None

    def set_source(self, pcd):
        self.source = pcd
    
    def set_target(self, pcd):
        self.target = pcd
    
    def set_clouds(self, source, target):
        self.source = source
        self.target = target
    
    def __get_transform_matirx(self, rot, t):
        T = np.eye(4, 4)
        T[:3, :3] = rot
        T[:3, 3] = t.T
        return T

    def draw_pcds(self):
        self.source.paint_uniform_color([1, 0.706, 0])
        self.target.paint_uniform_color([0, 0.651, 0.929])
        if self.result:
            self.result.paint_uniform_color([1, 0.235, 0.722])
            o3d.visualization.draw_geometries([self.source, self.target, self.result])
        else:
            o3d.visualization.draw_geometries([self.source, self.target])

    def __get_transformed_matrix(self, T):
        source_temp = copy.deepcopy(self.source)
        # target_temp = copy.deepcopy(target)
        source_temp.transform(T)
        return source_temp

    def register_pcds(self, source=None, target=None):
        if source is not None:
            self.source = source

        if target is not None:
            self.target = target
        
        tf_param = probreg.filterreg.registration_filterreg(self.source, self.target)
        T = self.__get_transform_matirx(tf_param.transformation.rot, tf_param.transformation.t)
        self.result = self.__get_transformed_matrix(T)
        return self.result

class PcdDataset(Dataset):
    def __init__(self, root_dir, points=512, valid=False, folder="train", transform=default_transforms(), with_normals=True, save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        self.with_normals = with_normals
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
        points = np.asarray(pcd.points)
        points = torch.tensor(points)

        normals = []

        if self.with_normals == True:
            pcd.estimate_normals(fast_normal_computation=False)
            pcd.normalize_normals()
            pcd.orient_normals_consistent_tangent_plane(100)

            # o3d.visualization.draw_geometries([pcd])

            # print(len(points))
            if self.save_path is not None:
                if len(points) < self.points:
                    alpha = 0.03
                    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        pcd, alpha)
                    # o3d.visualization.draw_geometries([pcd, rec_mesh])

                    num_points_sample = self.points

                    pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 
                    points = pcd_sampled.points
                
                    normals = np.asarray(pcd_sampled.normals)
                else:
                    normals = np.asarray(pcd.normals)
            else:
                normals = np.asarray(pcd.normals)

            normals = torch.Tensor(normals)

        pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=self.files[idx]['category'])

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f.name.strip(), idx)
            if self.save_path is not None:
                name = str(time.time())
                name = name.replace('.', '')
                name = str(name) + ".pcd"
                splits = f.name.strip().split("/")
                cat = splits[len(splits) - 3]
                total_path = self.save_path/cat/self.folder/name
                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(pointcloud.pos)
                pcd_save.normals =  o3d.utility.Vector3dVector(pointcloud.x)
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
    num_points = 206
    root = Path("/home/victor/workspace/catkin_ws/Dataset/")
    save_path = Path("/home/victor/workspace/catkin_ws/dataset_resampled/")
    transform = torch_geometric.transforms.FixedPoints(num_points, allow_duplicates=False)


    process_dataset(root, save_path, transform, num_points)

    # dataset = PcdDataset(root, save_path=save_path, transform=transform)
    # print(len(dataset))
    # print(dataset[1])
    # print("~~~" * 20)

    # loader = DenseDataLoader(dataset, batch_size=8)

    # for data in loader:
    #     print(data)
    #     print(data.y)
    #     break

# /home/victor/workspace/catkin_ws/dataset_camera