from nis import cat
from unicodedata import category
import numpy as np
import os

import torch
# from torch.utils.data import DataLoader
from torchvision import transforms, utils

from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.data.dataset import Dataset
from path import Path
import scipy.spatial.distance

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch_geometric.transforms

import open3d as o3d

def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])

class PcdDataset(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms(), with_normals=True):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        self.with_normals = with_normals
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
        #o3d.visualization.draw_geometries([pcd])

        points = np.asarray(pcd.points)
        points = torch.tensor(points)

        normals = []

        if self.with_normals == True:
            pcd.estimate_normals(fast_normal_computation=False)
            pcd.normalize_normals()
            pcd.orient_normals_consistent_tangent_plane(100)

            # o3d.visualization.draw_geometries([pcd])

        print(len(points))
        if len(points) < 2048:
            radii = [0.005, 0.01, 0.02, 0.04]
            alpha = 0.03
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha)


            #o3d.visualization.draw_geometries([pcd, rec_mesh])

            num_points_sample = 2048

            pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 
            points = pcd_sampled.points
        
            normals = np.asarray(pcd_sampled.normals)
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

        return pointcloud


if __name__ == '__main__':
    root = Path("/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/Dataset_camera/")
   
    transform = torch_geometric.transforms.FixedPoints(2048, allow_duplicates=False)

    dataset = PcdDataset(root, transform=transform)
    print(len(dataset))
    print(dataset[0])
    print("~~~" * 20)

    loader = DenseDataLoader(dataset, batch_size=8)

    for data in loader:
        print(data)
        print(data.y)
        break

# /home/victor/workspace/catkin_ws/dataset_camera