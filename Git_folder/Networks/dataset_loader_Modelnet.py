import os
import numpy as np
import itertools
import math, random
random.seed = 42

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from path import Path
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import open3d as o3d
from tqdm import tqdm



def read_off(file):
    ceva=file.readline().strip()
    if 'OFF' != ceva:
        #print('Not a valid OFF header')
        ceva=ceva.replace("OFF","")
        n_verts, n_faces, __ = tuple([int(s) for s in ceva.split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=50, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode='immediate'
                        )]
                    )
                ])]
        ),
        frames=frames
    )

    return fig


def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

# with open(path/"sofa/train/sofa_0002.off", 'r') as f:
#     verts, faces = read_off(f)
    
# i,j,k = np.array(faces).T
# x,y,z = np.array(verts).T
# print(len(x))

# visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='yellowgreen', opacity=0.50, i=i,j=j,k=k)]).show()
# visualize_rotate([go.Scatter3d(x=x, y=y, z=z, mode='markers')]).show()

# pcshow(x,y,z)

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points

# pointcloud = PointSampler(3000)((verts, faces))
# pcshow(*pointcloud.T)

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud


# norm_pointcloud = Normalize()(pointcloud)
# #pcshow(*norm_pointcloud.T)

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

# rot_pointcloud = RandRotation_z()(norm_pointcloud)
# noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
#pcshow(*noisy_rot_pointcloud.T)

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms(),save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.save_path = None
        self.files = []
        self.folder=folder

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
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        if self.save_path is not None:
            splits = f.name.strip().split("/")
            cat = splits[len(splits) - 3]
            name = splits[len(splits) - 1]

            name = name.replace('off','pcd')
            total_path = self.save_path/cat/self.folder/name
            pcd_save = o3d.geometry.PointCloud()
            pcd_save.points = o3d.utility.Vector3dVector(np.asarray(pointcloud))
            o3d.io.write_point_cloud(str(total_path), pcd_save, write_ascii=True)
        return {'pointcloud': pointcloud, 
                'category': self.classes[category]}




# train_ds = PointCloudData(path, transform=train_transforms,save_path=save_path)
# valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms,save_path=save_path)


def process_dataset(root, save_path, num_points=512):


    train_transforms = transforms.Compose([
                    PointSampler(num_points),
                    Normalize(),
                    ToTensor()
                    ])

    dataset_train = PointCloudData(root_dir=root, folder="train", transform=train_transforms, save_path=save_path)
    dataset_test =  PointCloudData(root_dir=root, folder="test", transform=train_transforms, save_path=save_path)

    print("Processing train data: ")
    for i in tqdm(range(len(dataset_train))):
        _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]


modelnet_num = 40
root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet"+str(modelnet_num)+"/raw"

path = Path(root)

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};
print(classes)
                

save_path = Path("/media/rambo/ssd2/Alex_data/RGCNN/PCD_DATA/Modelnet"+str(modelnet_num)+"/")
num_points = 2048
process_dataset(root=path, save_path=save_path,  num_points=num_points)




