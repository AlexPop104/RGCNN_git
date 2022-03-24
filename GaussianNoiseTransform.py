from typing import Optional, Union

import numpy as np
import open3d as o3d
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class GaussianNoiseTransform(BaseTransform):
    def __init__(self, mu: Optional[float] = 0, sigma: Optional[float] = 0.1, recompute_normals : bool = True):
        self.mu = mu
        self.sigma = sigma
        self.recompute_normals = recompute_normals

    def __call__(self, data: Union[Data, HeteroData]):
        noise = np.random.normal(self.mu, self.sigma, data.pos.shape)
        data.pos += noise

        if self.recompute_normals:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(data.pos)
            pcd_o3d.estimate_normals(fast_normal_computation=False)
            pcd_o3d.normalize_normals()
            if hasattr(data, 'normal'):
                data.normal = np.asarray(pcd_o3d.normals)
            else:
                data.x = np.asarray(pcd_o3d.normals)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
