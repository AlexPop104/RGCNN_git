from typing import Optional
from cv2 import threshold
from numpy import ndarray
import open3d as o3d
import numpy as np
from probreg import cpd
import probreg
import copy

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

                            
root = "/home/victor/workspace/catkin_ws/Dataset/Chair/train/"
source  = o3d.io.read_point_cloud(root+"1652277896568869.pcd")
target = o3d.io.read_point_cloud(root+"1652460790266222.pcd")

registrator = pcd_registration()
registrator.set_clouds(source, target)
registrator.draw_pcds()

print(registrator.register_pcds())
registrator.draw_pcds()



# def cpd_reg(source, target, th=None):
#     if th is None:
#         th = np.deg2rad(30.0)
#         c, s = np.cos(th), np.sin(th)
#         target.transform(np.array([ [  c,  -s, 0.0, 0.0],
#                                     [  s,   c, 0.0, 0.0],
#                                     [0.0, 0.0, 1.0, 0.0],
#                                     [0.0, 0.0, 0.0, 1.0]]))
#     # tf_param = probreg.l2dist_regs.registration_svr(source, target)
#     tf_param = probreg.filterreg.registration_filterreg(source, target)
#     print(tf_param.transformation.rot)
#     print(tf_param.transformation.t)
#     rot = tf_param.transformation.rot
#     t = tf_param.transformation.t

#     T = np.eye(4)
#     T[:3, :3] = rot
#     T[:3, 3] = t.T

#     # print(np.reshape(t_mat,[3,4]))

#     # tf_param, _, _ = cpd.registration_cpd(np.asarray(source.points), np.asarray(target.points), "rigid", w=0.5, maxiter=250)
#     result = copy.deepcopy(source)
#     # result = tf_param.transform(np.asarray(result.points))
#     draw_registration_result(source, target, T)
#     t_t = o3d.geometry.PointCloud()
#     t_t.points = o3d.utility.Vector3dVector(result)
#     source.paint_uniform_color([1, 0.706, 0])
#     t_t.paint_uniform_color([0, 0.651, 0.929])
#     o3d.visualization.draw_geometries([source, t_t])

# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])

# source.estimate_normals(fast_normal_computation=False)
# target.estimate_normals(fast_normal_computation=False)
# source.orient_normals_consistent_tangent_plane(100)
# target.orient_normals_consistent_tangent_plane(100)

# c_s = source.get_center()
# c_t = target.get_center()

# translate = np.asarray(c_s - c_t)

# translation = np.array([[1.0,0.0,0.0,translate[0]], [0.0,1.0,0.0,translate[1]], [0.0,0.0,1.0,translate[2]],[0.0,0.0,0.0,1.0]], dtype=np.float)
# translation = translation.astype(np.float)
# print(translation)
# o3d.visualization.draw_geometries([source, target])

# # target.transform(translation)

# #o3d.visualization.draw_geometries([source, target])

# source.paint_uniform_color([1, 0.706, 0])
# # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
# #                          [-0.139, 0.967, -0.215, 0.7],
# #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
# trans_init = np.asarray([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1.0] ])
# th = 0.002
# reg_p2p = o3d.pipelines.registration.registration_icp(
#     source, target, th, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
#     #, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
# print(reg_p2p)
# print(reg_p2p.transformation)

# #draw_registration_result(source, target, reg_p2p.transformation)

# cpd_reg(source, target)
