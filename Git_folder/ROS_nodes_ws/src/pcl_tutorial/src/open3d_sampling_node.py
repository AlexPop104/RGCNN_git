#!/home/alex/RGCNN_tensorflow/bin/python

import struct
from subprocess import call

from matplotlib.pyplot import axis
import rospy
import std_msgs.msg as msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointField
import ctypes
import os
import numpy as np
from numpy.random import default_rng
import sys
np.random.seed(0)
import open3d as o3d
import torch as t
from torch_geometric.nn import fps



counter = 0

class PointCloudPublisher:

    def __init__(self, file_name=""):

        self.point_cloud_messages = []
        self.points_lists = []
        self.index = -1
        self.i = 0
        self.header = msg.Header()
        self.header.frame_id = 'camera_depth_optical_frame'
        rng = default_rng()
        self.color = []
        for _ in range(50):
            self.color.append(rng.choice(254, size=3, replace=False).tolist())

        

        self.publishers = []
        rospy.init_node('PointCloudPublisher', anonymous=True)
        self.rate = rospy.Rate(1)
        self.fields = [PointField('x', 0,  PointField.FLOAT32, 1),
                       PointField('y', 4,  PointField.FLOAT32, 1),
                       PointField('z', 8,  PointField.FLOAT32, 1),
                       PointField('r', 12, PointField.FLOAT32, 1),
                       PointField('g', 16, PointField.FLOAT32, 1),
                       PointField('b', 20, PointField.FLOAT32, 1)
                       ]

        self.is_pc_loaded = False
        self.is_colored = False
        self.is_message_created = False
        self.points_list = []

    
    def _create_message(self):
        point_cloud_message = pcl2.create_cloud(self.header, self.fields, self.points_lists[-1])
        self.point_cloud_messages.append(point_cloud_message)

    
    def publish(self):
        try:
            while not rospy.is_shutdown():
                for i, pub in enumerate(self.publishers):
                    pub.publish(self.point_cloud_messages[i])
                    self.rate.sleep()
        except rospy.ROSInterruptException:
            exit()

    def get_pc(self):
        return self.point_clouds[self.index]

def callback(data,num_points):
    global counter 
    if counter % 15 ==0:
        counter = 0
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])

        gen = pcl2.read_points(data, skip_nans=True)

        init_data = list(gen)
        xyz = np.empty(shape=(len(init_data), 3))

        for i, x in enumerate(init_data):
            xyz[i] = np.array([x[0],x[1],x[2]])

        xyz = xyz[1:-1]
        

        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)


        points = np.asarray(pcd.points)
        points = t.tensor(points)

        pcd.estimate_normals(fast_normal_computation=False)
        pcd.orient_normals_consistent_tangent_plane(100)

        normals=np.asarray(pcd.normals)

        if len(points) < num_points:
                    alpha = 0.03
                    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        pcd, alpha)


                    #o3d.visualization.draw_geometries([pcd, rec_mesh])

                    num_points_sample = num_points

                    pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 

                    points = pcd_sampled.points
                    normals = np.asarray(pcd_sampled.normals)
        else:

                    nr_points_fps=num_points
                    nr_points=points.shape[0]

                    index_fps = fps(points, ratio=float(nr_points_fps/nr_points) , random_start=True)

                    index_fps=index_fps[0:num_points]

                    fps_points=points[index_fps]
                    fps_normals=normals[index_fps]

                    points=fps_points
                    normals = fps_normals

    test_pcd=np.concatenate((points,normals),axis=1)
    xyz = t.tensor(points)
    xyz = t.cat([xyz, t.tensor(np.asarray(normals))], dim=1)
    xyz = xyz.unsqueeze(0)

    print(test_pcd.shape)

    message = pcl2.create_cloud(header, fields, test_pcd)
    
    pub.publish(message)
    

   

def listener(num_points):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/no_floor_out", PointCloud2, callback=callback,callback_args=num_points)
    rospy.spin()


if __name__ == "__main__":
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]

    num_points =512
    header = msg.Header()
    header.frame_id = 'camera_depth_optical_frame'
    
    fields = [PointField('x', 0,  PointField.FLOAT32, 1),
                PointField('y', 4,  PointField.FLOAT32, 1),
                PointField('z', 8,  PointField.FLOAT32, 1),
                PointField('r', 12, PointField.FLOAT32, 1),
                PointField('g', 16, PointField.FLOAT32, 1),
                PointField('b', 20, PointField.FLOAT32, 1)
                ]
    color = []
    rng = default_rng()
    pub = rospy.Publisher("/Segmented_Point_Cloud", PointCloud2, queue_size=10)
    

    listener(num_points)
 


 

