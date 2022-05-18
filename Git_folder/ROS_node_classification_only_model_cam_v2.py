#!/home/alex/RGCNN_tensorflow/bin/python


import struct
from subprocess import call

from matplotlib.pyplot import axis, text
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
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

import torch as t
from classification_model_cam import cls_model
from visualization_msgs.msg import Marker

from std_msgs.msg import ColorRGBA

label_to_names = {0: 'bag',
                  1: 'can',
                  2: 'chair',
                  3: 'headphone',
                  4: 'shoe'}

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

def callback(data, model):
    xyz = np.array([[0, 0, 0,0,0,0]])
    norm = np.array([[0, 0, 0]])

    gen = pcl2.read_points(data,field_names = ("x", "y", "z" ,"r","g","b" ) ,skip_nans=True)
    

    init_data = list(gen)
    xyz = np.empty(shape=(len(init_data), 6))

    for i, x in enumerate(init_data):
        xyz[i] = np.array([x[0],x[1],x[2],x[3],x[4],x[5]])

    xyz = xyz[0:-1]

    ####################################

    points = xyz
    
    xyz = t.tensor(xyz)
    
    xyz = xyz.unsqueeze(0)

    #print(xyz.shape)



    pred,_ = model(xyz.to(t.float32).to(device))
    labels = pred.argmax(dim=-1)
    labels = labels.to('cpu')

    text = f'{label_to_names[labels.item()]}'
    print(text + f'{xyz.shape[1]}')
    
    text_marker.text = text
    text_pub.publish(text_marker)
    #print(labels.shape)

    
    #message = pcl2.create_cloud(header, fields, points)
    message = pcl2.create_cloud(header, fields, points)
    
    pub.publish(message)
    
def listener(model):
    rospy.Subscriber("/Segmented_Point_Cloud", PointCloud2, callback=callback, callback_args=model)
    rospy.spin()


if __name__ == "__main__":
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]

    rospy.init_node('listener', anonymous=True)

    header = msg.Header()
    header.frame_id = 'camera_depth_optical_frame'

    scale = Vector3(0.4, 0.4, 0.4)
    P = Pose(Point(0,0,0), Quaternion(0,0,0,1))

    text_marker = Marker()
    text_marker.header.frame_id = '/camera_depth_optical_frame'
    text_marker.header.stamp = rospy.Time.now()
    text_marker.action = Marker().ADD
    text_marker.type = Marker().TEXT_VIEW_FACING
    text_marker.id = 0
    text_marker.lifetime = rospy.Duration(0.0)

    text_marker.ns = "Text"
    text_marker.pose = P
    text_marker.scale = scale
    color = ColorRGBA()
    color.a = 1.0
    color.r = 0.8
    color.g = 0.1
    color.b = 0.1
    text_marker.color = color
    text_marker.text = "Placeholder text"

    ################################################################################

    fields = [PointField('x', 0,  PointField.FLOAT32, 1),
                PointField('y', 4,  PointField.FLOAT32, 1),
                PointField('z', 8,  PointField.FLOAT32, 1),
                PointField('r', 12, PointField.FLOAT32, 1),
                PointField('g', 16, PointField.FLOAT32, 1),
                PointField('b', 20, PointField.FLOAT32, 1)
                ]
    color = []
    rng = default_rng()

    text_pub = rospy.Publisher("/Text", Marker, queue_size=10)
    pub = rospy.Publisher("/Final_pcd", PointCloud2, queue_size=10)
    
    for i in range(40):
        color.append(rng.choice(254, size=3, replace=False).tolist())

    ##############################################################################3333


    path_saved_model="/home/alex/Alex_documents/RGCNN_git/Git_folder/model_512_points.pt"


    device = 'cuda'

    num_points = 512
    modelnet_num = 5
    dropout=0.25

    model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, reg_prior=True)
    model.load_state_dict(t.load(path_saved_model))

    print(model.parameters)
    model.to(device)
    model.eval()

    listener(model)