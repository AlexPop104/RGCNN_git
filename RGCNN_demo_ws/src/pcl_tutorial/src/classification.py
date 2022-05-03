#!/home/victor/anaconda3/envs/thesis_env/bin/python3
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
import GaussianNoiseTransform
from classification_model import cls_model
from seg_model_rambo_v2 import seg_model

model_path = "/home/victor/workspace/catkin_ws/src/RGCNN_demo_ws/src/pcl_tutorial/models/"
model_name = "model140.pt"
model_file = model_path + model_name
device = 'cuda'

label_to_names = {0: 'airplane',
                        1: 'bathtub',
                        2: 'bed',
                        3: 'bench',
                        4: 'bookshelf',
                        5: 'bottle',
                        6: 'bowl',
                        7: 'car',
                        8: 'chair',
                        9: 'cone',
                        10: 'cup',
                        11: 'curtain',
                        12: 'desk',
                        13: 'door',
                        14: 'dresser',
                        15: 'flower_pot',
                        16: 'glass_box',
                        17: 'guitar',
                        18: 'keyboard',
                        19: 'airplane',   # lamp
                        20: 'laptop',
                        21: 'mantel',
                        22: 'monitor',
                        23: 'night_stand',
                        24: 'person',
                        25: 'piano',
                        26: 'plant',
                        27: 'radio',
                        28: 'range_hood',
                        29: 'sink',
                        30: 'sofa',
                        31: 'stairs',
                        32: 'stool',
                        33: 'table',
                        34: 'tent',
                        35: 'toilet',
                        36: 'tv_stand',
                        37: 'vase',
                        38: 'wardrobe',
                        39: 'xbox'}

class PointCloudPublisher:

    def __init__(self, file_name=""):
        # self.path = os.path.dirname(os.path.abspath(__file__))
        # self.file_name = file_name
        # self.point_clouds = np.load(os.path.join(self.path, file_name))

        self.point_cloud_messages = []
        self.points_lists = []
        self.index = -1
        self.i = 0
        self.header = msg.Header()
        self.header.frame_id = 'map'
        rng = default_rng()
        self.color = []
        for _ in range(50):
            self.color.append(rng.choice(254, size=3, replace=False).tolist())

        '''
        self.color = [[255, 255, 255],
                      [255, 102, 102],
                      [255, 255, 102],
                      [51 , 255, 51 ],
                      [51 , 255, 255],
                      [102, 102, 255],
                      [255, 102, 178]]
        '''

        self.publishers = []
        rospy.init_node('PointCloudPublisher', anonymous=True)
        self.rate = rospy.Rate(1)
        self.fields = [PointField('x', 0,  PointField.FLOAT32, 1),
                       PointField('y', 4,  PointField.FLOAT32, 1),
                       PointField('z', 8,  PointField.FLOAT32, 1),
                       PointField('r', 12, PointField.FLOAT32, 1),
                       PointField('g', 16, PointField.FLOAT32, 1),
                       PointField('b', 20, PointField.FLOAT32, 1)]

        self.is_pc_loaded = False
        self.is_colored = False
        self.is_message_created = False
        self.points_list = []

    def _load_point_cloud(self, pcd):
        self.point_clouds = pcd

    def _color_point_cloud(self, labels):
        for i in range(len(self.points_lists[-1])):
            for j in range(3):
                self.points_lists[-1][i].append(self.color[int(labels[i])][j])

    def _load_point_cloud(self):
        points_list = []
        curr_pc = self.point_clouds[self.index]
        for pc in curr_pc:
            points_list.append([pc[0]+self.translation[0], pc[1]+self.translation[1], pc[2]])
        self.points_lists.append(points_list)
    
    def _create_message(self):
        point_cloud_message = pcl2.create_cloud(self.header, self.fields, self.points_lists[-1])
        self.point_cloud_messages.append(point_cloud_message)

    def add_point_cloud(self, labels, index, translation=[0,0]):
        self.index = index
        self.i = self.i+1
        self.translation = translation
        pub = rospy.Publisher('Segmented_Point_Cloud' + str(self.index) + '_' + str(self.i), PointCloud2, queue_size=10)
        self.publishers.append(pub)
        self._load_point_cloud()
        self._color_point_cloud(labels)
        self._create_message()

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
    xyz = np.array([[0, 0, 0]])
    rgb = np.array([[0, 0, 0]])

    gen = pcl2.read_points(data, skip_nans=True)

    init_data = list(gen)

    for x in init_data:
        '''
        test = x[3]

        s = struct.pack('>f', test)
        i = struct.unpack('>l', s)[0]

        pack = ctypes.c_uint32(i).value
        
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)

        rgb = np.append(rgb, [[r,g,b]], axis = 0)

        '''
        xyz = np.append(xyz, [[x[0],x[1],x[2]]], axis = 0)


    # Only get the first 2048 points. MUST BE CHANGED AS PCD MAY HAVE LESS THAN 2048 POINTS
    # xyz = xyz[0:num_points]
    # xyz = xyz[1:-1]
    # max_x = np.absolute(xyz[:,0]).max()
    # max_y = np.absolute(xyz[:,1]).max()
    # max_z = np.absolute(xyz[:,2]).max()
    # # print(f'{max_x}, {max_y}, {max_z}')

    # xyz[:,0] = (xyz[:,0] / max_x + 0.5) 
    # xyz[:,1] = (xyz[:,1] / max_y - 0.5)
    # xyz[:,2] = (xyz[:,2] / max_z - 0.5)
    # print(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(fast_normal_computation=False)
    pcd.normalize_normals()
    points = xyz
    xyz = t.tensor(xyz)
   
    xyz = t.cat([xyz, t.tensor(np.asarray(pcd.normals))], dim=1)
    
    cat = t.tensor(1).to(device)
    xyz = xyz.unsqueeze(0)


    #if xyz.shape[1] == num_points: 
    pred,_ = model(xyz.to(t.float32).to(device))
    # labels = pred.argmax(dim=2).squeeze(0)
    labels = pred.argmax(dim=-1)
    labels = labels.to('cpu')
    # rospy.loginfo(labels.shape)
    print(f'{label_to_names[labels.item()]}, {xyz.shape[1]}')
    # print(labels.shape)

    # aux_label = np.zeros([num_points, 3])
    # for i in range(num_points):
    #     aux_label[i] = color[int(labels[i])]

    # print(aux_label) 

    #points = np.append(points, aux_label, axis=1)
    

    # print(points)

    '''
    for i in range(points.shape[0]):
        for j in range(3):
            points[i] = np.append(points[i], color[int(labels[i])][j])
    '''
    
    message = pcl2.create_cloud(header, fields, points)
    pub.publish(message)

    '''
    else: 
        
        rospy.loginfo(xyz.shape)
    '''

def listener(model):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/no_floor_out", PointCloud2, callback=callback, callback_args=model)
    rospy.spin()


if __name__ == "__main__":
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]
    num_points = 512
    header = msg.Header()
    header.frame_id = 'map'
    # fields = [PointField('x', 0,  PointField.FLOAT32, 1),
    #             PointField('y', 4,  PointField.FLOAT32, 1),
    #             PointField('z', 8,  PointField.FLOAT32, 1),
    #             PointField('r', 12, PointField.FLOAT32, 1),
    #             PointField('g', 16, PointField.FLOAT32, 1),
    #             PointField('b', 20, PointField.FLOAT32, 1)]
    fields = [PointField('x', 0,  PointField.FLOAT32, 1),
                PointField('y', 4,  PointField.FLOAT32, 1),
                PointField('z', 8,  PointField.FLOAT32, 1),
                ]
    color = []
    rng = default_rng()
    pub = rospy.Publisher("Segmented_Point_Cloud", PointCloud2, queue_size=10)
    for i in range(40):
        color.append(rng.choice(254, size=3, replace=False).tolist())

    model = cls_model(num_points, F, K, M, class_num=40)
    model.load_state_dict(t.load(model_file))
    model.to(device)
    model.eval()
    listener(model)

