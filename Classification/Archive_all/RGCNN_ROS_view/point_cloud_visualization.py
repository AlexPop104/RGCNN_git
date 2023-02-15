#!/usr/bin/env python
import rospy
import std_msgs.msg as msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointField

import os
import numpy as np
from numpy.random import default_rng

class PointCloudPublisher:
    def __init__(self, file_name):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.file_name = file_name
        self.point_clouds = np.load(os.path.join(self.path, file_name))

        self.point_cloud_messages = []
        self.points_lists = []
        self.index = -1
        self.i = 0
        self.header = msg.Header()
        self.header.frame_id = 'map'
        rng = default_rng()
        self.color = []
        for i in range(49):
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

    def _color_point_cloud(self, labels):
        for i in range(len(self.points_lists[-1])):
            for j in range(3):
                self.points_lists[-1][i].append(self.color[int(labels[i])][j])

    def _load_point_cloud(self):
        points_list = []
        curr_pc = self.point_clouds[self.index]
        for pc in curr_pc:
            points_list.append([pc[0], pc[1], pc[2]])
        self.points_lists.append(points_list)
    
    def _create_message(self):
        point_cloud_message = pcl2.create_cloud(self.header, self.fields, self.points_lists[-1])
        self.point_cloud_messages.append(point_cloud_message)

    def add_point_cloud(self, labels, index):
        self.index = index
        pub = rospy.Publisher('Segmented_Point_Cloud' + str(self.index), PointCloud2, queue_size=10)
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

if __name__=='__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    file_name = 'seg_test_seg.npy'
    labels_test = np.load(os.path.join(path, file_name))
    print(np.amax(labels_test))
    file_name = 'seg_test_pos.npy'
    pcp    = PointCloudPublisher(file_name=file_name)
    labels = np.random.randint(low=0, high=6, size=2048)

    pcp.add_point_cloud(labels_test[0].tolist(), 0)
    pcp.add_point_cloud(labels_test[1].tolist(), 1)
    pcp.add_point_cloud(labels_test[2].tolist(), 2)
    pcp.publish()