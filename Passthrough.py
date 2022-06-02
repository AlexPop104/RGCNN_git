from cmath import sqrt
import open3d as o3d
import numpy as np

path_pointcloud="/home/alex/Alex_documents/RGCNN_git/1651654246540199.pcd"
cloud = o3d.io.read_point_cloud(path_pointcloud)

cloud.paint_uniform_color([0.5, 0.5, 0.5])

aabb = cloud.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)

#o3d.visualization.draw_geometries([cloud,aabb])

# ceva=aabb.get_print_info()
# ceva2=aabb.get_rotation_matrix_from_axis_angle

# ceva3=aabb.rotation


centroid= o3d.geometry.PointCloud.get_center(cloud)



# print("Centroid")
# print(centroid)

L=0.05
l=0.05
h=0.05

measures=[h/2,L/2,l/2]

centroid_point=o3d.geometry.PointCloud()
centroid_point.points=o3d.utility.Vector3dVector([centroid])

centroid_point.paint_uniform_color([0, 1, 0])



corners=[]

corner_1=centroid+np.multiply([-1,-1,-1],measures)
corner_2=centroid+np.multiply([-1,-1,1],measures)
corner_3=centroid+np.multiply([-1,1,1],measures)
corner_4=centroid+np.multiply([-1,1,-1],measures)
corner_5=centroid+np.multiply([1,-1,-1],measures)
corner_6=centroid+np.multiply([1,-1,1],measures)
corner_7=centroid+np.multiply([1,1,1],measures)
corner_8=centroid+np.multiply([1,1,-1],measures)

corner_list=[corner_1,corner_2,corner_3,corner_4,corner_5,corner_6,corner_7,corner_8]

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],[0,6],[1,7],[2,4],[3,5]]
colors = [[1, 0, 0] for i in range(len(lines))]



corners.append(corner_1)
corners.append(corner_2)
corners.append(corner_3)
corners.append(corner_4)
corners.append(corner_5)
corners.append(corner_6)
corners.append(corner_7)
corners.append(corner_8)


line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(np.array(corners))
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)


points = np.asarray(cloud.points)

x_range = np.logical_and(points[:,0] >=corner_1[0] ,points[:,0] <= corner_7[0])
y_range = np.logical_and(points[:,1] >= corner_1[1] ,points[:,1] <= corner_7[1])
z_range = np.logical_and(points[:,2] >= corner_1[2] ,points[:,2] <= corner_7[2])
axis=1
pass_through_filter = np.logical_and(x_range,np.logical_and(y_range,z_range))

cloud_2=o3d.geometry.PointCloud()

cloud_2.points = o3d.utility.Vector3dVector(points[pass_through_filter])

cloud_2.paint_uniform_color([0, 0, 1])

#o3d.visualization.draw_geometries([centroid_point,line_set,cloud])
#o3d.visualization.draw_geometries([centroid_point,line_set,cloud,cloud_2])




corner_points=np.asarray(corners)

ceva=np.multiply(np.ones(corner_points.shape),centroid)

corner_points=corner_points-ceva

ones_matrix=np.ones((np.asarray(corners).shape[0],1))

corners_enhanced=np.concatenate((corner_points,ones_matrix),axis=1 )
#print(corners_enhanced)
#size_corners=np.ones(np.asarray(corners).shape)

angle=30

Transf_matrix_x=np.matrix([[1 ,0,0, 0],[0 , np.cos(angle), -np.sin(angle),0],[0, np.sin(angle), np.cos(angle) ,0],[0, 0 ,0 ,1]])
Transf_matrix_y=np.matrix([[np.cos(angle) , 0, np.sin(angle),0],[0 ,1,0, 0],[-np.sin(angle), 0, np.cos(angle) ,0],[0, 0 ,0 ,1]])
Transf_matrix_z=np.matrix([[np.cos(angle), -np.sin(angle),0,0],[np.sin(angle), np.cos(angle),0 ,0],[ 0,0,1, 0],[0, 0 ,0 ,1]])

#print(Transf_matrix_x)
Transf_matrix=Transf_matrix_y

New_corners=np.dot(corners_enhanced,Transf_matrix)[:,0:3]
New_corners=New_corners+ceva
#print(New_corners)

lines_2 = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],[0,6],[1,7],[2,4],[3,5]]
colors_2 = [[0.2, 0.2, 0.5] for i in range(len(lines_2))]


line_set_2 = o3d.geometry.LineSet()
line_set_2.points = o3d.utility.Vector3dVector(New_corners)
line_set_2.lines = o3d.utility.Vector2iVector(lines_2)
line_set_2.colors = o3d.utility.Vector3dVector(colors_2)



# print("Maximum Minimum")
# print(Maximum)
# print(Minimum)

#lines_3 = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],[0,6],[1,7],[2,4],[3,5]]
lines_3 = [[4, 0],[4,1]]
colors_3 = [[1, 0., 0.] for i in range(len(lines_3))]


line_set_3 = o3d.geometry.LineSet()
line_set_3.points = o3d.utility.Vector3dVector(New_corners)
line_set_3.lines = o3d.utility.Vector2iVector(lines_3)
line_set_3.colors = o3d.utility.Vector3dVector(colors_3)

o3d.visualization.draw_geometries([line_set_2,line_set_3])


plane=np.zeros((6,4))

###################Plane 1

# line1=corner_2-corner_1
# line2=corner_2-corner_3

line1=New_corners[1]-New_corners[0]
line2=New_corners[1]-New_corners[2]


normals_plane=np.cross(line1,line2)

verification1=np.dot(normals_plane,line1.T)
verification2=np.dot(normals_plane,line2.T)

print(verification1)
print(verification2)

#d=-np.dot(normals_plane,corner_2)

d=-np.dot(normals_plane,New_corners[1].T)


plane[0,0:3]=normals_plane
plane[0,3]=d

print(plane)

distance_1=np.abs(np.dot(normals_plane,centroid)+d)/ np.sqrt(np.dot(normals_plane,normals_plane.T))

###################Plane 2

# line1=corner_6-corner_5
# line2=corner_6-corner_7

line1=New_corners[5]-New_corners[4]
line2=New_corners[5]-New_corners[6]

normals_plane=np.cross(line1,line2)

verification1=np.dot(normals_plane,line1.T)
verification2=np.dot(normals_plane,line2.T)

print(verification1)
print(verification2)

#d=-np.dot(normals_plane,corner_6)

d=-np.dot(normals_plane,New_corners[5].T)

plane[1,0:3]=normals_plane
plane[1,3]=d

print(plane)

distance_2=np.abs(np.dot(normals_plane,centroid)+d)/ np.sqrt(np.dot(normals_plane,normals_plane.T))

###################Plane 3

# line1=corner_5-corner_1
# line2=corner_5-corner_2

line1=New_corners[4]-New_corners[0]
line2=New_corners[4]-New_corners[1]

normals_plane=np.cross(line1,line2)

verification1=np.dot(normals_plane,line1.T)
verification2=np.dot(normals_plane,line2.T)

print(verification1)
print(verification2)

#d=-np.dot(normals_plane,corner_5)

d=-np.dot(normals_plane,New_corners[4].T)

plane[2,0:3]=normals_plane
plane[2,3]=d

print(plane)

distance_3=np.abs(np.dot(normals_plane,centroid)+d)/ np.sqrt(np.dot(normals_plane,normals_plane.T))
###################Plane 4

line1=corner_7-corner_3
line2=corner_7-corner_8

line1=New_corners[6]-New_corners[2]
line2=New_corners[6]-New_corners[7]


normals_plane=np.cross(line1,line2)

verification1=np.dot(normals_plane,line1.T)
verification2=np.dot(normals_plane,line2.T)

print(verification1)
print(verification2)

#d=-np.dot(normals_plane,corner_7)

d=-np.dot(normals_plane,New_corners[6].T)

plane[3,0:3]=normals_plane
plane[3,3]=d

print(plane)

distance_4=np.abs(np.dot(normals_plane,centroid)+d)/ np.sqrt(np.dot(normals_plane,normals_plane.T))
###################Plane 5

line1=corner_5-corner_1
line2=corner_5-corner_8

line1=New_corners[4]-New_corners[0]
line2=New_corners[4]-New_corners[7]


normals_plane=np.cross(line1,line2)

verification1=np.dot(normals_plane,line1.T)
verification2=np.dot(normals_plane,line2.T)

print(verification1)
print(verification2)

#d=-np.dot(normals_plane,corner_5)

d=-np.dot(normals_plane,New_corners[4].T)

plane[4,0:3]=normals_plane
plane[4,3]=d

print(plane)

distance_5=np.abs(np.dot(normals_plane,centroid)+d)/ np.sqrt(np.dot(normals_plane,normals_plane.T))
###################Plane 6

line1=corner_6-corner_2
line2=corner_6-corner_3


line1=New_corners[5]-New_corners[1]
line2=New_corners[5]-New_corners[2]


normals_plane=np.cross(line1,line2)

verification1=np.dot(normals_plane,line1.T)
verification2=np.dot(normals_plane,line2.T)

print(verification1)
print(verification2)

#d=-np.dot(normals_plane,corner_6)

d=-np.dot(normals_plane,New_corners[5].T)

plane[5,0:3]=normals_plane
plane[5,3]=d

print(plane)

distance_6=np.abs(np.dot(normals_plane,centroid)+d)/ np.sqrt(np.dot(normals_plane,normals_plane.T))



print(distance_1+distance_2+distance_3+distance_4+distance_5+distance_6)
print(L+l+h)


points_range=np.zeros(points.shape[0])



for i in range(6):
    points_range =points_range+ np.divide(np.abs(np.dot(points,plane[i,0:3].T)+plane[i,3]),np.sqrt(np.dot(plane[i,0:3],plane[i,0:3].T)))
    print(points_range)

points_range=points_range<=(L+l+h)

cloud_3=o3d.geometry.PointCloud()

cloud_3.points = o3d.utility.Vector3dVector(points[points_range])

cloud_3.paint_uniform_color([0, 1, 1])

o3d.visualization.draw_geometries([centroid_point,line_set_2,cloud,cloud_3])