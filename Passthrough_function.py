from cmath import sqrt
import open3d as o3d
import numpy as np



def Rot_matrix_x(angle=0):
    return np.matrix([[1 ,0,0, 0],[0 , np.cos(angle), -np.sin(angle),0],[0, np.sin(angle), np.cos(angle) ,0],[0, 0 ,0 ,1]])

def Rot_matrix_y(angle=0):
    return np.matrix([[np.cos(angle) , 0, np.sin(angle),0],[0 ,1,0, 0],[-np.sin(angle), 0, np.cos(angle) ,0],[0, 0 ,0 ,1]])

def Rot_matrix_z(angle=0):
    return np.matrix([[np.cos(angle), -np.sin(angle),0,0],[np.sin(angle), np.cos(angle),0 ,0],[ 0,0,1, 0],[0, 0 ,0 ,1]])


def Passthrough_custom(L=1,l=1,h=1,angle_x=0,angle_y=0,angle_z=0,centroid_x=1,centroid_y=1,centroid_z=1,color_box=[0.2, 0.2, 0.5],color_pass_cloud=[0., 1., 1.],cloud=o3d.geometry.PointCloud()):
    centroid=np.asarray([centroid_x,  centroid_y , centroid_z])

    centroid_point=o3d.geometry.PointCloud()
    centroid_point.points=o3d.utility.Vector3dVector([centroid])

    centroid_point.paint_uniform_color([0, 1, 0])

    Transf_matrix=np.dot(np.dot(Rot_matrix_x(angle=angle_x),Rot_matrix_y(angle=angle_y)),Rot_matrix_z(angle=angle_z))


    measures=[h/2,L/2,l/2]




    corners=[]

    corner_1=centroid+np.multiply([-1,-1,-1],measures)
    corner_2=centroid+np.multiply([-1,-1,1],measures)
    corner_3=centroid+np.multiply([-1,1,1],measures)
    corner_4=centroid+np.multiply([-1,1,-1],measures)
    corner_5=centroid+np.multiply([1,-1,-1],measures)
    corner_6=centroid+np.multiply([1,-1,1],measures)
    corner_7=centroid+np.multiply([1,1,1],measures)
    corner_8=centroid+np.multiply([1,1,-1],measures)

    corners.append(corner_1)
    corners.append(corner_2)
    corners.append(corner_3)
    corners.append(corner_4)
    corners.append(corner_5)
    corners.append(corner_6)
    corners.append(corner_7)
    corners.append(corner_8)


    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],[0,6],[1,7],[2,4],[3,5]]
    colors = [[1, 0, 0] for i in range(len(lines))]

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



    New_corners=np.dot(corners_enhanced,Transf_matrix)[:,0:3]
    New_corners=New_corners+ceva
    #print(New_corners)

    #color_box=[0.2, 0.2, 0.5]

    lines_2 = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],[0,6],[1,7],[2,4],[3,5]]
    colors_2 = [color_box for i in range(len(lines_2))]


    line_set_2 = o3d.geometry.LineSet()
    line_set_2.points = o3d.utility.Vector3dVector(New_corners)
    line_set_2.lines = o3d.utility.Vector2iVector(lines_2)
    line_set_2.colors = o3d.utility.Vector3dVector(colors_2)

    ##############################################################################
    ##Vizualizing lines in chosen planes

    #lines_3 = [[0, 1], [1, 2], [2, 3], [3, 0], [4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],[0,6],[1,7],[2,4],[3,5]]  ### Previous lines


    # lines_3 = [[4, 0],[4,1]]  # New selected lines
    # colors_3 = [[1, 0., 0.] for i in range(len(lines_3))]


    # line_set_3 = o3d.geometry.LineSet()
    # line_set_3.points = o3d.utility.Vector3dVector(New_corners)
    # line_set_3.lines = o3d.utility.Vector2iVector(lines_3)
    # line_set_3.colors = o3d.utility.Vector3dVector(colors_3)

    # o3d.visualization.draw_geometries([line_set_2,line_set_3])

    ##################################3
    ##Computing all planes using 3 chosen points


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

    points_range_final=points_range<=(L+l+h)

    cloud_3=o3d.geometry.PointCloud()

    cloud_3.points = o3d.utility.Vector3dVector(points[points_range_final])

    #color_pass_cloud=[0., 1., 1.]

    cloud_3.paint_uniform_color(color_pass_cloud)

    #o3d.visualization.draw_geometries([centroid_point,line_set_2,cloud,cloud_3])

    

    return cloud_3,line_set_2



path_pointcloud="/home/alex/Alex_documents/RGCNN_git/1651654246540199.pcd"
cloud = o3d.io.read_point_cloud(path_pointcloud)


cloud.paint_uniform_color([0.5, 0.5, 0.5])

centroid= o3d.geometry.PointCloud.get_center(cloud)


print("Centroid")
print(centroid)

L_1=0.15
l_1=0.15
h_1=0.10

angle_x_1=20
angle_y_1=10
angle_z_1=5

centroid_x_1=-0.2874577
centroid_y_1= -0.14024239-0.1
centroid_z_1=1.21766081-0.05

color_box_1=[0.2, 0.2, 0.5]
color_pass_cloud_1=[0., 1., 1.]


L_2=0.15
l_2=0.15
h_2=0.10

angle_x_2=20
angle_y_2=10
angle_z_2=5

centroid_x_2=-0.2874577
centroid_y_2= -0.14024239+0.1
centroid_z_2=1.21766081-0.05

color_box_2=[0.45, 0.1, 0.9]
color_pass_cloud_2=[1, 0.3, 0.5]



pf_filter1=o3d.geometry.PointCloud()
pf_filter2=o3d.geometry.PointCloud()
pf_filter3=o3d.geometry.PointCloud()

box_1=o3d.geometry.PointCloud()
box_2=o3d.geometry.PointCloud()
box_3=o3d.geometry.PointCloud()


pf_filter1,box_1=Passthrough_custom(L=L_1,l=l_1,h=h_1,angle_x=angle_x_1,angle_y=angle_y_1,angle_z=angle_z_1,centroid_x=centroid_x_1,centroid_y=centroid_y_1,centroid_z=centroid_z_1,color_box=color_box_1,color_pass_cloud=color_pass_cloud_1,cloud=cloud)
pf_filter2,box_2=Passthrough_custom(L=L_2,l=l_2,h=h_2,angle_x=angle_x_2,angle_y=angle_y_2,angle_z=angle_z_2,centroid_x=centroid_x_2,centroid_y=centroid_y_2,centroid_z=centroid_z_2,color_box=color_box_2,color_pass_cloud=color_pass_cloud_2,cloud=cloud)
o3d.visualization.draw_geometries([cloud,box_1,pf_filter1,box_2,pf_filter2])