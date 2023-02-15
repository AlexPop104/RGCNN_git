import numpy as np
import open3d as o3d
import os
import Passthrough_function as pf



def Rot_matrix_x(angle=0):
    return np.matrix([[1 ,0,0, 0],[0 , np.cos(angle), -np.sin(angle),0],[0, np.sin(angle), np.cos(angle) ,0],[0, 0 ,0 ,1]])

def Rot_matrix_y(angle=0):
    return np.matrix([[np.cos(angle) , 0, np.sin(angle),0],[0 ,1,0, 0],[-np.sin(angle), 0, np.cos(angle) ,0],[0, 0 ,0 ,1]])

def Rot_matrix_z(angle=0):
    return np.matrix([[np.cos(angle), -np.sin(angle),0,0],[np.sin(angle), np.cos(angle),0 ,0],[ 0,0,1, 0],[0, 0 ,0 ,1]])

# Helper parameters class containing variables that will change in the callback function
class params():
    # voxels counter that will stop the voxel mesh generation when there are no more voxels in the voxel grid

  

    centroid_x_1=-0.2874577
    centroid_y_1= -0.14024239-0.1
    centroid_z_1=1.21766081-0.05

    angle_x_1=20
    angle_y_1=10
    angle_z_1=5


    L_1=0.15
    l_1=0.15
    h_1=0.10

    counter = 0
    vox_mesh=o3d.geometry.TriangleMesh()
    out_pcd=o3d.geometry.PointCloud()

# Voxel builder callback function
def build_voxels(vis):

    print("Selection (0 for no change, 1 for change)")
    selection=int(input())

    if(selection>0):

        print("Centroid_x, Centroid_y, Centroid_z")
        params.centroid_x_1= float(input())
        params.centroid_y_1= float(input())
        params.centroid_z_1= float(input())

        print(params.centroid_x_1)
        print(params.centroid_y_1)
        print(params.centroid_z_1)

        # print("angle_x, angle_y, angle_z")
        # params.angle_x_1= float(input())
        # params.angle_y_1= float(input())
        # params.angle_z_1= float(input())

        # print(params.angle_x_1)
        # print(params.angle_y_1)
        # print(params.angle_z_1)

        # print("L, l, h")
        # params.L_1= float(input())
        # params.l_1= float(input())
        # params.h_1= float(input())

        # print(params.L_1)
        # print(params.l_1)
        # print(params.h_1)


    color_box_1=[0.2, 0.2, 0.5]
    color_pass_cloud_1=[0., 1., 1.]

    pf_filter1=o3d.geometry.PointCloud()
    box_1=o3d.geometry.PointCloud()

    ######################
    ##################
    ############33
    ##Need to pass th input point cloud

    pf_filter1,box_1=pf.Passthrough_custom(L=params.L_1,l=params.l_1,h=params.h_1,angle_x=params.angle_x_1,angle_y=params.angle_y_1,angle_z=params.angle_z_1,centroid_x=params.centroid_x_1,centroid_y=params.centroid_y_1,centroid_z=params.centroid_z_1,color_box=color_box_1,color_pass_cloud=color_pass_cloud_1,cloud=pcd)
    
    vis.add_geometry(pf_filter1)
    vis.add_geometry(box_1)
    vis.update_geometry(pf_filter1)
    vis.update_geometry(box_1)
    vis.update_renderer()



# Initialize a point cloud object
pcd = o3d.geometry.PointCloud()
path_pointcloud="/home/alex/Alex_documents/RGCNN_git/1651654246540199.pcd"
pcd = o3d.io.read_point_cloud(path_pointcloud)


# Add the points, colors and normals as Vectors
pcd.estimate_normals(fast_normal_computation=False)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
# Initialize a visualizer object
vis = o3d.visualization.Visualizer()
# Create a window, name it and scale it
vis.create_window(window_name='Bunny Visualize', width=800, height=600)
# Create a point cloud that will contain the voxel centers as points

vis.add_geometry(pcd)

# Invoke the callback function
vis.register_animation_callback(build_voxels)
# We run the visualizater
vis.run()
# Once the visualizer is closed destroy the window and clean up
vis.destroy_window()