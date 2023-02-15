import numpy as np

import random

arr = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]])
arr_s = np.array([[14, 15, 16], [20, 21, 22]])


x = arr == np.split(arr_s,np.shape(arr_s)[0])
res = np.sum(x, axis=0)

nr_points=200000
nr_voxel=100000

points= np.random.rand(nr_points,3)
sampled_list = np.random.choice(nr_points, nr_voxel)
points_voxel=points[sampled_list,:]

print(points.shape)
print(points_voxel.shape)

x = points == np.split(points_voxel,np.shape(points_voxel)[0])
res = np.sum(x, axis=0)
print(res)

# for i in range(points_voxel.shape[0]):

#     ceva= np.array(points_voxel[i,:])   
#     new_points= np.tile(ceva,points.shape[0])
#     new_points= np.reshape(new_points,(points.shape[0],points.shape[1]))

#     dif=points-new_points
#     dif=np.multiply(dif,dif)

#     dif=np.sum(dif,axis=1)

#     minimum =np.argmin(dif, axis=-1)

#     positions.append(minimum)


# for i in range(points_voxel.shape[0]):

#     ceva= np.array(points_voxel[i,:])   
    
#     minimum=np.where((points == ceva).all(axis=1))


#     positions.append(minimum[0])

#     print(i)
#     print(minimum[0])



print("ceva")







