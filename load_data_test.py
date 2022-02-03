import os
import sys
import numpy as np
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath('')
sys.path.append(BASE_DIR)
# ROOT_DIR = BASE_DIR
ROOT_DIR = os.path.join(BASE_DIR, os.pardir)
print(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')

train_file_name = 'modelnet40_train.txt'
train_file = os.path.join(DATA_DIR, train_file_name)

shape_ids = {}
shape_ids['train'] = [line.rstrip() for line in open(train_file)] # this gets the filenames of the objects
# print(shape_ids['train'])
split = 'train'
shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]] # this gets the objects names
# print(shape_names)

# get all the classes
cat_file = os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')
cat = [line.rstrip() for line in open(cat_file)]
# print(cat[0:10])

# Put the clasees into a 'dict'?
classes = dict(zip(cat, range(len(cat))))  
# print(classes)

# get the label and path to the obj. Ex.: [('airplane', 'home/.../airplane/airplane_0001.txt')]
datapath = [(shape_names[i], os.path.join(DATA_DIR, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]
# print(datapath[0][0])

d_size = len(datapath)
print(d_size)

fn1 = datapath[0]

point_set = np.loadtxt(fn1[1], delimiter=',').astype(np.float32)
print(point_set.shape)
for i in tqdm(range(1, 100)):
    fn = datapath[i]
    cls = classes[datapath[0][0]]
    cls = np.array([cls]).astype(np.int32)
    point_set = np.append(point_set, np.loadtxt(fn[1], delimiter=',').astype(np.float32), axis=0)

print(point_set.shape)

plt.plot(point_set[0:100][0:2])
plt.ylabel('some numbers')
plt.show()

np.save('train_data_test.npy', point_set)


loaded_point_set = np.load('train_data_test.npy')
plt.plot(loaded_point_set[0:100][0:2])
plt.ylabel('some numbers')
plt.show()