import numpy as np
import time, json
import os



path = '/home/alex/Alex_documents/RGCNN/models'

f1 = os.path.join(path, 'label_test.npy')
f2 = os.path.join(path, 'label_val.npy')

arr1 = np.load(f1)
arr2 = np.load(f2)

print(arr1.shape)
print(arr2.shape)

print(arr1)
print(arr2)