import os
import sys
import numpy as np
from time import sleep
from tqdm import tqdm
import json

BASE_DIR = os.path.abspath('')
sys.path.append(BASE_DIR)
# ROOT_DIR = BASE_DIR
ROOT_DIR = os.path.join(BASE_DIR, os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class data_handler(object):
    """
    This class helps to load .txt files and save them as .npy files (much faster to load).
    
    ~~~~~~~~~~~~~~~~ CURRENTLY ONLY TESTED WITH THE MODELNET40 DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    def __init__(self, load, save, limit=100):
        """
        load    - string:   file to load
        save    - string:   file save name
        limit   - int:      how many files to load per set
        """
        self.load = load
        self.save = save
        self.limit = limit

        cat_file = os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')
        cat = [line.rstrip() for line in open(cat_file)]
        self.classes = dict(zip(cat, range(len(cat))))
        self.point_set = np.array([])
        self.class_set = np.array([])

    def load_file(self):
        load_file = os.path.join(DATA_DIR, self.load)
        shape_ids = [line.rstrip() for line in open(load_file)]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids] # this gets the objects names
        datapath = [(shape_names[i], os.path.join(DATA_DIR, shape_names[i], shape_ids[i])+'.txt') for i in range(len(shape_ids))]
        d_size = len(datapath)
        curr_limit = min(d_size, self.limit)
        
        fn1 = datapath[0]
        # print(fn1)
        point_set = np.loadtxt(fn1[1], delimiter=',').astype(np.float32)
        class_set = self.classes[fn1[0]]
        class_set = np.array([class_set]).astype(np.int32)
        class_set = np.full([point_set.shape[0], 1], class_set)
        print(point_set.shape)
        for i in tqdm(range(1, curr_limit)):
            fn = datapath[i]
            cls = self.classes[datapath[i][0]]
            cls = np.array([cls]).astype(np.int32)
            curr_file_data = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            class_set = np.append(class_set, np.full([curr_file_data.shape[0],1], cls), axis=0)
            point_set = np.append(point_set, curr_file_data, axis=0)
        print(point_set.shape)
        print(class_set.shape)
        self.class_set = class_set
        self.point_set = point_set
    
    def save_file(self):
        np.save("data_%s" % self.save, self.point_set)
        np.save("label_%s" % self.save, self.class_set)

    def prepare_data(self):
        self.load_file()
        self.save_file()
        print("\tSAVE COMPLETED!")


class segmentation_dataset(object):
    def __init__(self, root, npoints, split='train', normalize=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normalize = normalize

        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in self.cat.items()}

        self.meta = {}

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
       
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
      
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
                
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-4])
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
            
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        for cat in sorted(self.seg_classes.keys()):
            print(cat, self.seg_classes[cat])
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
    
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            normal = data[:,3:6]
            seg = data[:,-1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice,:]
        return point_set, normal, seg, cls, data[choice,0:6]
        '''
        if self.classification:
            return point_set, normal, cls
        else:
            if self.return_cls_label:
                return point_set, normal, seg, cls
            else:
                return point_set, normal, seg
        '''

def save_data(path, data_set, npoints):
    d = segmentation_dataset(root=path, split=data_set, npoints=npoints)
    _,_,seg,_,data = d[0]
    all_data = np.array(data)
    all_seg = np.array(seg)

    print("Loading %s data:" % data_set)
    for i in tqdm(range(1, len(d))):
        _,_,seg,_,data = d[i]
        all_data = np.append(all_data, data, axis=0)
        all_seg = np.append(all_seg, seg, axis=0)
    print(all_data.shape)
    print(all_seg.shape)
    
    np.save("data_%s.npy" % data_set , all_data)
    np.save("label_%s.npy" % data_set , all_seg)




if __name__ == '__main__':


    save_data(path='/home/alex/Alex_documents/RGCNN/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', data_set='test', npoints=100)
    save_data(path='/home/alex/Alex_documents/RGCNN/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', data_set='train', npoints=100)
    save_data(path='/home/alex/Alex_documents/RGCNN/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', data_set='val', npoints=100)

    '''
    dh = data_handler('modelnet40_train.txt', 'train.npy', 1000)
    dh.prepare_data()

    dh2 = data_handler('modelnet40_test.txt', 'test.npy', 1000)
    dh2.prepare_data()
    test_data = np.load('data_train.npy')
    test_label = np.load('label_train.npy')
    print(test_data.shape)
    print(test_label.shape)
    
    dh3 = data_handler('modelnet40_test.txt', 'val.npy', 2000)
    dh3.prepare_data()

    test_data = np.load('data_val.npy')
    test_label = np.load('label_val.npy')
    print(test_data.shape)
    print(test_label.shape)
    '''
    '''
    d = segmentation_dataset(root='/home/victor/workspace/thesis_ws/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='train', npoints=3000)
    ps, normal, seg, cls, data = d[0]
    
    all_data = np.array(data)
    all_seg = np.array(seg)
    print("Loading training data: \n")
    for i in tqdm(range(1, len(d))):
        _,_,seg,_,data = d[i]
        all_data = np.append(all_data, data, axis=0)
        all_seg = np.append(all_seg, seg, axis=0)
    print(all_data.shape)
    print(all_seg.shape)
    
    np.save("data_train.npy", all_data)
    np.save("label_train.npy", all_seg)

    d2 = segmentation_dataset(root='/home/victor/workspace/thesis_ws/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='test', npoints=3000)
    ps, normal, seg, cls, data = d2[0]
    
    all_data = np.array(data)
    all_seg = np.array(seg)
    print("Loading test data: \n")
    for i in tqdm(range(1, len(d2))):
        _,_,seg,_,data = d2[i]
        all_data = np.append(all_data, data, axis=0)
        all_seg = np.append(all_seg, seg, axis=0)

    np.save("data_test.npy", all_data)
    np.save("label_test.npy", all_seg)


    d3 = segmentation_dataset(root='/home/victor/workspace/thesis_ws/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='val', npoints=3000)
    ps, normal, seg, cls, data = d3[0]
    
    all_data = np.array(data)
    all_seg = np.array(seg)
    print("Loading val data: \n")
    for i in tqdm(range(1, len(d3))):
        _,_,seg,_,data = d3[i]
        all_data = np.append(all_data, data, axis=0)
        all_seg = np.append(all_seg, seg, axis=0)

    np.save("data_val.npy", all_data)
    np.save("label_val.npy", all_seg)
    '''