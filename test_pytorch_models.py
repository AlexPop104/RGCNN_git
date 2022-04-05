from collections import defaultdict
from unittest import loader

from cls_model_rambo import cls_model
from seg_model_rambo_v2 import seg_model
import torch as t
import numpy as np
from GaussianNoiseTransform import GaussianNoiseTransform

from torch.nn.functional import one_hot

from torch_geometric.datasets import ModelNet
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import Compose
from torch_geometric.loader import DenseDataLoader
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import FixedPoints

from torch_geometric.transforms import NormalizeScale
t.manual_seed(0)

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                    'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                    'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                    'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                    'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


label_to_cat = {}
for key in seg_classes.keys():
    for label in seg_classes[key]:
        label_to_cat[label] = key
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

# PATH = "/home/victor/workspace/thesis_ws/github/RGCNN_git/models/17_03_22_22:00:42/2048p_normal_model50.pt"
# PATH = "/home/victor/workspace/thesis_ws/github/RGCNN_git/models/ModelNet/18_03_22_11:50:40/1024_40_chebconv_model50.pt"
        

position_global_original = np.zeros((1, 2048, 3))
label_original_original = np.zeros((1, 2048))
label_predicted_original = np.zeros((1, 2048))


position_global_noisy = np.zeros((1, 2048, 3))
label_original_noisy = np.zeros((1, 2048))
label_predicted_noisy = np.zeros((1, 2048))


def test_modelnet_model(PATH, num_points=1024, batch_size=2, modelnet_num=40, dropout=1, one_layer=False, reg_prior=True, mu=0, sigma=0.01):

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    transforms_original = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])
    transforms_noisy    = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale(), GaussianNoiseTransform(mu, sigma)])

    root = "/media/rambo/ssd2/Alex_data/RGCNN/ModelNet" + str(modelnet_num)

    dataset_original = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms_original)
    dataset_noisy    = ModelNet(root=root, name=str(modelnet_num), train=False, transform=transforms_noisy)

    loader_original  = DenseDataLoader(dataset_original, batch_size=batch_size, shuffle=True, pin_memory=True)
    loader_noisy     = DenseDataLoader(dataset_noisy, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = cls_model(num_points, [0],[0],[0], modelnet_num, dropout=dropout, one_layer=one_layer, reg_prior=reg_prior)
    model = model.to(device)
    model.load_state_dict(t.load(PATH))
    model.eval()

    print("Testing on " + str(device))

    @t.no_grad()
    def test_model(loader, model, noisy=True):
        total_correct = 0
        
        first = False
        for data in loader:         
            x = t.cat([data.pos, data.normal], dim=2)
            x = x.to(device)
            print(x.shape)
            logits, _ = model(x)
            pred = logits.argmax(dim=-1)
            total_correct += int((pred==data.y.to(device)).sum())

        return total_correct / len(loader.dataset)

    accuracy_original = test_model(loader=loader_original, model=model)
    # accuracy_noisy = test_model(loader=loader_noisy, model=model)

    print(f"Original:   {accuracy_original}%")
    # print(f"Noisy:      {accuracy_noisy}%")

def test_shapenet_model(PATH, num_points=2048, batch_size=2, input_dim=22, dropout=1, one_layer=False, reg_prior=False, mu=0, sigma=0.1):
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    root = "/media/rambo/ssd2/Alex_data/RGCNN/ShapeNet/"
    
    transforms_original = Compose([FixedPoints(num_points), GaussianNoiseTransform(0, 0, recompute_normals=True)])
    transforms_noisy    = Compose([FixedPoints(num_points), GaussianNoiseTransform(mu, sigma, recompute_normals=True)])

    dataset_original = ShapeNet(root=root, split="test", transform=transforms_original)
    dataset_noisy    = ShapeNet(root=root, split="test", transform=transforms_noisy)

    loader_original  = DenseDataLoader(dataset_original,    batch_size=batch_size, shuffle=False, pin_memory=True)
    loader_noisy     = DenseDataLoader(dataset_noisy,       batch_size=batch_size, shuffle=False, pin_memory=True)

    model = seg_model(num_points, [0,0], [0,0], [0,0], input_dim, dropout=1, reg_prior=True, relus=[128, 512, 1024, 512, 128, 50], b2relu=True)
    model.load_state_dict(t.load(PATH))
    print(model.state_dict)
    model.to(device)
    model.eval()

    @t.no_grad()
    def test_model(loader, model, noisy):        
        size = len(loader.dataset)
        predictions = np.empty((size, num_points))
        labels = np.empty((size, num_points))
        total_correct = 0
        indexes = [3, 100, 1000]
        for i, data in enumerate(loader):         
            cat = data.category
            x = t.cat([data.pos.type(t.float32), data.x.type(t.float32)], dim=2)  ### Pass this to the model
            y = data.y
            logits, _, _ = model(x.to(device), cat.to(device))
            logits = logits.to('cpu')
            pred = logits.argmax(dim=2)
            # print(pred)
            # print(f"TEST: {int((pred == data.y.to(device)).sum())}")
            if i in indexes:
                if noisy == False:
                    global position_global_original, label_original_original, label_predicted_original 
                    position_global_original = np.append(position_global_original, data.pos, axis=0)
                    label_original_original = np.append(label_original_original, y, axis = 0)
                    label_predicted_original = np.append(label_predicted_original, pred, axis=0)
                else:
                    global position_global_noisy, label_original_noisy, label_predicted_noisy 
                    position_global_noisy= np.append(position_global_noisy, data.pos, axis=0)
                    label_original_noisy = np.append(label_original_noisy, y, axis = 0)
                    label_predicted_noisy = np.append(label_predicted_noisy, pred, axis=0)            
            
            total_correct += int((pred == y).sum())
            start = i * batch_size
            stop  = start + batch_size
            predictions[start:stop] = pred
            lab = data.y
            labels[start:stop] = lab.reshape([-1, num_points])
        tot_iou = []
        cat_iou = defaultdict(list)
        for i in range(predictions.shape[0]):
            segp = predictions[i, :]
            segl = labels[i, :]
            cat = label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            cat_iou[cat].append(np.mean(part_ious))
            tot_iou.append(np.mean(part_ious))

        ncorrects = np.sum(predictions == labels)
        accuracy  = ncorrects * 100 / (len(loader.dataset) * num_points)
        # print(f"\tAccuracy: {accuracy}, ncorrect: {ncorrects} / {len(dataset_test) * num_points}")
        # print(f"\tIoU: \t{np.mean(tot_iou)*100}")
        return accuracy, cat_iou, tot_iou, ncorrects

    acc_o, cat_iou_o, tot_iou_o, ncorrects_o = test_model(loader_original, model, False)
    acc_n, cat_iou_n, tot_iou_n, ncorrects_n = test_model(loader_noisy, model, True)
    
    print(f"Original acc:   {acc_o}% - ncorrect: {ncorrects_o} : {len(loader_original.dataset) * num_points}")
    for key, value in cat_iou_o.items():
                print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
    print(f"Total IoU: { np.mean(tot_iou_o) * 100}")
    
    print("~~~" *20)

    print(f"Noisy acc:      {acc_n}% - ncorrect: {ncorrects_n} : {len(loader_noisy.dataset) * num_points}")
    for key, value in cat_iou_n.items():
                print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
    print(f"Total IoU: { np.mean(tot_iou_n) * 100}")
    

if __name__ == '__main__':
    # PATH = '/home/victor/workspace/thesis_ws/github/RGCNN_git/models/ModelNet/18_03_22_19:04:50/1024_40_chebconv_model50.pt'
    # test_modelnet_model(PATH, mu=0, sigma=0.01)
    
    PATH = '/home/victor/workspace/thesis_ws/github/RGCNN_git/models/30_03_22_10:41:18/2048p_model_v2100.pt'
    mu = 0
    sigma = 0.05
    test_shapenet_model(PATH, num_points=2048, mu=mu, sigma=sigma)
    print(f"Mu:     {mu}")
    print(f"Sigma:  {sigma}")

    
    root = '/home/victor/workspace/thesis_ws/github/RGCNN_git/'
    np.save(root + "positions_noisy.npy", position_global_noisy)
    np.save(root + "label_original_noisy.npy", label_original_noisy)
    np.save(root + "label_predicted_noisy.npy", label_predicted_noisy)

    np.save(root + "positions_original.npy", position_global_original)
    np.save(root + "label_original_original.npy", label_original_original)
    np.save(root + "label_predicted_original.npy", label_predicted_original)
    