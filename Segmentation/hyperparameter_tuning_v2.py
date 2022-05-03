from collections import defaultdict
import sys

sys.path.append('/home/alex/Alex_documents/RGCNN_git')

from torch_geometric.transforms import FixedPoints
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DenseDataLoader
from torch.nn.functional import one_hot, relu
from torch import float32
import torch as t
import torch
import time
import os
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose
from utils import GaussianNoiseTransform
from utils import compute_loss
from utils import get_weights

from datetime import datetime

from seg_model_rambo_v2 import seg_model

import ray
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

torch.manual_seed(0)

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
               'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
               'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
               'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

label_to_cat = {}
for key in seg_classes.keys():
    for label in seg_classes[key]:
        label_to_cat[label] = key

device = 'cuda' if t.cuda.is_available() else 'cpu'


def fit(model, optimizer, loader, criterion, regularization):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        cat = data.category
        y = data.y.type(torch.LongTensor)
        x = t.cat([data.pos.type(torch.float32),
                  data.x.type(torch.float32)], dim=2)
        # out, L are for regularization
        logits, out, L = model(x.to(device), cat.to(device))
        logits = logits.permute([0, 2, 1])

        loss = compute_loss(logits, y, out, L, criterion, s=regularization)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if i % 100 == 0:
        #     print(f"{i}: curr loss: {loss}")
    return total_loss * loader.batch_size / len(loader.dataset)


def test(model, loader):
    model.eval()
    size = len(loader.dataset)
    predictions = np.empty((size, model.vertice))
    labels = np.empty((size, model.vertice))
    total_correct = 0

    for i, data in enumerate(loader):
        cat = data.category
        x = t.cat([data.pos.type(torch.float32),
                  data.x.type(torch.float32)], dim=2)
        y = data.y
        logits, _, _ = model(x.to(device), cat.to(device))
        logits = logits.to('cpu')
        pred = logits.argmax(dim=2)

        total_correct += int((pred == y).sum())
        start = i * loader.batch_size
        stop = start + loader.batch_size
        predictions[start:stop] = pred
        lab = data.y
        labels[start:stop] = lab.reshape([-1, model.vertice])

    tot_iou = []
    cat_iou = defaultdict(list)
    for i in range(predictions.shape[0]):
        segp = predictions[i, :]
        segl = labels[i, :]
        cat = label_to_cat[segl[0]]
        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

        for l in seg_classes[cat]:
            # part is not present, no prediction as well
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                part_ious[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        cat_iou[cat].append(np.mean(part_ious))
        tot_iou.append(np.mean(part_ious))

    ncorrects = np.sum(predictions == labels)
    accuracy = ncorrects * 100 / (len(loader.dataset) * model.vertice)

    return accuracy, cat_iou, tot_iou, ncorrects


def train(config, checkpoint_dir=None):
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S_%f")
    parent_directory = "/home/alex/Alex_documents/RGCNN_git/data/logs/Trained_Models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    #root = "/home/alex/Alex_documents/RGCNN_git/data/Datasets/ShapeNet/"
    root = "/mnt/ssd1/Alex_data/RGCNN/ShapeNet/"
    num_points = 1024
    batch_size = 4
    num_epochs = 80
    weight_decay = 1e-9  # 1e-9
    decay_rate = 0.8

    writer = SummaryWriter(log_dir="/home/alex/Alex_documents/RGCNN_git/data/hyperparameter_tuning_runs/tests/3/" + 'seg_'+str(num_points) +
                            '_'+str(config['dropout'])+'_'+str(config['learning_rate'])+'_'+str(config['regularization']), filename_suffix='_reg')

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]
    
    transforms = Compose([FixedPoints(num_points), GaussianNoiseTransform(
        mu=0, sigma=0, recompute_normals=False)])

    dataset_train = ShapeNet(root=root, split="train", transform=transforms)
    dataset_test  = ShapeNet(root=root, split="test", transform=transforms)
    
    train_loader = DenseDataLoader(
        dataset_train, batch_size=batch_size,
        shuffle=True, pin_memory=False)

    test_loader = DenseDataLoader(
        dataset_test, batch_size=batch_size,
        shuffle=True)


    decay_steps   = len(dataset_train) / batch_size
    weights       = get_weights(dataset_train, num_points)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=float32).to('cuda'))

    model = seg_model(num_points, F, K, M,
                      dropout=config['dropout'],
                      one_layer=False,
                      reg_prior=True,
                      recompute_L=True,
                      b2relu=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
        weight_decay=weight_decay)

    my_lr_scheduler = lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay_rate)
    
    for epoch in range(1, num_epochs + 1):    
        start_time = time.time()
        loss = fit(model, optimizer, train_loader, criterion, regularization=config['regularization'])
        end_time = time.time()
        train_time = end_time - start_time
        
        writer.add_scalar('loss/train', loss, epoch)

        start_time = time.time()
        acc, cat_iou, tot_iou, ncorrect = test(model, test_loader)
        end_time = time.time()
        test_time = end_time - start_time

        for key, value in cat_iou.items():
            # print(
            #     key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
            writer.add_scalar(key + '/test', np.mean(value), epoch)

        writer.add_scalar("IoU/test", np.mean(tot_iou) * 100, epoch)
        writer.add_scalar("accuracy/test", acc, epoch)

        my_lr_scheduler.step()

        tune.report(epoch=epoch, tot_iou=np.mean(tot_iou), acc=acc, ncorrect=ncorrect, train_loss=loss)
        
        if epoch % 5 == 0:
             torch.save(model.state_dict(), path + '/' +
                       str(num_points) + 'p_model_v2_' + str(epoch) + '.pt')


if __name__ == "__main__":
    config = {
        "dropout": tune.uniform(0, 0.5),               # 0.25
        "learning_rate": tune.uniform(1e-4, 1e-2),     # 1e-3
        "regularization": tune.uniform(1e-10, 1e-8),   # 1e-9
    }

    hyperopt_search = BayesOptSearch(config, metric="tot_iou", mode="max")


    #ray.init(num_gpus=3)

    analysis = tune.run(train, num_samples=100, search_alg=hyperopt_search, scheduler=HyperBandScheduler(time_attr='epoch', metric='tot_iou', mode='max'), resources_per_trial={'gpu': 0.05}, name="Full_experiment", local_dir="/home/alex/Alex_documents/RGCNN_git/data/experiments/")

    df = analysis.results_df
    best_config = analysis.get_best_config(metric="tot_iou", mode="max")
    print(best_config)

    # train(parameters)
