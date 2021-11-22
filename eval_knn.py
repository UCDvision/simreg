import builtins
from collections import Counter, OrderedDict
from random import shuffle
import argparse
import os
from os.path import join
import random
import time
import sys
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import faiss

from tools import *
from models.resnet import resnet18, resnet50
from models.resnet50x4 import Resnet50_X4 as resnet50x4
from models.mobilenet import MobileNetV2 as mobilenet
from models.resnet_byol import resnet50 as byol_resnet50
from eval_linear import load_weights


parser = argparse.ArgumentParser(description='NN evaluation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'imagenet100', 'imagenet-lt'],
                    help='use full or subset of the dataset')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', type=str, default='alexnet',
                    choices=['resnet18', 'resnet50', 'mobilenet',
                             'sup_resnet50', 'byol_resnet50'])
parser.add_argument('--use_pred', action='store_true',
                    help='use mlp prediction head atop projection head')
parser.add_argument('--linear_pred', action='store_true',
                    help='use linear prediction layer for student')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='./outputs', type=str,
                    help='experiment output directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--epoch', default=130, type=int,
                    help='epoch number of loaded model')
parser.add_argument('-k', default=1, type=int,
                    help='k in kNN')
parser.add_argument('--debug', action='store_true',
                    help='whether in debug mode or not')


def main():
    global logger

    args = parser.parse_args()
    if not os.path.exists(args.weights):
        sys.exit("Checkpoint does not exist!")
    makedirs(args.save)

    if not args.debug:
        logger = get_logger(
            logpath=os.path.join(args.save, 'knn_logs'),
            filepath=os.path.abspath(__file__)
        )
        def print_pass(*args):
            logger.info(*args)
        builtins.print = print_pass

    print(args)

    main_worker(args)


def get_mlp(hidden_dims, out_dim, n_layers):
    layers = []
    # hidden_dims - input and output dimensions of all layers except the final output dimension
    for i in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_dims[-1], out_dim))
    mlp = nn.Sequential(*layers)
    return mlp


def get_linear_proj(inp_dim, out_dim):
    mlp = nn.Sequential(
        nn.BatchNorm1d(inp_dim),
        nn.Linear(inp_dim, out_dim),
    )
    return mlp


def get_model(args):

    model = None
    if args.arch == 'resnet50x4':
        model = resnet50x4()
        checkpoint = torch.load(args.weights)
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        print(model)
        print(msg.missing_keys)

    elif args.arch == 'resnet18':
        model = resnet18()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        if args.use_pred:
            ft_dim = 512
            if not args.linear_pred:
                predict_q = get_mlp([ft_dim, ft_dim * 2, ft_dim, ft_dim * 2], 2048, 4)
            else:
                predict_q = get_linear_proj(ft_dim, 2048)
            predict_q = torch.nn.DataParallel(predict_q).cuda()
        if args.load_cache:
            print('Loading features from cache, network not loaded!!!')
        else:
            checkpoint = torch.load(args.weights)
            if 'model' in checkpoint:
                sd = checkpoint['model']
            else:
                sd = checkpoint['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
            sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
            sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
            sd = {('module.'+k): v for k, v in sd.items()}
            msg = model.load_state_dict(sd, strict=False)
            print(model)
            print(msg.missing_keys)
            if args.use_pred:
                sd = {k: v for k, v in sd.items() if 'predict_q' in k}
                sd = {k.replace('predict_q.', ''): v for k, v in sd.items()}
                msg = predict_q.load_state_dict(sd, strict=False)
                print(predict_q)
                print('missing keys: ', msg.missing_keys)
                model = nn.Sequential(model, predict_q)

    elif args.arch == 'mobilenet':
        model = mobilenet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        if args.use_pred:
            ft_dim = 1280
            if not args.linear_pred:
                predict_q = get_mlp([ft_dim, ft_dim * 2, ft_dim, ft_dim * 2], 2048, 4)
            else:
                predict_q = get_linear_proj(ft_dim, 2048)
            predict_q = torch.nn.DataParallel(predict_q).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {('module.' + k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg.missing_keys)
        if args.use_pred:
            sd = {k: v for k, v in sd.items() if 'predict_q' in k}
            sd = {k.replace('predict_q.', ''): v for k, v in sd.items()}
            msg = predict_q.load_state_dict(sd, strict=False)
            print(predict_q)
            print(msg.missing_keys)
            model = nn.Sequential(model, predict_q)

    elif args.arch == 'resnet50':
        model = resnet50()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {('module.'+k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg.missing_keys)

    elif args.arch == 'byol_resnet50':
        model = byol_resnet50()
        model.fc = nn.Sequential()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k: v for k, v in sd.items() if 'predict_q' not in k}
        sd = {k: v for k, v in sd.items() if 'queue' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        pdb.set_trace()
        msg = model.load_state_dict(sd, strict=True)
        print(model)
        print(msg)
        model = torch.nn.DataParallel(model).cuda()
    else:
        sys.exit('architecture not supported!!!')

    for param in model.parameters():
        param.requires_grad = False

    return model


class ImageFolderEx(datasets.ImageFolder) :
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        path = self.samples[index][0].split('/')[-1]
        return index, path, sample, target


def get_loaders(dataset_dir, bs, workers, dataset='imagenet', args=None):
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolderEx(traindir, augmentation)
    val_dataset = ImageFolderEx(valdir, augmentation)

    if dataset == 'imagenet100':
        subset_classes(train_dataset, num_classes=100)
        subset_classes(val_dataset, num_classes=100)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    return train_loader, val_loader


def main_worker(args):

    start = time.time()
    # Get train/val loader 
    # ---------------------------------------------------------------
    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers, args.dataset, args)

    # Create and load the model
    # If you want to evaluate your model, modify this part and load your model
    # ------------------------------------------------------------------------
    # MODIFY 'get_model' TO EVALUATE YOUR MODEL
    model = get_model(args)
    if args.use_pred:
        print('Using Prediction Head!!!')
        args.save = join(args.save, 'pred_head')
        makedirs(args.save)

    # ------------------------------------------------------------------------
    # Forward training samples throw the model and cache feats
    # ------------------------------------------------------------------------
    cudnn.benchmark = True

    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        print('load train feats from cache =>')
        train_feats, train_labels, train_inds = torch.load(cached_feats)
    else:
        print('get train feats =>')
        train_feats, train_labels, train_inds, train_feats_dict = get_feats(train_loader, model, args.print_freq)
        # Uncomment this to save the feats - prevents recalculating them if evaluated again
        # torch.save((train_feats, train_labels, train_inds), cached_feats, _use_new_zipfile_serialization=False)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats) and False:
        print('load val feats from cache =>')
        val_feats, val_labels, val_inds = torch.load(cached_feats)
    else:
        print('get val feats =>')
        val_feats, val_labels, val_inds, val_names = get_feats(val_loader, model, args.print_freq)
        # Uncomment this to save the feats - prevents recalculating them if evaluated again
        # torch.save((val_feats, val_labels, val_inds), cached_feats, _use_new_zipfile_serialization=False)

    # ------------------------------------------------------------------------
    # Calculate NN accuracy on validation set
    # ------------------------------------------------------------------------

    train_feats = l2_normalize(train_feats)
    val_feats = l2_normalize(val_feats)

    for k in [1, 20]:
        acc, D = faiss_knn(train_feats, train_labels, val_feats, val_labels, k)
        nn_time = time.time() - start
        np.savetxt(join(args.save, 'nn_%d_acc_epoch_%03d.txt' % (k, args.epoch)), [acc])
        print(k)
        print('=> time : {:.2f}m'.format(nn_time/60.))
        print(' * Acc {:.2f}'.format(acc))


def l2_normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def faiss_knn(feats_train, targets_train, feats_val, targets_val, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    D, I = gpu_index.search(feats_val, k)

    pred = np.zeros(I.shape[0], dtype=np.int)
    conf_mat = np.zeros((1000, 1000), dtype=np.int)
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i] = max(votes, key=lambda x: x[1])[0]
        conf_mat[targets_val[i], pred[i]] += 1

    acc = 100.0 * (pred == targets_val).mean()
    assert acc == (100.0 * (np.trace(conf_mat) / np.sum(conf_mat)))

    return acc, D


def get_feats(loader, model, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, indices, ptr = None, None, None, 0
    all_names = []

    with torch.no_grad():
        end = time.time()
        for i, (index, names, images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = model(images).cpu()
            cur_indices = index.cpu()
            all_names.append(names)

            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()
                indices = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            indices.index_copy_(0, inds, cur_indices)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(progress.display(i))

    return feats, labels, indices, all_names


def subset_classes(dataset, num_classes=10):
    np.random.seed(1234)
    all_classes = sorted(dataset.class_to_idx.items(), key=lambda x: x[1])
    subset_classes = [all_classes[i] for i in np.random.permutation(len(all_classes))[:num_classes]]
    subset_classes = sorted(subset_classes, key=lambda x: x[1])
    dataset.classes_to_idx = {c: i for i, (c, _) in enumerate(subset_classes)}
    dataset.classes = [c for c, _ in subset_classes]
    orig_to_new_inds = {orig_ind: new_ind for new_ind, (_, orig_ind) in enumerate(subset_classes)}
    dataset.samples = [(p, orig_to_new_inds[i]) for p, i in dataset.samples if i in orig_to_new_inds]


if __name__ == '__main__':
    main()
