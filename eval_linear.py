import argparse
import os
import random
import time
import json
from os.path import join
import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from tools import *
from models.mobilenet import MobileNetV2
from models.resnet_byol import resnet50 as byol_resnet50

parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int,
                    help='evaluation frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', dest='weights', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                    help='lr drop schedule')
parser.add_argument('--use_cache', action='store_true',
                    help='use cached feats instead of backbone network')
parser.add_argument('--use_pred', action='store_true',
                    help='use mlp prediction head atop backbone')
parser.add_argument('--linear_pred', action='store_true',
                    help='use linear prediction head atop backbone')

best_acc1 = 0


def main():
    global logger

    args = parser.parse_args()
    if not os.path.exists(args.weights):
        sys.exit("Checkpoint does not exist!")

    if args.use_pred:
        print('Using Prediction Head!!!')
        args.save = join(args.save, 'pred_head')

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    args_file = join(args.save, 'train_args_linear.json')
    s = '*' * 50
    with open(args_file, 'a') as f:
        json.dump(s, f)
        json.dump(vars(args), f, indent=4)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

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


def load_weights(model, wts_path, args=None, predict_q=None):
    wts = torch.load(wts_path)
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    else:
        ckpt = wts

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    ckpt = {k.replace('encoder_q.', ''): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    msg = model.load_state_dict(state_dict)
    # print(model)
    print(msg)

    if args is not None:
        if args.use_pred:
            ckpt = {k: v for k, v in ckpt.items() if 'predict_q' in k}
            ckpt = {k.replace('predict_q.', ''): v for k, v in ckpt.items()}
            msg = predict_q.load_state_dict(ckpt, strict=False)
            print(predict_q)
            print(msg.missing_keys)


def get_linear_proj(inp_dim, out_dim):
    mlp = nn.Sequential(
        nn.BatchNorm1d(inp_dim),
        nn.Linear(inp_dim, out_dim),
    )
    return mlp


def get_model(arch, wts_path, args=None):
    if 'byol_resnet50' in arch:
        model = byol_resnet50()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif arch == 'mobilenet':
        model = MobileNetV2()
        model.fc = nn.Sequential()
        if args.use_pred:
            ft_dim = 1280
            if args.linear_pred:
                predict_q = get_linear_proj(ft_dim, 2048)
            else:
                predict_q = get_mlp([ft_dim, ft_dim * 2, ft_dim, ft_dim * 2], 2048, 4)
            load_weights(model, wts_path, args=args, predict_q=predict_q)
            model = nn.Sequential(model, predict_q)
        else:
            load_weights(model, wts_path)
    elif 'resnet' in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        if args.use_pred:
            ft_dim = 512
            if args.linear_pred:
                predict_q = get_linear_proj(ft_dim, 2048)
            else:
                predict_q = get_mlp([ft_dim, ft_dim * 2, ft_dim, ft_dim * 2], 2048, 4)
            load_weights(model, wts_path, args=args, predict_q=predict_q)
            model = nn.Sequential(model, predict_q)
        else:
            load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


class ImageFolderEx(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        path = self.samples[index][0].split('/')[-1]
        return path, sample, target


def main_worker(args):
    global best_acc1

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    if not args.use_cache:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    val_dataset = ImageFolderEx(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    train_val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    backbone = get_model(args.arch, args.weights, args=args)
    backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()

    cached_varmean = '%s/var_mean.pth.tar' % args.save
    cached_feat = '%s/train_feat.pth.tar' % args.save
    if not os.path.exists(cached_varmean):
        train_feats, train_labels = get_feats(train_val_loader, backbone, args)
        train_var, train_mean = torch.var_mean(train_feats, dim=0)
        torch.save((train_var, train_mean), cached_varmean)
        torch.save((train_feats, train_labels), cached_feat)
    else:
        train_var, train_mean = torch.load(cached_varmean)
        if args.use_cache:
            train_feats, train_labels = torch.load(cached_feat)
            print('loaded cache')

    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(get_channels(args.arch, args.use_pred), len(train_dataset.classes)),
    )
    linear = linear.cuda()

    optimizer = torch.optim.SGD(linear.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            msg = linear.load_state_dict(checkpoint['state_dict'])
            print(msg)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        _, acc_dict = validate(val_loader, backbone, linear, args)
        return

    st_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.use_cache:
            train(train_loader, backbone, linear, optimizer, epoch, args)
        else:
            train_cached(train_feats, train_labels, linear, optimizer, epoch, args)

        # evaluate on validation set
        if (epoch % args.eval_freq == 0) or (epoch == args.epochs - 1):
            acc1, _ = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()
        logger.info('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, is_best, args.save)
    sp_time = time.time()
    print('time: ', (sp_time - st_time) / 60.)


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch, pred=False):
    if arch == 'resnet50':
        c = 2048
    elif arch == 'byol_resnet50':
        c = 2048
    elif arch == 'resnet18':
        c = 512
        if pred:
            c = 2048
    elif arch == 'mobilenet':
        c = 1280
        if pred:
            c = 2048
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


def train_cached(train_feats, train_labels, linear, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_feats)//args.batch_size,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    # backbone.eval()
    linear.train()

    end = time.time()
    # for i, (images, target) in enumerate(train_loader):
    n_batches = len(train_feats) // args.batch_size
    indices = torch.arange(len(train_feats))
    perm = torch.randperm(len(train_feats))
    indices = indices[perm]
    # for i, idx in enumerate(indices):
    for idx in range(n_batches):
    # measure data loading time
        data_time.update(time.time() - end)

        ids = indices[(idx * args.batch_size): (idx + 1) * args.batch_size]
        output = train_feats[ids].cuda()
        target = train_labels[ids].cuda()
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), ids.size(0))
        top1.update(acc1[0], ids.size(0))
        top5.update(acc5[0], ids.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            logger.info(progress.display(idx))


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    feats_dict = {}

    with torch.no_grad():
        end = time.time()
        for i, (names, images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # store acc for each input
            for idx in range(len(output)):
                feats_dict[names[idx]] = output[idx].detach().cpu()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, feats_dict


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def get_feats(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = normalize(model(images)).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == '__main__':
    main()