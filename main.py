import builtins
import os
from os.path import join
import sys
import time
import argparse
import random
import pdb
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from PIL import ImageFilter

from simreg import SimReg
from dataloader import get_train_loader
from tools import adjust_learning_rate, AverageMeterv2 as AverageMeter, subset_classes, get_logger


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100'],
                        help='use full or subset of the dataset')
    parser.add_argument('--base_dir', default='./',
                        help='experiment root directory')
    parser.add_argument('--exp', default='./outputs',
                        help='experiment root directory')
    parser.add_argument('--debug', action='store_true',
                        help='whether in debug mode or not')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=130,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9,
                        help='SGD momentum')

    # model definition
    parser.add_argument('--arch_teacher', type=str, default='resnet50',
                        choices=['resnet50', 'byol_resnet50', 'resnet50x4', 'sup_resnet50'])
    parser.add_argument('--arch_student', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'mobilenet', 'byol_resnet50'])
    parser.add_argument('--n_mlp_layers', type=int, default=4,
                        help='number of layers in prediction MLP head')
    parser.add_argument('--linear_pred', action='store_true',
                        help='use linear prediction layer for student')
    parser.add_argument('--use_cache', action='store_true',
                        help='use cached features for teacher instead of loading network')
    parser.add_argument('--teacher_fc', action='store_true',
                        help='use pretrained projection head for teacher')

    # Augmentations
    parser.add_argument('--single_aug', action='store_true',
                        help='use single augmentation (same aug for both nets)')
    parser.add_argument('--weak_strong', action='store_true',
                        help='whether to strong/strong or weak/strong augmentation')
    parser.add_argument('--weak_weak', action='store_true',
                        help='whether to use weak/weak augmentation')
    parser.add_argument('--mse_nonorm', action='store_true',
                        help='calculate mse loss from un-normalized vectors')

    # Load model
    parser.add_argument('--weights', type=str,
                        help='path to weights file to initialize the student model from')
    parser.add_argument('--teacher_weights', type=str,
                        help='path to weights(trained model) file to initialize the teacher model from')
    parser.add_argument('--teacher_feats', type=str,
                        help='path to stored teacher training features, used instead of loading weights')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--restart', action='store_true',
                        help='restart training using ckpt - do not load optim parameters')

    opt = parser.parse_args()

    return opt


def main():
    args = parse_option()

    save_dir = join(args.base_dir, 'exp')
    args.ckpt_dir = join(save_dir, args.exp, 'checkpoints')
    args.logs_dir = join(save_dir, args.exp, 'logs')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args_file = join(args.logs_dir, 'train_args.json')
    s = '*' * 50
    with open(args_file, 'a') as f:
        json.dump(s, f)
        json.dump(vars(args), f, indent=4)

    if not args.debug:
        os.environ['PYTHONBREAKPOINT'] = '0'
        logger = get_logger(
            logpath=os.path.join(args.ckpt_dir, 'logs'),
            filepath=os.path.abspath(__file__)
        )

        def print_pass(*arg):
            logger.info(*arg)
        builtins.print = print_pass

    print(args)

    train_loader = get_train_loader(args)

    simreg = SimReg(
        args,
        args.arch_teacher,
        args.arch_student,
        args.teacher_weights,
    )
    simreg.data_parallel()
    simreg = simreg.cuda()
    print(simreg)

    params = [p for p in simreg.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.learning_rate,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.start_epoch = 1

    if args.weights:
        print('==> load weights from checkpoint: {}'.format(args.weights))
        ckpt = torch.load(args.weights)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        if 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt['state_dict']
        msg = simreg.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        print(msg)

    if args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        msg = simreg.load_state_dict(ckpt['state_dict'], strict=True)
        print(msg)
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1

    # routine
    if args.use_cache:
        print('Using cached features!!!')

    time0 = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        print(args.exp)

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train(epoch, train_loader, simreg, optimizer, args)

        time2 = time.time()
        print('epoch {}, epoch time {:.2f}, total time {:.2f}'.format(epoch, (time2 - time1)/60.,
                                                                      (time2 - time0)/(60*60.)))

        # saving the model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': simreg.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.ckpt_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, simreg, optimizer, opt):
    """
    one epoch training for SimReg
    """
    simreg.train()
    if not opt.use_cache:
        simreg.encoder_t.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, (indices, names, (im_q, im_t), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)

        # ===================forward=====================
        loss = simreg(im_q=im_q, im_t=im_t, names=names)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss=loss_meter))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()
