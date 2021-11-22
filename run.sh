#!/usr/bin/env bash

set -x
set -e

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --base_dir ./ \
  --exp exp_name \
  --arch_teacher resnet50\
  --arch_student resnet18\
  --n_mlp_layers 4\
  --learning_rate 0.05 \
  --epochs 130 \
  --single_aug\
  --weak_weak\
  --save_freq 10 \
  --teacher_weights path/to/teacher/weights \
  path/to/imagenet/dataset/root

