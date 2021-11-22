# SimReg: A Simple Regression Based Framework for Self-supervised Knowledge Distillation

Source code for the [paper](https://www.bmvc2021-virtualconference.com/assets/papers/1137.pdf) "SimReg: Regression as a Simple Yet Effective Tool for Self-supervised Knowledge Distillation".\
Paper accepted at _British Machine Vision Conference (BMVC), 2021_

## Overview

We present a simple framework to improve performance of regression based knowledge distillation from self-supervised teacher networks. The teacher is trained using a standard self-supervised learning (SSL) technique. The student network is then trained to directly regress the teacher features (using MSE loss on normalized features). Importantly, the student architecture contains an additional multi-layer perceptron (MLP) head atop the CNN backbone during the distillation (training) stage. A deeper architecture provides the student higher capacity to predict the teacher representations. This additional MLP head can be removed during inference without hurting downstream performance. This is especially surprising since only the output of the MLP is trained to mimic the teacher and the backbone CNN features have a high MSE loss with the teacher features. This observation allows us to obtain better student models by using deeper models during distillation without altering the inference architecture. The train and test stage architectures are shown in the figure below.

![](arch_reg.png)

## Requirements

All our experiments use the PyTorch library. We recommend installing the following package versions:
- python=3.7.6
- pytorch=1.4
- torchvision=0.5.0
- faiss-gpu=1.6.1 (required for k-NN evaluation alone)

Instructions for PyTorch installation can be found [here](https://pytorch.org/). 
GPU version of the FAISS package is necessary for k-NN evaluation of trained models. It can be installed using the following command:
```shell
pip install faiss-gpu
```

## Dataset

We use the ImageNet-1k dataset in our experiments. Download and prepare the dataset using the [PyTorch ImageNet training example code](https://github.com/pytorch/examples/tree/master/imagenet). The dataset path needs to be set in the bash scripts used for training and evaluation.

## Training

Distillation can be performed by running the following command:
```shell
bash run.sh
```
The defualt hyperparameters values are set to ones used in the paper. Modify the teacher and student architectures as necessary. Set the approapriate paths for the ImageNet dataset root and the experiment root. The current code will generate a directory named ```exp_dir``` containing ```checkpoints``` and ```logs``` sub-directories.

## Evaluation

Set the experiment name and checkpoint epoch in the evaluation bash scripts. The trained checkpoints are assumed to be stored as ```exp_dir/checkpoints/ckpt_epoch_<num>.pth```. Edit the ```weights``` argument to load model parameters from a custom checkpoint. 

### k-NN Evaluation

k-NN evaluation requires FAISS-GPU package installation. We evaluate the performance of the CNN backbone features. Run k-NN evaluation using:
```shell
bash knn_eval.sh
```
The image features and results for k-NN (k=1 and 20) evaluation are stored in ```exp_dir/features/``` path. 

### Linear Evaluation

Here, we train a single linear layer atop the CNN backbone using an SGD optimizer for 40 epochs. The evaluation can be performed using the following code:
```shell
bash lin_eval.sh
```
The evaluation results are stored in ```exp_dir/linear/``` path. Set the ```use_cache``` argument in the bash script to use cached features for evaluation. Using this argument will result in a single round of feature calculation for caching and 40 epochs of linear layer training using the cached features. While it usually results in slightly reduced performance, it can be used for faster evaluation of intermediate checkpoints.
