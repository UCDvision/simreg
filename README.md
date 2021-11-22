# SimReg: A Simple Regression Based Framework for Self-supervised Knowledge Distillation

Source code for the paper "SimReg: Regression as a Simple Yet Effective Tool for Self-supervised Knowledge Distillation".\
Paper accepted at _British Machine Vision Conference (BMVC), 2021_

# Overview

We present a simple framework to improve performance of regression based knowledge distillation from self-supervised teacher networks. The teacher is trained using a standard self-supervised learning (SSL) technique. The student network is then trained to directly regress the teacher features (using MSE loss on normalized features). Importantly, the student architecture contains an additional multi-layer perceptron (MLP) head atop the CNN backbone during the distillation (training) stage. A deeper architecture provides the student higher capacity to predict the teacher representations. This additional MLP head can be removed during inference without hurting downstream performance. This is especially surprising since only the output of the MLP is trained to mimic the teacher and the backbone CNN features have a high MSE loss with the teacher features. 
