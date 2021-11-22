exp='exp_name'
ep=130

CUDA_VISIBLE_DEVICES=0,1 python eval_linear.py\
    --arch 'resnet18'\
    --eval-freq 40\
    --weights $exp/checkpoints/ckpt_epoch_$ep.pth\
    --save $exp/linear \
    path/to/imagenet/dataset/root

echo $exp
echo 'epoch: '$ep
