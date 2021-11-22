exp='exp_name'
ep=130

CUDA_VISIBLE_DEVICES=0,1 python eval_knn.py\
    --arch 'resnet18'\
    --batch-size 512\
    --epoch $ep\
    -k 1\
    --weights $exp/checkpoints/ckpt_epoch_$ep.pth\
    --save $exp/features \
    path/to/imagenet/dataset/root

echo $exp
echo 'epoch: '$ep

