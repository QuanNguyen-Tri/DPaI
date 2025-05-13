
# add --wandb into the command if you want to save results with wandb
wandb_project=DPaI # project name
wandb_dir=/wandb/ # wandb save dir
wandb_key=123abcd4567xyz # wandb login

data_dir=/data/ # read data from
dataset=cifar10 # cifar10, cifar100, tiny-imagenet, imagenet
model=ResNet20 # ResNet20, vgg19_bn, ResNet18, ViT


alpha=0.99 # trade-off between node and path
beta=0.5 # trade-off between kernel and node
lr_score=0.005 # learning rate to update score parameters, during pruning
compression=2.5 # higher compression, higher sparsity / Compression: 0.5, 1.0, 1.5, 2.0 -> Sparisty: 68.37%, 90%, 96.37%, 99%
seed=0
num_steps=3000 # number of pruning step

CUDA_VISIBLE_DEVICES=0 python main.py --model $model --data $dataset --method npb --heuristic none --ablation none \
    --compression $compression --lr_score $lr_score --num_steps $num_steps --alpha $alpha --beta $beta --seed $seed \
    --epochs 160 --lr-drops 80 120 --lr 0.1 --l2 1e-4 --optimizer sgd --device cuda --amp --data_dir $data_dir \
    --wandb --wandb_project $wandb_project --wandb_dir $wandb_dir --wandb_key $wandb_key
