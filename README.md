# Network-Slimming
Network reproduction and expansion based on Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017).
This repository contains an official pytorch implementation for the following paper
Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017).
Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang.

Original implementation: slimming in Torch.
The code is based on pytorch-slimming. i support for ResNet and DenseNet.

Dependencies：
     pytorch v1.6, torchvision v0.7
     
Channel Selection Layer：
     I introduce channel selection layer to help the pruning of ResNet and DenseNet. This layer is easy to implement. It stores a parameter indexes which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.
     
Baseline：
     The dataset argument specifies which dataset to use: cifar10 or cifar100. The arch argument specifies the architecture to use: vgg,resnet or densenet. The depth is chosen to be the same as the networks used in the paper.
     
python main.py --dataset cifar10 --arch vgg --depth 19
python main.py --dataset cifar10 --arch resnet --depth 164
python main.py --dataset cifar10 --arch densenet --depth 40

Train with Sparsity:
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40
