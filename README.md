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
     
