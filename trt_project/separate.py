import torch
torch.cuda.set_device(0)
import os
import struct
import cv2
import json

import models
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--using_pruned_model', '-pruned', dest='pruned', action='store_true',
                    help='refine pruned model')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=20, type=int,
                    help='depth of the neural network')
parser.add_argument('--save', default=r'C:/Users/liujiwei/Desktop/Network-Slimming/logs/',
                    help='checkpoint__save')
parser.add_argument('--model_name', default="",
                    help='the name of the model')

args = parser.parse_args()

def read_cfg(savepath):
    with open(savepath, "r") as fp:
        data = [x.split()[0][1] if x == "'M'" else int(x) for x in fp.readlines()[1][1:-2].replace(",", "").split()]
    return data

def test(fileth):
    if os.path.exists(fileth)==False:
        os.makedirs(fileth)

def getWeights():
    if args.pruned:
        datanames = os.listdir(args.save)
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == '.txt':  # 目录下包含.txt的文件
                print(dataname)
                cfg = read_cfg(args.save+dataname)
        model = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset, cfg=cfg).cuda()
    else:
        model = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset).cuda()

    state_dict = torch.load(args.save + args.model_name + ".pth", map_location=lambda storage, loc: storage)["state_dict"]
    model.load_state_dict(state_dict)
    keys = [value for key, value in enumerate(state_dict)]
    weights = dict()
    for key in keys:
        weights[key] = state_dict[key]
    return weights, keys

def extract(weights,keys,weights_path):
    test(weights_path)
    for key in keys:
        value = weights[key]
        Shape = value.shape
        allsize = 1
        for idx in range(len(Shape)):
            allsize *= Shape[idx]

        Value = value.reshape(allsize)
        with open(weights_path + key + ".wts", "wb") as f:
            a = struct.pack("i", allsize)
            f.write(a)
            for i in range(allsize):
                a = struct.pack("f", Value[i])
                f.write(a)

def model_():
    if args.pruned:
        datanames = os.listdir(args.save)
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == '.txt':  # 目录下包含.txt的文件
                print(dataname)
                cfg = read_cfg(args.save+dataname)
                print(cfg)
        model = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset, cfg=cfg).cuda()
    else:
        model = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset).cuda()

    state_dict = torch.load(args.save + args.model_name + ".pth", map_location=lambda storage, loc: storage)["state_dict"]
    model.load_state_dict(state_dict)
    return model.cuda()

if __name__ == "__main__":
    weights,keys = getWeights()
    print(weights)
    print("---------------------")
    print(keys)
    extract(weights,keys,"D:/resnet50/weigths/pruned_resnet/")


    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dict1 = {}
    img = cv2.imread("0.jpg")
    blob = cv2.dnn.blobFromImage(img, 1.0, (32,32),(0.4914, 0.4822, 0.4465), swapRB=True)
    input = torch.Tensor(blob).cuda()
    model = model_()
    model.eval()
    output = model(input)
    pred = output.max(1)[1].detach().cpu().numpy()
    for v,k in enumerate(classes):
        dict1[v] = k
    print(output)
    print(pred)
    print(dict1[int(pred)])

